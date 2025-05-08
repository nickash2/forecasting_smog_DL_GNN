import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
from pathlib import Path
import json
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
from codecarbon import EmissionsTracker
from tqdm.auto import tqdm

# --- Add project root to sys.path ---
BASE_DIR = Path.cwd()
MODEL_PATH = BASE_DIR / "results" / "models"
DATA_DIR = BASE_DIR / "data" / "data_gnn"
ALL_DIR = DATA_DIR / "all"
RAW_DATA_DIR = BASE_DIR / "data" / "data_raw"
# ------------------------------------


from src.graph_modelling.datasets.no2_dataset import NO2DatasetLoader
from src.graph_modelling.training.train_utils import evaluate_index
import random

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_inference(cfg: DictConfig) -> float:
    """Runs inference on test set with pretrained model weights."""
    # Check if model config loaded correctly
    if cfg.model is None or "_target_" not in cfg.model:
        log.error(
            f"Failed to load model configuration. Available config: {OmegaConf.to_yaml(cfg)}"
        )
        if hasattr(cfg, "model") and cfg.model is not None:
            log.error(f"Model config exists but doesn't contain _target_: {cfg.model}")
        else:
            log.error("Model config is None")
        raise ValueError("Model configuration not found or invalid")

    # --- Setup ---
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    log.info(f"Hydra output directory: {output_dir}")
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Data Loading ---
    log.info("Initializing dataset loader...")
    # Get the correct path to the cache file
    cache_path = Path(cfg.data.cache_file)
    if not cache_path.is_absolute():
        cache_path = BASE_DIR / "data" / "data_gnn" / cfg.data.cache_file

    loader = NO2DatasetLoader(
        index=True,
        only_no2=cfg.data.only_no2,
        force_reload=cfg.data.force_reload,
        cache_file=str(cache_path),
        logger=log,
    )

    # Set all random seeds
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)  # Python's built-in random
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)  # Python hash seed

    # For reproducible CUDA operations (if available)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log.info(f"Setting random seed to {cfg.seed}")

    log.info(
        f"Loading dataset with lags={cfg.training.n_lags}, horizon={cfg.training.n_horizon}..."
    )
    train_loader, val_loader, test_loader, edges, edge_weights, lambda_max = (
        loader.get_index_dataset(
            lags=cfg.training.n_lags,
            batch_size=cfg.training.batch_size,
            shuffle=cfg.data.shuffle,
            allGPU=device,
            ratio=list(cfg.data.ratio),
            only_no2=cfg.data.only_no2,
            sample_size=cfg.data.sample_size,
            horizon=cfg.training.n_horizon,
            cache=True,
            step_size=cfg.training.n_step,
        )
    )
    log.info(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    # --- Model Initialization ---

    # Find out which model we're creating for better logging
    model_name = cfg.model._target_.split(".")[-1]
    data_mode = "no2_only" if cfg.data.only_no2 else "all_vars"

    # Create a friendly display name for logging and file naming
    friendly_model_name = f"{model_name}_{data_mode}"

    log.info(f"Initializing model: {model_name} with {data_mode} data")

    # In the model initialization section
    try:
        # Get model class to inspect its parameters
        model_class = hydra.utils.get_class(cfg.model._target_)
        import inspect

        # Get the parameters accepted by the model's __init__ method
        valid_params = list(inspect.signature(model_class.__init__).parameters.keys())
        if "self" in valid_params:
            valid_params.remove("self")

        # Make sure num_vars matches your data configuration
        num_nodes = 3
        num_vars = 1 if cfg.data.only_no2 else 7
        print("num_vars", num_vars)

        # Create dictionary with common parameters
        model_params = {
            "num_nodes": num_nodes,
            "num_vars": num_vars,
            "lags": cfg.training.n_lags,
            "horizon": cfg.training.n_horizon,
        }

        # Add parameters from cfg.model (like K, hidden_channels) ONLY if the model accepts them
        for k, v in cfg.model.items():
            if k != "_target_" and k in valid_params:
                model_params[k] = v

        # Filter to only include parameters this model accepts
        model_params = {k: v for k, v in model_params.items() if k in valid_params}

        # Instantiate the model
        model = hydra.utils.instantiate(
            cfg.model,
            **model_params,
            _recursive_=False,
        ).to(device)
        model = model.float()

        log.info(f"Model:\n{model}")

        # Load model weights
        # Check if the weights file exists
        if hasattr(cfg, "weights_file") and cfg.weights_file:
            weights_path = Path(cfg.weights_file)
            if not weights_path.is_absolute():
                weights_path = MODEL_PATH / cfg.weights_file

            log.info(f"Loading model weights from: {weights_path}")

            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights file not found: {weights_path}")

            model.load_state_dict(torch.load(weights_path, map_location=device))
            log.info(f"Successfully loaded weights from {weights_path}")
        else:
            raise ValueError("No weights file specified in configuration")

    except Exception as e:
        log.error(f"Error loading model or weights: {e}")
        raise e

    # --- Inference ---
    log.info(f"Starting inference for {friendly_model_name}...")
    city_names = ["Amsterdam", "Rotterdam", "Utrecht"]

    # Run multiple evaluations for timing and energy statistics
    n_eval_runs = cfg.get("n_eval_runs", 10)  # Default to 10 runs if not specified
    inference_times = []
    inference_energies = []
    inference_emissions = []

    # For calculating per-sample inference time
    total_test_samples = len(test_loader) * cfg.training.batch_size

    # Store first run results for visualization
    first_predictions = None
    first_targets = None
    first_test_loss = None

    log.info(f"Starting evaluation for {friendly_model_name} for {n_eval_runs} runs...")

    # Set model to evaluation mode
    model.eval()

    for run in range(n_eval_runs):
        with EmissionsTracker(log_level="error") as run_tracker:
            start_time = time.time()
            test_loss, predictions, targets = evaluate_index(
                model=model,
                test_loader=test_loader,
                edge_index=edges.to(device),
                edge_weight=edge_weights.to(device)
                if edge_weights is not None
                else None,
                device=device,
                loader=loader,
                cities=city_names,
                lambda_max=lambda_max,
            )
            end_time = time.time()

        # Store timing and energy data
        run_time = end_time - start_time
        per_sample_time = (
            run_time / total_test_samples
        )  # Time for one 24-hour prediction

        inference_times.append(run_time)
        inference_energies.append(run_tracker.final_emissions_data.energy_consumed)
        inference_emissions.append(run_tracker.final_emissions_data.emissions)

        log.info(
            f"Run {run + 1}/{n_eval_runs} - Time: {run_time:.4f}s, Per-sample: {per_sample_time:.6f}s"
        )

        # Store first run results for metrics and visualization
        if run == 0:
            first_predictions = predictions
            first_targets = targets
            first_test_loss = test_loss

    # Calculate statistics
    inference_time_mean = np.mean(inference_times)
    inference_time_std = np.std(inference_times)
    per_sample_time_mean = inference_time_mean / total_test_samples
    per_sample_time_std = inference_time_std / total_test_samples
    inference_energy_mean = np.mean(inference_energies)
    inference_energy_std = np.std(inference_energies)
    inference_emissions_mean = np.mean(inference_emissions)
    inference_emissions_std = np.std(inference_emissions)

    log.info(f"Test Loss for {friendly_model_name}: {first_test_loss:.4f}")
    log.info(
        f"Inference time: {inference_time_mean:.4f}s ± {inference_time_std:.4f}s (over {n_eval_runs} runs)"
    )
    log.info(
        f"Per-sample inference time: {per_sample_time_mean:.6f}s ± {per_sample_time_std:.6f}s"
    )
    log.info(
        f"Inference energy: {inference_energy_mean:.6f}kWh ± {inference_energy_std:.6f}kWh"
    )

    # Calculate additional metrics using first run results
    mse = first_test_loss
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(first_predictions - first_targets))

    # Create metrics dictionary
    metrics = {
        "test_loss": float(first_test_loss),
        "rmse": float(rmse),
        "mae": float(mae),
        "inference_time_s": float(inference_time_mean),
        "inference_time_std_s": float(inference_time_std),
        "per_sample_time_s": float(per_sample_time_mean),
        "per_sample_time_std_s": float(per_sample_time_std),
        "inference_runs": n_eval_runs,
        "inference_energy_kWh": float(inference_energy_mean),
        "inference_energy_std_kWh": float(inference_energy_std),
        "inference_emissions_gCO2": float(inference_emissions_mean),
        "inference_emissions_std_gCO2": float(inference_emissions_std),
    }

    # Print key metrics
    log.info(f"Test metrics for {friendly_model_name}:")
    log.info(f"  MSE: {mse:.4f}")
    log.info(f"  RMSE: {rmse:.4f}")
    log.info(f"  MAE: {mae:.4f}")

    # Denormalize predictions if possible
    if hasattr(loader, "denormalize_no2"):
        try:
            # Try to denormalize if the loader supports it
            predictions_orig = loader.denormalize_no2(first_predictions)
            targets_orig = loader.denormalize_no2(first_targets)
            log.info("Successfully denormalized predictions and targets")

            # Save denormalized predictions
            predictions_to_save = predictions_orig
            targets_to_save = targets_orig
        except Exception as e:
            log.warning(f"Could not denormalize predictions and targets: {e}")
            predictions_to_save = first_predictions
            targets_to_save = first_targets
    else:
        log.warning("Loader does not support denormalization")
        predictions_to_save = first_predictions
        targets_to_save = first_targets

    # Assuming your predictions and targets have shape [time, city]
    # If not, reshape them first
    if len(predictions_to_save.shape) == 1:
        n_cities = len(city_names)
        n_time = len(predictions_to_save) // n_cities
        predictions_to_save = predictions_to_save.reshape(n_time, n_cities)
        targets_to_save = targets_to_save.reshape(n_time, n_cities)

    # Create a dictionary with city-specific data
    city_data = {}
    for i, city in enumerate(city_names):
        city_data[city] = {
            "predictions": predictions_to_save[:, i].tolist(),
            "targets": targets_to_save[:, i].tolist(),
        }

    # Create results dictionary with better structure
    results = {
        "model_name": friendly_model_name,
        "city_data": city_data,
        "metrics": metrics,
        "city_names": city_names,
    }

    # Save results to JSON
    results_path = output_dir / f"{friendly_model_name}_inference_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    log.info(f"Inference results saved to {results_path}")

    # Return primary metric for Hydra sweep comparison
    return per_sample_time_mean


if __name__ == "__main__":
    os.environ["HYDRA_CONFIG_PATH"] = str(Path(__file__).parent / "conf")

    # Set base_dir to project root or another appropriate location
    base_dir = BASE_DIR
    OmegaConf.update(OmegaConf.create(), "base_dir", str(base_dir))

    run_inference()
