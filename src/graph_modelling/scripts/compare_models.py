import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from codecarbon import EmissionsTracker
import time

# --- Add project root to sys.path ---
BASE_DIR = Path.cwd()
MODEL_PATH = BASE_DIR / "results" / "models"
DATA_DIR = BASE_DIR / "data" / "data_gnn"
ALL_DIR = DATA_DIR / "all"
RAW_DATA_DIR = BASE_DIR / "data" / "data_raw"
# ------------------------------------

from src.graph_modelling.datasets.no2_dataset import NO2DatasetLoader
from src.graph_modelling.training.train_utils import train_model_index, evaluate_index
from src.graph_modelling.visualization.visualization import (
    plot_training_history,
    plot_predictions,
    set_base_dir,
)
import inspect

# A logger for this file
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def initialize_model(
    cfg: DictConfig,
    device: torch.device,
    edges: torch.Tensor,
    edge_weights: torch.Tensor | None,
):
    """Initializes the model based on the configuration."""
    # Find out which model we're creating for better logging
    model_name = cfg.model._target_.split(".")[-1]
    data_mode = "no2_only" if cfg.data.only_no2 else "all_vars"

    # Create a friendly display name for logging and file naming
    friendly_model_name = f"{model_name}_{data_mode}"

    log.info(f"Initializing model: {model_name} with {data_mode} data")

    try:
        # Get model class to inspect its parameters
        model_class = hydra.utils.get_class(cfg.model._target_)

        # Get the parameters accepted by the model's __init__ method
        valid_params = list(inspect.signature(model_class.__init__).parameters.keys())
        if "self" in valid_params:
            valid_params.remove("self")

        # Make sure num_vars matches your data configuration
        num_nodes = 3  # Assuming 3 cities for this dataset
        num_vars = 1 if cfg.data.only_no2 else 7
        log.info(
            f"Setting num_vars={num_vars} based on cfg.data.only_no2={cfg.data.only_no2}"
        )

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
        log.info(f"Instantiating {model_name} with parameters: {model_params}")

        # Instantiate with only the parameters this model accepts
        model = hydra.utils.instantiate(
            cfg.model,
            **model_params,
            _recursive_=False,
        ).to(device)
        model = model.float()

        # Ensure correct dtype for edges and weights
        edges = edges.long()
        if edge_weights is not None:
            edge_weights = edge_weights.float()

        log.info(f"Model:\n{model}")
        return model, edges, edge_weights, friendly_model_name

    except Exception as e:
        log.error(f"Error instantiating model: {e}")
        raise e


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> float:
    """Runs a single experiment configuration."""
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

    # Set random seed
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    log.info(f"Random seed set to {cfg.seed}")

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

    log.info(
        f"Loading dataset with lags={cfg.training.n_lags}, horizon={cfg.training.n_horizon}..."
    )

    # Check if time-based splitting is enabled
    use_time_split = cfg.data.get("use_time_split", False)

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
            use_time_split=use_time_split,
            dask_batching=False,
            target_offset=cfg.data.target_offset,
        )
    )

    log.info(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    skip_training = cfg.training.get("skip_training", False)
    if skip_training:
        # Check if we need to load pretrained model
        model, edges, edge_weights, friendly_model_name = initialize_model(
            cfg, device, edges, edge_weights
        )

        history = {}  # Initialize history
        load_path_str = cfg.training.get("load_model_path", None)  # Use .get for safety

        if load_path_str:
            load_path = Path(load_path_str)
            # Resolve relative paths against the base directory if needed
            if not load_path.is_absolute():
                # Decide base path: project root (BASE_DIR) or hydra output (output_dir)?
                # Using BASE_DIR seems more robust for predefined models.
                load_path = BASE_DIR / load_path

            if load_path.exists():
                log.info(f"Attempting to load pretrained model from: {load_path}")
                try:
                    model.load_state_dict(torch.load(load_path, map_location=device))
                    log.info(f"Successfully loaded pretrained model: {load_path}")
                    skip_training = True
                except Exception as e:
                    log.error(
                        f"Failed to load model state from {load_path}: {e}. Proceeding with training."
                    )
            else:
                log.warning(
                    f"Pretrained model path specified but not found: {load_path}. Proceeding with training."
                )
        else:
            log.info("No pretrained model path specified. Training from scratch.")

    else:
        # --- Model Initialization ---
        # Assuming 3 nodes for this specific dataset (Amsterdam, Rotterdam, Utrecht)

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
            valid_params = list(
                inspect.signature(model_class.__init__).parameters.keys()
            )
            if "self" in valid_params:
                valid_params.remove("self")

            # Make sure num_vars matches your data configuration
            num_nodes = 3
            num_vars = 1 if cfg.data.only_no2 else 7

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

            # Instantiate with only the parameters this model accepts
            model = hydra.utils.instantiate(
                cfg.model,
                **model_params,
                _recursive_=False,
            ).to(device)
            model = model.float()

            edges = edges.long()

            if edge_weights is not None:
                edge_weights = edge_weights.float()  # Edge weights should be float32

            log.info(f"Model:\n{model}")
        except Exception as e:
            log.error(f"Error instantiating model: {e}")
            raise e

        # --- Training ---
        log.info(f"Starting training for {friendly_model_name}...")
        tb_log_dir = output_dir / cfg.paths.tensorboard_subdir
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))

        try:
            with EmissionsTracker(log_level="error") as train_tracker:
                model, history = train_model_index(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    edge_index=edges.long().to(device),  # Ensure long dtype
                    edge_weight=edge_weights.float().to(device)
                    if edge_weights is not None
                    else None,  # Ensure float dtype
                    device=device,
                    epochs=cfg.training.n_epochs,
                    patience=cfg.training.patience,
                    writer=writer,
                    lambda_max=lambda_max,
                )

            # Plot training history with model name
            plots_dir = output_dir / cfg.paths.plot_subdir
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Use the visualization module's plotting with model name
            set_base_dir(
                Path(cfg.base_dir)
            )  # Set base directory for visualization module
            plot_training_history(history, model_name=friendly_model_name)
            log.info(f"Training history plot created for {friendly_model_name}")

        except KeyboardInterrupt:
            log.warning("Training interrupted by user.")
            history = {}  # Assign empty history

        finally:
            writer.close()

    # --- Evaluation ---
    city_names = ["Amsterdam", "Utrecht", "Rotterdam"]

    # Run multiple evaluations for timing and energy statistics
    n_eval_runs = cfg.optuna.n_eval_runs  # Number of evaluation runs
    inference_times = []
    inference_energies = []
    inference_emissions = []

    # Store first run results for visualization
    first_predictions = None
    first_targets = None
    first_test_loss = None
    log.info(f"Starting evaluation for {friendly_model_name} for {n_eval_runs} runs...")

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
        inference_times.append(run_time)
        inference_energies.append(run_tracker.final_emissions_data.energy_consumed)
        inference_emissions.append(run_tracker.final_emissions_data.emissions)

        # Store first run results for metrics and visualization
        if run == 0:
            first_predictions = predictions
            first_targets = targets
            first_test_loss = test_loss

    # Calculate statistics
    inference_time_mean = np.mean(inference_times)
    inference_time_std = np.std(inference_times)
    inference_energy_mean = np.mean(inference_energies)
    inference_energy_std = np.std(inference_energies)
    inference_emissions_mean = np.mean(inference_emissions)
    inference_emissions_std = np.std(inference_emissions)

    log.info(f"Test Loss for {friendly_model_name}: {first_test_loss:.4f}")
    log.info(
        f"Inference time: {inference_time_mean:.4f}s ± {inference_time_std:.4f}s (over {n_eval_runs} runs)"
    )
    log.info(
        f"Inference energy: {inference_energy_mean:.6f}kWh ± {inference_energy_std:.6f}kWh"
    )

    # Calculate additional metrics using first run results
    mse = first_test_loss
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(first_predictions - first_targets))

    if not skip_training:
        train_emissions = train_tracker.final_emissions_data

        metrics = {
            "inference_time_s": inference_time_mean,
            "inference_time_std_s": inference_time_std,
            "inference_runs": n_eval_runs,
            "inference_energy_kWh": inference_energy_mean,
            "inference_energy_std_kWh": inference_energy_std,
            "inference_emissions_gCO2": inference_emissions_mean,
            "inference_emissions_std_gCO2": inference_emissions_std,
            "training_energy_kWh": train_emissions.energy_consumed,
            "training_emissions_gCO2": train_emissions.emissions,
        }
    else:
        metrics = {
            "inference_time_s": inference_time_mean,
            "inference_time_std_s": inference_time_std,
            "inference_runs": n_eval_runs,
            "inference_energy_kWh": inference_energy_mean,
            "inference_energy_std_kWh": inference_energy_std,
            "inference_emissions_gCO2": inference_emissions_mean,
            "inference_emissions_std_gCO2": inference_emissions_std,
            "training_energy_kWh": None,
            "training_emissions_gCO2": None,
        }

    log.info(
        f"Test metrics for {friendly_model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}"
    )

    # Create prediction plots with model name and city-specific metrics
    if hasattr(loader, "denormalize_no2"):
        try:
            # Try to denormalize if the loader supports it
            predictions_orig = loader.denormalize_no2(first_predictions)
            targets_orig = loader.denormalize_no2(first_targets)
            log.info("Successfully denormalized predictions and targets")

            # Use visualization module's plot_predictions with model name
            plot_metrics = plot_predictions(
                predictions_orig,
                targets_orig,
                city_names=city_names,
                model_name=friendly_model_name,
                save_metrics=True,
                use_plotly=True,
                energy_metrics=metrics,  # Pass the energy metrics here
                BASE_DIR=BASE_DIR,
            )

            if plot_metrics:
                log.info(
                    "Prediction plots and detailed metrics created with per-city RMSE"
                )

        except Exception as e:
            log.warning(f"Could not create denormalized prediction plots: {e}")
    else:
        log.warning(
            "Loader does not support denormalization, skipping prediction plots"
        )

    # Save final model state
    model_path = output_dir / cfg.paths.model_save_name
    torch.save(model.state_dict(), model_path)
    log.info(f"Final model saved to {model_path}")

    return rmse


if __name__ == "__main__":
    os.environ["HYDRA_CONFIG_PATH"] = str(Path(__file__).parent / "conf")

    base_dir = BASE_DIR
    OmegaConf.update(OmegaConf.create(), "base_dir", str(base_dir))

    run_experiment()
