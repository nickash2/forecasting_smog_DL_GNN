import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
from pathlib import Path
import json
import logging
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
import numpy as np
import optuna
from tqdm.auto import tqdm
import sqlite3

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
from src.graph_modelling.training.optuna_utils import (
    setup_optuna_study,
    create_objective,
    save_study_results,
)

# A logger for this file
log = logging.getLogger(__name__)


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

    torch.manual_seed(cfg.seed)
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
    )

    log.info(
        f"Loading dataset with lags={cfg.training.n_lags}, horizon={cfg.training.n_horizon}..."
    )
    train_loader, val_loader, test_loader, edges, edge_weights = (
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

        # Store model creation parameters for optuna if needed
        model_creator_fn = lambda **params: hydra.utils.instantiate(
            cfg.model,
            **params,
            _recursive_=False,
        )

        edges = edges.long()
        if edge_weights is not None:
            edge_weights = edge_weights.float()  # Edge weights should be float32

        # Check if we should run Optuna optimization
        if hasattr(cfg, "optuna") and cfg.optuna.get("enabled", False):
            log.info(
                f"Starting Optuna hyperparameter optimization for {friendly_model_name}"
            )

            # Create study dir
            optuna_dir = output_dir / cfg.paths.optuna_subdir
            optuna_dir.mkdir(parents=True, exist_ok=True)

            # Ensure storage path is absolute
            storage_path = cfg.optuna.storage
            if not storage_path.startswith(
                ("sqlite:///", "mysql://", "postgresql:///")
            ):
                storage_path = f"sqlite:///{output_dir / storage_path}"

            # Set up study
            study = setup_optuna_study(
                study_name=f"{cfg.optuna.study_name}_{friendly_model_name}",
                storage=storage_path,
                direction=cfg.optuna.direction,
                load_if_exists=True,
                epochs=cfg.training.n_epochs,
            )

            # Create objective function
            base_config = {
                "model_params": model_params,
                "training": {
                    "n_epochs": cfg.training.n_epochs,
                    "patience": cfg.training.patience,
                },
            }

            objective = create_objective(
                model_fn=model_creator_fn,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                edge_index=edges,
                edge_weight=edge_weights,
                device=device,
                model_name=friendly_model_name,
                base_cfg=base_config,
                output_dir=optuna_dir,
            )

            log.info(f"Running {cfg.optuna.n_trials} trials for {friendly_model_name}")

            # Create a tqdm progress bar for Optuna trials
            with tqdm(
                total=cfg.optuna.n_trials,
                desc=f"Optuna trials for {friendly_model_name}",
            ) as pbar:
                # Define callback to update progress bar after each trial
                def tqdm_callback(study, trial):
                    pbar.update(1)
                    if (
                        trial.number > 0
                    ):  # After first trial, we have some results to show
                        pbar.set_postfix({"best_value": study.best_value})

                # Run optimization with the callback
                study.optimize(
                    objective,
                    n_trials=cfg.optuna.n_trials,
                    timeout=cfg.optuna.timeout,
                    callbacks=[tqdm_callback],
                )

            # Save results
            best_params = save_study_results(
                study=study, output_dir=output_dir, model_name=friendly_model_name
            )

            log.info(f"Best params for {friendly_model_name}: {best_params}")

            # Update model parameters with best found params
            for k, v in study.best_params.items():
                if k in model_params:
                    model_params[k] = v

            log.info(f"Using best parameters from Optuna for final model training")
        else:
            # Load best parameters from existing study database
            try:
                # Path to the existing database
                db_path = BASE_DIR / "no2_models_optuna_4hrs.db"
                storage_path = f"sqlite:///{db_path}"

                # Import sqlite3 for direct database query
                import sqlite3
                import re

                # Connect to the database to list available studies
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT study_name FROM studies")
                available_studies = [row[0] for row in cursor.fetchall()]
                conn.close()

                # Pattern to match the study name format
                # For example: no2_model_optimization_ASTGCN_Like_all_vars_20250430_200522
                model_pattern = f"no2_model_optimization_{model_name}_{data_mode}"
                matching_studies = [
                    s for s in available_studies if s.startswith(model_pattern)
                ]

                if matching_studies:
                    # Use the most recent study (assuming timestamp is at the end)
                    study_name = sorted(matching_studies)[-1]
                    log.info(f"Found matching study: {study_name}")

                    # Load the study
                    study = optuna.load_study(
                        study_name=study_name, storage=storage_path
                    )

                    if study.best_params:
                        # Update model parameters with best found params
                        log.info(
                            f"Found saved best parameters for {friendly_model_name}"
                        )
                        for k, v in study.best_params.items():
                            if k in model_params:
                                model_params[k] = v
                        log.info(
                            f"Using best parameters from saved study: {study.best_params}"
                        )
                    else:
                        log.info(
                            f"No best parameters found for {friendly_model_name}, using defaults"
                        )
                else:
                    log.info(f"No matching study found for pattern: {model_pattern}")
                    log.info(f"Available studies: {available_studies}")
                    log.info("Using default parameters from configuration")
            except Exception as e:
                log.info(f"Could not load parameters from study database: {e}")
                log.info("Using default parameters from configuration")

        # Instantiate the model (with best params if optuna was run)
        model = model_creator_fn(**model_params).to(device)
        model = model.float()

        log.info(f"Model:\n{model}")

    except Exception as e:
        log.error(f"Error instantiating model: {e}")
        raise e

    # --- Training ---
    log.info(f"Starting training for {friendly_model_name}...")
    tb_log_dir = output_dir / cfg.paths.tensorboard_subdir
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    # Extract learning rate and weight decay - use values from study if available,
    # otherwise fall back to config
    learning_rate = study.best_params.get("lr")
    weight_decay = study.best_params.get(
        "weight_decay", cfg.training.get("weight_decay")
    )

    log.info(
        f"Training with learning_rate={learning_rate}, weight_decay={weight_decay}"
    )

    try:
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
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        # Save training history
        history_path = output_dir / cfg.paths.history_save_name
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        log.info(f"Training history saved to {history_path}")

        # Plot training history with model name
        plots_dir = output_dir / cfg.paths.plot_subdir
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Use the visualization module's plotting with model name
        set_base_dir(output_dir)  # Set base directory for visualization module
        plot_training_history(history, model_name=friendly_model_name)
        log.info(f"Training history plot created for {friendly_model_name}")

    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")
        history = {}  # Assign empty history

    finally:
        writer.close()

    # --- Evaluation ---
    log.info(f"Starting evaluation for {friendly_model_name}...")
    city_names = ["Amsterdam", "Rotterdam", "Utrecht"]

    test_loss, predictions, targets = evaluate_index(
        model=model,
        test_loader=test_loader,
        edge_index=edges.to(device),
        edge_weight=edge_weights.to(device) if edge_weights is not None else None,
        device=device,
        loader=loader,
        cities=city_names,
    )
    log.info(f"Test Loss for {friendly_model_name}: {test_loss:.4f}")

    # Calculate additional metrics
    mse = test_loss  # Already MSE
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    log.info(
        f"Test metrics for {friendly_model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}"
    )

    # Create prediction plots with model name and city-specific metrics
    if hasattr(loader, "denormalize_no2"):
        try:
            # Try to denormalize if the loader supports it
            predictions_orig = loader.denormalize_no2(predictions)
            targets_orig = loader.denormalize_no2(targets)
            log.info("Successfully denormalized predictions and targets")

            # Use visualization module's plot_predictions with model name
            plot_metrics = plot_predictions(
                predictions_orig,
                targets_orig,
                city_names=city_names,
                model_name=friendly_model_name,
                save_metrics=True,
                use_plotly=True,
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

    # Return primary metric for Hydra sweep comparison
    return rmse


if __name__ == "__main__":
    os.environ["HYDRA_CONFIG_PATH"] = str(Path(__file__).parent / "conf")

    # Set base_dir to project root or another appropriate location
    base_dir = BASE_DIR
    OmegaConf.update(OmegaConf.create(), "base_dir", str(base_dir))

    run_experiment()
