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

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ------------------------------------

from src.graph_modelling.datasets.no2_dataset import NO2DatasetLoader
from src.graph_modelling.training.train_utils import train_model_index, evaluate_index
from src.graph_modelling.visualization.visualization import (
    plot_training_history,
    plot_predictions,
)

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> float:
    """Runs a single experiment configuration."""

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
        cache_path = PROJECT_ROOT / "data" / "data_gnn" / cfg.data.cache_file

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
    num_nodes = 3
    num_vars = cfg.data.num_vars

    # Find out which model we're creating for better logging
    model_name = cfg.model._target_.split(".")[-1]
    data_mode = "no2_only" if cfg.data.only_no2 else "all_vars"
    log.info(f"Initializing model: {model_name} with {data_mode}")

    try:
        model = hydra.utils.instantiate(
            cfg.model,
            num_nodes=num_nodes,
            num_vars=num_vars,
            lags=cfg.training.n_lags,
            horizon=cfg.training.n_horizon,
            _recursive_=False,
        ).to(device)
        log.info(f"Model:\n{model}")
    except Exception as e:
        log.error(f"Error instantiating model: {e}")
        raise e

    # --- Training ---
    log.info("Starting training...")
    tb_log_dir = output_dir / cfg.paths.tensorboard_subdir
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    try:
        model, history = train_model_index(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            edge_index=edges.to(device),
            edge_weight=edge_weights.to(device) if edge_weights is not None else None,
            device=device,
            epochs=cfg.training.n_epochs,
            patience=cfg.training.patience,
            writer=writer,
        )
        # Save training history
        history_path = output_dir / cfg.paths.history_save_name
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        log.info(f"Training history saved to {history_path}")

        # Plot training history
        plots_dir = output_dir / cfg.paths.plot_subdir
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / "training_history.png"

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(history["train_loss"], label="Training Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training History - {model_name} ({data_mode})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        log.info(f"Training history plot saved to {plot_path}")

    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")
        history = {}  # Assign empty history

    finally:
        writer.close()

    # --- Evaluation ---
    log.info("Starting evaluation...")
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
    log.info(f"Test Loss: {test_loss:.4f}")

    # Calculate additional metrics
    mse = test_loss  # Already MSE
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    log.info(f"Test metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # --- Save Results ---
    # Save test metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_mse": float(mse),
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "model_name": model_name,
        "data_mode": data_mode,
        "num_vars": int(num_vars),
    }

    metrics_path = output_dir / cfg.paths.metrics_save_name
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    log.info(f"Test metrics saved to {metrics_path}")

    # Create prediction plots
    predictions_plot_path = plots_dir / "predictions.png"

    # Assuming predictions and targets are numpy arrays with shape [timesteps, cities]
    if hasattr(loader, "denormalize_no2"):
        try:
            # Try to denormalize if the loader supports it
            predictions_orig = loader.denormalize_no2(predictions)
            targets_orig = loader.denormalize_no2(targets)
            log.info("Successfully denormalized predictions and targets")

            # Call the plotting function
            plt.figure(figsize=(12, 4 * num_nodes))

            for i in range(num_nodes):
                city_name = city_names[i] if i < len(city_names) else f"City {i}"
                plt.subplot(num_nodes, 1, i + 1)
                plt.plot(targets_orig[:, i], "b-", label="Actual")
                plt.plot(predictions_orig[:, i], "r--", label="Predicted")
                plt.title(f"NO2 Predictions for {city_name}")
                plt.xlabel("Time (hours)")
                plt.ylabel("NO2 (μg/m³)")
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Add error metrics
                city_mse = np.mean((predictions_orig[:, i] - targets_orig[:, i]) ** 2)
                city_rmse = np.sqrt(city_mse)
                city_mae = np.mean(np.abs(predictions_orig[:, i] - targets_orig[:, i]))
                plt.text(
                    0.02,
                    0.90,
                    f"RMSE: {city_rmse:.2f}\nMAE: {city_mae:.2f}",
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor="white", alpha=0.7),
                )

            plt.tight_layout()
            plt.savefig(predictions_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            log.info(f"Prediction plots saved to {predictions_plot_path}")

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
    base_dir = PROJECT_ROOT / "results" / "model_comparison"
    OmegaConf.update(OmegaConf.create(), "base_dir", str(base_dir))

    run_experiment()
