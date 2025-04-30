import optuna
import logging
from optuna.pruners import MedianPruner
from pathlib import Path
import torch
import yaml
import json
from typing import Dict, Any, Optional, Callable
import numpy as np
import os

# Configure logger
logger = logging.getLogger(__name__)


def setup_optuna_study(
    study_name: str,
    storage: str,
    direction: str = "minimize",
    load_if_exists: bool = True,
    pruner: Optional[optuna.pruners.BasePruner] = None,
):
    """
    Set up an Optuna study with the specified name and storage.

    Args:
        study_name: Name of the study
        storage: Storage URL (e.g., 'sqlite:///optuna.db')
        direction: 'minimize' or 'maximize' the objective
        load_if_exists: Whether to load the study if it already exists
        pruner: Optuna pruner to use (default: MedianPruner)

    Returns:
        optuna.Study: The created or loaded study
    """
    if pruner is None:
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=load_if_exists,
            pruner=pruner,
        )
        logger.info(f"Study '{study_name}' is ready. Storage: {storage}")
        return study
    except Exception as e:
        logger.error(f"Error setting up Optuna study: {e}")
        raise


def define_model_param_space(trial, model_name: str) -> Dict[str, Any]:
    """
    Define hyperparameter search space based on the model type.

    Args:
        trial: Optuna trial object
        model_name: Name of the model to optimize

    Returns:
        Dict containing hyperparameters for the model
    """
    # Base parameters for all models
    params = {
        "lr": trial.suggest_float("lr", 1e-6, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
    }

    # Model-specific parameters
    if "spatial_only_gcn" in model_name.lower():
        params.update(
            {
                "hidden_channels": trial.suggest_int(
                    "hidden_channels", 16, 128, step=16
                ),
            }
        )

    elif "temporal_only_gru" in model_name.lower():
        params.update(
            {
                "hidden_channels": trial.suggest_int(
                    "hidden_channels", 16, 128, step=16
                ),
            }
        )

    elif "attention_gconvgru" in model_name.lower():
        params.update(
            {
                "hidden_channels": trial.suggest_int(
                    "hidden_channels", 32, 128, step=16
                ),
                "attention_mlp_hidden": trial.suggest_int(
                    "attention_mlp_hidden", 8, 64, step=8
                ),
            }
        )

    elif "batched_gconvgru_index" in model_name.lower():
        params.update(
            {
                "hidden_channels": trial.suggest_int(
                    "hidden_channels", 32, 128, step=16
                ),
            }
        )

    elif "astgcn_like" in model_name.lower():
        params.update(
            {
                "block_channels": trial.suggest_int("block_channels", 16, 64, step=16),
                "gru_channels": trial.suggest_int("gru_channels", 16, 64, step=16),
                "num_blocks": trial.suggest_int("num_blocks", 1, 3),
                "d_k": trial.suggest_int("d_k", 16, 64, step=16),
            }
        )

    # elif "astgcn_seq2seq" in model_name.lower():
    #     params.update(
    #         {
    #             "block_channels": trial.suggest_int("block_channels", 16, 64, step=16),
    #             "gru_channels": trial.suggest_int("gru_channels", 16, 64, step=16),
    #             "num_blocks": trial.suggest_int("num_blocks", 1, 3),
    #             "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
    #         }
    #     )

    return params


def create_objective(
    model_fn: Callable,
    train_loader,
    val_loader,
    test_loader,
    edge_index,
    edge_weight,
    device,
    model_name: str,
    base_cfg: Dict[str, Any],
    output_dir: Path,
):
    """
    Create an objective function for Optuna optimization.
    """
    from src.graph_modelling.training.train_utils import (
        train_model_index,
        evaluate_index,
    )
    from torch.utils.tensorboard import SummaryWriter

    def objective(trial):
        # Set up tensorboard for this trial
        tb_log_dir = output_dir / f"tensorboard_logs/trial_{trial.number}"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))

        try:
            # Get hyperparameters for this trial
            params = define_model_param_space(trial, model_name)

            # Log parameters
            logger.info(f"Trial {trial.number}: {params}")

            # Create model with trial parameters
            model_params = base_cfg["model_params"].copy()
            # Remove lr and weight_decay from model params
            lr = params.pop("lr")
            weight_decay = params.pop("weight_decay")

            model_params.update(params)
            model = model_fn(**model_params).to(device)
            model = model.float()

            # Training parameters
            n_epochs = base_cfg.get("training", {}).get("n_epochs", 150)
            patience = base_cfg.get("training", {}).get("patience", 5)

            # Optimizer parameters
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            # Custom train function with Optuna pruning
            best_val_loss = float("inf")
            patience_counter = 0
            history = {"train_loss": [], "val_loss": [], "epochs": []}

            # Training loop with pruning
            for epoch in range(1, n_epochs + 1):
                # Train for one epoch
                model.train()
                train_losses = []

                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = (
                        x_batch.to(device).float(),
                        y_batch.to(device).float(),
                    )
                    optimizer.zero_grad()

                    y_hat = model(
                        x_batch, edge_index.to(device), edge_weight.to(device)
                    )

                    # Handle target shape differences
                    if y_batch.shape[2] != y_hat.shape[2]:
                        B, H, NF = y_batch.shape
                        num_nodes = 3  # Assuming 3 cities
                        num_vars = NF // num_nodes

                        y_batch_reshaped = y_batch.view(B, H, num_nodes, num_vars)
                        y_batch_no2 = y_batch_reshaped[:, :, :, 0]

                        loss = torch.nn.functional.mse_loss(y_hat, y_batch_no2)
                    else:
                        loss = torch.nn.functional.mse_loss(y_hat, y_batch)

                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                # Validate
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch, y_batch = (
                            x_batch.to(device).float(),
                            y_batch.to(device).float(),
                        )
                        y_hat = model(
                            x_batch, edge_index.to(device), edge_weight.to(device)
                        )

                        # Handle target shape differences
                        if y_batch.shape[2] != y_hat.shape[2]:
                            B, H, NF = y_batch.shape
                            num_nodes = 3
                            num_vars = NF // num_nodes

                            y_batch_reshaped = y_batch.view(B, H, num_nodes, num_vars)
                            y_batch_no2 = y_batch_reshaped[:, :, :, 0]

                            val_loss = torch.nn.functional.mse_loss(y_hat, y_batch_no2)
                        else:
                            val_loss = torch.nn.functional.mse_loss(y_hat, y_batch)

                        val_losses.append(val_loss.item())

                avg_train_loss = sum(train_losses) / len(train_losses)
                avg_val_loss = sum(val_losses) / len(val_losses)

                # Update history
                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)
                history["epochs"].append(epoch)

                # Report to tensorboard
                writer.add_scalars(
                    "Loss", {"train": avg_train_loss, "val": avg_val_loss}, epoch
                )

                # Report to Optuna for pruning
                trial.report(avg_val_loss, epoch)

                # Handle pruning
                if trial.should_prune():
                    writer.close()
                    raise optuna.exceptions.TrialPruned()

                # Early stopping logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(
                            f"Trial {trial.number}: Early stopping at epoch {epoch}"
                        )
                        model.load_state_dict(best_model_state)
                        break

            # Evaluate on test set
            model.eval()
            test_loss, _, _ = evaluate_index(
                model=model,
                test_loader=test_loader,
                edge_index=edge_index.to(device),
                edge_weight=edge_weight.to(device),
                device=device,
            )

            # Save rmse values to optuna
            rmse_values = {
                "rmse": test_loss,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
            trial.set_user_attr("rmse_values", rmse_values)

            # Return validation loss as the objective value
            writer.close()
            return best_val_loss

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            if writer:
                writer.close()
            raise

    return objective


def save_study_results(study, output_dir: Path, model_name: str):
    """
    Save Optuna study results to disk.

    Args:
        study: Optuna study object
        output_dir: Directory to save results to
        model_name: Name of the model being optimized
    """
    # Create output directory
    results_dir = output_dir / "optuna_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    results = {
        "model_name": model_name,
        "best_params": best_params,
        "best_value": float(best_value),
        "best_trial_number": best_trial.number,
        "n_trials": len(study.trials),
    }

    # Save as JSON
    with open(results_dir / f"{model_name}_best_params.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save trial information
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_data = {
                "number": trial.number,
                "value": float(trial.value),
                "params": trial.params,
            }
            trials_data.append(trial_data)

    with open(results_dir / f"{model_name}_trials.json", "w") as f:
        json.dump(trials_data, f, indent=2)

    # Save importance of hyperparameters
    try:
        importance = optuna.importance.get_param_importances(study)
        importance_data = {k: float(v) for k, v in importance.items()}

        with open(results_dir / f"{model_name}_importance.json", "w") as f:
            json.dump(importance_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not compute parameter importance: {e}")

    logger.info(f"Saved optimization results for {model_name} to {results_dir}")
    return results
