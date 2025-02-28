import optuna

from .cross_validation import k_fold_cross_validation_expanding_hierarchical

from typing import Any, List, Dict, Tuple
import torch
from .grid_search import update_dict, print_current_config, save_dict


def objective(
    trial: optuna.Trial, hp: Dict[str, Any], train_dataset, hier: bool
) -> float:
    """
    Objective function for Optuna to minimize.
    """
    # Suggest hyperparameters from the search space
    config = {
        # "n_hours_u": trial.suggest_int("n_hours_u", 12, 48),
        # "n_hours_y": trial.suggest_int("n_hours_y", 12, 48),
        "hidden_layers": trial.suggest_int("hidden_layers", 2, 6),
        "hidden_units": trial.suggest_categorical("hidden_units", [32, 64, 128]),
        "batch_sz": trial.suggest_categorical("batch_sz", [16, 32, 64]),
        "lr_shared": trial.suggest_float("lr_shared", 1e-4, 1e-2, log=True),
        "w_decay": trial.suggest_float("w_decay", 1e-8, 1e-4, log=True),
        "patience": trial.suggest_int("patience", 10, 30),
    }

    # Update hyperparameters with suggested values
    config_dict = update_dict(hp, config)
    print_current_config(config_dict)

    # Perform k-fold validation or training
    val_loss, _ = k_fold_cross_validation_expanding_hierarchical(
        config_dict, train_dataset, hier=hier
    )

    return val_loss  # Optuna minimizes this value


def optuna_search(
    hp: Dict[str, Any],
    train_dataset: torch.utils.data.Dataset,
    n_trials: int = 50,
    hier: bool = True,
    db_name: str = "optuna_db",
    study_name: str = "optuna_study",
    seed: int = 10,
    hp_save_dir: str = "best_hp.json",
) -> Tuple[Dict[str, Any], float]:
    """
    Uses Optuna to find the best hyperparameters.
    """
    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///results/optuna_search/{db_name}.db",
        study_name=study_name,
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: objective(trial, hp, train_dataset, hier), n_trials=n_trials
    )

    # Get best hyperparameters
    best_hp = update_dict(hp, study.best_params)
    best_val_loss = study.best_value

    print("Best hyperparameters:", best_hp)
    print("Best validation loss:", best_val_loss)

    return best_hp, best_val_loss
