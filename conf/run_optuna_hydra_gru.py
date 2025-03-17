import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from src.modelling import (
    GRU,
    HGRU,
    get_dataframes,
    TimeSeriesDataset,
    EarlyStopper,
    PrintManager,
)
from src.modelling.train import train
from src.modelling.test import test
from src.modelling.grid_search import grid_search
from src.modelling.optuna import optuna_search
from src.modelling.plots import set_minmax_path, set_contaminants
from src.modelling.grid_search import update_dict
import pandas as pd
import optuna

# Set default device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def load_data(city_name, u: int = 72, y: int = 24, step: int = 24):
    """Load datasets."""
    train_input_frames = get_dataframes("train", "u", city_name)
    train_output_frames = get_dataframes("train", "y", city_name)

    val_input_frames = get_dataframes("val", "u", city_name)
    val_output_frames = get_dataframes("val", "y", city_name)

    test_input_frames = get_dataframes("test", "u", city_name)
    test_output_frames = get_dataframes("test", "y", city_name)

    train_dataset = TimeSeriesDataset(
        train_input_frames, train_output_frames, 5, u, y, y
    ).to(device)
    val_dataset = TimeSeriesDataset(val_input_frames, val_output_frames, 3, u, y, y).to(
        device
    )
    test_dataset = TimeSeriesDataset(
        test_input_frames, test_output_frames, 3, u, y, y
    ).to(device)

    return train_dataset, val_dataset, test_dataset


def train_model(hp, train_dataset, val_dataset, save_path, city_name):
    """Train the model using the given hyperparameters."""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_loader = DataLoader(train_dataset, batch_size=hp["batch_sz"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hp["batch_sz"], shuffle=False)

    model = hp["model_class"](
        hp["n_hours_u"],
        hp["n_hours_y"],
        hp["input_units"],
        hp["hidden_layers"],
        hp["hidden_units"],
        hp["output_units"],
    ).to(device)

    model, train_losses, val_losses = train(hp, train_loader, val_loader, verbose=True)
    df_losses = pd.DataFrame({"L_train": train_losses, "L_val": val_losses})
    df_losses.to_csv(
        f"{os.path.join(os.getcwd(), 'results/final_losses')}/losses_GRU_at_{city_name}_{current_time}.csv",
        sep=";",
        decimal=".",
        encoding="utf-8",
    )

    torch.save(model.state_dict(), save_path)
    return model


def grid_search_model(hp, hp_space, train_dataset, save_path):
    """Perform grid search for hyperparameter tuning."""
    train_dataset_full = ConcatDataset([train_dataset])
    model, best_hp, val_loss = grid_search(
        hp, hp_space, train_dataset_full, hier=False, verbose=True
    )

    torch.save(model.state_dict(), save_path)
    return model, best_hp


def test_model(model, test_dataset):
    """Evaluate the trained model."""
    test_loader = DataLoader(test_dataset.to(device), batch_size=16, shuffle=False)
    loss_fn = nn.MSELoss()

    test_error = test(model, loss_fn, test_loader)
    print(f"Test MSE: {test_error}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    SEED = cfg.seed
    # Paths
    BASE_DIR = Path(hydra.utils.get_original_cwd())
    SRC_DIR = BASE_DIR / "src"
    MODEL_PATH = SRC_DIR / "results" / "models"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model_save_path = MODEL_PATH / f"model_{cfg.model}_{cfg.cities.name}_metric.pth"
    CITY_NAME = cfg.cities.city
    MINMAX_PATH = (
        BASE_DIR.parent
        / "data"
        / "data_combined"
        / CITY_NAME.lower()
        / "contaminant_minmax.csv"
    )

    print("BASE_DIR: ", BASE_DIR)
    print("MODEL_PATH: ", MODEL_PATH)
    print("MINMAX_PATH: ", MINMAX_PATH)

    torch.manual_seed(SEED)  # set seed for reproducibility

    N_HOURS_U = 72  # number of hours to use for input
    N_HOURS_Y = 24  # number of hours to predict
    N_HOURS_STEP = 24  # "sampling rate" in hours of the data; e.g. 24
    # Load datasets
    os.chdir(SRC_DIR)
    train_dataset, val_dataset, test_dataset = load_data(
        CITY_NAME, u=N_HOURS_U, y=N_HOURS_Y, step=N_HOURS_STEP
    )

    hp = {
        "n_hours_u": N_HOURS_U,
        "n_hours_y": N_HOURS_Y,
        "model_class": HGRU if cfg.model == "HGRU" else GRU,  # changed to GRU
        "input_units": 8,  # train_dataset.__n_features_in__(),
        "hidden_layers": 4,
        "hidden_units": 128,
        # 'branches' : 2,  # predicting only no2 and o3
        "output_units": 2,  # train_dataset.__n_features_out__(),
        "Optimizer": torch.optim.Adam,
        "lr_shared": 1e-3,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {
            "mode": "min",
            "factor": 0.1,
            "patience": 3,
            "cooldown": 8,
            "verbose": True,
        },
        "w_decay": 1e-5,
        "loss_fn": torch.nn.MSELoss(),
        "epochs": 5000,
        "early_stopper": EarlyStopper,
        "patience": 15,
        "batch_sz": 16,
        "k_folds": 5,
    }
    if cfg.hp_tuning.grid_search:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        full_training = ConcatDataset([train_dataset, val_dataset])
        print("Starting grid search...")
        with PrintManager(
            f"results/grid_search_exe_s/grid_search_log_{current_time}.txt",
            "a",
            cfg.habrok,
        ):
            model, best_hp = grid_search_model(
                hp, cfg.hp_space, train_dataset, model_save_path
            )
        print("Best Hyperparameters:", best_hp)

    elif cfg.train:
        print("Starting training...")
        best_study = optuna.load_study(
            study_name="optuna_utrecht_16-33-31",
            storage="sqlite:///results/optuna_search/optuna_utrecht.db",
        )
        best_hp = update_dict(hp, best_study.best_params)
        print("Best Hyperparameters:", best_hp)

        model = train_model(
            best_hp,
            train_dataset,
            val_dataset,
            model_save_path,
            CITY_NAME.lower(),
        )

    elif cfg.hp_tuning.optuna:
        print("Starting optuna search...")
        print(model_save_path)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        full_training = ConcatDataset([train_dataset, val_dataset])
        with PrintManager(
            f"results/optuna_search/logs/optuna_log_{current_time}_{cfg.model}.txt",
            "a",
            cfg.habrok,
        ):
            best_hp, best_val_loss = optuna_search(
                hp,
                train_dataset,
                n_trials=cfg.hp_tuning.n_trials,
                hier=False,
                db_name=cfg.hp_tuning.db_name,
                study_name=cfg.hp_tuning.study_name,
                seed=SEED,
                hp_save_dir=cfg.hp_tuning.hp_save_dir,
            )
        print("Training with best hyperparameters...")
        print("Best Hyperparameters:", best_hp)
        model = train_model(best_hp, train_dataset, val_dataset, model_save_path)
        torch.save(model.state_dict(), model_save_path)

    else:
        print("No valid action selected. Use --help for options.")

    # Test the model if training or grid search was performed
    if cfg.train or cfg.hp_tuning.grid_search or cfg.hp_tuning.optuna:
        model.load_state_dict(torch.load(model_save_path))
        test_model(model, test_dataset)


if __name__ == "__main__":
    main()
