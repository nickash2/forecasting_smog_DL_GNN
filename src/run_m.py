import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from modelling import *
from modelling import GRU, HGRU
from modelling.plots import set_minmax_path, set_contaminants
from modelling.optuna import optuna_search

# Set default device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def load_data(city_name):
    """Load datasets."""
    train_input_frames = get_dataframes("train", "u", city_name)
    train_output_frames = get_dataframes("train", "y", city_name)

    val_input_frames = get_dataframes("val", "u", city_name)
    val_output_frames = get_dataframes("val", "y", city_name)

    test_input_frames = get_dataframes("test", "u", city_name)
    test_output_frames = get_dataframes("test", "y", city_name)

    train_dataset = TimeSeriesDataset(
        train_input_frames, train_output_frames, 5, 72, 24, 24
    )
    val_dataset = TimeSeriesDataset(val_input_frames, val_output_frames, 3, 72, 24, 24)
    test_dataset = TimeSeriesDataset(
        test_input_frames, test_output_frames, 3, 72, 24, 24
    )

    return train_dataset, val_dataset, test_dataset


def train_model(hp, train_dataset, val_dataset, save_path):
    """Train the model using the given hyperparameters."""
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
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    loss_fn = nn.MSELoss()

    test_error = test(model, loss_fn, test_loader)
    print(f"Test MSE: {test_error}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate air pollution models."
    )
    parser.add_argument("--train", action="store_true", help="Train a model")
    parser.add_argument(
        "--grid-search", action="store_true", help="Perform hyperparameter grid search"
    )
    parser.add_argument(
        "--optuna", action="store_true", help="Perform Optuna hyperparameter tuning"
    )
    parser.add_argument(
        "--city", type=str, default="Utrecht", help="City name for data selection"
    )
    parser.add_argument(
        "--epochs", type=int, default=5000, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--model", type=str, choices=["GRU", "HGRU"], default="GRU", help="Model type"
    )

    args = parser.parse_args()

    # Paths
    BASE_DIR = Path.cwd()
    MODEL_PATH = BASE_DIR / "results" / "models"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model_save_path = MODEL_PATH / f"model_{args.model}.pth"
    CITY_NAME = args.city
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

    torch.manual_seed(34)  # set seed for reproducibility

    N_HOURS_U = 72  # number of hours to use for input
    N_HOURS_Y = 24  # number of hours to predict
    N_HOURS_STEP = 24  # "sampling rate" in hours of the data; e.g. 24
    # means sample an I/O-pair every 24 hours
    # the contaminants and meteorological vars
    CONTAMINANTS = ["NO2", "O3"]  # 'PM10', 'PM25']

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_data(args.city)

    # Define hyperparameters
    hp = {
        "n_hours_u": N_HOURS_U,
        "n_hours_y": N_HOURS_Y,
        "model_class": GRU,  # changed to GRU
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
    # Hyperparameter search space for grid search
    hp_space = {
        "hidden_layers": [2, 4, 6],
        "hidden_units": [32, 64, 128],
        "lr_shared": [1e-3, 1e-4, 5e-4],
        "batch_sz": [16, 32, 64],
    }

    if args.grid_search:
        print("Starting grid search...")
        model, best_hp = grid_search_model(hp, hp_space, train_dataset, model_save_path)
        print("Best Hyperparameters:", best_hp)

    elif args.train:
        print("Starting training...")
        model = train_model(hp, train_dataset, val_dataset, model_save_path)

    elif args.optuna:
        print("Starting optuna search...")
        model = optuna_search(hp, train_dataset, n_trials=50, hier=False)

    else:
        print("No valid action selected. Use --help for options.")

    # Test the model if training or grid search was performed
    if args.train or args.grid_search or args.optuna:
        model.load_state_dict(torch.load(model_save_path))
        test_model(model, test_dataset)


if __name__ == "__main__":
    main()
