# %%
import torch
import pandas as pd
from torch_geometric.data import Data
from pathlib import Path
import os
import datetime
import pickle
import optuna
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from modelling import get_dataframes
from modelling.metrics.metricstracker import MetricsTracker
from graph_modelling.utils.load_data import (
    load_train_val_data,
    load_test_data,
    read_csv_files,
)
from graph_modelling.utils.tune_gnn import objective
from graph_modelling.models.temporalgnn import TemporalGNN
from graph_modelling.models.basicgnn import BasicGNN
from graph_modelling.models.attentiongnn import AttentionGNN
from graph_modelling.models.temporalattentiongnn import GATGRUGNN
from graph_modelling.utils.test_gnn import predict_and_evaluate
from graph_modelling.utils.train_gnn import train


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def inner_main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Use values from config instead of hardcoded values
    model_type = cfg.models.type
    N_HOURS_U = cfg.n_hours_u
    N_HOURS_Y = cfg.n_hours_y
    N_TRIALS = cfg.hp_tuning.n_trials
    N_EPOCHS = cfg.training.epochs

    # Get the original working directory (project root)
    orig_cwd = get_original_cwd()

    # Base paths (adjusting to work with Hydra's working directory)
    BASE_DIR = Path(orig_cwd).parent
    MODEL_PATH = BASE_DIR / "results" / "gnn_results" / "models"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    DATA_DIR = BASE_DIR / "data" / "data_combined"
    ALL_DIR = DATA_DIR / "all"

    print("BASE_DIR: ", BASE_DIR)
    print("MODEL_PATH: ", MODEL_PATH)
    print("ALL_DIR: ", ALL_DIR)

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Load the datasets
    with open(ALL_DIR / "geometric_pkl" / "train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open(ALL_DIR / "geometric_pkl" / "val_dataset.pkl", "rb") as f:
        val_dataset = pickle.load(f)
    with open(ALL_DIR / "geometric_pkl" / "test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    print("Loaded all datasets")

    input_dim = 7  # N_HOURS_U * input_features
    output_dim = 24  # N_HOURS_Y * output_predictions

    print(f"Using model type: {model_type}")

    num_nodes = 3
    window_size = N_HOURS_U  # 72
    n_features = input_dim  # e.g. 7

    for ds in (train_dataset, val_dataset, test_dataset):
        for data in ds:
            # data.x is (num_nodes, window_size * n_features)
            # reshape it back into (num_nodes, window_size, n_features)
            data.x_seq = data.x.view(num_nodes, window_size, n_features)

    patience = cfg.training.patience
    criterion = torch.nn.MSELoss()

    ## 4. Optuna Study
    study_name = f"{model_type}-gnn-tuning-{current_time}"
    storage_name = f"sqlite:///{cfg.hp_tuning.db_name}.db"

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            model_type,  # Pass model here
            train_dataset,  # Pass training data
            val_dataset,  # Pass validation data
            input_dim,
            output_dim,
            device=device,  # Pass device as keyword argument
            num_epochs=N_EPOCHS,  # reduced epochs for the demo
            N_HOURS_U=N_HOURS_U,
            N_HOURS_Y=N_HOURS_Y,
        ),
        n_trials=N_TRIALS,
    )

    # --- Print Best Results ---
    print("\n--- Optuna Study Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Value (Min Validation Loss): {best_trial.value:.6f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Create model based on the best trial parameters
    if model_type == "temporalgnn":
        final_model = TemporalGNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=best_trial.params["hidden_dim"],
            gcn_layers=best_trial.params["num_gcn"],
            rnn_layers=best_trial.params["rnn_layers"],
            rnn_dropout=best_trial.params["rnn_dropout"],
        ).to(device)

    elif model_type == "basicgnn":
        final_model = BasicGNN(
            seq_len=N_HOURS_U,
            num_features=input_dim,
            forecast_horizon=N_HOURS_Y,
            hidden_dim=best_trial.params["hidden_dim"],
            num_gcn=best_trial.params["num_gcn"],
        ).to(device)

    elif model_type == "attentiongnn":
        final_model = AttentionGNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=best_trial.params["hidden_dim"],
            num_layers=best_trial.params["num_gcn"],
            heads=best_trial.params["heads"],
            dropout=best_trial.params["dropout"],
        ).to(device)

    elif model_type == "temporalattentiongnn":
        final_model = GATGRUGNN(
            input_features=input_dim,
            seq_len=N_HOURS_U,
            forecast_horizon=N_HOURS_Y,
            hidden_dim=best_trial.params["hidden_dim"],
            gat_heads=best_trial.params["gat_heads"],
            gat_layers=best_trial.params["gat_layers"],
            rnn_layers=best_trial.params["gru_layers"],
            dropout=best_trial.params["dropout"],
        ).to(device)

    print(final_model)

    def custom_collate(data_list):
        # First create the batch as usual
        batch = Batch.from_data_list(data_list)

        # For x_seq, we need to maintain the 4D structure
        batch_size = len(data_list)
        num_nodes = data_list[0].x_seq.size(0)  # Typically 3
        seq_len = data_list[0].x_seq.size(1)  # 72
        n_features = data_list[0].x_seq.size(2)  # 7

        # Stack the x_seq tensors properly to get (batch_size, num_nodes, seq_len, features)
        x_seq_stacked = torch.stack([d.x_seq for d in data_list], dim=0)

        # Make sure it has the right shape
        assert x_seq_stacked.shape == (batch_size, num_nodes, seq_len, n_features), (
            f"Expected shape ({batch_size}, {num_nodes}, {seq_len}, {n_features}), got {x_seq_stacked.shape}"
        )

        # Assign to the batch
        batch.x_seq = x_seq_stacked

        return batch

    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=best_trial.params["lr"],
        weight_decay=best_trial.params["weight_decay"],
    )
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=best_trial.params["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=best_trial.params["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=best_trial.params["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
    )

    train_losses, val_losses = train(
        final_model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        N_EPOCHS,
        patience,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training and Validation Loss Over Epochs - {model_type}")
    plt.savefig(f"final_training_loss_{model_type}_{current_time}.png")
    plt.savefig(MODEL_PATH / f"final_training_loss_{model_type}_{current_time}.png")

    # Save the model
    torch.save(final_model.state_dict(), f"final_model_{model_type}_{current_time}.pt")
    torch.save(
        final_model.state_dict(),
        MODEL_PATH / f"final_model_{model_type}_{current_time}.pt",
    )

    # Load y_min and y_max
    with open(ALL_DIR / "geometric_pkl" / "y_min_max.pkl", "rb") as f:
        y_min, y_max = pickle.load(f)

    global_rmse, rmse_per_node, mean_no2_rmse = predict_and_evaluate(
        final_model, test_loader, device, output_dim, y_min, y_max, N_HOURS_Y
    )

    # Save results based on model type of rmse
    results = {
        "model_type": model_type,
        "global_rmse": global_rmse,
        "mean_no2_rmse": mean_no2_rmse,
        "rmse_per_node": rmse_per_node.tolist()
        if isinstance(rmse_per_node, torch.Tensor)
        else rmse_per_node,
    }

    # Save the results to a CSV file
    results_df = pd.DataFrame([results])
    results_df.to_csv(
        f"results_{model_type}_{current_time}.csv",
        index=False,
    )
    results_df.to_csv(
        MODEL_PATH / f"results_{model_type}_{current_time}.csv",
        index=False,
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    @hydra.main(version_base="1.3", config_path="../conf", config_name="config")
    def main(cfg):
        inner_main(cfg)
    
    main()