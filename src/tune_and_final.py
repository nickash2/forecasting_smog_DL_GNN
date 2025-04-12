import torch
import pandas as pd
from torch_geometric.data import Data
from pathlib import Path
import os

os.chdir(Path().cwd().parent)
from modelling import get_dataframes
from modelling.metrics.metricstracker import MetricsTracker
import datetime
from graph_modelling.utils.load_data import (
    load_train_val_data,
    load_test_data,
    read_csv_files,
)
from graph_modelling.utils.tune_gnn import objective
from graph_modelling.models.temporalgnn import TemporalGNN
from graph_modelling.models.basicgnn import BasicGNN
import pickle
import optuna
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from graph_modelling.utils.test_gnn import predict_and_evaluate
from graph_modelling.utils.train_gnn import train
import numpy as np
import argparse

HABROK = bool(0)  # set to True if using HABROK; it will print
# all stdout to a .txt file to log progress
BASE_DIR = Path.cwd()
MODEL_PATH = BASE_DIR / "results" / "gnn_results" / "models"
DATA_DIR = BASE_DIR / "data" / "data_combined"
ALL_DIR = DATA_DIR / "all"

print("BASE_DIR: ", BASE_DIR)
print("MODEL_PATH: ", MODEL_PATH)
print("ALL_DIR: ", ALL_DIR)

torch.manual_seed(34)  # set seed for reproducibility

N_HOURS_U = 72  # number of hours to use for input
N_HOURS_Y = 24  # number of hours to predict
N_HOURS_STEP = 24  # "sampling rate" in hours of the data; e.g. 24
# means sample an I/O-pair every 24 hours
# the contaminants and meteorological vars
CONTAMINANTS = ["NO2", "O3"]  # 'PM10', 'PM25']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


N_TRIALS = 1
N_EPOCHS = 5


# Load the datasets
with open(ALL_DIR / "geometric_pkl" / "train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)
with open(ALL_DIR / "geometric_pkl" / "val_dataset.pkl", "rb") as f:
    val_dataset = pickle.load(f)
with open(ALL_DIR / "geometric_pkl" / "test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)

print("Loaded all datasets")


input_dim = N_HOURS_U * 8
output_dim = N_HOURS_Y * 2
valid_types = ["temporalgnn", "basicgnn"]

# Use argparse to get the model type from the command line
parser = argparse.ArgumentParser(description="Tune and train a GNN model.")
parser.add_argument(
    "--model_type",
    type=str,
    default="basicgnn",  # Default model type if not specified
    choices=valid_types,
    help=f"The type of GNN model to use. Choose from: {valid_types}.",
)

args = parser.parse_args()
model_type = args.model_type

print(f"Using model type: {model_type}")

patience = 10

criterion = torch.nn.MSELoss()


## 4. Optuna Study
study_name = f"{model_type}-gnn-tuning-{current_time}"

storage_name = "sqlite:///gnn_tuning.db"

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
    ),
    n_trials=N_TRIALS,
)


# --- Print Best Results ---
print("\n--- Optuna Study Complete ---")
print(f"Number of finished trials: {len(study.trials)}")
print("Best trial:")
best_trial = study.best_trial

print("  Value (Min Validation Loss): {best_trial.value:.6f}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")


if model_type == "temporalgnn":
    # Load the best model
    final_model = TemporalGNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=best_trial.params["hidden_dim"],
        gcn_layers=best_trial.params["num_gcn"],
        rnn_layers=best_trial.params["rnn_layers"],
        rnn_dropout=best_trial.params["rnn_dropout"],
    ).to(device)

elif model_type == "basicgnn":
    # Load the best model
    final_model = BasicGNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=best_trial.params["hidden_dim"],
        num_gcn=best_trial.params["gcn_layers"],
    ).to(device)


optimizer = torch.optim.Adam(
    final_model.parameters(),
    lr=best_trial.params["lr"],
    weight_decay=best_trial.params["weight_decay"],
)
criterion = torch.nn.MSELoss()

train_loader = DataLoader(
    train_dataset, batch_size=best_trial.params["batch_size"], shuffle=False
)
val_loader = DataLoader(
    val_dataset, batch_size=best_trial.params["batch_size"], shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=best_trial.params["batch_size"], shuffle=False
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


plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Over Epochs")
plt.savefig(MODEL_PATH / f"final_training_loss_basicgnn_{current_time}.png")


# Load y_min and y_max
with open(ALL_DIR / "geometric_pkl" / "y_min_max.pkl", "rb") as f:
    y_min, y_max = pickle.load(f)


global_rmse, rmse_no2, rmse_o3, preds, targets = predict_and_evaluate(
    final_model, test_loader, device, output_dim, y_min, y_max, N_HOURS_Y
)

# Save results based on model type of rmse
results = {
    "model_type": model_type,
    "global_rmse": global_rmse,
    "rmse_no2": rmse_no2,
    "rmse_o3": rmse_o3,
}


# Save the results to a CSV file
results_df = pd.DataFrame([results])
results_df.to_csv(
    MODEL_PATH / f"results_{model_type}_{current_time}.csv",
    index=False,
)


# Save the model
torch.save(
    final_model.state_dict(),
    MODEL_PATH / f"final_model_{model_type}_{current_time}.pt",
)
