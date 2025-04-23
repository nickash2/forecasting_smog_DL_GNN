# %%
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
from graph_modelling.models.attentiongnn import AttentionGNN
from graph_modelling.models.temporalattentiongnn import GATGRUGNN
import pickle
import optuna
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from graph_modelling.utils.test_gnn import predict_and_evaluate
from graph_modelling.utils.train_gnn import train, init_scheduler
import numpy as np
import argparse
from torch_geometric.data import Batch

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


N_TRIALS = 70
N_EPOCHS = 500  # number of epochs to train for


# Load the datasets
with open(ALL_DIR / "geometric_pkl" / "train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)
with open(ALL_DIR / "geometric_pkl" / "val_dataset.pkl", "rb") as f:
    val_dataset = pickle.load(f)
with open(ALL_DIR / "geometric_pkl" / "test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)

print("Loaded all datasets")
# Add this after loading datasets to verify feature dimensions

# Debug: Check the feature dimensions
print(f"Feature dimensions check:")
sample_data = train_dataset[0]
print(f"  x_seq shape: {sample_data.x_seq.shape}")
print(f"  Number of features: {sample_data.x_seq.shape[2]}")
print(f"  y shape: {sample_data.y.shape}")

input_dim = 7 + 2  # Original features (7) + time encoding features (6)
output_dim = 24  # N_HOURS_Y * output_predictions
valid_types = ["temporalgnn", "basicgnn", "attentiongnn", "temporalattentiongnn"]
# %%
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

num_nodes = 3
window_size = N_HOURS_U  # 72
n_features = input_dim  # e.g. 7

# for ds in (train_dataset, val_dataset, test_dataset):
#     for data in ds:
#         # data.x is (num_nodes, window_size * n_features)
#         # reshape it back into (num_nodes, window_size, n_features)
#         data.x_seq = data.x.view(num_nodes, window_size, n_features)


patience = 20

criterion = torch.nn.MSELoss()


## 4. Optuna Study
study_name = (
    "basicgnn-gnn-tuning-20250418-150835"  # f"{model_type}-gnn-tuning-{current_time}"
)

storage_name = "sqlite:///src/gnn_hydra_tuning.db"

study = optuna.create_study(
    direction="minimize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
    pruner=optuna.pruners.HyperbandPruner(),
)
# print("\n--- All Studies in Database ---")
# all_study_names = optuna.study.get_all_study_names(storage=storage_name)
# print(f"Found {len(all_study_names)} studies:")

# for study_name in all_study_names:
#     try:
#         study = optuna.load_study(study_name=study_name, storage=storage_name)
#         print(f"\nStudy: {study_name}")
#         print(f"  Number of trials: {len(study.trials)}")
#         if len(study.trials) > 0:
#             print(f"  Best trial value: {study.best_value}")
#             print("  Best trial params:")
#             for key, value in study.best_params.items():
#                 print(f"    {key}: {value}")
#         else:
#             print("  No completed trials in this study")
#     except Exception as e:
#         print(f"  Error loading study {study_name}: {str(e)}")

# study.optimize(
#     lambda trial: objective(
#         trial,
#         model_type,  # Pass model here
#         train_dataset,  # Pass training data
#         val_dataset,  # Pass validation data
#         input_dim,
#         output_dim,
#         device=device,  # Pass device as keyword argument
#         num_epochs=N_EPOCHS,  # reduced epochs for the demo
#         N_HOURS_U=N_HOURS_U,
#         N_HOURS_Y=N_HOURS_Y,
#     ),
#     n_trials=N_TRIALS,
# )


# --- Print Best Results ---
print("\n--- Optuna Study Complete ---")
print(f"Number of finished trials: {len(study.trials)}")
print("Best trial:")
best_trial = study.best_trial

print(f"  Value (Min Validation Loss): {best_trial.value:.6f}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")


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
        hidden_dim=64,  # Increased from 32
        num_gcn=3,  # Increased from 2
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


# Use a more appropriate optimizer configuration to slow down training
optimizer = torch.optim.AdamW(  # AdamW instead of Adam for better weight decay
    final_model.parameters(),
    lr=1e-5,  # Reduced learning rate to slow down initial convergence
    weight_decay=0.01,  # Stronger weight decay to prevent overfitting
    eps=1e-8,  # For numerical stability
)

# Use MSE loss with an L1 component to prevent extreme predictions
criterion = torch.nn.MSELoss()

# Create a less aggressive scheduler that allows learning to progress more gradually
scheduler = init_scheduler(
    optimizer,
    factor=0.7,  # Gentler learning rate reduction (70% of previous)
    patience=15,  # Wait longer before reducing LR
    min_lr=1e-8,  # Don't go below this learning rate
)

# Create data loaders with smaller batch size for more gradient updates
train_loader = DataLoader(
    train_dataset,
    batch_size=32,  # Smaller batch size for more frequent updates
    shuffle=True,  # Shuffle training data
    drop_last=True,  # Drop last batch if incomplete
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,  # Same batch size for consistent comparison
    shuffle=False,  # No shuffling for validation
    drop_last=False,  # Keep all validation data
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False,
)

# Add diagnostic prints to verify model setup
print("\n=== Model Training Setup ===")
print(f"Model type: {model_type}")
print(f"Input dim: {input_dim}, Output dim: {output_dim}")
print(
    f"Hidden dim: {final_model.convs[0].lin.out_channels}, GCN layers: {len(final_model.convs)}"
)
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"Weight decay: {optimizer.param_groups[0]['weight_decay']}")
print(f"Batch size: 32")
print(
    f"Using LR scheduler: ReduceLROnPlateau (factor={scheduler.factor}, patience={scheduler.patience})"
)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Add sample batch debug output
sample_batch = next(iter(train_loader))
sample_val_batch = next(iter(val_loader))
print("\n=== Data Diagnostics ===")
print(f"Sample batch shapes:")
print(f"  x_seq: {sample_batch.x_seq.shape}")
print(f"  y: {sample_batch.y.shape}")

# Move batch to device before model forward pass
sample_batch = sample_batch.to(device)
model_out = final_model(sample_batch)
print(f"  Model output shape: {model_out.shape}")
print(
    f"  Model output stats: min={model_out.min().item():.4f}, max={model_out.max().item():.4f}, mean={model_out.mean().item():.4f}"
)
print(
    f"  Target stats: min={sample_batch.y.min().item():.4f}, max={sample_batch.y.max().item():.4f}, mean={sample_batch.y.mean().item():.4f}"
)

# Print validation stats too to compare distributions
sample_val_batch = sample_val_batch.to(device)
val_out = final_model(sample_val_batch)
print(
    f"  Val output stats: min={val_out.min().item():.4f}, max={val_out.max().item():.4f}, mean={val_out.mean().item():.4f}"
)
print(
    f"  Val target stats: min={sample_val_batch.y.min().item():.4f}, max={sample_val_batch.y.max().item():.4f}, mean={sample_val_batch.y.mean().item():.4f}"
)

# Calculate and print initial loss values before training
train_loss = criterion(model_out, sample_batch.y.reshape(-1, output_dim))
val_loss = criterion(val_out, sample_val_batch.y.reshape(-1, output_dim))
print(f"  Initial train loss: {train_loss.item():.6f}")
print(f"  Initial val loss: {val_loss.item():.6f}")
print("\n=== Starting Training ===")

# Train with better early stopping settings
train_losses, val_losses = train(
    final_model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    N_EPOCHS,
    patience=patience,  # More patient early stopping
    use_scheduler=True,
    scheduler=scheduler,
)


plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Over Epochs")
plt.savefig(MODEL_PATH / f"final_training_loss_{model_type}_{current_time}.png")

# Save the model
torch.save(
    final_model.state_dict(),
    MODEL_PATH / f"final_model_{model_type}_{current_time}.pt",
)


# Load y_min and y_max
with open(ALL_DIR / "geometric_pkl" / "y_min_max.pkl", "rb") as f:
    y_min, y_max = pickle.load(f)

global_rmse, rmse_per_node = predict_and_evaluate(
    final_model, test_loader, device, output_dim, y_min, y_max, N_HOURS_Y
)

# Save results based on model type of rmse
results = {
    "model_type": model_type,
    "global_rmse": global_rmse,
    "rmse_per_node": rmse_per_node,
}


# Save the results to a CSV file
results_df = pd.DataFrame([results])
results_df.to_csv(
    MODEL_PATH / f"results_{model_type}_{current_time}.csv",
    index=False,
)
