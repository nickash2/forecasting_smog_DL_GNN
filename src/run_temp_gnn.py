# %%
import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader  # <-- Added DataLoader here
from pathlib import Path
import os
import datetime
import optuna  # <-- Import Optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Keep existing imports
# from modelling import get_dataframes # Assuming this is not strictly needed for this script anymore
# from modelling.metrics.metricstracker import MetricsTracker
from graph_modelling.utils.load_data import (
    load_train_val_data,
    load_test_data,
    read_csv_files,
)
from graph_modelling.models.temporalgnn import TemporalGNN  # Keep model import

os.chdir(Path().cwd())  # Assuming the script is run from the correct directory now

# %%
HABROK = bool(0)
BASE_DIR = Path.cwd()
MODEL_PATH = BASE_DIR / "results" / "models"
DATA_DIR = BASE_DIR / "data" / "data_combined"
ALL_DIR = DATA_DIR / "all"

print("BASE_DIR: ", BASE_DIR)
print("MODEL_PATH: ", MODEL_PATH)
print("ALL_DIR: ", ALL_DIR)

# Ensure necessary directories exist (especially for saving models if needed later)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

torch.manual_seed(34)

N_HOURS_U = 72
N_HOURS_Y = 24
N_HOURS_STEP = 24
CONTAMINANTS = ["NO2", "O3"]
target_features = len(CONTAMINANTS)  # Define target_features based on CONTAMINANTS

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Tracker commented out, can be re-enabled if needed within the objective or outside
# tracker = MetricsTracker(...)


# %%
# --- Data Loading and Preprocessing Functions (Keep as they are) ---
def get_data_files(city_path, data_type):
    """
    Get all feature and label files for a city for a specific data type.

    Args:
        city_path (Path): Path to the city directory.
        data_type (str): Type of data (train, val, or test).

    Returns:
        tuple: Lists of feature and label files.
    """
    feature_files = sorted(
        [
            f
            for f in os.listdir(city_path)
            if f.startswith(data_type) and f.endswith("_u.csv")
        ]
    )

    label_files = sorted(
        [
            f
            for f in os.listdir(city_path)
            if f.startswith(data_type) and f.endswith("_y.csv")
        ]
    )

    return feature_files, label_files


def read_csv_files(
    city_path, feature_files, label_files, drop_datetime=True, city_name=None
):
    """
    Read feature and label CSV files.

    Args:
        city_path (Path): Path to the city directory.
        feature_files (list): List of feature file names.
        label_files (list): List of label file names.
        drop_datetime (bool, optional): Whether to drop DateTime column. Defaults to True.

    Returns:
        tuple: Lists of feature and label DataFrames.
    """
    feature_dfs = []
    label_dfs = []

    for feat_file, label_file in zip(feature_files, label_files):
        feat_df = pd.read_csv(os.path.join(city_path, feat_file), delimiter=";")
        label_df = pd.read_csv(os.path.join(city_path, label_file), delimiter=";")
        if city_name is not None:
            feat_df.insert(
                feat_df.columns.get_loc("DateTime") + 1, "city_name", city_name
            )
            label_df.insert(
                label_df.columns.get_loc("DateTime") + 1, "city_name", city_name
            )
        if drop_datetime:
            # Keep DateTime for sorting, drop later if needed before conversion to tensor
            # Let's assume the original load_gnn_data handles this correctly.
            pass  # We'll handle dropping later

        feature_dfs.append(feat_df)
        label_dfs.append(label_df)

    return feature_dfs, label_dfs


def minmax_normalize_arr(arr, arr_min, arr_max):
    # Normalize with provided min and max, with a small epsilon to avoid division by zero
    return (arr - arr_min) / (arr_max - arr_min + 1e-8)


cities = ["amsterdam", "rotterdam", "utrecht"]


def load_gnn_data(
    split_type="train", drop_datetime=False, save=False
):  # Keep drop_datetime False initially
    f = pd.DataFrame()
    l = pd.DataFrame()
    for idx, city in enumerate(cities):
        city_path = ALL_DIR / city
        if not city_path.exists():
            print(f"Warning: Directory not found {city_path}")
            continue  # Skip if city data dir doesn't exist
        x_files, y_files = get_data_files(city_path, split_type)
        if not x_files or not y_files:
            print(f"Warning: No {split_type} files found for {city}")
            continue
        x_dfs, y_dfs = read_csv_files(
            city_path,
            x_files,
            y_files,
            drop_datetime=False,
            city_name=idx,  # Keep DateTime for sorting
        )
        for element in x_dfs:
            f = pd.concat([f, element], axis=0)
        for element in y_dfs:
            l = pd.concat([l, element], axis=0)

    if f.empty or l.empty:
        raise ValueError(f"No data loaded for split type: {split_type}")

    if save:
        # Ensure parent directory exists before saving
        ALL_DIR.mkdir(parents=True, exist_ok=True)
        f.to_csv(ALL_DIR / f"{split_type}_u.csv", index=False, sep=";")
        l.to_csv(ALL_DIR / f"{split_type}_y.csv", index=False, sep=";")
    return f, l


# %%
# --- Load and Prepare Data (Outside Optuna Objective) ---
print("Loading data...")
X_train, y_train = load_gnn_data("train", drop_datetime=False)
X_val, y_val = load_gnn_data("val", drop_datetime=False)
X_test, y_test = load_gnn_data("test", drop_datetime=False)

# Combine temporarily for consistent processing if needed, but keep splits separate
X_all = pd.concat([X_train, X_val, X_test], axis=0)
y_all = pd.concat([y_train, y_val, y_test], axis=0)
print(f"Total X shape: {X_all.shape}, Total y shape: {y_all.shape}")

# Ensure data is sorted by time before reshaping
X_all_sorted = X_all.sort_values(by=["DateTime", "city_name"])
y_all_sorted = y_all.sort_values(by=["DateTime", "city_name"])

# Check if sorting resulted in data
if X_all_sorted.empty or y_all_sorted.empty:
    raise ValueError("Data is empty after sorting. Check loading and DateTime values.")


num_nodes = len(cities)
num_timesteps_all = len(X_all_sorted) // num_nodes
num_features = X_all_sorted.shape[1] - 2  # Exclude DateTime, city_name
target_features = y_all_sorted.shape[1] - 2  # Exclude DateTime, city_name


print(
    f"Num Timesteps: {num_timesteps_all}, Num Nodes: {num_nodes}, Num Features: {num_features}, Target Features: {target_features}"
)

# Reshape to (num_timesteps, num_nodes, num_features)
x_all = X_all_sorted.drop(columns=["DateTime", "city_name"]).values.reshape(
    num_timesteps_all, num_nodes, num_features
)
y_all = y_all_sorted.drop(columns=["DateTime", "city_name"]).values.reshape(
    num_timesteps_all, num_nodes, target_features
)

# Convert to PyTorch tensor
x_all = torch.tensor(x_all, dtype=torch.float)
y_all = torch.tensor(y_all, dtype=torch.float)

print("Full Node Features Shape:", x_all.shape)
print("Full Target Shape:", y_all.shape)

# Define edge_index (assuming fully connected between the 3 cities)
edge_index = torch.tensor(
    [
        [0, 0, 1, 1, 2, 2],  # Source nodes
        [1, 2, 0, 2, 0, 1],  # Target nodes
    ],
    dtype=torch.long,
)


# --- Create Sliding Windows Function ---
def create_sliding_windows(X, Y, window_size, forecast_horizon):
    X_windows, Y_windows = [], []
    # Ensure we don't go out of bounds
    max_start_idx = len(X) - window_size - forecast_horizon
    if max_start_idx < 0:
        raise ValueError(
            f"Dataset too small ({len(X)} timesteps) for window_size={window_size} and forecast_horizon={forecast_horizon}"
        )

    for i in range(max_start_idx + 1):
        X_windows.append(X[i : i + window_size])
        Y_windows.append(Y[i + window_size : i + window_size + forecast_horizon])
    if not X_windows:  # Check if any windows were created
        return torch.empty(0), torch.empty(0)  # Return empty tensors
    return torch.stack(X_windows), torch.stack(Y_windows)


# Generate sliding window data
X_windows, Y_windows = create_sliding_windows(x_all, y_all, N_HOURS_U, N_HOURS_Y)

if X_windows.numel() == 0:
    raise ValueError(
        "No sliding windows created. Check data size and window parameters."
    )


print(
    f"X_windows shape: {X_windows.shape}"
)  # (num_samples, N_HOURS_U, num_nodes, num_features)
print(
    f"Y_windows shape: {Y_windows.shape}"
)  # (num_samples, N_HOURS_Y, num_nodes, target_features)

num_samples, window_size, _, num_features_check = X_windows.shape
_, forecast_horizon, _, target_features_check = Y_windows.shape

# Sanity checks
assert window_size == N_HOURS_U
assert forecast_horizon == N_HOURS_Y
assert num_features_check == num_features
assert target_features_check == target_features


# Flatten features and targets per node for GNN input/output
# Input: Flatten time window and features for each node
# Output: Flatten forecast horizon and targets for each node
X_windows_flat = X_windows.permute(0, 2, 1, 3).reshape(
    num_samples, num_nodes, window_size * num_features
)
Y_windows_flat = Y_windows.permute(0, 2, 1, 3).reshape(
    num_samples, num_nodes, forecast_horizon * target_features
)

print(
    f"X_windows_flat shape: {X_windows_flat.shape}"
)  # (num_samples, num_nodes, window_size*num_features)
print(
    f"Y_windows_flat shape: {Y_windows_flat.shape}"
)  # (num_samples, num_nodes, forecast_horizon*target_features)

input_dim = window_size * num_features
output_dim = (
    forecast_horizon * target_features
)  # This is the correct output dim for the model

# Create PyG Data objects
dataset = []
for i in range(num_samples):
    data = Data(
        x=X_windows_flat[i],  # shape: (num_nodes, input_dim)
        edge_index=edge_index,
        y=Y_windows_flat[i],  # shape: (num_nodes, output_dim)
    )
    dataset.append(data)

print(f"Created dataset with {len(dataset)} graphs.")

# Split dataset chronologically
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size : train_size + val_size]
test_dataset = dataset[train_size + val_size :]

print(
    f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
)

if not train_dataset:
    raise ValueError("Training dataset is empty after split.")
if not val_dataset:
    raise ValueError("Validation dataset is empty after split.")
if not test_dataset:
    raise ValueError("Test dataset is empty after split.")


# --- Compute Normalization Stats on Training Set ---
def get_min_max(dataset, attr_name):
    if not dataset:
        return None, None  # Handle empty dataset case
    all_data = torch.cat(
        [getattr(data, attr_name) for data in dataset], dim=0
    )  # Concatenate across nodes dimension
    arr = all_data.numpy()
    arr_min = arr.min(axis=0, keepdims=True)
    arr_max = arr.max(axis=0, keepdims=True)
    return arr_min, arr_max


x_min, x_max = get_min_max(train_dataset, "x")
y_min, y_max = get_min_max(train_dataset, "y")

# Check if min/max calculation worked
if x_min is None or y_min is None:
    raise ValueError("Could not compute min/max. Training data might be empty.")


print(
    "x_min shape:", x_min.shape, "x_max shape:", x_max.shape
)  # Should be (1, input_dim)
print(
    "y_min shape:", y_min.shape, "y_max shape:", y_max.shape
)  # Should be (1, output_dim)


# --- Normalize Datasets ---
def normalize_dataset(dataset, x_min, x_max, y_min, y_max):
    normalized_dataset = []
    for data in dataset:
        # Important: Create a *copy* to avoid modifying the original data object
        # which might be used by other references (less critical here, but good practice)
        data_copy = data.clone()

        # Normalize x:
        x_arr = data_copy.x.numpy()
        x_norm = minmax_normalize_arr(x_arr, x_min, x_max)
        data_copy.x = torch.tensor(x_norm, dtype=torch.float)

        # Normalize y:
        y_arr = data_copy.y.numpy()
        y_norm = minmax_normalize_arr(y_arr, y_min, y_max)
        data_copy.y = torch.tensor(y_norm, dtype=torch.float)
        normalized_dataset.append(data_copy)
    return normalized_dataset


print("Normalizing datasets...")
train_dataset_norm = normalize_dataset(train_dataset, x_min, x_max, y_min, y_max)
val_dataset_norm = normalize_dataset(val_dataset, x_min, x_max, y_min, y_max)
test_dataset_norm = normalize_dataset(test_dataset, x_min, x_max, y_min, y_max)


def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Define fixed parameters for training run
    num_epochs = 500
    patience = 10

    train_loader = DataLoader(train_dataset_norm, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset_norm, batch_size=batch_size, shuffle=False)

    # Create model instance for this trial
    model = TemporalGNN(
        input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    factor = trial.suggest_float(
        "plateau_factor", 0.1, 0.7, step=0.1
    )  # Factor to reduce LR by
    patience_sched = trial.suggest_int(
        "plateau_patience", 3, 10
    )  # Scheduler patience (epochs without improvement)
    min_lr = trial.suggest_float(
        "plateau_min_lr", 1e-8, 1e-6, log=True
    )  # Minimum LR threshold

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",  # Reduce LR when the monitored metric has stopped decreasing
        factor=factor,
        patience=patience_sched,
        min_lr=min_lr,
        verbose=False,  # Set to True if you want scheduler messages during training
    )
    print(
        f"  Using ReduceLROnPlateau: factor={factor}, patience={patience_sched}, min_lr={min_lr}"
    )

    print(f"\n--- Trial {trial.number} ---")
    print(
        f"Params: lr={lr:.6f}, hidden_dim={hidden_dim}, weight_decay={weight_decay:.6f}"
    )

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:  # Removed tqdm for cleaner Optuna logs
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            # Target shape is (batch_size * num_nodes, output_dim)
            y_target = batch.y.view(-1, output_dim)

            # Check shape consistency (important!)
            if out.shape != y_target.shape:
                print(
                    f"Warning: Shape mismatch in trial {trial.number}, epoch {epoch + 1}. Output: {out.shape}, Target: {y_target.shape}. Skipping batch."
                )
                continue  # Skip batch if shapes don't match

            loss = criterion(out, y_target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                y_target = batch.y.view(-1, output_dim)

                if out.shape != y_target.shape:
                    print(
                        f"Warning: Shape mismatch during validation trial {trial.number}, epoch {epoch + 1}. Skipping batch."
                    )
                    continue  # Skip batch

                loss = criterion(out, y_target)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)  # Step the scheduler with validation loss

        print(
            f"Trial {trial.number} - Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

        trial.report(avg_val_loss, epoch)  # Report intermediate value to Optuna

        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch + 1}.")
            raise optuna.exceptions.TrialPruned()

        # Early stopping based on validation loss improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Trial {trial.number} early stopping triggered at epoch {epoch + 1}."
            )
            break  # Stop training for this trial

    return best_val_loss


# %%
# --- Run Optuna Study ---
study_name = f"temporal-gnn-tuning-{current_time}"
storage_name = f"sqlite:///temp_gnn_tuning.db"  # Store results in a SQLite DB

study = optuna.create_study(
    direction="minimize",  # We want to minimize validation loss
    study_name=study_name,
    storage=storage_name,  # Persist study results
    load_if_exists=True,  # Resume study if it already exists
    pruner=optuna.pruners.HyperbandPruner(),
)

print(f"Using Optuna study name: {study_name}")
print(f"Results will be saved to: {storage_name}")

# Start optimization
n_trials = 1  # Number of trials Optuna should run
study.optimize(objective, n_trials=n_trials, timeout=3600)  # Added timeout (1 hour)

# --- Print Best Results ---
print("\n--- Optuna Study Complete ---")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial:")
best_trial = study.best_trial

print(f"  Value (Min Validation Loss): {best_trial.value:.6f}")
print(f"  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# %%
# --- Optional: Train Final Model with Best Hyperparameters and Evaluate on Test Set ---
print("\n--- Training Final Model with Best Hyperparameters ---")

best_params = study.best_params
final_lr = best_params["lr"]
final_hidden_dim = best_params["hidden_dim"]
final_weight_decay = best_params["weight_decay"]
# Use a slightly larger number of epochs or rely on early stopping for final training
final_num_epochs = 150
final_patience = 15  # Can use the same or slightly different patience
final_batch_size = 32  # Use the same batch size as in trials or the best one if tuned

# Create final DataLoaders (can combine train+val for final training if desired)
# Using only train_norm here, as is standard practice before test evaluation.
final_train_loader = DataLoader(
    train_dataset_norm, batch_size=final_batch_size, shuffle=True
)
final_val_loader = DataLoader(
    val_dataset_norm, batch_size=final_batch_size, shuffle=False
)  # Still use val for early stopping
final_test_loader = DataLoader(
    test_dataset_norm, batch_size=final_batch_size, shuffle=False
)

# Create final model
final_model = TemporalGNN(
    input_dim=input_dim, output_dim=output_dim, hidden_dim=final_hidden_dim
)
final_model = final_model.to(device)
final_optimizer = torch.optim.Adam(
    final_model.parameters(), lr=final_lr, weight_decay=final_weight_decay
)
final_criterion = torch.nn.MSELoss()

best_final_val_loss = float("inf")
final_patience_counter = 0
final_train_losses = []
final_val_losses = []

for epoch in range(final_num_epochs):
    # Training
    final_model.train()
    epoch_train_loss = 0
    with tqdm(
        final_train_loader,
        desc=f"Final Train Epoch {epoch + 1}/{final_num_epochs}",
        unit="batch",
        leave=False,
    ) as pbar:
        for batch in pbar:
            batch = batch.to(device)
            final_optimizer.zero_grad()
            out = final_model(batch)
            y_target = batch.y.view(-1, output_dim)
            if out.shape != y_target.shape:
                continue  # Skip problematic batches
            loss = final_criterion(out, y_target)
            loss.backward()
            final_optimizer.ste
