import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import Adam
import sys
from tqdm import tqdm

# TensorBoard imports
from torch.utils.tensorboard import SummaryWriter
import datetime

# Import for recurrent GNN
from torch_geometric_temporal.nn.recurrent import GConvGRU

# Import for custom BatchedGConvGRU
from batched_graph_gru import BatchedGConvGRU

# Import for StructuredRecurrentGNN
from structured_recurrent_gnn import StructuredRecurrentGNN


def train_model(
    model,
    train_loader,
    val_loader,
    edges,
    edge_weights,
    device,
    epochs=100,
    patience=10,
    writer=None,
):
    """
    Train the GNN model using batched data loaders

    Args:
        model: The GNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        edges: Edge indices for the graph
        edge_weights: Edge weights for the graph
        device: Device to train on (cuda/cpu)
        epochs: Maximum number of training epochs
        patience: Early stopping patience

    Returns:
        Trained model and training history
    """
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # Ensure edges and weights are the correct data type
    edges = edges.to(device).long()
    edge_weights = edge_weights.to(device).float()

    # For tracking metrics
    best_val_loss = float("inf")
    counter = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": []}

    print(f"Training on {device}...")
    print(f"Edge shape: {edges.shape}, Edge weights shape: {edge_weights.shape}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0

        # Using tqdm for progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (x, y) in enumerate(progress_bar):
            # Move data to device
            x = x.to(device).float()
            y = y.to(device).float()

            # Print shapes for debugging (first batch of first epoch only)
            if epoch == 0 and batch_idx == 0:
                print(f"Input shape: {x.shape}, Target shape: {y.shape}")

            # Forward pass
            optimizer.zero_grad()
            y_hat = model(x, edges, edge_weights)  # Shape: [batch_size, 3, 1]

            # Extract NO2 values for 3 cities from the last timestep
            # Assuming the first 3 features of each timestep are NO2 for the 3 cities
            y_target = y[:, -1, :3]  # Shape: [batch_size, 3]

            # Reshape model output to match target shape
            y_hat = y_hat.squeeze(-1)  # Shape: [batch_size, 3]

            # Print shapes for debugging
            if epoch == 0 and batch_idx == 0:
                print(
                    f"Model output shape: {y_hat.shape}, Target NO2 shape: {y_target.shape}"
                )

            # Calculate loss
            loss = criterion(y_hat, y_target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            batch_count += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        # Calculate average training loss
        avg_train_loss = train_loss / batch_count
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).float()
                y = y.to(device).float()

                # Forward pass
                y_hat = model(x, edges, edge_weights)  # Shape: [batch_size, 3, 1]

                # Extract NO2 values for 3 cities from the last timestep (same as training)
                y_target = y[:, -1, :3]  # Shape: [batch_size, 3]

                # Reshape model output to match target shape
                y_hat = y_hat.squeeze(-1)  # Shape: [batch_size, 3]

                # Calculate loss on the extracted target
                loss = criterion(y_hat, y_target)

                # Update metrics
                val_loss += loss.item()
                val_batch_count += 1
        # Calculate average validation loss
        avg_val_loss = val_loss / val_batch_count
        history["val_loss"].append(avg_val_loss)

        # Print epoch results
        print(
            f"Epoch {epoch + 1}/{epochs} - Train loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}"
        )
        # Log to TensorBoard
        writer.add_scalars(
            "Loss",
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            epoch,
        )

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_PATH / "best_batched_gnn_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model if we found one
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def evaluate_model(model, test_loader, edges, edge_weights, device, loader=None):
    """
    Evaluate the model on test data

    Args:
        model: Trained GNN model
        test_loader: DataLoader for test data
        edges: Edge indices for the graph
        edge_weights: Edge weights for the graph
        device: Device to evaluate on (cuda/cpu)
        loader: NO2DatasetLoader for denormalization

    Returns:
        MSE score and predictions
    """
    model.eval()
    edges = edges.to(device).long()
    edge_weights = edge_weights.to(device).float()

    criterion = torch.nn.MSELoss()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    batch_count = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x = x.to(device).float()
            y = y.to(device).float()

            # Forward pass
            y_hat = model(x, edges, edge_weights)  # Shape: [batch_size, 3, 1]

            # Extract NO2 values for the 3 cities
            y_target = y[:, -1, :3]  # Shape: [batch_size, 3]

            # Reshape model output
            y_hat = y_hat.squeeze(-1)  # Shape: [batch_size, 3]

            # Store predictions and targets
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())

            # Calculate loss
            loss = criterion(y_hat, y_target)
            test_loss += loss.item()
            batch_count += 1

    # Calculate average loss
    test_loss /= batch_count

    # Combine all predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test RMSE: {np.sqrt(test_loss):.4f}")

    # If we have a loader with denormalization capability, use it
    if loader is not None and hasattr(loader, "denormalize_no2"):
        try:
            # Reshape for denormalization if needed
            orig_shape = all_preds.shape
            all_preds_reshaped = all_preds.reshape(-1, 1)
            all_targets_reshaped = all_targets.reshape(-1, 1)

            # Denormalize
            all_preds_inv = loader.denormalize_no2(all_preds_reshaped)
            all_targets_inv = loader.denormalize_no2(all_targets_reshaped)

            # Reshape back
            all_preds_inv = all_preds_inv.reshape(orig_shape)
            all_targets_inv = all_targets_inv.reshape(orig_shape)

            # Calculate MSE on original scale
            mse_original = ((all_preds_inv - all_targets_inv) ** 2).mean()
            print(f"Test MSE (original scale): {mse_original:.4f}")
            print(f"Test RMSE (original scale): {np.sqrt(mse_original):.4f} μg/m³")

            # Return denormalized predictions and targets
            return test_loss, all_preds_inv, all_targets_inv

        except Exception as e:
            print(f"Error during denormalization: {e}")

    return test_loss, all_preds, all_targets


def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        str(BASE_DIR / "results" / "batched_training_history.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_predictions(predictions, targets, city_names=None):
    """
    Plot test predictions against actual values for each city

    Args:
        predictions: Model predictions (denormalized)
        targets: Actual target values (denormalized)
        city_names: Names of cities for the plot labels
    """
    if city_names is None:
        city_names = ["Amsterdam", "Rotterdam", "Utrecht"]

    # Determine the number of cities (nodes) from the data shape
    n_cities = predictions.shape[1] if len(predictions.shape) > 1 else 1
    n_samples = len(predictions) // n_cities if n_cities > 1 else len(predictions)

    # Create time steps for x-axis
    time_steps = np.arange(n_samples)

    # Create subplots - one for each city
    fig, axes = plt.subplots(n_cities, 1, figsize=(12, 4 * n_cities))

    # Make axes iterable even if there's only one city
    if n_cities == 1:
        axes = [axes]

    # Plot predictions vs actual values for each city
    for i in range(n_cities):
        city_name = city_names[i] if i < len(city_names) else f"City {i}"

        if n_cities > 1:
            # Extract predictions and targets for this city
            city_preds = predictions[:, i]
            city_targets = targets[:, i]
        else:
            city_preds = predictions
            city_targets = targets

        # Plot the data
        axes[i].plot(time_steps, city_targets, "b-", label="Actual")
        axes[i].plot(time_steps, city_preds, "r--", label="Predicted")
        axes[i].set_title(f"NO2 Predictions for {city_name}")
        axes[i].set_xlabel("Time (hours)")
        axes[i].set_ylabel("NO2 (μg/m³)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Add error metrics in the plot
        mse = np.mean((city_preds - city_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(city_preds - city_targets))

        # Display metrics on the plot
        axes[i].text(
            0.02,
            0.92,
            f"RMSE: {rmse:.2f} μg/m³\nMAE: {mae:.2f} μg/m³",
            transform=axes[i].transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(
        str(BASE_DIR / "results" / "batched_test_predictions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Also create a scatter plot of predicted vs actual values
    plt.figure(figsize=(10, 8))

    # Different colors for each city
    colors = ["blue", "red", "green", "orange", "purple"]

    for i in range(n_cities):
        if n_cities > 1:
            city_preds = predictions[:, i]
            city_targets = targets[:, i]
        else:
            city_preds = predictions
            city_targets = targets

        plt.scatter(
            city_targets,
            city_preds,
            alpha=0.5,
            color=colors[i % len(colors)],
            label=city_names[i] if i < len(city_names) else f"City {i}",
        )

    # Add reference line (perfect predictions)
    max_val = max(np.max(predictions), np.max(targets))
    min_val = min(np.min(predictions), np.min(targets))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.8)

    plt.xlabel("Actual NO2 (μg/m³)")
    plt.ylabel("Predicted NO2 (μg/m³)")
    plt.title("Predicted vs Actual NO2 Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    plt.savefig(
        str(BASE_DIR / "results" / "batched_prediction_scatter.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    """Main function to process data and train the model using built-in batching"""
    # Initialize the NO2DatasetLoader with index=True to enable batched processing
    N_LAGS = 72
    BATCH_SIZE = 32

    print("Initializing NO2 dataset loader with batching capability...")
    # Set force_reload=False to ensure the cached dataset is used
    loader = NO2DatasetLoader(
        index=True,
        only_no2=False,
        force_reload=False,
        cache_file="index_dataset_l72_b32_r0.7-0.15-0.15_s0.2.pkl",
    )

    print(f"Loading dataset with {N_LAGS} lags, using built-in batching...")
    # The cache file should be named something like "index_dataset_l72_b32_r0.7-0.15-0.15_s0.2.pkl"
    # This matches what we see in the data_gnn directory
    train_loader, val_loader, test_loader, edges, edge_weights = (
        loader.get_index_dataset(
            lags=N_LAGS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            only_no2=False,
            sample_size=0.20,
            horizon=24,  # Specify the horizon parameter with a value (same as lags for 1-step prediction)
            ratio=(0.7, 0.15, 0.15),
            cache=True,  # Make sure caching is enabled
            cache_suffix=None,  # No additional suffix needed
        )
    )
    print("Batched dataset loaders created successfully.")

    # Set device for GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get feature dimensionality (number of lag timesteps per variable)
    # In the batched format, we need to handle this differently
    # A single batch from the dataloader contains [batch_size, features]
    # Get a sample from the data to determine dimensions
    sample_batch = next(iter(train_loader))
    sample_x, sample_y = sample_batch

    print(f"Sample batch shape - X: {sample_x.shape}, y: {sample_y.shape}")

    # Calculate number of variables and features
    # For NO2 dataset with default settings, we have 7 variables (NO2 + weather vars)
    # Each with N_LAGS timesteps
    num_vars = 7 if not loader.only_no2 else 1

    # Define model and move to device
    model = StructuredRecurrentGNN(
        node_features=N_LAGS,  # Each variable has N_LAGS timesteps
        num_vars=num_vars,
        num_lags=N_LAGS,
        hidden_channels=64,
        out_channels=3,  # Predicting NO2 for 3 cities
        k=3,
    ).to(device)

    # Set up TensorBoard for visualization
    log_dir = (
        BASE_DIR
        / "results"
        / "logs"
        / f"batched_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Add model graph to TensorBoard (if possible with batched data)
    try:
        writer.add_graph(
            model,
            input_to_model=(
                sample_x[:1].to(device),
                edges.to(device),
                edge_weights.to(device),
            ),
        )
    except Exception as e:
        print(f"Could not add model graph to TensorBoard: {e}")

    # Train the model using the batched data loaders
    print("Starting model training with batched data...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        edges=edges,
        edge_weights=edge_weights,
        device=device,
        epochs=100,
        patience=10,
        writer=writer,
    )

    # Close TensorBoard writer
    writer.close()

    # Plot training history
    plot_training_history(history)

    # Evaluate model on test data
    print("Evaluating model with batched test data...")
    test_loss, predictions, targets = evaluate_model(
        model=model,
        test_loader=test_loader,
        edges=edges,
        edge_weights=edge_weights,
        device=device,
        loader=loader,
    )

    # Plot predictions if denormalized values were returned
    if predictions is not None and targets is not None:
        plot_predictions(predictions, targets, city_names=cities)
        print("Prediction plots saved to results directory")

        # Save predictions for later analysis
        np.savez(
            str(BASE_DIR / "results" / "batched_predictions.npz"),
            predictions=predictions,
            targets=targets,
        )

    # Save the final model
    torch.save(model.state_dict(), MODEL_PATH / "final_batched_gnn_model.pt")
    print("Model evaluation complete!")


if __name__ == "__main__":
    sys.path.append(str(Path.cwd()))
    from graph_modelling.datasets.no2_dataset import NO2DatasetLoader

    # Set paths and constants
    BASE_DIR = Path.cwd().parent
    MODEL_PATH = BASE_DIR / "results" / "models"
    DATA_DIR = BASE_DIR / "data" / "data_gnn"
    ALL_DIR = DATA_DIR / "all"
    RAW_DATA_DIR = BASE_DIR / "data" / "data_raw"

    print("BASE_DIR: ", BASE_DIR)
    print("MODEL_PATH: ", MODEL_PATH)
    print("ALL_DIR: ", ALL_DIR)

    # Set random seed for reproducibility
    torch.manual_seed(34)

    # Constants
    N_HOURS_U = 72  # number of hours to use for input
    N_HOURS_Y = 24  # number of hours to predict
    N_HOURS_STEP = 24  # "sampling rate" in hours
    CONTAMINANTS = ["NO2"]  # Only predicting NO2
    WEATHER_VARS = [
        "P",
        "SQ",
        "WD",
        "Wvh",
        "dewP",
        "temp",
    ]  # Weather variables to include
    cities = ["amsterdam", "rotterdam", "utrecht"]
    SAVE_DIR = ALL_DIR / "geometric_pkl"

    # Ensure directories exist
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "scalers").mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "results").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "results" / "graph_plots").mkdir(parents=True, exist_ok=True)

    # Run main function
    main()
