import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import Adam
import sys
from tqdm import tqdm

# Add TensorFlow and TensorBoard imports
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import datetime

# Import for recurrent GNN
from torch_geometric_temporal.nn.recurrent import GConvGRU


class BasicGNN(torch.nn.Module):
    def __init__(
        self, num_node_features, hidden_channels=16, out_channels=3
    ):  # Changed output channels to 3
        super(BasicGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(
            hidden_channels, out_channels
        )  # Output 3 channels to match target

    def forward(self, x, edge_index, edge_weight=None):
        # First Graph Convolution
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Graph Convolution
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        # Output layer
        x = self.out(x)
        return x


# Define a recurrent GNN model for spatio-temporal dependencies
class RecurrentGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels=32, out_channels=3, k=2):
        super(RecurrentGNN, self).__init__()
        self.recurrent = GConvGRU(node_features, hidden_channels, k)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Apply the GRU with graph convolution
        h = self.recurrent(x, edge_index, edge_weight)
        # Apply ReLU activation
        h = F.relu(h)
        # Apply linear layer for output
        h = self.linear(h)
        return h


# Function to visualize the graph
def visualize_graph(edge_index, edge_attr=None, node_labels=None):
    """
    Visualize the graph structure with NetworkX

    Args:
        edge_index: Tensor containing the edge indices
        edge_attr: Tensor containing edge weights/attributes (optional)
        node_labels: List of node labels (optional)
    """
    # Create a new graph
    G = nx.DiGraph()

    # Add nodes
    num_nodes = max(edge_index[0].max(), edge_index[1].max()) + 1
    G.add_nodes_from(range(num_nodes))

    # Add edges with their weights
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        weight = 1.0 if edge_attr is None else edge_attr[i].item()
        G.add_edge(src, dst, weight=weight)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Generate positions for the nodes
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=700)

    # Draw edges with width proportional to weight
    edge_widths = (
        [G[u][v]["weight"] / 10 for u, v in G.edges()]
        if edge_attr is not None
        else [1.0 for _ in G.edges()]
    )
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowsize=20, alpha=0.7)

    # Add node labels
    if node_labels is None:
        node_labels = {i: f"City {i}" for i in range(num_nodes)}
    else:
        node_labels = {i: label for i, label in enumerate(node_labels)}
    nx.draw_networkx_labels(G, pos, node_labels, font_size=12)

    # Add edge weights as labels
    if edge_attr is not None:
        edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title("NO2 Monitoring Network Graph", fontsize=15)
    plt.axis("off")

    # Save the graph visualization
    plt.savefig(
        str(BASE_DIR / "results" / "graph_plots" / "graph_visualization.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def train_model(
    model,
    train_loader,
    val_loader,
    edges,
    edge_weights,
    device,
    epochs=100,
    patience=10,
):
    """
    Train the GNN model and return the trained model

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
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Ensure edges and weights are the correct data type
    edges = edges.to(device).long()  # Edge indices should be long
    edge_weights = edge_weights.to(device).float()  # Edge weights should be float

    # For tracking metrics
    best_val_loss = float("inf")
    counter = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"Training on {device}...")
    print(f"Edge shape: {edges.shape}, Edge weights shape: {edge_weights.shape}")

    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            # Convert data types to ensure consistency
            x = x.to(device).float()
            y = y.to(device).float()

            # Print shapes for debugging (first batch of first epoch only)
            if epoch == 0 and batch_idx == 0:
                print(f"Input shape: {x.shape}, Target shape: {y.shape}")

            optimizer.zero_grad()

            # Forward pass
            out = model(x, edges, edge_weights)

            # Check if shapes match, if not, adjust the output to match target shape
            if out.shape != y.shape:
                print(f"Shape mismatch: out={out.shape}, y={y.shape}")
                # If the model predicts only one channel but target has 3 channels
                # We need to make sure predictions match targets in dimensions
                if out.shape[-1] < y.shape[-1]:
                    # Expand model output to match target dimensions
                    out = out.expand(-1, -1, y.shape[-1])
                    print(f"Expanded output shape to: {out.shape}")

            loss = criterion(out, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x = x.to(device).float()
                y = y.to(device).float()
                out = model(x, edges, edge_weights)
                loss = criterion(out, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_PATH / "best_recurrent_gnn_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH / "best_recurrent_gnn_model.pt"))
    return model, history


def evaluate_model(model, test_loader, edges, edge_weights, device, scaler=None):
    """
    Evaluate the model on test data

    Args:
        model: Trained GNN model
        test_loader: DataLoader for test data
        edges: Edge indices for the graph
        edge_weights: Edge weights for the graph
        device: Device to evaluate on (cuda/cpu)
        scaler: Optional scaler to inverse transform predictions

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

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device).float()
            y = y.to(device).float()

            # Forward pass
            out = model(x, edges, edge_weights)

            # Store predictions and targets
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())

            # Calculate loss
            loss = criterion(out, y)
            test_loss += loss.item()

    # Calculate average loss
    test_loss /= len(test_loader)

    # Combine all predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    print(f"Test MSE: {test_loss:.4f}")

    # If we have a scaler, inverse transform and recalculate metrics
    if scaler is not None:
        try:
            # Reshape for inverse transform if needed
            orig_shape = all_preds.shape
            all_preds_reshaped = all_preds.reshape(-1, 1)
            all_targets_reshaped = all_targets.reshape(-1, 1)

            # Inverse transform
            all_preds_inv = scaler.inverse_transform(all_preds_reshaped)
            all_targets_inv = scaler.inverse_transform(all_targets_reshaped)

            # Reshape back
            all_preds_inv = all_preds_inv.reshape(orig_shape)
            all_targets_inv = all_targets_inv.reshape(orig_shape)

            # Calculate MSE on original scale
            mse_original = ((all_preds_inv - all_targets_inv) ** 2).mean()
            print(f"Test MSE (original scale): {mse_original:.4f}")
        except Exception as e:
            print(f"Error during inverse transformation: {e}")

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
        str(BASE_DIR / "results" / "training_history.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    """Main function to process data and train the model"""
    from torch_geometric_temporal.signal import temporal_signal_split

    loader = NO2DatasetLoader(only_no2=True)
    print("Loading dataset...")
    dataset = loader.get_dataset(lags=24, only_no2=True, sample_size=0.20)
    print("Dataset loaded.")

    print("Splitting dataset into train and test sets...")
    # Create a validation set by splitting the training data
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
    # Further split train into train and validation
    train_dataset, val_dataset = temporal_signal_split(train_dataset, train_ratio=0.8)

    # Get number of features from first snapshot
    snapshot = train_dataset[0]
    num_node_features = snapshot.x.size(1)

    # Set device for GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model and move to device
    model = RecurrentGNN(
        node_features=num_node_features, hidden_channels=64, out_channels=1, k=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Initialize history
    history = {"train_loss": [], "val_loss": []}

    # Set up TensorBoard for visualization
    log_dir = (
        BASE_DIR
        / "results"
        / "logs"
        / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Add model graph to TensorBoard
    # Get a sample snapshot for tracing the model
    sample_x = snapshot.x.to(device)
    sample_edge_index = snapshot.edge_index.to(device)
    sample_edge_attr = snapshot.edge_attr.to(device)
    writer.add_graph(model, (sample_x, sample_edge_index, sample_edge_attr))

    # Training loop
    model.train()
    num_epochs = 50

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        train_loss = 0.0
        num_training_batches = 0

        for time, snapshot in enumerate(train_dataset):
            x = snapshot.x.to(device)
            y = snapshot.y.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)

            optimizer.zero_grad()
            y_hat = model(x, edge_index, edge_attr)
            loss = torch.mean((y_hat.squeeze() - y) ** 2)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_training_batches += 1

        avg_train_loss = train_loss / num_training_batches
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for time, snapshot in enumerate(val_dataset):
                x = snapshot.x.to(device)
                y = snapshot.y.to(device)
                edge_index = snapshot.edge_index.to(device)
                edge_attr = snapshot.edge_attr.to(device)

                y_hat = model(x, edge_index, edge_attr)
                loss = torch.mean((y_hat.squeeze() - y) ** 2)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        history["val_loss"].append(avg_val_loss)

        # Log metrics to TensorBoard - both train and val loss on same chart
        writer.add_scalars(
            "Loss", {"train": avg_train_loss, "validation": avg_val_loss}, epoch
        )

        # Log learning rate
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # Log model parameters as histograms
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f"Parameters/{name}", param.data, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}"
        )

    writer.close()

    plot_training_history(history)

    # Evaluate model on test data
    print("Evaluating model on test data...")
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for time, snapshot in enumerate(test_dataset):
            x = snapshot.x.to(device)
            y = snapshot.y.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)

            y_hat = model(x, edge_index, edge_attr)
            loss = torch.mean((y_hat.squeeze() - y) ** 2)

            # Store predictions and targets for unscaling
            all_preds.append(y_hat.squeeze().cpu().numpy())
            all_targets.append(y.cpu().numpy())

            test_loss += loss.item()

    # Average test loss
    test_loss /= time + 1
    print(f"Test MSE (scaled): {test_loss:.6f}")
    print(f"Test RMSE (scaled): {np.sqrt(test_loss):.6f}")

    # Convert lists to numpy arrays
    all_preds = np.concatenate([pred.reshape(-1) for pred in all_preds])
    all_targets = np.concatenate([target.reshape(-1) for target in all_targets])

    # Calculate unscaled metrics
    try:
        # Use loader's built-in denormalization function
        unscaled_preds = loader.denormalize_no2(all_preds)
        unscaled_targets = loader.denormalize_no2(all_targets)

        # Calculate unscaled MSE and RMSE
        unscaled_mse = np.mean((unscaled_preds - unscaled_targets) ** 2)
        unscaled_rmse = np.sqrt(unscaled_mse)

        print(f"Test MSE (unscaled): {unscaled_mse:.4f}")
        print(f"Test RMSE (unscaled): {unscaled_rmse:.4f} μg/m³")

        # Save predictions for visualization if needed
        np.savez(
            str(BASE_DIR / "results" / "predictions.npz"),
            predictions=unscaled_preds,
            targets=unscaled_targets,
        )

    except Exception as e:
        print(f"Error during denormalization: {e}")

    # Save the final model
    torch.save(model.state_dict(), MODEL_PATH / "final_recurrent_gnn_model.pt")
    print("Model evaluation complete!")


if __name__ == "__main__":
    sys.path.append(str(Path.cwd()))
    from graph_modelling.datasets.no2_dataset import NO2DatasetLoader

    HABROK = bool(0)
    BASE_DIR = Path.cwd().parent
    MODEL_PATH = BASE_DIR / "results" / "models"
    DATA_DIR = BASE_DIR / "data" / "data_gnn"
    ALL_DIR = DATA_DIR / "all"
    RAW_DATA_DIR = BASE_DIR / "data" / "data_raw"

    print("BASE_DIR: ", BASE_DIR)
    print("MODEL_PATH: ", MODEL_PATH)
    print("ALL_DIR: ", ALL_DIR)

    torch.manual_seed(34)

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

    # Create directories if they don't exist
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "results").mkdir(parents=True, exist_ok=True)

    # Run main function
    main()
