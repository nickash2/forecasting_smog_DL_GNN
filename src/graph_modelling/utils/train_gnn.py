import torch.optim as optim
from torch_geometric.data import DataLoader
import torch
import sys

sys.path.append("../../")

from modelling.metrics.metricstracker import MetricsTracker


def init_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def init_metrics_tracker(BASE_DIR):
    tracker = MetricsTracker(
        experiment_name="GNN",
        log_dir=BASE_DIR / "src" / "results" / "energy_logs",
        track_energy=True,
        track_tensorboard=True,
        track_memory=True,
        verbose=True,
    )
    return tracker


def train_epoch(
    epoch: int, train_data, model, criterion, optimizer, val_data, tracker=None
):
    return _train_epoch_impl(epoch, train_data, model, criterion, optimizer, val_data)


def _train_epoch_impl(epoch: int, train_data, model, criterion, optimizer, val_data):
    model.train()
    optimizer.zero_grad()

    # Training step for the single graph (one batch)
    optimizer.zero_grad()
    output = model(train_data)
    loss = criterion(output, train_data.y)
    loss.backward()
    optimizer.step()

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_output = model(val_data)
        val_loss += criterion(val_output, val_data.y).item()

    # Calculate average validation loss
    val_loss /= 1  # Since you're passing just one graph (i.e., no batching)

    return {"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss}


def train_gnn(
    model, train_data, val_data, epochs=100, lr=0.01, device="cuda", BASE_DIR=None
):
    optimizer = init_optimizer(model, lr)
    criterion = torch.nn.MSELoss()

    # Initialize tracker if BASE_DIR is provided
    tracker = init_metrics_tracker(BASE_DIR) if BASE_DIR else None

    for epoch in range(epochs):
        result, tracking_data = train_epoch(
            epoch,
            train_data.to(device),
            model,
            criterion,
            optimizer,
            val_data.to(device),
            tracker,
        )
        print(
            f"Epoch {epoch}, Train Loss: {result['train_loss']}, Val Loss: {result['val_loss']}"
        )
