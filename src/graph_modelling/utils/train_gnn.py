import torch.optim as optim
from torch_geometric.data import DataLoader
import torch
import sys
from tqdm import tqdm

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
    model, train_loader, optimizer, criterion, device, epoch, num_epochs, output_dim
):
    """Trains the model for a single epoch.

    Args:
        model: The PyTorch model.
        train_loader: The data loader for the training data.
        optimizer: The optimizer used for training.
        criterion: The loss function.
        device: The device to run the training on (CPU or GPU).
        epoch: The current epoch number.
        num_epochs: The total number of epochs.
        output_dim: The dimension of the output layer (calculated based on the flattened target).

    Returns:
        The average training loss for the epoch.
    """
    model.train()
    epoch_train_loss = 0

    with tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit="batch"
    ) as pbar:
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            y_target = batch.y.view(-1, output_dim)

            if out.shape != y_target.shape:
                print(f"Shape mismatch: output {out.shape}, target {y_target.shape}")
                continue

            loss = criterion(out, y_target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            pbar.set_postfix(loss=epoch_train_loss / (pbar.n + 1))

    avg_train_loss = epoch_train_loss / len(train_loader)
    return avg_train_loss


def validate_epoch(model, val_loader, criterion, device, epoch, num_epochs, output_dim):
    """Validates the model for a single epoch.

    Args:
        model: The PyTorch model.
        val_loader: The data loader for the validation data.
        criterion: The loss function.
        device: The device to run the validation on (CPU or GPU).
        epoch: The current epoch number.
        num_epochs: The total number of epochs.
        output_dim: The dimension of the output layer (calculated based on the flattened target).

    Returns:
        The average validation loss for the epoch.
    """
    model.eval()
    epoch_val_loss = 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", unit="batch"
        ) as pbar_val:
            for batch in pbar_val:
                batch = batch.to(device)
                out = model(batch)
                y_target = batch.y.view(-1, output_dim)

                if out.shape != y_target.shape:
                    print(
                        f"Shape mismatch during validation: output {out.shape}, target {y_target.shape}"
                    )
                    continue

                loss = criterion(out, y_target)
                epoch_val_loss += loss.item()
                pbar_val.set_postfix(loss=epoch_val_loss / (pbar_val.n + 1))

    avg_val_loss = epoch_val_loss / len(val_loader)
    return avg_val_loss


# --- Main training loop ---
def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    patience,
):
    """Trains and validates the model for a specified number of epochs, with early stopping.

    Args:
        model: The PyTorch model.
        train_loader: The data loader for the training data.
        val_loader: The data loader for the validation data.
        optimizer: The optimizer used for training.
        criterion: The loss function.
        device: The device to run the training on (CPU or GPU).
        num_epochs: The total number of epochs.
        patience: The patience for early stopping.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    output_dim = model.output_dim

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            num_epochs,
            output_dim,
        )
        avg_val_loss = validate_epoch(
            model, val_loader, criterion, device, epoch, num_epochs, output_dim
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] Training Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}"
        )

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return train_losses, val_losses
