import torch.optim as optim
from torch_geometric.data import DataLoader
import torch
import sys
from tqdm import tqdm

sys.path.append("../../")

from modelling.metrics.metricstracker import MetricsTracker


def init_optimizer(model, lr, weight_decay=1e-4):
    """Initialize the optimizer with optional weight decay for regularization"""
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def init_scheduler(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6):
    """Initialize a learning rate scheduler for better convergence

    Args:
        optimizer: The optimizer to schedule
        mode: 'min' for reducing on metric decrease, 'max' for increasing
        factor: Factor by which to reduce learning rate
        patience: Number of epochs with no improvement after which LR will be reduced
        min_lr: Lower bound on the learning rate
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        verbose=True,
        min_lr=min_lr,
    )
    return scheduler


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
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    num_epochs,
    output_dim,
    val_loader=None,
):
    """Trains the model for a single epoch."""
    model.train()
    epoch_train_loss = 0

    # At the beginning of training, sample and print some values
    if epoch == 0 and val_loader is not None:
        first_batch = next(iter(train_loader))
        first_val_batch = next(iter(val_loader))

        # Check input ranges
        print(
            f"Training input range: [{first_batch.x_seq.min():.4f}, {first_batch.x_seq.max():.4f}]"
        )
        print(
            f"Validation input range: [{first_val_batch.x_seq.min():.4f}, {first_val_batch.x_seq.max():.4f}]"
        )

        # Check target ranges
        print(
            f"Training target range: [{first_batch.y.min():.4f}, {first_batch.y.max():.4f}]"
        )
        print(
            f"Validation target range: [{first_val_batch.y.min():.4f}, {first_val_batch.y.max():.4f}]"
        )

    with tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit="batch"
    ) as pbar:
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)  # Shape: [batch_size*num_nodes, forecast_horizon]

            # Since we're only predicting NO2 now, y is simpler
            # Reshape y to match model output [batch*nodes, forecast_horizon]
            y_target = batch.y.view(-1, output_dim)

            # Check shape match (should be simpler now)
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
    """Validates the model for a single epoch."""
    model.eval()
    epoch_val_loss = 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", unit="batch"
        ) as pbar_val:
            for batch in pbar_val:
                batch = batch.to(device)
                out = model(batch)

                # Since we're only predicting NO2 now, y is simpler
                # Reshape y to match model output [batch*nodes, forecast_horizon]
                y_target = batch.y.reshape(-1, output_dim)

                # Check shape match
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
    use_scheduler=True,
    scheduler=None,
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
        use_scheduler: Whether to use learning rate scheduling.
        scheduler: Optional pre-configured scheduler. If None but use_scheduler is True, will create default scheduler.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    output_dim = model.output_dim

    # Create scheduler if requested but not provided
    if use_scheduler and scheduler is None:
        scheduler = init_scheduler(optimizer)
        print(f"Initialized learning rate scheduler: {scheduler.__class__.__name__}")

    for epoch in range(num_epochs):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        avg_train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            num_epochs,
            output_dim,
            val_loader=val_loader,
        )
        avg_val_loss = validate_epoch(
            model, val_loader, criterion, device, epoch, num_epochs, output_dim
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] Training Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f} | LR: {current_lr:.8f}"
        )

        # Step the scheduler based on validation loss
        if use_scheduler and scheduler is not None:
            scheduler.step(avg_val_loss)

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model checkpoint
            best_model_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
            }
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            # Restore best model before stopping
            if "best_model_state" in locals():
                model.load_state_dict(best_model_state["model_state_dict"])
                print(f"Restored best model from epoch {best_model_state['epoch'] + 1}")
            break

    return train_losses, val_losses
