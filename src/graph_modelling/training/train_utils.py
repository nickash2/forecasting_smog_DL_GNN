import torch
import numpy as np
from tqdm import tqdm
from ..visualization.visualization import plot_predictions
from codecarbon import EmissionsTracker


def train_model_index(
    model,
    train_loader,
    val_loader,
    edge_index,
    edge_weight,
    device,
    epochs=50,
    patience=5,
    writer=None,
    learning_rate=1e-5,
    weight_decay=1e-6,
    lambda_max=None,
):
    """
    Train model using index-based dataloaders

    Args:
        model: Model to train
        train_loader, val_loader: DataLoaders with training and validation data
        device: PyTorch device
        epochs: Max number of epochs to train
        patience: Early stopping patience (epochs without improvement)
        only_no2: Whether only NO2 values should be used as targets
    """
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-8,
    )
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "epochs": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for x_batch, y_batch in train_pbar:
            # Move data to device
            x_batch, y_batch = x_batch.to(device).float(), y_batch.to(device).float()

            edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

            optimizer.zero_grad()
            y_hat = model(x_batch, edge_index, edge_weight, lambda_max=lambda_max)

            # Extract only NO2 values as targets when using all variables
            if y_batch.shape[2] != y_hat.shape[2]:
                # Model is expecting (B, horizon, num_nodes=3)
                # Target is (B, horizon, num_nodes*num_vars=21)

                # Reshape to (B, horizon, num_nodes=3, num_vars=7)
                B, H, NF = y_batch.shape
                num_nodes = 3  # Assuming 3 cities
                num_vars = NF // num_nodes  # Calculate based on actual dimensions

                # Reshape and extract just NO2 (first variable) for each node
                y_batch_reshaped = y_batch.view(B, H, num_nodes, num_vars)
                # Take only NO2 (index 0) for all nodes
                y_batch_no2 = y_batch_reshaped[:, :, :, 0]

                # Calculate loss using only NO2 values
                loss = criterion(y_hat, y_batch_no2)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            train_pbar.set_postfix(
                {"train_loss": sum(train_losses) / len(train_losses)}
            )

        # Validation phase
        val_loss = validate_model(
            model, val_loader, criterion, device, edge_index, edge_weight, lambda_max
        )
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Track current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        # Update history
        avg_train_loss = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"].append(epoch)

        writer.add_scalars("Loss", {"train": avg_train_loss, "val": val_loss}, epoch)

        print(
            f"Epoch {epoch}: Train Loss {avg_train_loss:.6f}, Val Loss {val_loss:.6f} - LR {current_lr:.3g}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model if needed
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(f"Validation loss did not improve {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping after {epoch} epochs")
                model.load_state_dict(best_model_state)
                break

    return model, history


def validate_model(
    model, val_loader, criterion, device, edge_index, edge_weight, lambda_max
):
    """Run validation and return average loss"""
    model.eval()
    val_losses = []
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device).float(), y_batch.to(device).float()

            y_hat = model(x_batch, edge_index, edge_weight, lambda_max=lambda_max)

            if y_batch.shape[2] != y_hat.shape[2]:
                # Reshape to get just NO2 values for each node
                B, H, NF = y_batch.shape
                num_nodes = 3
                num_vars = NF // num_nodes  # Calculate based on actual dimensions

                y_batch_reshaped = y_batch.view(B, H, num_nodes, num_vars)
                y_batch_no2 = y_batch_reshaped[:, :, :, 0]

                loss = criterion(y_hat, y_batch_no2)

            val_losses.append(loss.item())

    return sum(val_losses) / len(val_losses)


def evaluate_index(
    model,
    test_loader,
    edge_index,
    edge_weight,
    device,
    loader=None,
    cities=["amsterdam", "rotterdam", "utrecht"],
    lambda_max=None,
):
    model.eval()
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
    criterion = torch.nn.MSELoss()

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device).float(), y_batch.to(device).float()

            y_hat = model(x_batch, edge_index, edge_weight, lambda_max=lambda_max)

            if y_batch.shape[2] != y_hat.shape[2]:
                B, H, NF = y_batch.shape
                num_nodes = 3
                num_vars = NF // num_nodes  # Calculate based on actual dimensions

                y_batch_reshaped = y_batch.view(B, H, num_nodes, num_vars)
                y_batch_no2 = y_batch_reshaped[:, :, :, 0]

                loss = criterion(y_hat, y_batch_no2)

                all_preds.append(y_hat.cpu().numpy())
                all_targets.append(y_batch_no2.cpu().numpy())

            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test MSE (scaled): {avg_loss:.6f}")
    print(f"Test RMSE (scaled): {np.sqrt(avg_loss):.6f}")

    all_preds_array = np.vstack(all_preds)
    all_targets_array = np.vstack(all_targets)

    # Unscale predictions if loader is provided
    if loader is not None:
        try:
            unscaled_preds = loader.denormalize_no2(all_preds_array)
            unscaled_targets = loader.denormalize_no2(all_targets_array)

            unscaled_mse = np.mean((unscaled_preds - unscaled_targets) ** 2)
            unscaled_rmse = np.sqrt(unscaled_mse)

            print(f"Test MSE (unscaled): {unscaled_mse:.4f}")
            print(f"Test RMSE (unscaled): {unscaled_rmse:.4f} μg/m³")

        except Exception as e:
            print(f"Error during denormalization or plotting: {e}")

    return avg_loss, all_preds_array, all_targets_array
