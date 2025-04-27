import torch
import numpy as np
from tqdm import tqdm
from ..visualization.visualization import plot_predictions


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
):
    """
    Same as before, but `train_loader` yields (x_batch, y_batch),
    and `edge_index` / `edge_weight` are passed in once.
    """
    model.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # ——— Training ———
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in tqdm(
            train_loader, desc=f"Train Epoch {epoch + 1}", unit="batch"
        ):
            # x_batch: (B, lags, N*F),  y_batch: (B, horizon, N)
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            y_hat = model(x_batch, edge_index, edge_weight)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)
        history["train_loss"].append(avg_train)

        # ——— Validation ———
        model.eval()
        running_val = 0.0

        with torch.no_grad():
            for x_batch, y_batch in tqdm(
                val_loader, desc=f"Val Epoch {epoch + 1}", unit="batch"
            ):
                x_batch = x_batch.to(device).float()
                y_batch = y_batch.to(device).float()

                y_hat = model(x_batch, edge_index, edge_weight)
                running_val += criterion(y_hat, y_batch).item()

        avg_val = running_val / len(val_loader)
        history["val_loss"].append(avg_val)

        print(f"Epoch {epoch + 1} — train: {avg_train:.6f}, val: {avg_val:.6f}")

        # ——— Early stopping ———
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            counter = 0
            best_state = model.state_dict()
        else:
            counter += 1
            print(f"Val did not improve - {counter}/{patience}")
            if counter >= patience:
                print(f"Stopping early at epoch {epoch + 1}")
                break

        writer.add_scalars("Loss", {"train": avg_train, "validation": avg_val}, epoch)

    # Load best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_index(
    model,
    test_loader,
    edge_index,
    edge_weight,
    device,
    loader=None,
    cities=["amsterdam", "rotterdam", "utrecht"],
):
    model.eval()
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    criterion = torch.nn.MSELoss()

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            y_hat = model(x_batch, edge_index, edge_weight)
            total_loss += criterion(y_hat, y_batch).item()

            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

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

            if cities is not None:
                plot_predictions(unscaled_preds, unscaled_targets, city_names=cities)
                print("Prediction plots saved to results directory")

        except Exception as e:
            print(f"Error during denormalization or plotting: {e}")

    return avg_loss, all_preds_array, all_targets_array
