import torch
from torch.utils.data import DataLoader
from typing import Tuple, List


def predict_and_evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dim: int,
    y_min: List[float],
    y_max: List[float],
    N_HOURS_Y: int,
) -> Tuple[float, float, float]:
    """
    Predicts and evaluates the model on the test dataset, computing global RMSE and pollutant-specific RMSE for NO2 and O3.

    Args:
        model: The trained PyTorch model.
        test_loader: The DataLoader for the test dataset.
        device: The device to run the evaluation on (CPU or GPU).
        output_dim: The dimension of the output layer (forecast horizon * number of target features).
        y_min: A list of minimum values used for normalization, per feature.
        y_max: A list of maximum values used for normalization, per feature.
        N_HOURS_Y: The forecast horizon (number of hours predicted).

    Returns:
        A tuple containing the global RMSE, RMSE for NO2, and RMSE for O3.  All as floats.
    """

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            out = model(batch)  # out shape: (batch_size * 3, output_dim)
            y_target = batch["y"].view(-1, output_dim)

            all_preds.append(out)
            all_targets.append(y_target)

    # Concatenate over all batches
    all_preds = torch.cat(all_preds, dim=0)  # shape: (N, output_dim)
    all_targets = torch.cat(all_targets, dim=0)  # shape: (N, output_dim)

    # Convert y_min and y_max to torch tensors.
    y_min_tensor = torch.tensor(y_min, dtype=torch.float).to(device)  # Move to device
    y_max_tensor = torch.tensor(y_max, dtype=torch.float).to(device)  # Move to device

    # Ensure the min/max tensors can broadcast over predictions and targets.
    preds_unnorm = all_preds * (y_max_tensor - y_min_tensor) + y_min_tensor
    targets_unnorm = all_targets * (y_max_tensor - y_min_tensor) + y_min_tensor

    # --- Compute RMSE ---
    # Global RMSE over all forecast values:
    global_rmse = torch.sqrt(
        torch.mean((preds_unnorm - targets_unnorm) ** 2)
    ).item()  # Extract float

    # Reshape for pollutant-specific RMSE
    preds_reshaped = preds_unnorm.view(-1, N_HOURS_Y, 2)
    targets_reshaped = targets_unnorm.view(-1, N_HOURS_Y, 2)

    print(f"Global RMSE (unnormalized): {global_rmse:.4f}")

    return global_rmse, preds_reshaped, targets_reshaped
