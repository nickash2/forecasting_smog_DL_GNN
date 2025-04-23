import torch
from torch.utils.data import DataLoader
from typing import Tuple, List


def predict_and_evaluate(
    model, test_loader, device, output_dim, y_min, y_max, forecast_horizon
):
    """
    Predicts and evaluates the model on the test dataset, computing global RMSE and per-node NO2 RMSE.

    Args:
        model: The trained PyTorch model.
        test_loader: The DataLoader for the test dataset.
        device: The device to run the evaluation on (CPU or GPU).
        output_dim: The dimension of the output layer (forecast horizon).
        y_min: The minimum values used for normalization, shape [1, num_nodes, 1].
        y_max: The maximum values used for normalization, shape [1, num_nodes, 1].
        forecast_horizon: The forecast horizon (number of hours predicted).

    Returns:
        A tuple containing:
            - Global RMSE (float)
            - Per-node NO2 RMSE (list of floats)
    """
    model.eval()
    all_preds = []
    all_targets = []

    # Move y_min and y_max to the correct device once outside the loop
    y_min = torch.tensor(y_min, dtype=torch.float).to(device)
    y_max = torch.tensor(y_max, dtype=torch.float).to(device)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)

            # Reshape target to match output
            y_target = batch.y.reshape(-1, output_dim)

            # Denormalize predictions and targets (all on the same device now)
            preds = out * (y_max[0, 0] - y_min[0, 0]) + y_min[0, 0]
            targets = y_target * (y_max[0, 0] - y_min[0, 0]) + y_min[0, 0]

            # Move to CPU before appending to list
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate RMSE for each node
    rmse_per_node = []
    for node in range(3):  # 3 cities
        node_preds = all_preds[node::3]
        node_targets = all_targets[node::3]
        rmse = torch.sqrt(((node_preds - node_targets) ** 2).mean())
        rmse_per_node.append(rmse.item())

    # Global RMSE
    global_rmse = torch.sqrt(((all_preds - all_targets) ** 2).mean()).item()

    print(f"Global RMSE: {global_rmse:.4f}")
    print(
        f"RMSE per city: Amsterdam={rmse_per_node[0]:.4f}, Rotterdam={rmse_per_node[1]:.4f}, Utrecht={rmse_per_node[2]:.4f}"
    )

    return global_rmse, rmse_per_node
