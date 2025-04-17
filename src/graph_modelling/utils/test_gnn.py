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
    num_nodes: int = 3,
) -> Tuple[float, torch.Tensor, float]:
    """
    Predicts and evaluates the model on the test dataset, computing global RMSE and per-node NO2 RMSE.

    Args:
        model: The trained PyTorch model.
        test_loader: The DataLoader for the test dataset.
        device: The device to run the evaluation on (CPU or GPU).
        output_dim: The dimension of the output layer (forecast horizon).
        y_min: The minimum values used for normalization, shape [1, num_nodes, 1].
        y_max: The maximum values used for normalization, shape [1, num_nodes, 1].
        N_HOURS_Y: The forecast horizon (number of hours predicted).
        num_nodes: The number of graph nodes.

    Returns:
        A tuple containing:
            - Global RMSE (float)
            - Per-node NO2 RMSE (torch.Tensor of shape [num_nodes])
            - Mean NO2 RMSE across all nodes (float)
    """

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            out = model(batch)  # shape: (batch_size * num_nodes, output_dim)
            y_target = batch["y"].view(-1, output_dim)

            all_preds.append(out)
            all_targets.append(y_target)

    # Concatenate over all batches
    all_preds = torch.cat(all_preds, dim=0)  # (N, output_dim)
    all_targets = torch.cat(all_targets, dim=0)  # (N, output_dim)

    # Reshape to (batch, num_nodes, N_HOURS_Y)
    total_graphs = all_preds.shape[0] // num_nodes
    preds_reshaped = all_preds.view(total_graphs, num_nodes, N_HOURS_Y)
    targets_reshaped = all_targets.view(total_graphs, num_nodes, N_HOURS_Y)

    # Prepare y_min/y_max for unnormalization
    y_min_tensor = (
        torch.tensor(y_min, dtype=torch.float32, device=device).squeeze(0).squeeze(-1)
    )  # (num_nodes,)
    y_max_tensor = (
        torch.tensor(y_max, dtype=torch.float32, device=device).squeeze(0).squeeze(-1)
    )  # (num_nodes,)
    y_min_tensor = y_min_tensor.view(1, num_nodes, 1)  # (1, num_nodes, 1)
    y_max_tensor = y_max_tensor.view(1, num_nodes, 1)  # (1, num_nodes, 1)

    # Unnormalize
    preds_unnorm = preds_reshaped * (y_max_tensor - y_min_tensor) + y_min_tensor
    targets_unnorm = targets_reshaped * (y_max_tensor - y_min_tensor) + y_min_tensor

    # Per-node RMSE (for NO2 only, assuming it's the only feature)
    rmse_per_node = torch.sqrt(
        torch.mean((preds_unnorm - targets_unnorm) ** 2, dim=(0, 2))
    )  # shape: (num_nodes,)
    mean_no2_rmse = rmse_per_node.mean().item()

    # Global RMSE over all nodes & time
    global_rmse = torch.sqrt(torch.mean((preds_unnorm - targets_unnorm) ** 2)).item()

    # Print results
    for i, rmse in enumerate(rmse_per_node):
        print(f"Node {i}: NO2 RMSE = {rmse.item():.4f}")
    print(f"\nGlobal RMSE (all nodes, all time steps): {global_rmse:.4f}")
    print(f"Mean NO2 RMSE across nodes: {mean_no2_rmse:.4f}")

    return global_rmse, rmse_per_node, mean_no2_rmse
