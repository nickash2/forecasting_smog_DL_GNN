import torch
from modelling.denormalise import denormalise
import torch.nn.functional as F
from typing import Any, Dict
import numpy as np


def test_gnn_eparately(
    model: Any,
    loss_fn: Any,
    test_loader: Any,
    denorm: bool = False,
    path: str = None,
    components=["NO2", "O3"],
) -> Dict[str, float]:
    """
    Evaluates on test set and returns test loss

    :param model: model to evaluate, must be some PyTorch type model
    :param loss_fn: loss function to use, PyTorch defined, or PyTorch inherited
    :param test_loader: DataLoader to get batches from
    :param denorm: whether to denormalise the data before calculating loss
    :param path: path to the file containing the minmax values for the data
    :return: dictionary with contaminant names as keys and losses as values
    """
    model.eval()
    test_losses = [np.float64(0) for _ in components]

    with torch.no_grad():
        for batch_test_u, batch_test_y in test_loader:
            pred = model(batch_test_u)
            if denorm:
                pred = denormalise(pred, path)
                batch_test_y = denormalise(batch_test_y, path)

            for comp in range(len(components)):
                test_losses[comp] += loss_fn(
                    pred[:, :, comp], batch_test_y[:, :, comp]
                ).item()

    for comp in range(len(components)):
        test_losses[comp] /= len(test_loader)
    return {comp: loss for comp, loss in zip(components, test_losses)}
