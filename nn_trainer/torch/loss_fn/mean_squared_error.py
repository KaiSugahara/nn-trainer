import torch
from torch import nn


def mean_squared_error(model: nn.Module, Xs: tuple[torch.Tensor, ...], y: torch.Tensor) -> torch.Tensor:
    """Calculate mean squared error

    Args:
        model (nn.Module): Current model.
        Xs (tuple[torch.Tensor, ...]): Input array.
        y (torch.Tensor): Target array.

    Returns:
        torch.Tensor: loss.
    """

    # Prediction
    pred = model(*Xs)  # type: ignore

    # MSE
    loss = torch.mean((pred - y) ** 2)

    return loss
