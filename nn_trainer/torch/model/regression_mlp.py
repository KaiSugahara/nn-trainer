import torch
from torch import nn


class RegressionMLP(nn.Module):
    """Multilayer perceptron for regression task"""

    def __init__(self, input_layer_dim: int, hidden_layer_dims: list[int], output_layer_dim: int):
        """Initialize layers and parameters

        Args:
            input_layer_dim (int): the number of neurons in the input layer.
            hidden_layer_dims (list[int]): the numbers of neurons in the hidden layers.
            output_layer_dim (int): the number of neurons in the output layer.
        """

        super().__init__()

        dims = [input_layer_dim] + hidden_layer_dims + [output_layer_dim]

        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_features=dims[i], out_features=dims[i + 1]), nn.ReLU())
                for i in range(len(dims) - 2)
            ]
        )
        self.output_layer = nn.Linear(in_features=dims[-2], out_features=dims[-1])

    def forward(self, *Xs: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model

        Args:
            Xs (tuple[torch.Tensor, ...]): Input arrays.

        Returns:
            torch.Tensor: Output array.
        """

        X = Xs[0]
        X = self.hidden_layers(X)
        X = self.output_layer(X)

        return X
