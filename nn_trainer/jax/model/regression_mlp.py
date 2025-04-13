import jax
from flax import nnx


class RegressionMLP(nnx.Module):
    """Multilayer perceptron for regression task"""

    def __init__(self, input_layer_dim: int, hidden_layer_dims: list[int], output_layer_dim: int, rngs: nnx.Rngs):
        """Initialize layers and parameters

        Args:
            input_layer_dim (int): the number of neurons in the input layer.
            hidden_layer_dims (list[int]): the numbers of neurons in the hidden layers.
            output_layer_dim (int): the number of neurons in the output layer.
            rngs (nnx.Rngs): rng key.
        """

        super().__init__()

        dims = [input_layer_dim] + hidden_layer_dims + [output_layer_dim]

        self.hidden_layers = nnx.Sequential(
            *[
                nnx.Sequential(nnx.Linear(in_features=dims[i], out_features=dims[i + 1], rngs=rngs), nnx.relu)
                for i in range(len(dims) - 2)
            ]
        )
        self.output_layer = nnx.Linear(in_features=dims[-2], out_features=dims[-1], rngs=rngs)

    def __call__(self, *Xs: jax.Array) -> jax.Array:
        """Defines the forward pass of the model

        Args:
            Xs (tuple[jax.Array, ...]): Input arrays.

        Returns:
            jax.Array: Output array.
        """

        X = Xs[0]
        X = self.hidden_layers(X)
        X = self.output_layer(X)

        return X
