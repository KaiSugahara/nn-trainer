import jax
import jax.numpy as jnp
from flax import nnx


def mean_squared_error(model: nnx.Module, Xs: tuple[jax.Array, ...], y: jax.Array) -> jax.Array:
    """Calculate mean squared error

    Args:
        model (nnx.Module): Current model.
        Xs (tuple[jax.Array, ...]): Input array.
        y (jax.Array): Target array.

    Returns:
        jax.Array: loss.
    """

    # Prediction
    pred = model(*Xs)  # type: ignore

    # MSE
    loss = jnp.mean((pred - y) ** 2)

    return loss
