import jax
import polars as pl
from flax import nnx

from nn_trainer.flax.evaluator import BaseEvaluator
from nn_trainer.flax.loss_fn import mean_squared_error


class MeanSquaredErrorEvaluator(BaseEvaluator):
    """Evaluator using Mean Squared Error"""

    def __init__(self, X_df: pl.DataFrame, y_df: pl.DataFrame):
        """Initialize

        Args:
            X_df (pl.DataFrame): Input DataFrame.
            y_df (pl.DataFrame): Target DataFrame.
        """

        self.Xs = (jax.device_put(X_df.to_numpy().copy()),)
        self.y = jax.device_put(y_df.to_numpy().copy())

    def evaluate(self, model: nnx.Module) -> tuple[float, dict[str, float]]:
        """Calculate loss and metrics for the model

        Args:
            model (nnx.Module): Model.

        Returns:
            float: Loss.
            dict[str, float]: Metrics.
        """

        loss = mean_squared_error(model, self.Xs, self.y).item()

        return loss, {"MSE": loss}
