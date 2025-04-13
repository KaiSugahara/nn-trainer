import polars as pl
import torch
from torch import nn

from nn_trainer.torch.evaluator import BaseEvaluator
from nn_trainer.torch.loss_fn import mean_squared_error


class MeanSquaredErrorEvaluator(BaseEvaluator):
    """Evaluator using Mean Squared Error"""

    def __init__(self, X_df: pl.DataFrame, y_df: pl.DataFrame):
        """Initialize

        Args:
            X_df (pl.DataFrame): Input DataFrame.
            y_df (pl.DataFrame): Target DataFrame.
        """

        self.Xs = (torch.from_numpy(X_df.to_numpy().copy()),)
        self.y = torch.from_numpy(y_df.to_numpy().copy())

    def evaluate(self, model: nn.Module) -> tuple[float, dict[str, float]]:
        """Calculate loss and metrics for the model

        Args:
            model (nn.Module): Model.

        Returns:
            float: Loss.
            dict[str, float]: Metrics.
        """

        loss = mean_squared_error(model, self.Xs, self.y).item()

        return loss, {"MSE": loss}
