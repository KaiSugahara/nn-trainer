from torch import nn


class BaseEvaluator:
    def evaluate(self, model: nn.Module) -> tuple[float, dict[str, float]]:
        """Calculate loss and metrics for the model

        Args:
            model (nn.Module): Model.

        Returns:
            float: Loss.
            dict[str, float]: Metrics.
        """

        raise NotImplementedError
