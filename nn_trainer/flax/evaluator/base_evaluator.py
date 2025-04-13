from flax import nnx


class BaseEvaluator:
    def evaluate(self, model: nnx.Module) -> tuple[float, dict[str, float]]:
        """Calculate loss and metrics for the model

        Args:
            model (nnx.Module): Model.

        Returns:
            float: Loss.
            dict[str, float]: Metrics.
        """

        raise NotImplementedError
