import math
from typing import Self

import jax
import numpy as np
import polars as pl

from nn_trainer.flax.loader import BaseLoader


class MiniBatchLoader(BaseLoader):
    def __init__(
        self,
        X_df: pl.DataFrame,
        y_df: pl.DataFrame,
        batch_size: int,
        seed: int,
    ):
        # Construct Generator
        rng = np.random.default_rng(seed)

        # Num. of data
        assert X_df.height == y_df.height
        data_size = X_df.height

        # Num. of batch
        batch_num = math.floor(data_size / batch_size)

        self.X_df = X_df
        self.y_df = y_df
        self.batch_size = batch_size
        self.rng = rng
        self.data_size = data_size
        self.batch_num = batch_num

    def setup_epoch(self) -> Self:
        """Data preparation before the start of every epoch"""

        # Shuffle rows
        shuffled_indices = self.rng.permutation(self.data_size)
        X = self.X_df[shuffled_indices, :].to_numpy().copy()
        y = self.y_df[shuffled_indices, :].to_numpy().copy()

        # Create Array
        self.X = jax.device_put(X)
        self.y = jax.device_put(y)

        return self

    def __iter__(self) -> Self:
        """Initialize iteration"""

        # Initialize batch index
        self.batch_index = 0

        return self

    def __len__(self) -> int:
        """Returns the number of batches

        Returns:
            int: The number of batches
        """

        return self.batch_num

    def __next__(self) -> tuple[tuple[jax.Array, ...], jax.Array]:
        """Returns data from the current batch

        Returns:
            tuple[jax.Array]: Input array.
            jax.Array: Target array.
        """

        if self.batch_index >= self.batch_num:
            raise StopIteration()

        else:
            # Extract the {batch_index}-th mini-batch
            start_index = self.batch_size * self.batch_index
            end_index = self.batch_size * (self.batch_index + 1)
            X, y = self.X[start_index:end_index], self.y[start_index:end_index]

            # Update batch index
            self.batch_index += 1

            return (X,), y
