import math
from typing import Self

import numpy as np
import polars as pl
import torch

from nn_trainer.torch.loader import BaseLoader


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
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

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

    def __next__(self) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Returns data from the current batch

        Returns:
            tuple[torch.Tensor]: Input array.
            torch.Tensor: Target array.
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
