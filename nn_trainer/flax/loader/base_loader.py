from abc import ABC, abstractmethod
from typing import Self

import jax


class BaseLoader(ABC):
    @abstractmethod
    def setup_epoch(self) -> Self:
        """Data preparation before the start of every epoch"""

    @abstractmethod
    def __iter__(self) -> Self:
        """Initialize iteration"""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of batches

        Returns:
            int: The number of batches
        """

    @abstractmethod
    def __next__(self) -> tuple[tuple[jax.Array, ...], jax.Array]:
        """Returns data from the current batch

        Returns:
            tuple[jax.Array]: Input array.
            jax.Array: Target array.
        """
