from abc import ABC, abstractmethod
from typing import Self

import torch


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
    def __next__(self) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Returns data from the current batch

        Returns:
            tuple[torch.Tensor]: Input array.
            torch.Tensor: Target array.
        """
