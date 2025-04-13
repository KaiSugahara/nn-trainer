from abc import ABC, abstractmethod
from typing import Self


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
