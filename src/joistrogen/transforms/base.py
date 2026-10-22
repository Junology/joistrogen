from abc import ABC, abstractmethod

import numpy as np


class Transformer(ABC):
    """Abstract base class for coordinate transformations."""

    @abstractmethod
    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Transform an array of shape (n, d_in) to shape (n, d_out)."""
