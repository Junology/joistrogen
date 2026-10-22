from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class DistributionSampler(ABC):
    """Abstract base class for multivariate random vector samplers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the intrinsic output dimension d of the distribution."""

    @abstractmethod
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generate n independent random vectors."""
