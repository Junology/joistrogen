from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


def format_float(value: float, sig_figs: Optional[int] = None) -> str:
    if sig_figs is None:
        return str(value)
    return format(float(value), f".{sig_figs}g")


class OutputFormatter(ABC):
    """Abstract base class for output serializers."""

    @abstractmethod
    def write(self, samples: np.ndarray, outfile_path: str, sig_figs: Optional[int] = None) -> None:
        """Serialize samples to a file."""
