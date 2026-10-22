import numpy as np

from .base import Transformer


class IdentityTransformer(Transformer):
    """Pass-through transformer."""

    def transform(self, samples: np.ndarray) -> np.ndarray:
        arr = np.asarray(samples, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr.copy()
