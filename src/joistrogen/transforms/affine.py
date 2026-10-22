import numpy as np

from ..cli.grammar import parse_transform
from .base import Transformer


class AffineTransformer(Transformer):
    """Apply an affine transformation represented by a matrix in homogeneous coordinates."""

    def __init__(self, entries, dimension: int):
        values = [float(v) for v in entries]
        if dimension <= 0:
            raise ValueError("Affine transformation dimension must be positive.")
        if len(values) == 0 or len(values) % (dimension + 1) != 0:
            raise ValueError(
                f"Affine transform length must be a non-zero multiple of d + 1 = {dimension + 1}."
            )
        rows = len(values) // (dimension + 1)
        matrix = np.asarray(values, dtype=float).reshape(rows, dimension + 1)
        self._matrix = matrix
        self._dimension = dimension

    @classmethod
    def from_spec(cls, spec: str, dimension: int):
        entries = parse_transform(spec, dimension)
        return cls(entries, dimension)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def transform(self, samples: np.ndarray) -> np.ndarray:
        arr = np.asarray(samples, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self._dimension:
            raise ValueError(
                f"Sample dimension {arr.shape[1]} does not match transformation input dimension {self._dimension}."
            )
        augmented = np.hstack([arr, np.ones((arr.shape[0], 1), dtype=float)])
        return augmented @ self._matrix.T
