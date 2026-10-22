from typing import Optional, Sequence

import numpy as np
from scipy import stats

from .base import DistributionSampler


class GaussianCopulaSampler(DistributionSampler):
    """Sampler for Gaussian copula distributions over intervals [a_i, b_i]."""

    def __init__(self, a: Sequence[float], b: Sequence[float], correlation: Sequence[float] | np.ndarray):
        lower = np.asarray(a, dtype=float)
        upper = np.asarray(b, dtype=float)
        if lower.ndim != 1 or upper.ndim != 1:
            raise ValueError("GC(...) bounds must be 1D arrays.")
        if lower.size != upper.size or lower.size == 0:
            raise ValueError("GC(...) bounds must be non-empty and match in length.")
        if not np.all(lower < upper):
            raise ValueError("Each lower bound a_i must be strictly less than its upper bound b_i.")
        self._a = lower
        self._b = upper
        self._correlation = self._reconstruct_correlation(correlation)
        if self._correlation.shape[0] != self._a.size:
            raise ValueError("Correlation matrix dimension does not match the interval bounds dimension.")

    @property
    def dimension(self) -> int:
        return self._a.size

    @staticmethod
    def _reconstruct_correlation(correlation: Sequence[float] | np.ndarray) -> np.ndarray:
        corr = np.asarray(correlation, dtype=float)
        if corr.ndim == 2:
            matrix = corr.copy()
        elif corr.ndim == 1:
            d = int((1 + np.sqrt(1 + 8 * corr.size)) / 2)
            if d * (d - 1) // 2 != corr.size:
                raise ValueError("Correlation vector does not match a valid strict upper-triangular specification.")
            matrix = np.eye(d, dtype=float)
            idx = np.triu_indices(d, k=1)
            matrix[idx] = corr
            matrix = matrix + matrix.T - np.diag(np.diag(matrix))
        else:
            raise ValueError("Correlation must be a 1D or 2D array.")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Correlation matrix must be square.")
        if np.any(np.abs(np.diag(matrix) - 1.0) > 1e-12):
            raise ValueError("Correlation matrix diagonal entries must all be 1.0.")
        eigvals = np.linalg.eigvalsh(matrix)
        if np.min(eigvals) < -1e-12:
            raise ValueError("Correlation matrix must be positive semidefinite.")
        return matrix

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if n < 0:
            raise ValueError("Sample count n must be non-negative.")
        generator = np.random.default_rng() if rng is None else rng
        z = generator.multivariate_normal(np.zeros(self.dimension), self._correlation, size=n)
        u = stats.norm.cdf(z)
        return self._a + u * (self._b - self._a)
