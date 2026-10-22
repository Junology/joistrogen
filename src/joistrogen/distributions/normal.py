from typing import Iterable, Optional, Sequence

import numpy as np

from .base import DistributionSampler


class JointNormalSampler(DistributionSampler):
    """Sampler for multivariate normal distributions."""

    def __init__(self, mean: Sequence[float], covariance: Sequence[float] | np.ndarray):
        mean_array = np.asarray(mean, dtype=float)
        if mean_array.ndim != 1 or mean_array.size == 0:
            raise ValueError("JointNormalSampler mean must be a non-empty 1D array.")
        self._mean = mean_array
        self._covariance = self._reconstruct_covariance(covariance)
        if self._covariance.shape[0] != self._mean.size:
            raise ValueError("Covariance dimension does not match the mean vector dimension.")

    @property
    def dimension(self) -> int:
        return self._mean.size

    @staticmethod
    def _reconstruct_covariance(covariance: Sequence[float] | np.ndarray) -> np.ndarray:
        cov = np.asarray(covariance, dtype=float)
        if cov.ndim == 2:
            matrix = cov.copy()
        elif cov.ndim == 1:
            d = int((np.sqrt(1 + 8 * cov.size) - 1) / 2)
            if d * (d + 1) // 2 != cov.size:
                raise ValueError("Covariance vector does not match a valid upper-triangular specification.")
            matrix = np.zeros((d, d), dtype=float)
            idx = np.triu_indices(d)
            matrix[idx] = cov
            matrix = matrix + matrix.T - np.diag(np.diag(matrix))
        else:
            raise ValueError("Covariance must be a 1D or 2D array.")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Covariance matrix must be square.")

        eigvals = np.linalg.eigvalsh(matrix)
        if np.min(eigvals) < -1e-12:
            raise ValueError("Covariance matrix must be positive semidefinite.")
        return matrix

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if n < 0:
            raise ValueError("Sample count n must be non-negative.")
        generator = np.random.default_rng() if rng is None else rng
        z = generator.standard_normal((n, self.dimension))
        chol = np.linalg.cholesky(self._covariance)
        return z @ chol.T + self._mean
