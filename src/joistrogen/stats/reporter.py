from typing import Optional

import numpy as np

from ..formatters.base import format_float


class StatsReporter:
    """Compute and format variance and correlation diagnostics for sample matrices."""

    @staticmethod
    def _variance(samples: np.ndarray) -> np.ndarray:
        return np.var(samples, axis=0, ddof=1)

    @staticmethod
    def _correlation(samples: np.ndarray) -> np.ndarray:
        return np.corrcoef(samples, rowvar=False)

    def report(self, samples: np.ndarray, sig_figs: Optional[int] = None) -> str:
        arr = np.asarray(samples, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        variance = self._variance(arr)
        corr = self._correlation(arr)
        var_text = "Variance per coordinate:\n" + "[" + ", ".join(format_float(float(v), sig_figs) for v in variance) + "]"
        corr_rows = [
            "[" + ", ".join(format_float(float(x), sig_figs) for x in row) + "]"
            for row in corr
        ]
        corr_text = "Correlation matrix:\n" + "\n".join(corr_rows)
        return var_text + "\n" + corr_text
