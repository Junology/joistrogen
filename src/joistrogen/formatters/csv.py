import csv
from pathlib import Path
from typing import Optional

import numpy as np

from .base import OutputFormatter, format_float


class CsvFormatter(OutputFormatter):
    """CSV serializer for matrices of samples."""

    def write(self, samples: np.ndarray, outfile_path: str, sig_figs: Optional[int] = None) -> None:
        arr = np.asarray(samples, dtype=float)
        path = Path(outfile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for row in arr:
                writer.writerow([format_float(float(value), sig_figs) for value in row])
