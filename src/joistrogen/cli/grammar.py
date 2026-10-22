import re
from typing import Dict, List, Optional

import numpy as np


_NUMBER_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def _split_values(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    values = [part for part in re.split(r"[\s,]+", raw) if part]
    return values


def parse_distribution(spec: str) -> Dict[str, object]:
    if spec is None:
        raise ValueError("Distribution specification is required.")
    text = spec.strip()
    if not text:
        raise ValueError("Distribution specification is required.")

    normal_match = re.fullmatch(r"N\s*\((?P<mean>[^;]+)\s*;\s*(?P<cov>[^)]+)\)", text, flags=re.IGNORECASE)
    if normal_match:
        mean_values = _split_values(normal_match.group("mean"))
        cov_values = _split_values(normal_match.group("cov"))
        if not mean_values:
            raise ValueError("Mean vector in N(...) must not be empty.")
        d = len(mean_values)
        expected = d * (d + 1) // 2
        if len(cov_values) != expected:
            raise ValueError(
                f"N(...) covariance requires {expected} values for dimension {d}, got {len(cov_values)}."
            )
        return {"kind": "normal", "mean": mean_values, "covariance": cov_values}

    copula_match = re.fullmatch(
        r"GC\s*\(\s*(?P<a>[^;]+)\s*;\s*(?P<b>[^;]+)\s*;\s*(?P<corr>[^)]+)\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if copula_match:
        a_values = _split_values(copula_match.group("a"))
        b_values = _split_values(copula_match.group("b"))
        corr_values = _split_values(copula_match.group("corr"))
        if not a_values or not b_values:
            raise ValueError("GC(...) interval bounds must not be empty.")
        if len(a_values) != len(b_values):
            raise ValueError("GC(...) lower and upper bounds must have the same length.")
        d = len(a_values)
        expected = d * (d - 1) // 2
        if len(corr_values) != expected:
            raise ValueError(
                f"GC(...) correlation requires {expected} values for dimension {d}, got {len(corr_values)}."
            )
        return {"kind": "copula", "a": a_values, "b": b_values, "correlation": corr_values}

    raise ValueError(f"Unsupported distribution specification: {spec!r}")


def parse_transform(spec: Optional[str], dimension: Optional[int] = None) -> List[float]:
    if spec is None:
        if dimension is None:
            return []
        matrix = np.zeros((dimension, dimension + 1), dtype=float)
        np.fill_diagonal(matrix[:, :-1], 1.0)
        return matrix.reshape(-1).tolist()

    text = spec.strip()
    if not text:
        if dimension is None:
            return []
        matrix = np.zeros((dimension, dimension + 1), dtype=float)
        np.fill_diagonal(matrix[:, :-1], 1.0)
        return matrix.reshape(-1).tolist()

    if not text.startswith("(") or not text.endswith(")"):
        raise ValueError("Transform specification must be enclosed in parentheses, e.g. (1, 0, 0, 1).")

    inner = text[1:-1].strip()
    values = _split_values(inner)
    if not values:
        raise ValueError("Transform specification cannot be empty.")
    parsed = [float(value) for value in values]
    if dimension is not None:
        if len(parsed) % (dimension + 1) != 0:
            raise ValueError(
                f"Affine transform entries must be a multiple of (d + 1) = {dimension + 1}. "
                f"Got {len(parsed)} entries."
            )
    return parsed
