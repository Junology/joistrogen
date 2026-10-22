from __future__ import annotations

import argparse
import re
from typing import Sequence

import numpy as np

from .distributions.factory import DistributionFactory
from .formatters.base import format_float
from .formatters.registry import FormatterRegistry
from .transforms.affine import AffineTransformer
from .transforms.identity import IdentityTransformer


def parse_distribution_spec(spec: str) -> tuple[str, str]:
    if spec is None:
        raise ValueError("Distribution specification is required.")
    text = spec.strip()
    if not text:
        raise ValueError("Distribution specification is required.")
    match = re.fullmatch(r"(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*\((?P<arguments>.*)\)", text)
    if match is None:
        raise ValueError(
            "Distribution specification must follow <name>(<arguments>), "
            "for example N(0,0;1,0,1)."
        )
    return match.group("name"), match.group("arguments").strip()


def parse_transform_spec(spec: str | None, dimension: int | None = None) -> list[float]:
    if spec is None or not spec.strip():
        if dimension is None:
            return []
        matrix = [[0.0 for _ in range(dimension + 1)] for _ in range(dimension)]
        for idx in range(dimension):
            matrix[idx][idx] = 1.0
        return [entry for row in matrix for entry in row]

    text = spec.strip()
    if not text.startswith("(") or not text.endswith(")"):
        raise ValueError("Transform specification must be enclosed in parentheses, e.g. (1, 0, 0, 1).")

    values = [part for part in re.split(r"[\s,]+", text[1:-1].strip()) if part]
    if not values:
        raise ValueError("Transform specification cannot be empty.")
    parsed = [float(value) for value in values]
    if dimension is not None and len(parsed) % (dimension + 1) != 0:
        raise ValueError(
            f"Affine transform entries must be a multiple of (d + 1) = {dimension + 1}. "
            f"Got {len(parsed)} entries."
        )
    return parsed


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return ivalue


def build_parser() -> argparse.ArgumentParser:
    formats = FormatterRegistry.available_formats()
    formats_help = ", ".join(formats)

    parser = argparse.ArgumentParser(description="Generate samples from joint distributions.")
    parser.add_argument(
        "-n",
        "--number",
        type=positive_int,
        default=128,
        help="Number of generated samples (default: 128).",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=formats,
        default="csv",
        help=f"Output format (default: csv). Available: {formats_help}.",
    )
    parser.add_argument(
        "-s",
        dest="sig_figs",
        type=int,
        default=None,
        help="Optional significant digits used in output formatting.",
    )
    parser.add_argument(
        "--distribution",
        default="N(0,0;1,0,1)",
        help="Distribution specification: <name>(<arguments>).",
    )
    parser.add_argument(
        "--transform",
        default=None,
        help="Affine transformation matrix in flattened row-major form, enclosed in parentheses.",
    )
    parser.add_argument("outfile", help="Destination output file.")
    return parser


def parse_args(argv: Sequence[str] | None = None):
    return build_parser().parse_args(argv)


def _resolve_transform(spec: str | None, dimension: int):
    if spec is None:
        return IdentityTransformer()
    entries = parse_transform_spec(spec, dimension)
    return AffineTransformer(entries, dimension)


def run(args) -> int:
    distribution_name, distribution_arguments = parse_distribution_spec(args.distribution)
    sampler = DistributionFactory.build(distribution_name, distribution_arguments)
    samples = sampler.sample(args.number)

    transform = _resolve_transform(args.transform, sampler.dimension)
    samples = transform.transform(samples)

    formatter = FormatterRegistry.get(args.output_format)
    formatter.write(samples, args.outfile, args.sig_figs)

    means = np.mean(samples, axis=0)
    variances = np.var(samples, axis=0, ddof=1)
    correlations = np.corrcoef(samples, rowvar=False)

    print(f"Mean: [{', '.join(format_float(float(v), args.sig_figs) for v in means)}]")
    print(f"Variance: [{', '.join(format_float(float(v), args.sig_figs) for v in variances)}]")
    print("Correlation matrix:")
    correlation_rows = [
        f"[{', '.join(format_float(float(x), args.sig_figs) for x in row)}]"
        for row in correlations
    ]
    for row in correlation_rows:
        print(row)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
