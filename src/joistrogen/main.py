from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .cli.parser import build_parser
from .distributions.factory import DistributionFactory
from .formatters.registry import FormatterRegistry
from .stats.reporter import StatsReporter
from .transforms.affine import AffineTransformer
from .transforms.identity import IdentityTransformer


def _resolve_transform(spec: str | None, dimension: int):
    if spec is None:
        return IdentityTransformer()
    return AffineTransformer.from_spec(spec, dimension)


def run(args) -> int:
    sampler = DistributionFactory.build(args.distribution)
    samples = sampler.sample(args.number)

    transform = _resolve_transform(args.transform, sampler.dimension)
    samples = transform.transform(samples)

    formatter = FormatterRegistry.get(args.output_format)
    formatter.write(samples, args.outfile, args.sig_figs)

    reporter = StatsReporter()
    print(reporter.report(samples, args.sig_figs))
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
