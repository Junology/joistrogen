import argparse


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return ivalue


def build_parser() -> argparse.ArgumentParser:
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
        choices=["csv"],
        default="csv",
        help="Output format (default: csv).",
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
        help="Distribution specification: N(...) or GC(...).",
    )
    parser.add_argument(
        "--transform",
        default=None,
        help="Affine transformation matrix in flattened row-major form, enclosed in parentheses.",
    )
    parser.add_argument("outfile", help="Destination output file.")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)
