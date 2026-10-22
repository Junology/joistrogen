from joistrogen.cli.grammar import parse_distribution, parse_transform
from joistrogen.cli.parser import build_parser
from joistrogen.main import main


def test_parser_handles_default_and_attached_flags():
    parser = build_parser()
    args = parser.parse_args(["-n128", "-fcsv", "-s3", "out.csv"])
    assert args.number == 128
    assert args.output_format == "csv"
    assert args.sig_figs == 3
    assert args.outfile == "out.csv"

    default_args = parser.parse_args(["out.csv"])
    assert default_args.number == 128
    assert default_args.output_format == "csv"
    assert default_args.distribution == "N(0,0;1,0,1)"


def test_distribution_and_transform_grammars():
    parsed = parse_distribution("N(0,0;1,0,1)")
    assert parsed["kind"] == "normal"
    assert parsed["mean"] == ["0", "0"]
    assert parsed["covariance"] == ["1", "0", "1"]

    copula = parse_distribution("GC(0,0;1,1;0.5)")
    assert copula["kind"] == "copula"
    assert copula["a"] == ["0", "0"]
    assert copula["b"] == ["1", "1"]
    assert copula["correlation"] == ["0.5"]

    assert parse_transform("(1,0,0,1)") == [1.0, 0.0, 0.0, 1.0]
    assert parse_transform(None, 2) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_main_creates_csv_output(tmp_path):
    outfile = tmp_path / "samples.csv"
    exit_code = main(["-n", "8", str(outfile)])
    assert exit_code == 0
    assert outfile.exists()
    with outfile.open() as handle:
        lines = handle.read().strip().splitlines()
    assert len(lines) == 8
