from joistrogen.main import build_parser, main, parse_distribution_spec, parse_transform_spec


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

    cetz_args = parser.parse_args(["-fcetz", "out.typ"])
    assert cetz_args.output_format == "cetz"


def test_distribution_and_transform_grammars():
    name, arguments = parse_distribution_spec("N(0,0;1,0,1)")
    assert name == "N"
    assert arguments == "0,0;1,0,1"

    copula_name, copula_arguments = parse_distribution_spec("GC(0,0;1,1;0.5)")
    assert copula_name == "GC"
    assert copula_arguments == "0,0;1,1;0.5"

    assert parse_transform_spec("(1,0,0,1)") == [1.0, 0.0, 0.0, 1.0]
    assert parse_transform_spec(None, 2) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_main_creates_csv_output(tmp_path, capsys):
    outfile = tmp_path / "samples.csv"
    exit_code = main(["-n", "8", str(outfile)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert outfile.exists()
    assert "Mean per coordinate:" in captured.out
    assert "Variance per coordinate:" in captured.out
    assert "Correlation matrix:" in captured.out
    with outfile.open() as handle:
        lines = handle.read().strip().splitlines()
    assert len(lines) == 8
