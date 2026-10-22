import numpy as np

from joistrogen.formatters.base import format_float
from joistrogen.formatters.csv import CsvFormatter


def test_format_float_rounds_with_sig_figs():
    assert format_float(123.456, 3) == "123"
    assert format_float(0.004567, 3) == "0.00457"


def test_csv_formatter_writes_rows(tmp_path):
    outfile = tmp_path / "out.csv"
    samples = np.array([[1.234, 5.678], [9.101, 2.345]])
    CsvFormatter().write(samples, str(outfile), sig_figs=3)
    path_text = outfile.read_text()
    assert "1.23" in path_text
    assert "5.68" in path_text
