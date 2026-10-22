from pathlib import Path
from typing import Optional

import numpy as np

from .base import OutputFormatter, format_float

class CeTZFormatter(OutputFormatter):
    """Plotting sample points in Typst using CeTZ."""

    def write(self, samples: np.ndarray, outfile_path: str, sig_figs: Optional[int] = None) -> None:
        if samples.ndim != 2:
            raise ValueError("Samples must be exactly 2 dimensional.")
        arr = np.asarray(samples, dtype=float)
        path = Path(outfile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as out:
            out.writelines([
                '#import \"@preview/cetz:0.5.3\"\n',
                '#let plot-points(..args) = {\n',
                '  cetz.canvas({\n',
                '    import cetz.draw: *\n',
                '    let points = ((0,1),(1,1),(1.5,2),(0.5,0.5))\n',
                '    let radius = 0.0625\n',
                '    set-style(stroke: (paint: red, thickness: 0.5pt))\n',
                '    for pt in args.pos() {\n',
                '      line((rel: (-radius,0), to: pt), (rel: (2 * radius,0)))\n',
                '      line((rel: (0,-radius), to: pt), (rel: (0,2 * radius)))\n',
                '    }\n',
                '  })\n',
                '}\n',
                '\n',
                '#plot-points(\n',
            ])
            for row in arr:
                out.write(f"  ({', '.join(format_float(float(value), sig_figs) for value in row)}),\n")
            out.write(')\n')
