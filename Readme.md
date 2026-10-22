# joistrogen

Python script generating random samples in designated joint distributions.

The project is released under MIT license (see (LICENSE)).

## Dependency
- `NumPy`
- `SciPy.stats`


## usage

```
joistrogen [options] outfile
```

The generated samples are written out to `outfile` in the format specified in `options` (default: `csv`).
The program also shows the mean and variance of each coordinate of generated samples, and the correlations, in `stdout`.

## options

In the following `<Xn>...` denotes a comma separeted sequence of numbers.

- `--distribution <specifier>`: the distribution in which samples are generated (default: `N(0,0;1,0,1)`). The following are available:
  - `N(<μn>...;<Cmn>...)`: the joint normal distribution of the mean vector `(<μn>...)` and the covariance matrix whose upper triangular part is specified by `<Cmn>` (in row-major).
  - `GC(<an>...;<bn>...;<rmn>...)`: the Gauussian copula in the cube spanned by the coordinates `(<an>...)` and `(<bn>...)` with the correlation matrix whose strict upper triangular part is specified by `<rmn>` (in row-major).
- `-f<fmt>` `--format <fmt>`: the output format (default: `csv`)
  - `csv`: each row is an individual samples
  - `cetz`: Typst CeTZ script plotting sample points
- `-n<num>` `--number <num>`: the number of generated samples (default: `128`)
- `-s<n>`: the significant digits in the final output.
- `--transform (<tmn>...)`: the transformation matrix applied to all the samples. The parenthesized arguments `<tmn>` are the entries of the matrix in the row-major order which has (d+1) columns if the samples are in dimension d (default: the identity affine transformation matrix).
