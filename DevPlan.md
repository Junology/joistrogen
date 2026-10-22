# joistrogen — Development Plan & Architecture Specification

## 1. Executive Summary & Objective

`joistrogen` is a command-line utility for generating random multidimensional vectors following joint probability distributions, applying affine spatial transformations, calculating summary statistics, and exporting the results into structured files.

### Core Technical Constraints
- **License**: MIT License.
- **Allowed Runtime Dependencies**:
  - Python Standard Library (`argparse`, `re`, `math`, `csv`, `abc`, `typing`, `sys`, `pathlib`).
  - `NumPy` (vectorized matrix computations, linear algebra, random number generation).
  - `SciPy.stats` (specifically `scipy.stats.norm` for Gaussian Copula CDF evaluation).
- **Strict Prohibition**: No additional third-party dependencies (e.g., Click, Pandas, SymPy, SciPy copula libraries) may be introduced.

---

## 2. Project Directory Structure

All library logic is encapsulated inside `src/joistrogen/`, keeping the repository root minimal and separating execution concerns from domain logic:

```
joistrogen/
├── joistrogen                        # Executable CLI entrypoint (Python 3 script)
├── LICENSE                           # MIT License
├── Readme.md                         # Project documentation
├── DEVELOPMENT_PLAN.md               # This planning specification
├── tests/
│   ├── __init__.py
│   ├── test_cli.py                   # CLI arguments & grammar parsing tests
│   ├── test_distributions.py         # Sampling accuracy & matrix reconstruction tests
│   ├── test_transforms.py            # Affine transformation tests
│   └── test_formatters.py            # Serialization & sig-fig precision tests
└── src/
    └── joistrogen/
        ├── __init__.py
        ├── main.py                   # Pipeline orchestrator
        ├── cli/
        │   ├── __init__.py
        │   ├── parser.py             # argparse configuration & token pre-processing
        │   └── grammar.py            # Custom grammar parsers for N(...), GC(...), and transform
        ├── distributions/
        │   ├── __init__.py
        │   ├── base.py               # DistributionSampler ABC
        │   ├── normal.py             # Joint normal distribution implementation
        │   ├── copula.py             # Gaussian copula implementation
        │   └── factory.py            # Sampler factory
        ├── transforms/
        │   ├── __init__.py
        │   ├── base.py               # Transformer ABC
        │   ├── affine.py             # Homogeneous affine matrix transformer
        │   └── identity.py           # Identity (no-op) transformer
        ├── formatters/
        │   ├── __init__.py
        │   ├── base.py               # OutputFormatter ABC
        │   ├── csv.py                # CSV output formatter implementation
        │   └── registry.py           # Extensible format registry for future formats
        └── stats/
            ├── __init__.py
            └── reporter.py           # Variance and correlation matrix calculations
```

---

## 3. Architecture & Interface Abstractions

To ensure future extensibility (e.g., adding formats such as JSON, Parquet, or TSV, or adding student-t copulas and mixture distributions) without modifying the main execution pipeline, key layers are decoupled using Abstract Base Classes (`abc.ABC`).

```
                      ┌───────────────────────┐
                      │      CLI Options      │
                      └──────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │     main.py Engine     │
                     └──────────┬─────────────┘
          ┌─────────────────────┼──────────────────────┐
          │                     │                      │
          ▼                     ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐
│DistributionSampler│  │   Transformer    │  │  OutputFormatter   │
│      (ABC)       │  │      (ABC)       │  │       (ABC)        │
└─────────┬────────┘  └─────────┬────────┘  └──────────┬─────────┘
          │                     │                      │
   ┌──────┴──────┐              │               ┌──────┴──────┐
   │             │              │               │             │
┌─────┐       ┌──────┐    ┌───────────┐     ┌───────┐     ┌───────┐
│ N(…)│       │GC(…) │    │  Affine   │     │  CSV  │     │[Future│
└─────┘       └──────┘    └───────────┘     └───────┘     │Formats│
                                                          └───────┘
```

### 3.1. Distribution Sampler Abstraction (`distributions/`)

```python
# src/joistrogen/distributions/base.py
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class DistributionSampler(ABC):
    """Abstract base class for multivariate random vector samplers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the intrinsic output dimension d of the distribution."""
        pass

    @abstractmethod
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate n independent random vectors.

        :param n: Number of samples to draw.
        :param rng: NumPy Generator instance.
        :return: Array of shape (n, dimension).
        """
        pass
```

- **`JointNormalSampler`**:
  - Reconstructs $d \times d$ covariance matrix $\Sigma$ from upper triangular entries $\frac{d(d+1)}{2}$.
  - Validates positive semi-definiteness ($M \succeq 0$) via Cholesky decomposition or eigenvalue decomposition (`np.linalg.eigvalsh`).
  - Samples via $X = \boldsymbol{\mu} + Z L^T$, where $L L^T = \Sigma$ and $Z \sim \mathcal{N}(0, I_d)$.
- **`GaussianCopulaSampler`**:
  - Reconstructs correlation matrix $R$ ($R_{ii} = 1.0$) from strict upper triangular entries $\frac{d(d-1)}{2}$.
  - Validates cube intervals $[a_i, b_i]$ (ensuring $a_i < b_i$) and positive semi-definiteness of $R$.
  - Draws standard normal vectors $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, R)$, converts to uniform marginals $U_i = \Phi(Z_i)$ using `scipy.stats.norm.cdf`, and scales onto $[a_i, b_i]$ via $X_i = a_i + U_i (b_i - a_i)$.

### 3.2. Coordinate Transformation Abstraction (`transforms/`)

```python
# src/joistrogen/transforms/base.py
from abc import ABC, abstractmethod
import numpy as np

class Transformer(ABC):
    """Abstract base class for coordinate transformations."""

    @abstractmethod
    def transform(self, samples: np.ndarray) -> np.ndarray:
        """
        Transforms input matrix of shape (n, d_in) to (n, d_out).
        """
        pass
```

- **`AffineTransformer`**:
  - Accepts a flattened parameter list and input dimension $d$.
  - Validates that the number of coefficients is a multiple of $(d + 1)$, forming an $m \times (d + 1)$ matrix $T$.
  - Augments coordinates into homogeneous space: $\tilde{X} = \begin{bmatrix} X & \mathbf{1}_n \end{bmatrix}$.
  - Computes linear projection: $Y = \tilde{X} T^T \in \mathbb{R}^{n \times m}$.
- **`IdentityTransformer`**: Returns samples unmodified when `--transform` is omitted.

### 3.3. Output Formatter Abstraction & Registry (`formatters/`)

```python
# src/joistrogen/formatters/base.py
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class OutputFormatter(ABC):
    """Abstract base class for output serializing."""

    @abstractmethod
    def write(self, samples: np.ndarray, outfile_path: str, sig_figs: Optional[int] = None) -> None:
        """Serializes samples array to disk at outfile_path."""
        pass
```

```python
# src/joistrogen/formatters/registry.py
from typing import Dict, Type
from .base import OutputFormatter
from .csv import CsvFormatter

class FormatterRegistry:
    """Registry pattern allowing addition of new formats without changing pipeline code."""
    _registry: Dict[str, Type[OutputFormatter]] = {
        "csv": CsvFormatter,
    }

    @classmethod
    def register(cls, name: str, formatter_cls: Type[OutputFormatter]) -> None:
        cls._registry[name.lower()] = formatter_cls

    @classmethod
    def get(cls, name: str) -> OutputFormatter:
        fmt_cls = cls._registry.get(name.lower())
        if not fmt_cls:
            supported = ", ".join(cls._registry.keys())
            raise ValueError(f"Unsupported format '{name}'. Available formats: {supported}")
        return fmt_cls()
```

---

## 4. CLI Grammar & Parser Design

The script must handle custom syntactical constructs without external parser generators:

### 4.1. Attached and Space-Separated Flags
Standard `argparse` supports both `-fcsv` and `-f csv`, as well as `-n128` and `-n 128`.
- `-n, --number`: Integer, default `128`.
- `-f, --format`: String, default `csv`.
- `-s`: Integer (significant digits).
- `--distribution`: String, default `N(0,0;1,0,1)`.
- `--transform`: String (matrix parameters enclosed in parentheses).
- `outfile`: Positional string argument for destination file.

### 4.2. Grammar Specifications & Validation

| Argument | Grammar Pattern | Dimension Constraints & Semantics |
| :--- | :--- | :--- |
| `N(...)` | `^N\(\s*(?P<mean>[^;]+)\s*;\s*(?P<cov>[^)]+)\s*\)$` | Let mean vector length be $d$. Covariance must contain $\frac{d(d+1)}{2}$ numbers. Upper triangle is populated in row-major order: $C_{i,j}$ for $i \le j$, and $C_{j,i} = C_{i,j}$. |
| `GC(...)` | `^GC\(\s*(?P<a>[^;]+)\s*;\s*(?P<b>[^;]+)\s*;\s*(?P<corr>[^)]+)\s*\)$` | Vectors $\mathbf{a}$ and $\mathbf{b}$ must both have length $d$ with $a_i < b_i$. Correlation must contain $\frac{d(d-1)}{2}$ numbers in $[-1, 1]$. Strict upper triangle populated row-major ($R_{i,j}$ for $i < j$, $R_{ii} = 1.0$). |
| `--transform` | `^\(\s*(?P<entries>[\s\S]+)\s*\)$` | Strip parentheses, split on commas/whitespace. Total entries $K$ must be a non-zero multiple of $(d + 1)$. Target dimension $m = \frac{K}{d + 1}$. Reshaped into $T \in \mathbb{R}^{m \times (d+1)}$. |

---

## 5. Summary Statistics & Diagnostic Reporting

Before serializing to `outfile`, the pipeline outputs diagnostic metrics directly to `stdout`:

1. **Sample Coordinate Variances**:
   $$\sigma_j^2 = \frac{1}{n - 1} \sum_{i=1}^n (X_{i,j} - \bar{X}_j)^2, \quad \text{for } j = 1, \dots, m$$
   Computed via `np.var(samples, axis=0, ddof=1)`.
2. **Empirical Correlation Matrix**:
   Computed via `np.corrcoef(samples, rowvar=False)`.
3. **Format**:
   Cleanly aligned text representation displaying vector variance and a formatted symmetric correlation matrix.

---

## 6. Significant Digits & Formatting Engine

When `-s <n>` is specified:
- Every coordinate value $x$ is converted using:
  ```python
  def format_float(val: float, sig_figs: Optional[int]) -> str:
      if sig_figs is None:
          return str(val)
      return f"{val:.{sig_figs}g}"
  ```
- Output files reflect the formatted string representations directly to preserve precision settings.

---

## 7. Phased Implementation Roadmap

### Phase 1: Foundation, CLI & Syntax Parsers
- Implement `src/joistrogen/cli/parser.py` using `argparse`.
- Implement `src/joistrogen/cli/grammar.py` using regular expressions to parse `N(...)`, `GC(...)`, and `(<tmn>...)`.
- Create unit tests verifying syntax errors, token extraction, and dimensional validation errors.

### Phase 2: Distribution Engine & Math Validation
- Implement `DistributionSampler`, `JointNormalSampler`, and `GaussianCopulaSampler`.
- Implement symmetric matrix reconstruction and positive semi-definiteness checks (`np.linalg.eigvalsh(M) >= -1e-12`).
- Implement marginal uniform-to-range transformations using `scipy.stats.norm.cdf`.
- Write unit tests verifying sample shapes, empirical bounds, and error handling for non-PSD matrices.

### Phase 3: Spatial Transformations
- Implement `Transformer`, `AffineTransformer`, and `IdentityTransformer`.
- Handle homogeneous coordinate augmentation $[X, \mathbf{1}]$ and dot product vectorization.
- Add unit tests verifying dimension changes (e.g., $d=2 \to m=3$, and translation/scaling accuracy).

### Phase 4: Statistical Diagnostics & Output Formatting
- Implement `StatsReporter` for coordinate variance and empirical correlation reporting to `stdout`.
- Implement `OutputFormatter` and `CsvFormatter` with support for significant digits (`-s`).
- Implement `FormatterRegistry` to allow clean extension for future file formats.

### Phase 5: Pipeline Integration & CLI Entrypoint
- Wire the complete pipeline inside `src/joistrogen/main.py`.
- Create the executable script `./joistrogen` with `#!/usr/bin/env python3`.
- Validate CLI exit codes (0 for success, non-zero for parsing or statistical errors).

### Phase 6: Verification & Edge Case Testing
- Test end-to-end default command: `joistrogen output.csv`.
- Test attached flags (`-n256`, `-fcsv`, `-s3`).
- Test high-dimension inputs, degenerate covariance matrices, and invalid transformation shapes.
- Verify test suite passes without third-party dependencies outside `NumPy` and `SciPy.stats`.