# pyshapley2

[![PyPI version](https://badge.fury.io/py/pyshapley2.svg)](https://badge.fury.io/py/pyshapley2)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/luzhiyu-econ/pyshapley2/actions/workflows/publish.yml/badge.svg)](https://github.com/luzhiyu-econ/pyshapley2/actions)

Python replication of Stata's [`shapley2`](https://ideas.repec.org/c/boc/bocode/s457543.html) command (Chavez Juarez, 2013).

Computes the **Shapley-Owen decomposition** of any regression fit statistic (R², adjusted R², log-likelihood, AIC, …) across independent variables or user-defined variable groups, with optional **parallel computation** support.

---

## Installation

```bash
# Recommended (parallel + plot + progress)
pip install "pyshapley2[all]"

# Core only (serial, no extras)
pip install pyshapley2
```

Optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| `parallel` | `joblib` | `n_jobs != 1` |
| `plot` | `matplotlib` | `.plot()` |
| `progress` | `tqdm` | `verbose=1` |
| `all` | all of above | everything |
| `dev` | above + pytest, ruff | development |

---

## Quick Start

```python
import pandas as pd
from pyshapley2 import shapley2

# Sample data
df = pd.read_csv("your_data.csv")

# Basic R² decomposition
result = shapley2(df, depvar="wage", indepvars=["edu", "exp", "tenure"])
result.summary()
```

Output (1:1 replica of Stata's table format):

```
Shapley-Owen decomposition  |  depvar: wage  |  stat: r2  |  command: ols
Observations: 500  |  Subsets: 8  |  K=3

Factor     │ Shapley value │ Per cent  │Shapley value │  Per cent
           │  (estimate)   │(estimate) │ (normalized) │(normalized)
───────────┼───────────────┼───────────┼──────────────┼─────────────
edu        │       0.35420 │    51.23 % │      0.31876 │      46.12 %
exp        │       0.27816 │    40.25 % │      0.25034 │      36.21 %
tenure     │       0.05918 │     8.56 % │      0.05326 │       7.70 %
───────────┼───────────────┼───────────┼──────────────┼─────────────
Residual   │      -0.00204 │    -0.04 % │              │
───────────┼───────────────┼───────────┼──────────────┼─────────────
TOTAL      │       0.68954 │   100.00 % │      0.68954 │     100.00 %
───────────┼───────────────┼───────────┼──────────────┼─────────────
```

---

## Features

### All `stat` options

| `stat=` | Meaning | Stata equivalent |
|---|---|---|
| `"r2"` | R² | `e(r2)` |
| `"r2_a"` | Adjusted R² | `e(r2_a)` |
| `"ll"` | Log-likelihood | `e(ll)` |
| `"aic"` | AIC | computed |
| `"bic"` | BIC | computed |
| `"rmse"` | Root MSE | computed |

Custom extractor via `stat_func`:

```python
result = shapley2(df, "y", ["x1", "x2", "x3"], stat_func=lambda r: r.rsquared)
```

### All `command` options

| `command=` | Model | Stata equivalent |
|---|---|---|
| `"ols"` / `"reg"` | OLS | `regress` |
| `"logit"` | Logit | `logit` |
| `"probit"` | Probit | `probit` |
| `"poisson"` | Poisson | `poisson` |
| `"glm"` | GLM | `glm` |
| callable | Custom | any `e()` command |

### Group decomposition (Stata `group()` option)

```python
result = shapley2(
    df, "wage", ["edu", "exp", "tenure", "age"],
    stat="r2",
    groups={
        "Human Capital":  ["edu", "exp"],
        "Job Tenure":     ["tenure"],
        "Demographics":   ["age"],
    },
)
result.summary()
```

### Parallel computation

```python
# Use all available CPU cores
result = shapley2(
    df, "wage", ["x1", "x2", "x3", "x4", "x5"],
    stat="r2",
    n_jobs=-1,       # -1 = all cores; N = exactly N processes
    backend="loky",  # "loky" (default) | "threading" | "multiprocessing"
    verbose=1,       # show progress bar (requires tqdm)
)
```

> **When to use parallel?**  
> Parallel is beneficial when K ≥ 10 (≥ 1,024 regressions).  
> For small K (≤ 8), the process-spawning overhead outweighs the benefit.

### Visualization

```python
fig, ax = result.plot(
    kind="norm_pct",   # "pct" | "norm_pct" | "shapley" | "norm"
    figsize=(8, 5),
)
fig.savefig("shapley_decomp.pdf", dpi=300)
```

---

## Stata → Python mapping

| Stata | Python |
|---|---|
| `shapley2, stat(r2)` | `shapley2(df, "y", ["x1","x2"], stat="r2")` |
| `shapley2, stat(r2) command(logit)` | `shapley2(..., stat="ll", command="logit")` |
| `shapley2, stat(r2) group(x1 x2, x3)` | `shapley2(..., groups={"G1":["x1","x2"],"G2":["x3"]})` |
| `shapley2, stat(r2) force` | `shapley2(..., force=True)` |
| *(not available in Stata)* | `shapley2(..., n_jobs=-1)` |

---

## Result object attributes

```python
result.table           # pd.DataFrame: shapley, shapley_pct, shapley_norm, shapley_norm_pct
result.full_stat       # float: full-model stat (e.g. R²)
result.residual        # float: full_stat − sum(shapley)
result.K               # int: number of variables/groups
result.runs            # int: number of regressions run (2^K)
result.n_obs           # int: number of observations used
result.summary()       # prints Stata-style table, returns str
result.plot()          # matplotlib bar chart
result.to_dict()       # serializable dict
```

---

## Algorithm

Shapley2 implements the **Shapley-Owen regression decomposition** (also known as the LMG method):

1. **Enumerate** all 2^K subsets of K variables/groups.
2. **Regress** the outcome on each subset; record the fit statistic.
3. **OLS** (with intercept): regress the vector of fit statistics on the binary inclusion indicators; slope coefficients are the Shapley values.
4. **Normalize**: compute four output forms (raw, relative %, normalized, normalized %).

This is a 1:1 algorithmic replication of [Stata's `shapley2` v1.1](https://ideas.repec.org/c/boc/bocode/s457543.html).

---

## Validation against Stata

Results are verified to match Stata's `shapley2` (v1.1) output to ≥ 5 decimal places on two public benchmark datasets.

### Test 1 — mtcars (individual variables)

**Data**: [Motor Trend Cars Road Tests (1974)](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/mtcars.csv), N = 32
**Model**: `regress mpg hp wt disp`
**Stata**: `reg mpg hp wt disp` → `shapley2, stat(r2)`

| Variable | Shapley (est.) | % (est.) | Shapley (norm.) | % (norm.) |
|----------|---------------|----------|-----------------|-----------|
| hp | 0.18805 | 22.74% | 0.22511 | 27.23% |
| wt | 0.27959 | 33.81% | 0.33469 | 40.48% |
| disp | 0.22307 | 26.98% | 0.26704 | 32.30% |
| Residual | 0.13612 | 16.46% | — | — |
| **TOTAL** | **0.82684** | **100%** | **0.82684** | **100%** |

```python
import pandas as pd
from pyshapley2 import shapley2

df = pd.read_csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/mtcars.csv")
result = shapley2(df, "mpg", ["hp", "wt", "disp"], stat="r2")
result.summary()
```

### Test 2 — Boston Housing (grouped variables)

**Data**: [Boston Housing (Harrison & Rubinfeld, 1978)](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/Boston.csv), N = 506
**Model**: `regress medv lstat rm dis ptratio`
**Stata**: `reg medv lstat rm dis ptratio` → `shapley2, stat(r2) group(lstat,rm,dis ptratio)`

| Group | Variables | Shapley (est.) | % (est.) | Shapley (norm.) | % (norm.) |
|-------|-----------|---------------|----------|-----------------|-----------|
| Group 1 | lstat | 0.29427 | 42.63% | 0.31257 | 45.28% |
| Group 2 | rm | 0.23205 | 33.61% | 0.24648 | 35.71% |
| Group 3 | dis, ptratio | 0.12358 | 17.90% | 0.13126 | 19.01% |
| Residual | — | 0.04041 | 5.85% | — | — |
| **TOTAL** | — | **0.69031** | **100%** | **0.69031** | **100%** |

```python
import pandas as pd
from pyshapley2 import shapley2

df = pd.read_csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/Boston.csv")
result = shapley2(
    df, "medv", ["lstat", "rm", "dis", "ptratio"],
    stat="r2",
    groups={
        "lstat":       ["lstat"],
        "rm":          ["rm"],
        "dis_ptratio": ["dis", "ptratio"],
    },
)
result.summary()
```

---

## References

- Chavez Juarez, F. (2013). **shapley2: Stata module to compute Shapley values from regressions**. Statistical Software Components S457543, Boston College.
- Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games*, 2, 307–317.
- Owen, G. (1977). Values of games with a priori unions. *Essays in Mathematical Economics and Game Theory*, 76–88.
- Kruskal, W. (1987). Relative importance by averaging over orderings. *American Statistician*, 41(1), 6–10.

---

## License

MIT © 2026 luzhiyu-econ
