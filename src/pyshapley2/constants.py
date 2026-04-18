"""
pyshapley2.constants
====================
Built-in fit-statistic extractors for statsmodels result objects.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Stat extractor registry
# key: stat string (lowercase); value: f(statsmodels_result) -> float
# ---------------------------------------------------------------------------
STAT_EXTRACTORS: dict[str, object] = {
    # OLS
    "r2":       lambda res: res.rsquared,
    "r2_a":     lambda res: res.rsquared_adj,
    "r2_adj":   lambda res: res.rsquared_adj,
    # General
    "ll":       lambda res: res.llf,
    "aic":      lambda res: res.aic,
    "bic":      lambda res: res.bic,
    "rmse":     lambda res: float(np.sqrt(res.mse_resid)),
    # GLM
    "deviance": lambda res: getattr(res, "deviance", float("nan")),
    "pearson":  lambda res: getattr(res, "pearson_chi2", float("nan")),
}
