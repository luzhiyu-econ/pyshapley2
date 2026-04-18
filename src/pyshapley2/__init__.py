"""
pyshapley2
==========
Python replication of Stata's ``shapley2`` command (Chavez Juarez, 2013).

Computes Shapley-Owen decomposition of any regression fit statistic
(R², log-likelihood, AIC, …) across independent variables or variable groups.

Basic usage::

    from pyshapley2 import shapley2

    result = shapley2(df, "wage", ["edu", "exp", "tenure"], stat="r2")
    result.summary()
    result.plot()

Parallel usage::

    result = shapley2(
        df, "wage", ["edu", "exp", "tenure"],
        stat="r2",
        n_jobs=-1,       # use all CPU cores
    )
"""

from __future__ import annotations

from ._core import shapley2
from ._result import Shapley2Result
from .constants import STAT_EXTRACTORS

__all__ = [
    "shapley2",
    "Shapley2Result",
    "STAT_EXTRACTORS",
]

__version__ = "0.1.0"
__author__  = "luzhiyu-econ"
__email__   = "zhiyu.lu.econ@icloud.com"
