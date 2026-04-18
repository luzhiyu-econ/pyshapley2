"""
pyshapley2._core
================
Core computation engine: enumerate subset regressions and derive Shapley values via OLS.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ._estimators import build_estimator
from ._result import Shapley2Result
from .constants import STAT_EXTRACTORS


# ---------------------------------------------------------------------------
# Single subset regression (top-level function, picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _run_one_subset(
    args: tuple,
) -> tuple[int, float]:
    """
    Run one subset regression and return (index, stat_value).

    Defined as a top-level function (not a lambda or closure) to support
    pickling with multiprocessing / loky backends.
    """
    (
        i,
        subset,
        group_vars,
        data_dict,
        columns,
        depvar,
        command,
        fit_kwargs,
        stat_key,
        stat_func_pkl,
    ) = args

    # Reconstruct DataFrame for this subset
    df = pd.DataFrame(data_dict, columns=columns)

    # Rebuild estimator
    from ._estimators import build_estimator
    run_model = build_estimator(command, fit_kwargs)

    # Rebuild stat extractor
    if stat_func_pkl is not None:
        import pickle
        get_stat = pickle.loads(stat_func_pkl)
    else:
        get_stat = STAT_EXTRACTORS[stat_key]

    # Resolve variables included in this subset
    included_vars = [
        v
        for j, grp in enumerate(group_vars)
        for v in grp
        if subset[j] == 1
    ]

    if not included_vars:
        return i, 0.0

    try:
        res = run_model(df, depvar, included_vars)
        return i, float(get_stat(res))
    except Exception as exc:
        warnings.warn(
            f"Subset {included_vars} failed and was skipped: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return i, float("nan")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def shapley2(
    data: pd.DataFrame,
    depvar: str,
    indepvars: list[str],
    *,
    stat: str = "r2",
    command: Union[str, Callable] = "ols",
    groups: Optional[dict[str, list[str]]] = None,
    force: bool = False,
    fit_kwargs: Optional[dict] = None,
    stat_func: Optional[Callable] = None,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 0,
) -> Shapley2Result:
    """
    Shapley-Owen decomposition of a regression fit statistic.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset (only complete observations are used).
    depvar : str
        Dependent variable column name.
    indepvars : list[str]
        Independent variable column names.
    stat : str, default ``"r2"``
        Fit statistic to decompose. Built-in options:

        * ``"r2"``      — OLS R²
        * ``"r2_a"``    — Adjusted R²
        * ``"ll"``      — Log-likelihood
        * ``"aic"``     — AIC (lower is better; Shapley values will be negative)
        * ``"bic"``     — BIC (same as AIC)
        * ``"rmse"``    — Root MSE (same as AIC)

        Override with ``stat_func`` to use any custom scalar.
    command : str or callable, default ``"ols"``
        Estimation command. Built-in strings:

        * ``"ols"`` / ``"reg"``
        * ``"logit"``
        * ``"probit"``
        * ``"poisson"``
        * ``"glm"``

        A callable with signature ``f(data, depvar, indepvars) -> statsmodels result``
        is also accepted.
    groups : dict[str, list[str]], optional
        Variable grouping. Keys are group names; values are lists of variable names.
        When set, Shapley values are computed at the group level.

        Example::

            groups={"Human Capital": ["edu", "exp"], "Physical Capital": ["capital"]}

    force : bool, default ``False``
        Allow more than 20 variables/groups (requires ``force=True``).
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to the estimator's ``fit()`` call.
    stat_func : callable, optional
        Custom stat extractor with signature ``f(result) -> float``.
        Overrides ``stat`` when provided.
    n_jobs : int, default ``1``
        Number of parallel workers.

        * ``1``  — serial (default)
        * ``-1`` — use all available CPU cores
        * ``N``  — use exactly N workers

        Requires ``joblib`` (``pip install pyshapley2[parallel]``).
    backend : str, default ``"loky"``
        joblib backend: ``"loky"``, ``"threading"``, or ``"multiprocessing"``.
    verbose : int, default ``0``
        Verbosity level:

        * ``0`` — silent
        * ``1`` — progress bar (requires tqdm)
        * ``2`` — print each subset regression result

    Returns
    -------
    Shapley2Result
        Result object with ``.table``, ``.summary()``, ``.plot()``, etc.

    Raises
    ------
    ValueError
        If K > 20 and ``force=False``, or if ``stat`` is not recognised.
    ImportError
        If ``n_jobs != 1`` and ``joblib`` is not installed.

    Examples
    --------
    Basic OLS R² decomposition::

        import pandas as pd
        from pyshapley2 import shapley2

        result = shapley2(df, "wage", ["edu", "exp", "tenure"], stat="r2")
        result.summary()

    Grouped decomposition with parallel execution::

        result = shapley2(
            df, "wage", ["edu", "exp", "tenure", "age"],
            stat="r2",
            groups={"Human Capital": ["edu", "exp"], "Other": ["tenure", "age"]},
            n_jobs=-1,
        )

    Notes
    -----
    Algorithm (Shapley-Owen decomposition):

    1. Enumerate all 2^K subsets of K variables/groups.
    2. Regress the outcome on each subset; record the fit statistic.
    3. Fit OLS of the subset statistics on binary inclusion indicators;
       slope coefficients are the Shapley values.
       (Shapley 1953; Owen 1977; Kruskal 1987)
    4. Compute four output forms: raw, relative %, normalised, normalised %.

    References
    ----------
    Chavez Juarez, F. (2013). shapley2: Stata module to compute Shapley values
    from regressions. Statistical Software Components.

    Shapley, L. S. (1953). A value for n-person games. Contributions to the
    Theory of Games, 2, 307–317.

    Owen, G. (1977). Values of games with a priori unions. Essays in
    Mathematical Economics and Game Theory, 76–88.
    """

    fit_kwargs = fit_kwargs or {}

    # Drop incomplete observations
    all_cols = list({depvar} | set(indepvars))
    data = data[all_cols].dropna().reset_index(drop=True)

    # Resolve grouping structure
    if groups is not None:
        group_names = list(groups.keys())
        group_vars  = list(groups.values())
        K = len(group_names)
        labels = group_names
    else:
        K = len(indepvars)
        group_names = None
        group_vars  = [[v] for v in indepvars]
        labels = list(indepvars)

    # Guard against excessive subset count
    runs = 2 ** K
    if K > 20 and not force:
        raise ValueError(
            f"K={K} variables/groups would require {runs} regressions. "
            "Set force=True to proceed."
        )
    if K > 30:
        raise ValueError(f"K={K} exceeds the hard limit of 30. Aborting.")

    # Resolve stat extractor
    if stat_func is not None:
        get_stat  = stat_func
        stat_key  = "__custom__"
    elif stat.lower() in STAT_EXTRACTORS:
        get_stat  = STAT_EXTRACTORS[stat.lower()]
        stat_key  = stat.lower()
    else:
        raise ValueError(
            f"Unknown stat='{stat}'. "
            f"Built-in options: {list(STAT_EXTRACTORS.keys())}. "
            "Pass a callable via stat_func for custom statistics."
        )

    # Fit full model to obtain reference stat
    run_model  = build_estimator(command, fit_kwargs)
    full_vars  = [v for grp in group_vars for v in grp]
    full_res   = run_model(data, depvar, full_vars)
    full_stat  = float(get_stat(full_res))

    # Enumerate all 2^K subsets
    subsets      = list(itertools.product([0, 1], repeat=K))
    results_col  = np.full(len(subsets), np.nan)

    # Serialize stat_func for multiprocessing
    stat_func_pkl: Optional[bytes] = None
    if stat_func is not None and n_jobs != 1:
        try:
            import pickle
            stat_func_pkl = pickle.dumps(stat_func)
        except Exception:
            warnings.warn(
                "stat_func could not be pickled; falling back to serial mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            n_jobs = 1

    # Build argument list for parallel workers
    # Convert DataFrame to dict for cross-process serialization
    data_dict = data.to_dict("list")
    columns   = list(data.columns)

    # Callable commands fall back to serial mode
    if callable(command):
        if n_jobs != 1:
            warnings.warn(
                "Callable commands do not support parallel execution; "
                "falling back to serial mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            n_jobs = 1
        command_arg = command
    else:
        command_arg = command

    task_args = [
        (
            i,
            subsets[i],
            group_vars,
            data_dict,
            columns,
            depvar,
            command_arg,
            fit_kwargs,
            stat_key,
            stat_func_pkl,
        )
        for i in range(len(subsets))
    ]

    # Run subset regressions
    if n_jobs == 1:
        # Serial path
        iter_subsets = enumerate(subsets)
        if verbose == 1:
            from tqdm import tqdm
            iter_subsets = tqdm(list(iter_subsets), desc="Shapley subsets", unit="subset")

        for i, subset in iter_subsets:
            included_vars = [
                v for j, grp in enumerate(group_vars)
                for v in grp if subset[j] == 1
            ]
            if not included_vars:
                results_col[i] = 0.0
                continue
            try:
                res = run_model(data, depvar, included_vars)
                val = float(get_stat(res))
            except Exception as exc:
                warnings.warn(
                    f"Subset {included_vars} failed and was skipped: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                val = float("nan")
            results_col[i] = val
            if verbose >= 2:
                print(f"  [{i:4d}/{runs}] {str(included_vars):<60s} {stat}={val:.6f}")

    else:
        # Parallel path via joblib
        from joblib import Parallel, delayed

        joblib_verbose = 10 if verbose >= 1 else 0
        pairs = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=joblib_verbose,
        )(delayed(_run_one_subset)(args) for args in task_args)

        for idx, val in pairs:
            results_col[idx] = val

    # Fit OLS of subset stats on binary indicators; slope coefficients are the Shapley values
    combo_mat   = np.array(subsets, dtype=float)                    # shape (2^K, K)
    combo_mat_c = sm.add_constant(combo_mat, has_constant=False)    # shape (2^K, K+1)
    ols_res     = sm.OLS(results_col, combo_mat_c).fit()
    shapley     = ols_res.params[1:]                                # exclude intercept; K slope coefficients

    shap_sum         = float(shapley.sum())
    shapley_rel      = shapley / full_stat
    shapley_norm     = (full_stat / shap_sum) * shapley
    shapley_rel_norm = (full_stat / shap_sum) * shapley_rel
    residual         = full_stat - shap_sum

    return Shapley2Result(
        labels           = labels,
        depvar           = depvar,
        indepvars        = indepvars,
        stat             = stat,
        command          = command if isinstance(command, str) else command.__name__,
        full_stat        = full_stat,
        shapley          = shapley,
        shapley_rel      = shapley_rel,
        shapley_norm     = shapley_norm,
        shapley_rel_norm = shapley_rel_norm,
        residual         = residual,
        groups           = groups,
        K                = K,
        runs             = runs,
        n_obs            = len(data),
    )
