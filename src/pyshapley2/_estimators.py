"""
pyshapley2._estimators
======================
Maps command name strings to callables with a unified signature.
"""

from __future__ import annotations

from typing import Callable, Union

import statsmodels.formula.api as smf


_COMMAND_ALIASES: dict[str, str] = {
    "reg":       "ols",
    "regress":   "ols",
    "logistic":  "logit",
    "pois":      "poisson",
}


def build_estimator(
    command: Union[str, Callable],
    fit_kwargs: dict,
) -> Callable:
    """
    Return a callable with signature ``f(data, depvar, indepvars) -> statsmodels result``.

    Parameters
    ----------
    command : str or callable
        Estimation command name or a custom callable.
    fit_kwargs : dict
        Extra keyword arguments forwarded to ``fit()``.

    Returns
    -------
    callable
        Function with signature ``(data, depvar, indepvars) -> result``.
    """
    if callable(command):
        def _passthrough(data, depvar, indepvars):
            return command(data, depvar, indepvars, **fit_kwargs)
        return _passthrough

    cmd = _COMMAND_ALIASES.get(command.lower(), command.lower())

    def _estimator(data, depvar, indepvars):
        formula = f"{depvar} ~ {' + '.join(indepvars)}"
        if cmd == "ols":
            return smf.ols(formula, data=data).fit(**fit_kwargs)
        elif cmd == "logit":
            return smf.logit(formula, data=data).fit(disp=0, **fit_kwargs)
        elif cmd == "probit":
            return smf.probit(formula, data=data).fit(disp=0, **fit_kwargs)
        elif cmd == "poisson":
            return smf.poisson(formula, data=data).fit(disp=0, **fit_kwargs)
        elif cmd == "glm":
            return smf.glm(formula, data=data, **fit_kwargs).fit()
        else:
            raise ValueError(
                f"Unknown command '{command}'. "
                "Built-in options: 'ols', 'logit', 'probit', 'poisson', 'glm', "
                "or pass a callable."
            )

    return _estimator
