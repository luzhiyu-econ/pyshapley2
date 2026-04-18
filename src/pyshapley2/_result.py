"""
pyshapley2._result
==================
Container, display, and visualisation for Shapley-Owen decomposition results.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class Shapley2Result:
    """
    Container for Shapley-Owen decomposition results.

    Attributes
    ----------
    table : pd.DataFrame
        Four-column result table indexed by variable/group name:

        * ``shapley``          — raw Shapley value
        * ``shapley_pct``      — percentage contribution (raw)
        * ``shapley_norm``     — normalised Shapley value
        * ``shapley_norm_pct`` — percentage contribution (normalised)

    full_stat : float
        Full-model fit statistic (e.g. R²).
    residual : float
        Difference between full_stat and the sum of all Shapley values.
    K : int
        Number of variables or groups.
    runs : int
        Number of subset regressions executed (= 2^K).
    n_obs : int
        Number of observations used in the regressions.
    """

    def __init__(
        self,
        *,
        labels: list[str],
        depvar: str,
        indepvars: list[str],
        stat: str,
        command: str,
        full_stat: float,
        shapley: np.ndarray,
        shapley_rel: np.ndarray,
        shapley_norm: np.ndarray,
        shapley_rel_norm: np.ndarray,
        residual: float,
        groups: Optional[dict],
        K: int,
        runs: int,
        n_obs: int,
    ) -> None:
        self.labels           = labels
        self.depvar           = depvar
        self.indepvars        = indepvars
        self.stat             = stat
        self.command          = command
        self.full_stat        = full_stat
        self.shapley          = shapley
        self.shapley_rel      = shapley_rel
        self.shapley_norm     = shapley_norm
        self.shapley_rel_norm = shapley_rel_norm
        self.residual         = residual
        self.groups           = groups
        self.K                = K
        self.runs             = runs
        self.n_obs            = n_obs

        self.table = pd.DataFrame(
            {
                "shapley":          shapley,
                "shapley_pct":      shapley_rel * 100,
                "shapley_norm":     shapley_norm,
                "shapley_norm_pct": shapley_rel_norm * 100,
            },
            index=labels,
        )

    # Format and print results table
    def summary(self) -> str:
        """
        Print a Stata-style decomposition table and return it as a string.

        Returns
        -------
        str
            Formatted table text.
        """
        SEP = "─" * 11 + "┼" + "─" * 15 + "┼" + "─" * 11 + "┼" + "─" * 14 + "┼" + "─" * 13

        header_1 = (
            f"{'Factor':<11}│{'Shapley value':^15}│{'Per cent':^11}│"
            f"{'Shapley value':^14}│{'Per cent':^13}"
        )
        header_2 = (
            f"{'':<11}│{'(estimate)':^15}│{'(estimate)':^11}│"
            f"{'(normalized)':^14}│{'(normalized)':^13}"
        )

        lines = [
            f"Shapley-Owen decomposition  |  depvar: {self.depvar}"
            f"  |  stat: {self.stat}  |  command: {self.command}",
            f"Observations: {self.n_obs:,}  |  Subsets: {self.runs:,}  |  K={self.K}",
            "",
            header_1,
            header_2,
            SEP,
        ]

        for lbl, row in self.table.iterrows():
            name = str(lbl)[:10]
            lines.append(
                f"{name:<11}│{row['shapley']:>14.5f} │"
                f"{row['shapley_pct']:>9.2f} % │"
                f"{row['shapley_norm']:>13.5f} │"
                f"{row['shapley_norm_pct']:>11.2f} %"
            )

        lines.append(SEP)

        res_pct = 100.0 * (1.0 - float(np.sum(self.shapley)) / self.full_stat)
        lines.append(
            f"{'Residual':<11}│{self.residual:>14.5f} │"
            f"{res_pct:>9.2f} % │{'':14}│{'':13}"
        )
        lines.append(SEP)
        lines.append(
            f"{'TOTAL':<11}│{self.full_stat:>14.5f} │"
            f"{100.0:>9.2f} % │{self.full_stat:>13.5f} │"
            f"{100.0:>11.2f} %"
        )
        lines.append(SEP)

        if self.groups:
            lines.append("")
            lines.append("Groups:")
            for gname, gvars in self.groups.items():
                lines.append(f"  {gname}: {', '.join(gvars)}")

        out = "\n".join(lines)
        print(out)
        return out

    # Visualization
    def plot(
        self,
        kind: str = "norm_pct",
        ax=None,
        color: str = "#2196F3",
        neg_color: str = "#F44336",
        figsize: tuple = None,
        title: Optional[str] = None,
        **kwargs,
    ):
        """
        Plot a horizontal bar chart of Shapley contributions.

        Parameters
        ----------
        kind : str, default ``"norm_pct"``
            Column to display:

            * ``"pct"``      — shapley_pct (raw %)
            * ``"norm_pct"`` — shapley_norm_pct (normalised %)
            * ``"shapley"``  — raw Shapley values
            * ``"norm"``     — normalised Shapley values
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on; creates new figure if None.
        color : str
            Bar colour for positive values.
        neg_color : str
            Bar colour for negative values.
        figsize : tuple, optional
            Figure size. Auto-computed from K when omitted.
        title : str, optional
            Plot title. Auto-generated when omitted.
        **kwargs
            Additional keyword arguments forwarded to ``ax.barh()``.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        """
        import matplotlib.pyplot as plt

        _col_map = {
            "pct":      ("shapley_pct",      f"Shapley contribution % (raw · {self.stat})"),
            "norm_pct": ("shapley_norm_pct", f"Shapley contribution % (normalised · {self.stat})"),
            "shapley":  ("shapley",           f"Shapley value (raw · {self.stat})"),
            "norm":     ("shapley_norm",      f"Shapley value (normalised · {self.stat})"),
        }
        if kind not in _col_map:
            raise ValueError(f"kind must be one of {list(_col_map.keys())}; got '{kind}'.")

        col, xlabel = _col_map[kind]
        vals        = self.table[col].sort_values(ascending=True)
        colors      = [color if v >= 0 else neg_color for v in vals]

        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize or (8, max(3.0, self.K * 0.55 + 1.2))
            )
        else:
            fig = ax.figure

        ax.barh(vals.index.astype(str), vals.values, color=colors, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_title(
            title
            or f"Shapley-Owen decomposition | depvar: {self.depvar} | stat: {self.stat}"
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        return fig, ax

    # Special methods
    def __repr__(self) -> str:
        return (
            f"<Shapley2Result  depvar={self.depvar!r}  stat={self.stat!r}"
            f"  K={self.K}  runs={self.runs}  n_obs={self.n_obs}>\n"
            + self.table.to_string(float_format="{:.5f}".format)
        )

    def to_dict(self) -> dict:
        """Return a serialisable dict of all result fields."""
        return {
            "depvar":    self.depvar,
            "stat":      self.stat,
            "command":   self.command,
            "full_stat": self.full_stat,
            "residual":  self.residual,
            "K":         self.K,
            "runs":      self.runs,
            "n_obs":     self.n_obs,
            "table":     self.table.to_dict(),
        }
