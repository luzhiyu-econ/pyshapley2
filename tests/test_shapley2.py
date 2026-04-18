"""
tests/test_shapley2.py
======================
Unit tests and numerical validation for pyshapley2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyshapley2 import Shapley2Result, shapley2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Reproducible synthetic dataset: n=300, 3 independent variables."""
    rng = np.random.default_rng(42)
    n   = 300
    x1  = rng.standard_normal(n)
    x2  = rng.standard_normal(n) + 0.4 * x1
    x3  = rng.standard_normal(n)
    y   = 2.0 * x1 + 1.5 * x2 + 0.5 * x3 + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


@pytest.fixture(scope="module")
def binary_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_df.copy()
    df["y_bin"] = (df["y"] > 0).astype(float)
    return df


# ---------------------------------------------------------------------------
# Basic OLS
# ---------------------------------------------------------------------------

class TestBasicOLS:
    def test_returns_result_object(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        assert isinstance(res, Shapley2Result)

    def test_table_shape(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        assert res.table.shape == (3, 4)

    def test_table_columns(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        assert list(res.table.columns) == [
            "shapley", "shapley_pct", "shapley_norm", "shapley_norm_pct"
        ]

    def test_table_index(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        assert list(res.table.index) == ["x1", "x2", "x3"]

    def test_runs_count(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        assert res.runs == 2 ** 3

    def test_k(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        assert res.K == 3

    def test_n_obs(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        assert res.n_obs == 300


# ---------------------------------------------------------------------------
# Numerical accuracy
# ---------------------------------------------------------------------------

class TestNumerics:
    def test_shapley_positive(self, sample_df):
        """All Shapley values should be positive given the DGP."""
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        assert res.table["shapley"].gt(0).all()

    def test_x1_largest_contribution(self, sample_df):
        """x1 should have the largest contribution (DGP coefficient = 2.0)."""
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        assert res.table.loc["x1", "shapley"] == res.table["shapley"].max()

    def test_norm_pct_sums_to_100(self, sample_df):
        """Normalised percentage contributions must sum to 100."""
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        total = res.table["shapley_norm_pct"].sum()
        assert abs(total - 100.0) < 1e-6

    def test_norm_sums_to_full_stat(self, sample_df):
        """Normalised Shapley values must sum to the full-model stat."""
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        total = res.table["shapley_norm"].sum()
        assert abs(total - res.full_stat) < 1e-8

    def test_full_stat_range(self, sample_df):
        """Full-model R² must be in (0, 1]."""
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        assert 0 < res.full_stat <= 1.0

    def test_adjusted_r2(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2_a")
        assert 0 < res.full_stat <= 1.0

    def test_residual_calculation(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        expected_residual = res.full_stat - float(res.shapley.sum())
        assert abs(res.residual - expected_residual) < 1e-10


# ---------------------------------------------------------------------------
# Grouped Shapley
# ---------------------------------------------------------------------------

class TestGrouped:
    def test_grouped_result(self, sample_df):
        res = shapley2(
            sample_df, "y", ["x1", "x2", "x3"],
            stat="r2",
            groups={"G1": ["x1", "x2"], "G2": ["x3"]},
        )
        assert res.K == 2
        assert list(res.table.index) == ["G1", "G2"]

    def test_grouped_runs(self, sample_df):
        res = shapley2(
            sample_df, "y", ["x1", "x2", "x3"],
            stat="r2",
            groups={"G1": ["x1", "x2"], "G2": ["x3"]},
        )
        assert res.runs == 4   # 2^2

    def test_grouped_g1_dominates(self, sample_df):
        """G1 (x1+x2) should contribute more than G2 (x3)."""
        res = shapley2(
            sample_df, "y", ["x1", "x2", "x3"],
            stat="r2",
            groups={"G1": ["x1", "x2"], "G2": ["x3"]},
        )
        assert res.table.loc["G1", "shapley"] > res.table.loc["G2", "shapley"]


# ---------------------------------------------------------------------------
# Estimation commands
# ---------------------------------------------------------------------------

class TestCommands:
    def test_logit(self, binary_df):
        res = shapley2(
            binary_df, "y_bin", ["x1", "x2", "x3"],
            stat="ll",
            command="logit",
        )
        assert isinstance(res, Shapley2Result)
        assert res.full_stat < 0   # log-likelihood is negative

    def test_probit(self, binary_df):
        res = shapley2(
            binary_df, "y_bin", ["x1", "x2", "x3"],
            stat="ll",
            command="probit",
        )
        assert isinstance(res, Shapley2Result)

    def test_custom_callable(self, sample_df):
        """Custom estimator callable should be accepted."""
        import statsmodels.formula.api as smf

        def my_ols(data, depvar, indepvars):
            formula = f"{depvar} ~ {' + '.join(indepvars)}"
            return smf.ols(formula, data=data).fit()

        res = shapley2(
            sample_df, "y", ["x1", "x2", "x3"],
            stat="r2",
            command=my_ols,
        )
        assert isinstance(res, Shapley2Result)


# ---------------------------------------------------------------------------
# Custom stat_func
# ---------------------------------------------------------------------------

class TestStatFunc:
    def test_custom_stat_func(self, sample_df):
        res = shapley2(
            sample_df, "y", ["x1", "x2", "x3"],
            stat_func=lambda r: r.rsquared,
        )
        assert isinstance(res, Shapley2Result)

    def test_custom_matches_builtin(self, sample_df):
        res_builtin = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        res_custom  = shapley2(
            sample_df, "y", ["x1", "x2", "x3"],
            stat_func=lambda r: r.rsquared,
        )
        np.testing.assert_allclose(
            res_builtin.shapley, res_custom.shapley, rtol=1e-6
        )


# ---------------------------------------------------------------------------
# Parallel consistency (requires joblib)
# ---------------------------------------------------------------------------

class TestParallel:
    pytest.importorskip("joblib")

    def test_parallel_matches_serial(self, sample_df):
        res_serial   = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2", n_jobs=1)
        res_parallel = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2", n_jobs=2)
        np.testing.assert_allclose(
            res_serial.shapley,
            res_parallel.shapley,
            rtol=1e-6,
            err_msg="Parallel and serial results do not match.",
        )

    def test_parallel_all_cores(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2"], stat="r2", n_jobs=-1)
        assert isinstance(res, Shapley2Result)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unknown_stat_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unknown stat"):
            shapley2(sample_df, "y", ["x1", "x2"], stat="not_a_stat")

    def test_too_many_vars_raises(self, sample_df):
        """More than 20 variables without force=True should raise ValueError."""
        many_vars = [f"x{i}" for i in range(21)]
        big_df = sample_df.copy()
        for v in many_vars:
            big_df[v] = np.random.randn(len(big_df))
        with pytest.raises(ValueError, match="force=True"):
            shapley2(big_df, "y", many_vars, stat="r2")

    def test_unknown_command_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unknown command"):
            shapley2(sample_df, "y", ["x1", "x2"], command="tobit")


# ---------------------------------------------------------------------------
# Output interface
# ---------------------------------------------------------------------------

class TestOutputs:
    def test_summary_returns_str(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        out = res.summary()
        assert isinstance(out, str)
        assert "TOTAL" in out
        assert "Residual" in out

    def test_repr(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        r   = repr(res)
        assert "Shapley2Result" in r
        assert "x1" in r

    def test_to_dict(self, sample_df):
        res = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        d   = res.to_dict()
        assert "table" in d
        assert d["K"] == 3

    def test_plot_returns_fig_ax(self, sample_df):
        pytest.importorskip("matplotlib")
        res      = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        fig, ax  = res.plot()
        assert fig is not None
        assert ax is not None

    @pytest.mark.parametrize("kind", ["pct", "norm_pct", "shapley", "norm"])
    def test_plot_kinds(self, sample_df, kind):
        pytest.importorskip("matplotlib")
        res     = shapley2(sample_df, "y", ["x1", "x2", "x3"], stat="r2")
        fig, ax = res.plot(kind=kind)
        assert fig is not None
