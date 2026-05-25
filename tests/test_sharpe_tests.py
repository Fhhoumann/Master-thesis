from __future__ import annotations

import numpy as np
import pandas as pd

from core.benchmark import (
    build_pairwise_sharpe_test_table,
    calibrate_block_size_algorithm_3_1,
    ledoit_wolf_sharpe_difference_test,
    prepare_pairwise_excess_returns,
)
from core.benchmark.sharpe_tests import generate_var1_pseudo_sequence, stationary_bootstrap_indices


def test_prepare_pairwise_excess_returns_aligns_and_subtracts_rf() -> None:
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    strategy_a = pd.DataFrame(
        {
            "date": dates,
            "net_return": [0.02, 0.01, 0.03],
            "cost_convention": ["one_way"] * 3,
            "cost_bps": [0.0] * 3,
        }
    )
    strategy_b = pd.DataFrame(
        {
            "date": dates,
            "net_return": [0.01, 0.015, 0.02],
            "cost_convention": ["one_way"] * 3,
            "cost_bps": [0.0] * 3,
        }
    )
    rf = pd.DataFrame({"date": dates, "rf": [0.001, 0.002, 0.0015]})

    out = prepare_pairwise_excess_returns(strategy_a, strategy_b, rf)
    assert list(out.columns) == ["date", "month_key", "excess_a", "excess_b"]
    assert out.shape[0] == 3
    assert np.isclose(float(out.loc[0, "excess_a"]), 0.019)
    assert np.isclose(float(out.loc[1, "excess_b"]), 0.013)


def test_ledoit_wolf_sharpe_difference_test_returns_finite_statistics() -> None:
    rng = np.random.default_rng(7)
    excess = np.column_stack(
        [
            rng.normal(loc=0.012, scale=0.04, size=120),
            rng.normal(loc=0.006, scale=0.04, size=120),
        ]
    )
    result = ledoit_wolf_sharpe_difference_test(
        excess,
        strategy_a="A",
        strategy_b="B",
        block_size=4,
        bootstrap_resamples=99,
        seed=7,
    )

    assert result.n_obs == 120
    assert np.isfinite(result.delta_sharpe)
    assert np.isfinite(result.se_delta)
    assert np.isfinite(result.ci_lower)
    assert np.isfinite(result.ci_upper)
    assert 0.0 <= result.p_value <= 1.0


def test_stationary_bootstrap_and_var1_generation_preserve_shape() -> None:
    rng = np.random.default_rng(3)
    data = np.column_stack(
        [
            rng.normal(loc=0.01, scale=0.03, size=40),
            rng.normal(loc=0.008, scale=0.03, size=40),
        ]
    )
    idx = stationary_bootstrap_indices(39, avg_block_size=5.0, rng=rng)
    assert idx.shape == (39,)
    assert idx.min() >= 0
    assert idx.max() < 39

    pseudo = generate_var1_pseudo_sequence(data, residual_bootstrap_avg_block_size=5.0, rng=rng)
    assert pseudo.shape == data.shape
    assert np.isfinite(pseudo).all()


def test_algorithm_3_1_calibration_returns_valid_choice() -> None:
    rng = np.random.default_rng(5)
    excess = np.column_stack(
        [
            rng.normal(loc=0.012, scale=0.04, size=60),
            rng.normal(loc=0.009, scale=0.04, size=60),
        ]
    )
    result = calibrate_block_size_algorithm_3_1(
        excess,
        candidate_block_sizes=[1, 2, 4],
        pseudo_sequences=9,
        bootstrap_resamples=19,
        seed=5,
    )

    assert result.selected_block_size in {1, 2, 4}
    assert result.calibration_table.shape[0] == 3
    assert set(result.calibration_table["block_size"]) == {1, 2, 4}
    assert np.isfinite(result.calibration_table["g_hat"]).all()


def test_build_pairwise_sharpe_test_table_runs_for_three_strategies() -> None:
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(11)

    def _strategy(mean: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": dates,
                "net_return": rng.normal(loc=mean, scale=0.03, size=len(dates)),
                "cost_convention": ["one_way"] * len(dates),
                "cost_bps": [0.0] * len(dates),
            }
        )

    strategies = {
        "EN": _strategy(0.006),
        "RF": _strategy(0.008),
        "Torch": _strategy(0.01),
    }
    rf = pd.DataFrame({"date": dates, "rf": np.full(len(dates), 0.001)})

    out = build_pairwise_sharpe_test_table(
        strategies,
        rf,
        strategy_order=["EN", "RF", "Torch"],
        block_size=4,
        bootstrap_resamples=49,
        seed=11,
    )

    assert out.shape[0] == 3
    assert set(out["strategy_a"]) == {"EN", "EN", "RF"}
    assert set(out["strategy_b"]) == {"RF", "Torch"}


def test_build_pairwise_sharpe_test_table_with_calibration_adds_columns() -> None:
    dates = pd.date_range("2000-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(13)

    def _strategy(mean: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": dates,
                "net_return": rng.normal(loc=mean, scale=0.03, size=len(dates)),
                "cost_convention": ["one_way"] * len(dates),
                "cost_bps": [0.0] * len(dates),
            }
        )

    strategies = {"EN": _strategy(0.006), "RF": _strategy(0.008), "Torch": _strategy(0.01)}
    rf = pd.DataFrame({"date": dates, "rf": np.full(len(dates), 0.001)})
    out = build_pairwise_sharpe_test_table(
        strategies,
        rf,
        strategy_order=["EN", "RF", "Torch"],
        bootstrap_resamples=29,
        calibrate_block_size=True,
        candidate_block_sizes=[1, 2, 4],
        calibration_pseudo_sequences=7,
        calibration_bootstrap_resamples=11,
        seed=13,
    )

    assert out.shape[0] == 3
    assert out["block_size_calibrated"].all()
    assert "calibration_g_hat" in out.columns
