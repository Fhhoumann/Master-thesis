from __future__ import annotations

import numpy as np
import pandas as pd

from core.metrics import infer_periods_per_year, summarize_backtest_frame, summarize_returns


def test_infer_periods_per_year_detects_monthly_and_yearly() -> None:
    monthly = pd.date_range("2020-01-31", periods=6, freq="ME")
    yearly = pd.to_datetime(["2019-12-31", "2020-12-31", "2021-12-31"])

    assert infer_periods_per_year(monthly) == 12.0
    assert infer_periods_per_year(yearly) == 1.0


def test_summarize_backtest_uses_date_aware_frequency() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-31", "2020-12-31", "2021-12-31"]),
            "cost_convention": ["one_way", "one_way", "one_way"],
            "cost_bps": [10.0, 10.0, 10.0],
            "net_return": [0.10, 0.00, -0.05],
            "turnover_one_way": [0.0, 0.2, 0.1],
            "turnover_round_trip": [0.0, 0.4, 0.2],
        }
    )

    summary = summarize_backtest_frame(frame)
    row = summary.iloc[0]
    assert float(row["periods_per_year"]) == 1.0
    assert np.isclose(float(row["annual_volatility"]), float(np.std(frame["net_return"], ddof=0)))


def test_summarize_returns_monthly_default_still_works() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.00])
    out = summarize_returns(returns)
    assert float(out["periods_per_year"]) == 12.0
    assert np.isfinite(float(out["annual_return"]))


def test_summarize_backtest_uses_excess_sharpe_when_rf_provided() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]),
            "cost_convention": ["one_way"] * 4,
            "cost_bps": [0.0] * 4,
            "net_return": [0.02, 0.01, -0.01, 0.03],
            "turnover_one_way": [0.1, 0.1, 0.1, 0.1],
            "turnover_round_trip": [0.2, 0.2, 0.2, 0.2],
        }
    )
    rf = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]),
            "rf": [0.005, 0.005, 0.005, 0.005],
        }
    )
    raw = summarize_backtest_frame(frame).iloc[0]
    excess = summarize_backtest_frame(frame, risk_free_frame=rf).iloc[0]
    assert float(excess["sharpe"]) != float(raw["sharpe"])
    assert np.isfinite(float(excess["sharpe"]))
