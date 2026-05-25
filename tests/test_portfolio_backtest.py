from __future__ import annotations

import numpy as np
import pandas as pd

from core.portfolio import missing_returns_by_month, portfolio_returns_from_weights


def test_portfolio_returns_excludes_missing_and_renormalizes() -> None:
    date = pd.Timestamp("2020-01-31")
    weights = pd.DataFrame(
        {
            "date": [date, date],
            "asset_id": [1, 2],
            "weight": [0.5, -0.5],
        }
    )
    realized = pd.DataFrame(
        {
            "date": [date, date],
            "asset_id": [1, 2],
            "ret_1m": [0.10, np.nan],
        }
    )

    out = portfolio_returns_from_weights(weights, realized)
    gross = float(out.loc[out["date"] == date, "gross_return"].iloc[0])
    assert np.isclose(gross, 0.10)


def test_portfolio_returns_all_missing_for_date_yields_zero() -> None:
    date = pd.Timestamp("2020-01-31")
    weights = pd.DataFrame(
        {
            "date": [date, date],
            "asset_id": [1, 2],
            "weight": [0.5, -0.5],
        }
    )
    realized = pd.DataFrame(
        {
            "date": [date, date],
            "asset_id": [1, 2],
            "ret_1m": [np.nan, np.nan],
        }
    )

    out = portfolio_returns_from_weights(weights, realized)
    gross = float(out.loc[out["date"] == date, "gross_return"].iloc[0])
    assert np.isclose(gross, 0.0)


def test_portfolio_returns_fill_zero_policy_keeps_original_weights() -> None:
    date = pd.Timestamp("2020-01-31")
    weights = pd.DataFrame(
        {
            "date": [date, date],
            "asset_id": [1, 2],
            "weight": [0.5, -0.5],
        }
    )
    realized = pd.DataFrame(
        {
            "date": [date, date],
            "asset_id": [1, 2],
            "ret_1m": [0.10, np.nan],
        }
    )

    out = portfolio_returns_from_weights(weights, realized, missing_return_policy="fill_zero")
    gross = float(out.loc[out["date"] == date, "gross_return"].iloc[0])
    assert np.isclose(gross, 0.05)


def test_missing_returns_by_month_extracts_unique_date_rows() -> None:
    date = pd.Timestamp("2020-01-31")
    frame = pd.DataFrame(
        {
            "date": [date, date],
            "gross_return": [0.1, 0.1],
            "n_assets_total": [2, 2],
            "n_missing_return": [1, 1],
            "n_valid_return": [1, 1],
            "missing_return_rate": [0.5, 0.5],
            "gross_weight_before": [1.0, 1.0],
            "gross_weight_valid": [0.5, 0.5],
            "dropped_weight_gross": [0.5, 0.5],
            "renorm_scale": [2.0, 2.0],
            "missing_return_policy": ["exclude_renorm", "exclude_renorm"],
            "cost_convention": ["one_way", "round_trip"],
            "cost_bps": [10.0, 10.0],
            "net_return": [0.1, 0.1],
        }
    )

    out = missing_returns_by_month(frame)
    assert out.shape[0] == 1
    assert float(out.loc[0, "missing_return_rate_pct"]) == 50.0
