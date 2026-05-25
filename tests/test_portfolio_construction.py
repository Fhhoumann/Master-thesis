from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.cv.splitters import TimeSeriesSplit
from core.portfolio import (
    construct_constrained_long_only_from_scores,
    construct_long_short_from_scores,
    normalize_weights_by_date,
)


def test_construct_long_short_from_scores_assigns_tails() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31")] * 10,
            "asset_id": list(range(10)),
            "prediction": np.arange(10, dtype=float),
        }
    )

    out = construct_long_short_from_scores(frame, top_quantile=0.2)
    lowest = out.nsmallest(2, "prediction")["weight"].to_numpy()
    highest = out.nlargest(2, "prediction")["weight"].to_numpy()
    middle = out.iloc[2:8]["weight"].to_numpy()

    assert np.allclose(lowest, -0.5)
    assert np.allclose(highest, 0.5)
    assert np.allclose(middle, 0.0)


def test_construct_long_short_from_scores_validates_quantile() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31")],
            "asset_id": [1, 2],
            "prediction": [0.1, -0.1],
        }
    )
    with pytest.raises(ValueError):
        construct_long_short_from_scores(frame, top_quantile=0.0)
    with pytest.raises(ValueError):
        construct_long_short_from_scores(frame, top_quantile=0.6)


def test_normalize_weights_by_date_targets_gross_and_neutrality() -> None:
    frame = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2020-01-31"),
                pd.Timestamp("2020-01-31"),
                pd.Timestamp("2020-01-31"),
                pd.Timestamp("2020-02-29"),
                pd.Timestamp("2020-02-29"),
                pd.Timestamp("2020-02-29"),
            ],
            "asset_id": [1, 2, 3, 1, 2, 3],
            "weight": [0.6, -0.2, -0.4, 2.0, -1.0, -1.0],
        }
    )

    out = normalize_weights_by_date(
        frame,
        weight_column="weight",
        clip=0.5,
        gross_target=1.0,
        dollar_neutral=True,
    )
    grouped = out.groupby("date", observed=True)["weight"]
    gross = grouped.apply(lambda s: float(s.abs().sum()))
    net = grouped.sum()

    assert np.allclose(gross.to_numpy(), 1.0)
    assert np.allclose(net.to_numpy(), 0.0, atol=1e-10)


def test_construct_constrained_long_only_from_scores_respects_cap_and_sum() -> None:
    train_dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    valid_date = pd.Timestamp("2020-04-30")
    asset_ids = np.arange(1, 11)

    predictions = pd.DataFrame(
        {
            "date": [valid_date] * len(asset_ids),
            "asset_id": asset_ids,
            "prediction": np.linspace(0.01, 0.10, len(asset_ids)),
            "fold_id": [0] * len(asset_ids),
        }
    )
    historical_rows = []
    for date_idx, date in enumerate(train_dates, 1):
        for asset_id in asset_ids:
            historical_rows.append(
                {
                    "date": date,
                    "asset_id": asset_id,
                    "ret_1m": 0.001 * asset_id + 0.0005 * date_idx,
                }
            )
    historical_returns = pd.DataFrame(historical_rows)
    split = TimeSeriesSplit(
        train_dates=np.array(train_dates),
        validation_dates=np.array([valid_date]),
        fold_id=0,
    )

    out = construct_constrained_long_only_from_scores(
        predictions,
        historical_returns=historical_returns,
        splits=[split],
        top_quantile=1.0,
        weight_cap=0.10,
        covariance_shrinkage=0.5,
        risk_aversion=5.0,
        optimizer_steps=100,
        optimizer_step_size=0.05,
    )
    weights = out["weight"].to_numpy()

    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert np.all(weights >= -1e-10)
    assert np.all(weights <= 0.10 + 1e-6)
