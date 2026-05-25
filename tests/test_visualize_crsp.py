from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from apps.visualize.main import (
    _asset_label,
    _build_crsp_monthly_series,
    _load_crsp_lookup,
    _rank_eligible_crsp_ids,
)


def test_rank_eligible_crsp_ids_uses_id_level_month_coverage() -> None:
    permno = np.array(
        [
            [10001.0, 10001.0, 20002.0, 20002.0],
            [20002.0, 10001.0, 10001.0, 10001.0],
            [np.nan, 30003.0, 30003.0, 30003.0],
        ],
        dtype=np.float64,
    )
    returns_current = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.00, 0.10, 0.20, 0.30],
            [np.nan, 0.50, 0.60, np.nan],
        ],
        dtype=np.float64,
    )
    month_end = [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-04-30"),
    ]

    coverage = _rank_eligible_crsp_ids(
        permno,
        returns_current,
        month_end,
        min_months=3,
        max_ids=2,
    )

    assert coverage["asset_id"].tolist() == [10001, 20002]
    assert coverage["months_observed"].tolist() == [4, 3]


def test_build_crsp_monthly_series_aggregates_and_compounds() -> None:
    permno = np.array(
        [
            [111.0, 111.0, 111.0],
            [111.0, 222.0, 111.0],
        ],
        dtype=np.float64,
    )
    returns_current = np.array(
        [
            [0.10, 0.20, -0.10],
            [0.30, 0.40, 0.10],
        ],
        dtype=np.float64,
    )
    month_end = [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
        pd.Timestamp("2020-03-31"),
    ]

    series = _build_crsp_monthly_series(
        permno,
        returns_current,
        month_end,
        np.array([111, 222], dtype=np.int64),
        aggregator="median",
    )

    series_111 = series.loc[series["asset_id"] == 111].reset_index(drop=True)
    assert series_111.shape[0] == 3
    assert np.allclose(series_111["ret_agg"].to_numpy(), np.array([0.20, 0.20, 0.00]))
    assert np.allclose(series_111["row_count"].to_numpy(), np.array([2, 1, 2]))
    assert np.allclose(series_111["ret_std"].to_numpy(), np.array([0.10, 0.00, 0.10]))
    assert np.allclose(series_111["cumret"].to_numpy(), np.array([0.20, 0.44, 0.44]))

    series_222 = series.loc[series["asset_id"] == 222].reset_index(drop=True)
    assert series_222.shape[0] == 1
    assert np.isclose(float(series_222.iloc[0]["ret_agg"]), 0.40)
    assert np.isclose(float(series_222.iloc[0]["cumret"]), 0.40)


def test_load_crsp_lookup_and_asset_label(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.csv"
    pd.DataFrame(
        {
            "asset_id": [10101, 20202],
            "ticker": ["AAPL", "MSFT"],
            "company_name": ["Apple Inc.", "Microsoft Corp."],
        }
    ).to_csv(lookup_path, index=False)

    lookup = _load_crsp_lookup(lookup_path)

    assert _asset_label(10101, lookup) == "10101 | AAPL | Apple Inc."
    assert _asset_label(20202, lookup) == "20202 | MSFT | Microsoft Corp."
    assert _asset_label(99999, lookup) == "99999"
