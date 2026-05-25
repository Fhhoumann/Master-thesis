from __future__ import annotations

import numpy as np
import pandas as pd

from core.benchmark import (
    build_ew_market,
    build_umd_momentum,
    compute_attribution_table,
    run_factor_regressions,
)
from core.config import RegimeConfig


def test_build_ew_market_returns_mean_by_date() -> None:
    panel = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2020-01-31"),
                pd.Timestamp("2020-01-31"),
                pd.Timestamp("2020-02-29"),
                pd.Timestamp("2020-02-29"),
            ],
            "asset_id": [1, 2, 1, 2],
            "ret_1m": [0.01, -0.01, 0.03, 0.01],
        }
    )

    out = build_ew_market(panel)
    assert out["benchmark_name"].nunique() == 1
    jan = out.loc[out["date"] == pd.Timestamp("2020-01-31"), "gross_return"].iloc[0]
    feb = out.loc[out["date"] == pd.Timestamp("2020-02-29"), "gross_return"].iloc[0]
    assert np.isclose(jan, 0.0)
    assert np.isclose(feb, 0.02)


def test_build_umd_momentum_produces_long_short_series() -> None:
    dates = pd.date_range("2000-01-31", periods=18, freq="ME")
    asset_ids = np.arange(10000, 10025)
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date in dates:
        for rank, asset_id in enumerate(asset_ids):
            rows.append(
                {
                    "date": date,
                    "asset_id": float(asset_id),
                    "ret_1m": (rank - 12) / 1000.0,
                }
            )
    panel = pd.DataFrame(rows)

    out = build_umd_momentum(panel, top_q=0.10)
    assert not out.empty
    assert out["benchmark_name"].eq("umd_12_1_ls").all()
    assert float(out["gross_return"].mean()) > 0.0


def test_run_factor_regressions_capm_recovers_beta() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2001-01-31", periods=48, freq="ME")
    factors = pd.DataFrame(
        {
            "date": dates,
            "mkt_rf": rng.normal(0.005, 0.02, size=len(dates)),
            "rf": np.full(len(dates), 0.002),
        }
    )
    noise = rng.normal(0.0, 0.001, size=len(dates))
    returns = pd.Series(
        (
            factors["rf"]
            + 0.001
            + 1.2 * factors["mkt_rf"]
            + noise
        ).to_numpy(),
        index=dates,
    )

    stats = run_factor_regressions(returns, factors, model="capm", min_obs=24)
    assert stats["status"] == "ok"
    assert int(stats["n_obs"]) == len(dates)
    assert np.isfinite(float(stats["alpha_monthly"]))
    assert abs(float(stats["beta_mkt_rf"]) - 1.2) < 0.15


def test_compute_attribution_table_includes_subperiods() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("1999-01-31", periods=36, freq="ME")
    factors = pd.DataFrame(
        {
            "date": dates,
            "mkt_rf": rng.normal(0.004, 0.018, size=len(dates)),
            "rf": np.full(len(dates), 0.0015),
        }
    )
    period_return = (
        factors["rf"]
        + 0.001
        + 0.8 * factors["mkt_rf"]
        + rng.normal(0.0, 0.0015, size=len(dates))
    )
    returns_frame = pd.DataFrame(
        {
            "run_key": "run_a",
            "date": dates,
            "period_return": period_return,
            "return_type": "net",
            "cost_convention": "one_way",
            "cost_bps": 25.0,
        }
    )
    regimes = [
        RegimeConfig(name="1990s", start=pd.Timestamp("1990-01-01").date(), end=pd.Timestamp("1999-12-31").date()),
        RegimeConfig(name="2000+", start=pd.Timestamp("2000-01-01").date(), end=pd.Timestamp("2024-12-31").date()),
    ]

    out = compute_attribution_table(
        returns_frame,
        factors,
        regimes,
        models=("capm",),
        min_obs=6,
    )

    assert out.shape[0] == 3
    assert set(out["regime"]) == {"full_sample", "1990s", "2000+"}
    full_capm = out[(out["regime"] == "full_sample") & (out["model"] == "capm")].iloc[0]
    assert full_capm["status"] == "ok"
    assert int(full_capm["n_obs"]) == 36
