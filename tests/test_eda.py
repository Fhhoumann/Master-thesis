from __future__ import annotations

import pandas as pd

from apps.eda.main import (
    _assign_regime,
    _build_benchmark_comparison_table,
    _strategy_baseline_selector,
)
from core.config import EDAConfig, load_eda_config


def test_load_eda_config_default_file() -> None:
    cfg = load_eda_config("configs/eda/default.yaml")
    assert cfg.mode in {"full", "quick"}
    assert cfg.baseline_cost_convention == "one_way"
    assert cfg.baseline_cost_bps == 25.0
    assert len(cfg.regimes) == 2
    assert cfg.regimes[0].name == "1990s"


def test_assign_regime_labels() -> None:
    cfg = EDAConfig()
    dates = pd.Series(
        [
            pd.Timestamp("1995-06-30"),
            pd.Timestamp("2005-01-31"),
            pd.Timestamp("2017-07-31"),
            pd.Timestamp("2024-10-31"),
            pd.Timestamp("2025-01-31"),
        ]
    )
    regimes = _assign_regime(dates, cfg.regimes)
    assert list(regimes.astype(str)) == [
        "1990s",
        "2000+",
        "2000+",
        "2000+",
        "outside_regimes",
    ]


def test_strategy_baseline_selector_prefers_exact_match() -> None:
    cfg = EDAConfig(baseline_cost_convention="one_way", baseline_cost_bps=25.0)
    summary = pd.DataFrame(
        {
            "cost_convention": ["one_way", "one_way", "round_trip"],
            "cost_bps": [10.0, 25.0, 25.0],
            "sharpe": [0.1, 0.2, 0.15],
        }
    )
    selected = _strategy_baseline_selector(summary, cfg)
    assert selected.shape[0] == 1
    assert selected.iloc[0]["cost_convention"] == "one_way"
    assert float(selected.iloc[0]["cost_bps"]) == 25.0


def test_benchmark_comparison_uses_per_run_baseline_fallback() -> None:
    cfg = EDAConfig(baseline_cost_convention="one_way", baseline_cost_bps=25.0)
    strategy_returns = pd.DataFrame(
        {
            "run_key": ["run_a"] * 3 + ["run_b"] * 3,
            "date": pd.to_datetime(
                [
                    "2000-01-31",
                    "2000-02-29",
                    "2000-03-31",
                    "2000-01-31",
                    "2000-02-29",
                    "2000-03-31",
                ]
            ),
            "period_return": [0.01, 0.00, 0.02, -0.01, 0.01, 0.00],
        }
    )
    benchmark_returns = pd.DataFrame(
        {
            "benchmark_name": ["ew_market"] * 3,
            "date": pd.to_datetime(["2000-01-31", "2000-02-29", "2000-03-31"]),
            "gross_return": [0.00, 0.01, 0.01],
        }
    )

    out = _build_benchmark_comparison_table(strategy_returns, benchmark_returns, cfg.regimes)
    assert set(out["run_key"]) == {"run_a", "run_b"}
    assert set(out["scope"]) == {"full_sample", "subperiod"}
    assert out["periods_per_year"].eq(12.0).all()
