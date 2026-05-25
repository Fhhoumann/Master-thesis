from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config.schemas import FeatureConfig
from core.features.technical import (
    FEATURE_SET_REGISTRY,
    FEATURE_COLUMNS,
    SUPERVISOR_MACRO_INTERACTION_COLUMNS,
    generate_features,
)


def test_generate_features_target_alignment_and_readiness() -> None:
    dates = pd.date_range("2000-01-31", periods=40, freq="ME")
    ret_a = np.linspace(0.01, 0.04, 40)
    ret_b = np.linspace(-0.02, 0.03, 40)
    vol_a = np.linspace(100, 200, 40)
    vol_b = np.linspace(300, 500, 40)
    vol_b[:3] = np.nan

    panel = pd.DataFrame(
        {
            "date": np.concatenate([dates, dates]),
            "asset_id": np.concatenate([np.repeat(10001, 40), np.repeat(10002, 40)]),
            "ret_1m": np.concatenate([ret_a, ret_b]),
            "volume_1m": np.concatenate([vol_a, vol_b]),
        }
    )

    cfg = FeatureConfig(min_history=5)
    result = generate_features(panel, cfg)
    frame = result.frame.sort_values(["asset_id", "date"]).reset_index(drop=True)

    asset_a = frame.loc[frame["asset_id"] == 10001].reset_index(drop=True)
    assert np.isclose(asset_a.loc[10, "target_ret_1m"], ret_a[11])
    assert np.isnan(asset_a.loc[len(asset_a) - 1, "target_ret_1m"])

    assert (asset_a.loc[:10, "feature_ready"] == False).all()  # noqa: E712
    assert asset_a.loc[30:, "feature_ready"].all()

    for column in FEATURE_COLUMNS:
        assert column in frame.columns

    assert frame["volume_1m"].isna().sum() == 0


def test_generate_features_merges_macro_main_effects_by_month(tmp_path: Path) -> None:
    dates = pd.date_range("2000-01-31", periods=40, freq="ME")
    panel = pd.DataFrame(
        {
            "date": np.concatenate([dates, dates]),
            "asset_id": np.concatenate([np.repeat(10001, 40), np.repeat(10002, 40)]),
            "ret_1m": np.concatenate([np.linspace(0.01, 0.04, 40), np.linspace(-0.02, 0.03, 40)]),
            "volume_1m": np.concatenate([np.linspace(100, 200, 40), np.linspace(300, 500, 40)]),
        }
    )
    macro_path = tmp_path / "macro.parquet"
    macro = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-28", periods=40, freq="BME"),
            "macro_vix": np.linspace(10.0, 20.0, 40),
            "macro_epu": np.linspace(100.0, 140.0, 40),
            "macro_credit_spread": np.linspace(1.0, 2.0, 40),
            "macro_term_spread": np.linspace(0.5, -0.5, 40),
        }
    )
    macro.to_parquet(macro_path, index=False)

    cfg = FeatureConfig(min_history=5, macro_factors_path=macro_path)
    result = generate_features(panel, cfg)

    for column in cfg.macro_columns:
        assert column in result.feature_columns
        assert column in result.frame.columns

    january = result.frame.loc[result.frame["date"] == pd.Timestamp("2000-01-31")]
    assert january["macro_vix"].nunique() == 1
    assert january["macro_vix"].iloc[0] == pytest.approx(10.0)
    assert january["macro_epu"].iloc[0] == pytest.approx(100.0)
    assert "macro_main_effects" in FEATURE_SET_REGISTRY


def test_generate_features_adds_supervisor_interaction_block(tmp_path: Path) -> None:
    dates = pd.date_range("2000-01-31", periods=40, freq="ME")
    panel = pd.DataFrame(
        {
            "date": np.concatenate([dates, dates]),
            "asset_id": np.concatenate([np.repeat(10001, 40), np.repeat(10002, 40)]),
            "ret_1m": np.concatenate([np.linspace(0.01, 0.04, 40), np.linspace(-0.02, 0.03, 40)]),
            "volume_1m": np.concatenate([np.linspace(100, 200, 40), np.linspace(300, 500, 40)]),
        }
    )
    macro_path = tmp_path / "macro.parquet"
    macro = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-28", periods=40, freq="BME"),
            "macro_vix": np.linspace(10.0, 20.0, 40),
            "macro_epu": np.linspace(100.0, 140.0, 40),
            "macro_credit_spread": np.linspace(1.0, 2.0, 40),
            "macro_term_spread": np.linspace(0.5, -0.5, 40),
        }
    )
    macro.to_parquet(macro_path, index=False)

    cfg = FeatureConfig(min_history=5, macro_factors_path=macro_path)
    result = generate_features(panel, cfg)

    for column in SUPERVISOR_MACRO_INTERACTION_COLUMNS:
        assert column in result.feature_columns
        assert column in result.frame.columns

    ready = result.frame.loc[result.frame["feature_ready"]].iloc[0]
    assert ready["mom_12_1_x_macro_epu"] == pytest.approx(ready["mom_12_1"] * ready["macro_epu"])
    assert ready["log_volume_x_macro_credit_spread"] == pytest.approx(
        ready["log_volume"] * ready["macro_credit_spread"]
    )


def test_generate_features_adds_binary_ma_signal_3_12() -> None:
    dates = pd.date_range("2000-01-31", periods=24, freq="ME")
    ret_up = np.full(24, 0.02)
    ret_down = np.full(24, -0.02)
    volume = np.linspace(100, 200, 24)

    panel = pd.DataFrame(
        {
            "date": np.concatenate([dates, dates]),
            "asset_id": np.concatenate([np.repeat(10001, 24), np.repeat(10002, 24)]),
            "ret_1m": np.concatenate([ret_up, ret_down]),
            "volume_1m": np.concatenate([volume, volume]),
        }
    )

    cfg = FeatureConfig(min_history=12)
    result = generate_features(panel, cfg)
    frame = result.frame.sort_values(["asset_id", "date"]).reset_index(drop=True)

    assert "ma_signal_3_12" in result.feature_columns
    assert "ma_signal_3_12" in frame.columns

    recent_up = frame.loc[
        (frame["asset_id"] == 10001) & (frame["date"] == pd.Timestamp("2001-12-31")),
        "ma_signal_3_12",
    ].iloc[0]
    recent_down = frame.loc[
        (frame["asset_id"] == 10002) & (frame["date"] == pd.Timestamp("2001-12-31")),
        "ma_signal_3_12",
    ].iloc[0]

    assert recent_up == pytest.approx(1.0)
    assert recent_down == pytest.approx(0.0)
