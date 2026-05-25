from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from core.config.schemas import FeatureConfig
from core.features.macro import merge_macro_features

if TYPE_CHECKING:
    from core.io.datacube import DataCube

BASE_FEATURE_COLUMNS = [
    "mom_12_1",
    "mom_6_1",
    "mom_3_1",
    "tsmom_12",
    "tsmom_sign_12",
    "reversal_1m",
    "ma_gap_12m",
    "ma_crossover_6_12",
    "ma_signal_3_12",
    "vol_3m",
    "vol_12m",
    "vol_regime_3_12",
    "log_volume",
    "abnormal_log_volume",
]

MACRO_FEATURE_COLUMNS = [
    "macro_vix",
    "macro_epu",
    "macro_credit_spread",
    "macro_term_spread",
]

SUPERVISOR_MACRO_INTERACTION_COLUMNS = [
    "mom_12_1_x_macro_epu",
    "mom_12_1_x_macro_vix",
    "tsmom_12_x_macro_epu",
    "tsmom_12_x_macro_vix",
    "ma_gap_12m_x_macro_epu",
    "ma_crossover_6_12_x_macro_vix",
    "reversal_1m_x_macro_epu",
    "reversal_1m_x_macro_vix",
    "reversal_1m_x_macro_credit_spread",
    "abnormal_log_volume_x_macro_epu",
    "abnormal_log_volume_x_macro_vix",
    "log_volume_x_macro_credit_spread",
    "vol_regime_3_12_x_macro_vix",
    "vol_regime_3_12_x_macro_term_spread",
    "vol_12m_x_macro_credit_spread",
]

FEATURE_SET_REGISTRY = {
    "baseline_technical": BASE_FEATURE_COLUMNS.copy(),
    "macro_main_effects": BASE_FEATURE_COLUMNS.copy() + MACRO_FEATURE_COLUMNS.copy(),
    "supervisor_macro_interactions": BASE_FEATURE_COLUMNS.copy()
    + MACRO_FEATURE_COLUMNS.copy()
    + SUPERVISOR_MACRO_INTERACTION_COLUMNS.copy(),
}

# Backwards-compatible alias used by the rest of the codebase for the original
# baseline technical feature list.
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS

NON_WINSORIZED_FEATURE_COLUMNS = {
    "tsmom_sign_12",
    "ma_signal_3_12",
}


@dataclass(slots=True)
class FeatureResult:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str


def get_feature_columns(feature_set_name: str = "baseline_technical") -> list[str]:
    try:
        return FEATURE_SET_REGISTRY[feature_set_name].copy()
    except KeyError as exc:
        known_sets = ", ".join(sorted(FEATURE_SET_REGISTRY))
        raise ValueError(
            f"Unknown feature set `{feature_set_name}`. Known sets: {known_sets}"
        ) from exc


def list_feature_sets() -> list[str]:
    return sorted(FEATURE_SET_REGISTRY)


def _winsorize_by_date(
    frame: pd.DataFrame, column: str, lower_q: float, upper_q: float
) -> pd.Series:
    def _clip_group(values: pd.Series) -> pd.Series:
        valid = values.dropna()
        if valid.empty:
            return values
        lower = float(valid.quantile(lower_q))
        upper = float(valid.quantile(upper_q))
        return values.clip(lower=lower, upper=upper)

    return frame.groupby("date", observed=True)[column].transform(_clip_group)


def _zscore_by_date(frame: pd.DataFrame, column: str) -> pd.Series:
    group = frame.groupby("date", observed=True)[column]
    mean = group.transform("mean")
    std = group.transform("std")
    return (frame[column] - mean) / (std + 1e-8)


def _rolling_product(series: pd.Series, window: int) -> pd.Series:
    return (1.0 + series).rolling(window=window, min_periods=window).apply(
        np.prod, raw=True
    ) - 1.0


def _binary_ma_signal(short_ma: np.ndarray | pd.Series, long_ma: np.ndarray | pd.Series) -> np.ndarray:
    short_arr = np.asarray(short_ma, dtype=np.float64)
    long_arr = np.asarray(long_ma, dtype=np.float64)
    signal = np.full(short_arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(short_arr) & np.isfinite(long_arr)
    signal[valid] = (short_arr[valid] >= long_arr[valid]).astype(np.float64)
    return signal


def _fill_volume_cross_section(volume: pd.Series, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"date": dates, "volume_1m": volume})
    medians = frame.groupby("date", observed=True)["volume_1m"].transform("median")
    filled = frame["volume_1m"].fillna(medians)
    return filled.fillna(0.0)


def _winsorize_feature_columns(frame: pd.DataFrame, feature_cfg: FeatureConfig) -> pd.DataFrame:
    for column in BASE_FEATURE_COLUMNS:
        if column in NON_WINSORIZED_FEATURE_COLUMNS:
            continue
        frame[column] = _winsorize_by_date(
            frame, column, feature_cfg.winsorize_lower, feature_cfg.winsorize_upper
        )
    return frame


def _add_supervisor_macro_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(MACRO_FEATURE_COLUMNS).issubset(frame.columns):
        return frame

    out = frame.copy()
    out["mom_12_1_x_macro_epu"] = out["mom_12_1"] * out["macro_epu"]
    out["mom_12_1_x_macro_vix"] = out["mom_12_1"] * out["macro_vix"]
    out["tsmom_12_x_macro_epu"] = out["tsmom_12"] * out["macro_epu"]
    out["tsmom_12_x_macro_vix"] = out["tsmom_12"] * out["macro_vix"]
    out["ma_gap_12m_x_macro_epu"] = out["ma_gap_12m"] * out["macro_epu"]
    out["ma_crossover_6_12_x_macro_vix"] = out["ma_crossover_6_12"] * out["macro_vix"]
    out["reversal_1m_x_macro_epu"] = out["reversal_1m"] * out["macro_epu"]
    out["reversal_1m_x_macro_vix"] = out["reversal_1m"] * out["macro_vix"]
    out["reversal_1m_x_macro_credit_spread"] = (
        out["reversal_1m"] * out["macro_credit_spread"]
    )
    out["abnormal_log_volume_x_macro_epu"] = out["abnormal_log_volume"] * out["macro_epu"]
    out["abnormal_log_volume_x_macro_vix"] = out["abnormal_log_volume"] * out["macro_vix"]
    out["log_volume_x_macro_credit_spread"] = out["log_volume"] * out["macro_credit_spread"]
    out["vol_regime_3_12_x_macro_vix"] = out["vol_regime_3_12"] * out["macro_vix"]
    out["vol_regime_3_12_x_macro_term_spread"] = (
        out["vol_regime_3_12"] * out["macro_term_spread"]
    )
    out["vol_12m_x_macro_credit_spread"] = out["vol_12m"] * out["macro_credit_spread"]
    return out


def _compute_direct_cube_features(datacube: DataCube) -> pd.DataFrame:
    returns_hist = datacube.returns[:, :-1, :]  # (asset, 120, time)
    returns_oos = datacube.returns[:, -1, :]    # (asset, time)
    volume_hist = datacube.volume[:, :-1, :]    # (asset, 120, time)

    n_assets, n_hist, n_time = returns_hist.shape
    if n_hist < 12:
        raise ValueError(f"direct_cube requires at least 12 historical slots, got {n_hist}")

    def _prod_from_history(window: np.ndarray) -> np.ndarray:
        out = np.full(window.shape[0], np.nan, dtype=np.float64)
        valid = np.isfinite(window).all(axis=1)
        if np.any(valid):
            out[valid] = np.prod(1.0 + window[valid], axis=1) - 1.0
        return out

    def _std_from_history(window: np.ndarray) -> np.ndarray:
        out = np.full(window.shape[0], np.nan, dtype=np.float64)
        valid = np.isfinite(window).all(axis=1)
        if np.any(valid):
            out[valid] = np.std(window[valid], axis=1, ddof=1)
        return out

    rows: list[pd.DataFrame] = []
    for t in range(n_time):
        hist = returns_hist[:, :, t]  # 120 months before OOS month
        vol_hist = volume_hist[:, :, t]
        ret_oos = returns_oos[:, t]

        mom_12_1 = _prod_from_history(hist[:, -12:-1])
        mom_6_1 = _prod_from_history(hist[:, -6:-1])
        mom_3_1 = _prod_from_history(hist[:, -3:-1])
        vol_3m = _std_from_history(hist[:, -3:])
        vol_12m = _std_from_history(hist[:, -12:])
        tsmom_12 = mom_12_1 / (vol_12m * np.sqrt(12.0) + 1e-8)
        tsmom_sign_12 = np.sign(mom_12_1)
        reversal_1m = -hist[:, -1]

        price = np.full_like(hist, np.nan, dtype=np.float64)
        valid_price = np.isfinite(hist).all(axis=1)
        if np.any(valid_price):
            price[valid_price] = np.cumprod(1.0 + hist[valid_price], axis=1)
        price_last = price[:, -1]
        sma_3 = np.nanmean(price[:, -3:], axis=1)
        sma_6 = np.nanmean(price[:, -6:], axis=1)
        sma_12 = np.nanmean(price[:, -12:], axis=1)
        ma_gap_12m = np.log(price_last + 1e-8) - np.log(sma_12 + 1e-8)
        ma_crossover_6_12 = sma_6 / (sma_12 + 1e-8) - 1.0
        ma_signal_3_12 = _binary_ma_signal(sma_3, sma_12)

        vol_regime_3_12 = vol_3m / (vol_12m + 1e-8) - 1.0

        curr_volume = vol_hist[:, -1]
        curr_volume = np.where(np.isfinite(curr_volume), np.maximum(curr_volume, 0.0), np.nan)
        log_volume = np.log1p(curr_volume)
        vol12 = np.where(np.isfinite(vol_hist[:, -12:]), np.maximum(vol_hist[:, -12:], 0.0), np.nan)
        log_volume_ma_12 = np.nanmean(np.log1p(vol12), axis=1)
        abnormal_log_volume = log_volume - log_volume_ma_12

        rows.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(datacube.month_end[t]),
                    "time_index": t,
                    "row_id": np.arange(n_assets, dtype=np.int32),
                    "asset_id": np.round(datacube.permno[:, t]),
                    "ret_1m": ret_oos,
                    "volume_1m": curr_volume,
                    "mom_12_1": mom_12_1,
                    "mom_6_1": mom_6_1,
                    "mom_3_1": mom_3_1,
                    "tsmom_12": tsmom_12,
                    "tsmom_sign_12": tsmom_sign_12,
                    "reversal_1m": reversal_1m,
                    "ma_gap_12m": ma_gap_12m,
                    "ma_crossover_6_12": ma_crossover_6_12,
                    "ma_signal_3_12": ma_signal_3_12,
                    "vol_3m": vol_3m,
                    "vol_12m": vol_12m,
                    "vol_regime_3_12": vol_regime_3_12,
                    "log_volume": log_volume,
                    "log_volume_ma_12": log_volume_ma_12,
                    "abnormal_log_volume": abnormal_log_volume,
                }
            )
        )

    frame = pd.concat(rows, ignore_index=True)
    frame["asset_id"] = pd.to_numeric(frame["asset_id"], errors="coerce")
    frame["has_asset_id"] = frame["asset_id"].notna()
    frame["has_finite_return"] = np.isfinite(frame["ret_1m"])
    frame["cs_mom_12_1"] = _zscore_by_date(frame, "mom_12_1")
    frame["target_ret_1m"] = frame["ret_1m"]
    frame["target_available"] = frame["target_ret_1m"].notna()
    frame["history_count"] = 120
    return frame


def generate_features_from_datacube(datacube: DataCube, feature_cfg: FeatureConfig) -> FeatureResult:
    frame = _compute_direct_cube_features(datacube)
    frame = _winsorize_feature_columns(frame, feature_cfg)
    frame, macro_columns = merge_macro_features(frame, feature_cfg)
    frame = _add_supervisor_macro_interactions(frame)
    feature_columns = (
        get_feature_columns("supervisor_macro_interactions")
        if macro_columns
        else get_feature_columns("baseline_technical")
    )
    frame["feature_ready"] = frame[feature_columns].notna().all(axis=1)
    return FeatureResult(frame=frame, feature_columns=feature_columns, target_column="target_ret_1m")


def generate_features(panel_df: pd.DataFrame, feature_cfg: FeatureConfig) -> FeatureResult:
    required = {"date", "asset_id", "ret_1m", "volume_1m"}
    missing = [name for name in required if name not in panel_df.columns]
    if missing:
        raise ValueError(f"panel_df is missing required columns: {missing}")

    frame = panel_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["asset_id", "date"], kind="stable")
    frame = frame[frame["asset_id"].notna()].copy()

    frame["volume_1m"] = _fill_volume_cross_section(frame["volume_1m"], frame["date"])
    frame["ret_1m"] = frame["ret_1m"].astype(float)

    grouped = frame.groupby("asset_id", observed=True, sort=False)

    frame["price_index"] = grouped["ret_1m"].transform(
        lambda s: (1.0 + s.fillna(0.0)).cumprod()
    )

    frame["mom_12_1"] = grouped["ret_1m"].transform(
        lambda s: _rolling_product(s.shift(1).fillna(0.0), 11)
    )
    frame["mom_6_1"] = grouped["ret_1m"].transform(
        lambda s: _rolling_product(s.shift(1).fillna(0.0), 5)
    )
    frame["cs_mom_12_1"] = _zscore_by_date(frame, "mom_12_1")
    frame["mom_3_1"] = grouped["ret_1m"].transform(
        lambda s: _rolling_product(s.shift(1).fillna(0.0), 2)
    )

    frame["vol_3m"] = grouped["ret_1m"].transform(lambda s: s.rolling(3, min_periods=3).std())
    frame["vol_12m"] = grouped["ret_1m"].transform(lambda s: s.rolling(12, min_periods=12).std())
    frame["tsmom_12"] = frame["mom_12_1"] / (frame["vol_12m"] * np.sqrt(12.0) + 1e-8)
    frame["tsmom_sign_12"] = np.sign(frame["mom_12_1"])

    lag_1 = grouped["ret_1m"].shift(1)
    lag_2 = grouped["ret_1m"].shift(2)
    frame["reversal_1m"] = -lag_1
    frame["reversal_2m"] = -(lag_1 + lag_2) / 2.0

    sma_3 = grouped["price_index"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    sma_6 = grouped["price_index"].transform(lambda s: s.rolling(6, min_periods=6).mean())
    sma_12 = grouped["price_index"].transform(lambda s: s.rolling(12, min_periods=12).mean())
    frame["ma_gap_12m"] = np.log(frame["price_index"] + 1e-8) - np.log(sma_12 + 1e-8)
    frame["ma_crossover_6_12"] = sma_6 / (sma_12 + 1e-8) - 1.0
    frame["ma_signal_3_12"] = _binary_ma_signal(sma_3, sma_12)

    frame["vol_regime_3_12"] = frame["vol_3m"] / (frame["vol_12m"] + 1e-8) - 1.0

    frame["log_volume"] = np.log1p(frame["volume_1m"].clip(lower=0.0))
    frame["log_volume_ma_12"] = grouped["log_volume"].transform(
        lambda s: s.rolling(12, min_periods=12).mean()
    )
    frame["abnormal_log_volume"] = frame["log_volume"] - frame["log_volume_ma_12"]

    frame = _winsorize_feature_columns(frame, feature_cfg)

    frame, macro_columns = merge_macro_features(frame, feature_cfg)
    frame = _add_supervisor_macro_interactions(frame)
    feature_columns = (
        get_feature_columns("supervisor_macro_interactions")
        if macro_columns
        else get_feature_columns("baseline_technical")
    )

    frame = frame.sort_values(["asset_id", "date"], kind="stable")
    grouped = frame.groupby("asset_id", observed=True, sort=False)
    frame["target_ret_1m"] = grouped["ret_1m"].shift(-1)
    frame["target_available"] = frame["target_ret_1m"].notna()
    frame["history_count"] = grouped.cumcount() + 1
    frame["feature_ready"] = (frame["history_count"] >= feature_cfg.min_history) & frame[
        feature_columns
    ].notna().all(axis=1)

    return FeatureResult(frame=frame, feature_columns=feature_columns, target_column="target_ret_1m")
