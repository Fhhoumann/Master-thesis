from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_panel(panel: pd.DataFrame) -> None:
    required = {"date", "asset_id", "ret_1m"}
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")


def build_ew_market(panel: pd.DataFrame) -> pd.DataFrame:
    _validate_panel(panel)
    work = panel[["date", "asset_id", "ret_1m"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["ret_1m"] = pd.to_numeric(work["ret_1m"], errors="coerce")
    work = work.dropna(subset=["date", "asset_id", "ret_1m"])
    out = (
        work.groupby("date", observed=True)["ret_1m"]
        .mean()
        .rename("gross_return")
        .reset_index()
        .sort_values("date")
    )
    out["benchmark_name"] = "ew_market"
    return out[["date", "benchmark_name", "gross_return"]]


def build_umd_momentum(panel: pd.DataFrame, top_q: float = 0.10) -> pd.DataFrame:
    if not (0.0 < top_q < 0.5):
        raise ValueError(f"`top_q` must be in (0, 0.5), got {top_q}")
    _validate_panel(panel)

    work = panel[["date", "asset_id", "ret_1m"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["ret_1m"] = pd.to_numeric(work["ret_1m"], errors="coerce")
    work = work.dropna(subset=["date", "asset_id"]).sort_values(["asset_id", "date"], kind="stable")

    grouped = work.groupby("asset_id", observed=True, sort=False)
    work["mom_12_1"] = grouped["ret_1m"].transform(
        lambda s: (1.0 + s.shift(1).fillna(0.0)).rolling(11, min_periods=11).apply(np.prod, raw=True) - 1.0
    )
    work = work.dropna(subset=["mom_12_1", "ret_1m"]).copy()

    def _date_long_short_return(group: pd.DataFrame) -> float:
        if group.shape[0] < 20:
            return float("nan")
        ranked = group.sort_values("mom_12_1")
        n = max(1, int(np.floor(ranked.shape[0] * top_q)))
        short = ranked.head(n)["ret_1m"].mean()
        long = ranked.tail(n)["ret_1m"].mean()
        return float(long - short)

    series = work.groupby("date", observed=True).apply(_date_long_short_return, include_groups=False)
    out = (
        series.rename("gross_return")
        .reset_index()
        .dropna(subset=["gross_return"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["benchmark_name"] = "umd_12_1_ls"
    return out[["date", "benchmark_name", "gross_return"]]


def build_hml_proxy(panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    # `panel` is accepted to keep interface symmetry with other builders.
    del panel
    required = {"date", "hml"}
    missing = [col for col in required if col not in factors.columns]
    if missing:
        raise ValueError(f"factors is missing required columns: {missing}")

    out = factors[["date", "hml"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["hml"] = pd.to_numeric(out["hml"], errors="coerce")
    out = out.dropna(subset=["date", "hml"]).sort_values("date").reset_index(drop=True)
    out = out.rename(columns={"hml": "gross_return"})
    out["benchmark_name"] = "hml_factor_proxy"
    return out[["date", "benchmark_name", "gross_return"]]

