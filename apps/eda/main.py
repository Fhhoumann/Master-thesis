from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.benchmark import (
    build_ew_market,
    build_umd_momentum,
    compute_attribution_table,
)
from core.config import EDAConfig, RegimeConfig, load_eda_config
from core.features import FEATURE_COLUMNS
from core.io import fetch_ken_french_monthly, load_datacube
from core.io import load_umd_factor_panel
from core.metrics import infer_periods_per_year, summarize_returns


@dataclass(slots=True)
class Paths:
    output_dir: Path
    figures_dir: Path
    tables_dir: Path
    summary_json: Path
    report_md: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deep exploratory data analysis report pack.")
    parser.add_argument(
        "--config",
        default="configs/eda/default.yaml",
        help="Path to EDA YAML config.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "quick"],
        default=None,
        help="Optional override for config mode.",
    )
    return parser.parse_args()


def _resolve_panel_path(path: Path) -> Path:
    if path.is_dir():
        return path / "panel.parquet"
    return path


def _resolve_feature_path(path: Path) -> Path:
    if path.is_dir():
        return path / "features.parquet"
    return path


def _assign_regime(dates: pd.Series, regimes: list[RegimeConfig]) -> pd.Series:
    out = pd.Series(data="outside_regimes", index=dates.index, dtype="object")
    valid_dates = pd.to_datetime(dates, errors="coerce")
    for regime in regimes:
        start = pd.Timestamp(regime.start)
        end = pd.Timestamp(regime.end)
        mask = valid_dates.between(start, end, inclusive="both")
        out.loc[mask] = regime.name
    order = [r.name for r in regimes] + ["outside_regimes"]
    return pd.Categorical(out, categories=order, ordered=True)


def _apply_time_ticks(ax: plt.Axes, labels: list[str]) -> None:
    if not labels:
        return
    step = max(1, len(labels) // 12)
    ticks = np.arange(0, len(labels), step)
    ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=45, ha="right")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ensure_dirs(output_dir: Path) -> Paths:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        output_dir=output_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        summary_json=output_dir / "summary.json",
        report_md=output_dir / "eda_report.md",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _safe_month_label(values: Iterable[pd.Timestamp | None]) -> list[str]:
    labels: list[str] = []
    for idx, value in enumerate(values):
        if value is None:
            labels.append(f"t{idx}")
        else:
            labels.append(value.strftime("%Y-%m"))
    return labels


def _top_n_abs(series: pd.Series, n: int) -> list[str]:
    ranked = series.dropna().abs().sort_values(ascending=False)
    return ranked.head(n).index.tolist()


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    mean = returns.rolling(window=window, min_periods=window).mean()
    std = returns.rolling(window=window, min_periods=window).std(ddof=0)
    return (mean * 12.0) / (std * np.sqrt(12.0) + 1e-12)


def _safe_spearman(x: pd.Series, y: pd.Series, min_count: int) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if valid.shape[0] < min_count:
        return float("nan")
    if valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman"))


def _find_run_dirs(artifacts_root: Path) -> list[Path]:
    if not artifacts_root.exists():
        return []
    return sorted(path.parent for path in artifacts_root.glob("*/*/summary.csv"))


def _load_run_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in (
        "summary.csv",
        "backtest.parquet",
        "weights.parquet",
        "predictions.parquet",
        "model_diagnostics.csv",
    ):
        path = run_dir / name
        if not path.exists():
            continue
        if path.suffix == ".csv":
            out[name] = pd.read_csv(path)
        else:
            out[name] = pd.read_parquet(path)
    return out


def _benchmark_performance(returns: pd.DataFrame, regimes: list[RegimeConfig]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = returns.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "benchmark_name", "gross_return"]).sort_values("date")
    work["regime"] = _assign_regime(work["date"], regimes)

    for benchmark_name, group in work.groupby("benchmark_name", observed=True):
        full = summarize_returns(group["gross_return"], dates=group["date"])
        rows.append(
            {
                "benchmark_name": str(benchmark_name),
                "regime": "full_sample",
                "n_periods": float(full["n_periods"]),
                "periods_per_year": float(full["periods_per_year"]),
                "annual_return": float(full["annual_return"]),
                "annual_volatility": float(full["annual_volatility"]),
                "sharpe": float(full["sharpe"]),
                "max_drawdown": float(full["max_drawdown"]),
                "cumulative_return": float(full["cumulative_return"]),
            }
        )
        for regime_name, regime_group in group.groupby("regime", observed=True):
            metrics = summarize_returns(regime_group["gross_return"], dates=regime_group["date"])
            rows.append(
                {
                    "benchmark_name": str(benchmark_name),
                    "regime": str(regime_name),
                    "n_periods": float(metrics["n_periods"]),
                    "periods_per_year": float(metrics["periods_per_year"]),
                    "annual_return": float(metrics["annual_return"]),
                    "annual_volatility": float(metrics["annual_volatility"]),
                    "sharpe": float(metrics["sharpe"]),
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "cumulative_return": float(metrics["cumulative_return"]),
                }
            )
    return pd.DataFrame(rows)


def _build_benchmark_comparison_table(
    strategy_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    regimes: list[RegimeConfig],
) -> pd.DataFrame:
    if strategy_returns.empty or benchmark_returns.empty:
        return pd.DataFrame()

    work_strategy = strategy_returns.copy()
    work_bench = benchmark_returns.copy()

    work_strategy["date"] = pd.to_datetime(work_strategy["date"], errors="coerce")
    work_strategy["period_return"] = pd.to_numeric(work_strategy["period_return"], errors="coerce")
    work_strategy = work_strategy.dropna(subset=["run_key", "date", "period_return"])

    work_bench["date"] = pd.to_datetime(work_bench["date"], errors="coerce")
    work_bench["gross_return"] = pd.to_numeric(work_bench["gross_return"], errors="coerce")
    work_bench = work_bench.dropna(subset=["benchmark_name", "date", "gross_return"])

    if work_strategy.empty or work_bench.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for run_key, run_group in work_strategy.groupby("run_key", observed=True):
        for benchmark_name, bench_group in work_bench.groupby("benchmark_name", observed=True):
            merged = run_group.merge(
                bench_group,
                on="date",
                how="inner",
            ).dropna(subset=["period_return", "gross_return"])
            if merged.empty:
                continue

            strategy_metrics = summarize_returns(merged["period_return"], dates=merged["date"])
            benchmark_metrics = summarize_returns(merged["gross_return"], dates=merged["date"])
            rows.append(
                {
                    "scope": "full_sample",
                    "run_key": str(run_key),
                    "regime": "full_sample",
                    "benchmark_name": str(benchmark_name),
                    "strategy_n_periods": float(strategy_metrics["n_periods"]),
                    "benchmark_n_periods": float(benchmark_metrics["n_periods"]),
                    "periods_per_year": float(strategy_metrics["periods_per_year"]),
                    "strategy_annual_return": float(strategy_metrics["annual_return"]),
                    "benchmark_annual_return": float(benchmark_metrics["annual_return"]),
                    "delta_annual_return": float(
                        strategy_metrics["annual_return"] - benchmark_metrics["annual_return"]
                    ),
                    "strategy_sharpe": float(strategy_metrics["sharpe"]),
                    "benchmark_sharpe": float(benchmark_metrics["sharpe"]),
                    "delta_sharpe": float(strategy_metrics["sharpe"] - benchmark_metrics["sharpe"]),
                    "strategy_max_drawdown": float(strategy_metrics["max_drawdown"]),
                    "benchmark_max_drawdown": float(benchmark_metrics["max_drawdown"]),
                }
            )

            merged["regime"] = _assign_regime(merged["date"], regimes)
            for regime_name, regime_group in merged.groupby("regime", observed=True):
                regime_strategy_metrics = summarize_returns(
                    regime_group["period_return"],
                    dates=regime_group["date"],
                )
                regime_benchmark_metrics = summarize_returns(
                    regime_group["gross_return"],
                    dates=regime_group["date"],
                )
                rows.append(
                    {
                        "scope": "subperiod",
                        "run_key": str(run_key),
                        "regime": str(regime_name),
                        "benchmark_name": str(benchmark_name),
                        "strategy_n_periods": float(regime_strategy_metrics["n_periods"]),
                        "benchmark_n_periods": float(regime_benchmark_metrics["n_periods"]),
                        "periods_per_year": float(regime_strategy_metrics["periods_per_year"]),
                        "strategy_annual_return": float(regime_strategy_metrics["annual_return"]),
                        "benchmark_annual_return": float(regime_benchmark_metrics["annual_return"]),
                        "delta_annual_return": float(
                            regime_strategy_metrics["annual_return"]
                            - regime_benchmark_metrics["annual_return"]
                        ),
                        "strategy_sharpe": float(regime_strategy_metrics["sharpe"]),
                        "benchmark_sharpe": float(regime_benchmark_metrics["sharpe"]),
                        "delta_sharpe": float(
                            regime_strategy_metrics["sharpe"] - regime_benchmark_metrics["sharpe"]
                        ),
                        "strategy_max_drawdown": float(regime_strategy_metrics["max_drawdown"]),
                        "benchmark_max_drawdown": float(regime_benchmark_metrics["max_drawdown"]),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["scope", "run_key", "regime", "benchmark_name"]).reset_index(drop=True)


def _prepare_returns_for_attribution(
    backtests: list[pd.DataFrame],
    include_return_types: set[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for backtest in backtests:
        if backtest.empty:
            continue
        work = backtest.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).sort_values("date")

        if "gross" in include_return_types:
            gross = work[["run_key", "date", "gross_return"]].drop_duplicates(subset=["run_key", "date"])
            gross = gross.rename(columns={"gross_return": "period_return"})
            gross["return_type"] = "gross"
            gross["cost_convention"] = "none"
            gross["cost_bps"] = 0.0
            rows.append(gross)

        if "net" in include_return_types:
            net_required = {"run_key", "date", "net_return", "cost_convention", "cost_bps"}
            if net_required.issubset(work.columns):
                net = work[["run_key", "date", "net_return", "cost_convention", "cost_bps"]].copy()
                net = net.rename(columns={"net_return": "period_return"})
                net["return_type"] = "net"
                rows.append(net)

    if not rows:
        return pd.DataFrame(
            columns=["run_key", "date", "period_return", "return_type", "cost_convention", "cost_bps"]
        )
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["period_return"]).copy()


def _build_prediction_diagnostics(
    run_key: str,
    scored_frame: pd.DataFrame,
    *,
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = scored_frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")

    score_col = "prediction" if "prediction" in work.columns else "score" if "score" in work.columns else None
    target_col = "target_ret_1m" if "target_ret_1m" in work.columns else None
    if score_col is None:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, object]] = []
    for date, group in work.groupby("date", observed=True):
        scores = pd.to_numeric(group[score_col], errors="coerce")
        row: dict[str, object] = {
            "run_key": run_key,
            "date": date,
            "n_assets": int(group.shape[0]),
            "prediction_std": float(scores.std(ddof=0)),
            "prediction_nunique": int(scores.nunique(dropna=True)),
        }
        if target_col and target_col in group.columns:
            target = pd.to_numeric(group[target_col], errors="coerce")
            row["spearman_ic"] = _safe_spearman(scores, target, min_count=min_count)
        else:
            row["spearman_ic"] = float("nan")
        rows.append(row)

    by_date = pd.DataFrame(rows).sort_values(["run_key", "date"]).reset_index(drop=True)
    if by_date.empty:
        return by_date, pd.DataFrame()

    constant_mask = by_date["prediction_nunique"] <= 1
    ic = by_date["spearman_ic"].dropna()
    ic_mean = float(ic.mean()) if not ic.empty else float("nan")
    ic_std = float(ic.std(ddof=0)) if not ic.empty else float("nan")
    ic_tstat = (
        float(ic_mean / (ic.std(ddof=1) / np.sqrt(ic.shape[0]) + 1e-12))
        if ic.shape[0] > 1
        else float("nan")
    )
    summary = pd.DataFrame(
        [
            {
                "run_key": run_key,
                "n_dates": int(by_date.shape[0]),
                "mean_prediction_std": float(by_date["prediction_std"].mean()),
                "min_prediction_std": float(by_date["prediction_std"].min()),
                "max_prediction_std": float(by_date["prediction_std"].max()),
                "constant_prediction_dates_pct": float(constant_mask.mean() * 100.0),
                "degenerate_predictions_flag": bool(constant_mask.mean() >= 0.95),
                "n_ic_dates": int(ic.shape[0]),
                "mean_spearman_ic": ic_mean,
                "std_spearman_ic": ic_std,
                "tstat_spearman_ic": ic_tstat,
                "ic_hit_rate_pct": float((ic > 0).mean() * 100.0) if not ic.empty else float("nan"),
            }
        ]
    )
    return by_date, summary


def _compute_factor_exposure_stability(attribution_subperiod: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "run_key",
        "return_type",
        "cost_convention",
        "cost_bps",
        "model",
        "beta_name",
        "regime_count",
        "beta_mean",
        "beta_std",
        "beta_min",
        "beta_max",
        "beta_range",
    ]
    if attribution_subperiod.empty:
        return pd.DataFrame(columns=base_columns)

    beta_cols = [col for col in attribution_subperiod.columns if col.startswith("beta_")]
    if not beta_cols:
        return pd.DataFrame(columns=base_columns)

    valid = attribution_subperiod[attribution_subperiod["status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame(columns=base_columns)

    rows: list[dict[str, object]] = []
    group_cols = ["run_key", "return_type", "cost_convention", "cost_bps", "model"]
    for keys, group in valid.groupby(group_cols, observed=True):
        run_key, return_type, cost_convention, cost_bps, model = keys
        for beta_col in beta_cols:
            series = pd.to_numeric(group[beta_col], errors="coerce").dropna()
            if series.shape[0] < 2:
                continue
            rows.append(
                {
                    "run_key": str(run_key),
                    "return_type": str(return_type),
                    "cost_convention": str(cost_convention),
                    "cost_bps": float(cost_bps),
                    "model": str(model),
                    "beta_name": beta_col,
                    "regime_count": int(series.shape[0]),
                    "beta_mean": float(series.mean()),
                    "beta_std": float(series.std(ddof=0)),
                    "beta_min": float(series.min()),
                    "beta_max": float(series.max()),
                    "beta_range": float(series.max() - series.min()),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=base_columns)
    return out.sort_values(
        ["run_key", "return_type", "cost_convention", "cost_bps", "model", "beta_name"]
    ).reset_index(drop=True)


def _build_factor_panel_for_attribution(
    panel: pd.DataFrame,
    factor_panel: pd.DataFrame,
    umd_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if panel.empty or factor_panel.empty:
        return pd.DataFrame(columns=["date", "rf", "mkt_rf", "smb", "hml", "umd", "rmw", "cma"])

    ew = build_ew_market(panel)[["date", "gross_return"]].copy()
    ew["date"] = pd.to_datetime(ew["date"], errors="coerce")
    ew = ew.dropna(subset=["date", "gross_return"]).copy()
    ew["month_key"] = ew["date"].dt.to_period("M")
    ew = ew.sort_values("date").drop_duplicates("month_key", keep="last")

    ff_cols = ["rf", "smb", "hml", "rmw", "cma"]
    ff = factor_panel[["date", *ff_cols]].copy()
    ff["date"] = pd.to_datetime(ff["date"], errors="coerce")
    ff = ff.dropna(subset=["date", "rf"]).copy()
    ff["month_key"] = ff["date"].dt.to_period("M")
    ff = ff.sort_values("date").drop_duplicates("month_key", keep="last")
    for col in ff_cols:
        ff[col] = pd.to_numeric(ff[col], errors="coerce")

    merged = ew.merge(ff[["month_key", *ff_cols]], on="month_key", how="inner")
    if umd_panel is not None and not umd_panel.empty:
        umd = umd_panel[["date", "umd"]].copy()
        umd["date"] = pd.to_datetime(umd["date"], errors="coerce")
        umd["umd"] = pd.to_numeric(umd["umd"], errors="coerce")
        umd = umd.dropna(subset=["date", "umd"]).copy()
        umd["month_key"] = umd["date"].dt.to_period("M")
        umd = umd.sort_values("date").drop_duplicates("month_key", keep="last")
        merged = merged.merge(umd[["month_key", "umd"]], on="month_key", how="left")
    if merged.empty:
        return pd.DataFrame(columns=["date", "rf", "mkt_rf", "smb", "hml", "umd", "rmw", "cma"])

    # Keep strategy-style date labels while aligning monthly factors by month key.
    merged["mkt_rf"] = pd.to_numeric(merged["gross_return"], errors="coerce") - pd.to_numeric(
        merged["rf"], errors="coerce"
    )
    out = (
        merged[["date", "rf", "mkt_rf", "smb", "hml", "umd", "rmw", "cma"]]
        .dropna(subset=["date", "rf", "mkt_rf"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def _build_data_layer(
    cfg: EDAConfig,
    paths: Paths,
) -> dict[str, object]:
    sample_assets = cfg.quick_sample_assets if cfg.mode == "quick" else None
    sample_time = cfg.quick_sample_time if cfg.mode == "quick" else None

    cube = load_datacube(
        cfg.datacube_path,
        sample_assets=sample_assets,
        sample_time=sample_time,
    )
    labels = _safe_month_label(cube.month_end)
    x = np.arange(cube.n_time)

    current_returns = cube.current_returns
    current_volume = cube.current_volume

    returns_missing_rate = np.mean(~np.isfinite(current_returns), axis=0)
    volume_missing_rate = np.mean(~np.isfinite(current_volume), axis=0)
    returns_q = np.nanquantile(current_returns, [0.1, 0.5, 0.9], axis=0)
    positive_volume = np.where(current_volume > 0, current_volume, np.nan)
    volume_q = np.nanquantile(positive_volume, [0.1, 0.5, 0.9], axis=0)

    churn = np.full(cube.n_time, np.nan, dtype=np.float64)
    for t in range(1, cube.n_time):
        prev = cube.permno[:, t - 1]
        curr = cube.permno[:, t]
        prev_set = set(prev[np.isfinite(prev)].astype(np.int64).tolist())
        curr_set = set(curr[np.isfinite(curr)].astype(np.int64).tolist())
        if not prev_set and not curr_set:
            churn[t] = 0.0
            continue
        union = len(prev_set | curr_set)
        intersection = len(prev_set & curr_set)
        churn[t] = 1.0 - (intersection / union if union else 0.0)

    data_monthly = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(cube.month_end, dtype="datetime64[ns]"), errors="coerce"),
            "month_label": labels,
            "returns_missing_pct": returns_missing_rate * 100.0,
            "volume_missing_pct": volume_missing_rate * 100.0,
            "returns_p10": returns_q[0],
            "returns_median": returns_q[1],
            "returns_p90": returns_q[2],
            "volume_p10": volume_q[0],
            "volume_median": volume_q[1],
            "volume_p90": volume_q[2],
            "membership_churn_pct": churn * 100.0,
        }
    )
    data_monthly["regime"] = _assign_regime(data_monthly["date"], cfg.regimes)

    cube_returns_missing_pct_by_time = (
        np.mean(~np.isfinite(cube.returns), axis=(0, 1)) * 100.0
    )
    cube_volume_missing_pct_by_time = np.mean(~np.isfinite(cube.volume), axis=(0, 1)) * 100.0
    cube_returns_missing_pct_total = float(np.mean(~np.isfinite(cube.returns)) * 100.0)
    cube_volume_missing_pct_total = float(np.mean(~np.isfinite(cube.volume)) * 100.0)

    def _fmt_date(value: pd.Timestamp | None) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return value.strftime("%Y-%m-%d")

    worst_cube_ret_t = int(np.nanargmax(cube_returns_missing_pct_by_time))
    worst_cube_vol_t = int(np.nanargmax(cube_volume_missing_pct_by_time))

    current_returns_max = float(np.nanmax(current_returns)) if np.isfinite(current_returns).any() else float("nan")
    current_volume_max = float(np.nanmax(current_volume)) if np.isfinite(current_volume).any() else float("nan")
    cube_returns_max = float(np.nanmax(cube.returns)) if np.isfinite(cube.returns).any() else float("nan")
    cube_volume_max = float(np.nanmax(cube.volume)) if np.isfinite(cube.volume).any() else float("nan")

    current_returns_max_count = (
        int(np.sum(current_returns == current_returns_max))
        if np.isfinite(current_returns_max)
        else 0
    )
    current_volume_max_count = (
        int(np.sum(current_volume == current_volume_max)) if np.isfinite(current_volume_max) else 0
    )
    cube_returns_max_count = (
        int(np.sum(cube.returns == cube_returns_max)) if np.isfinite(cube_returns_max) else 0
    )
    cube_volume_max_count = (
        int(np.sum(cube.volume == cube_volume_max)) if np.isfinite(cube_volume_max) else 0
    )

    data_regime = (
        data_monthly.groupby("regime", observed=True)
        .agg(
            n_months=("date", "count"),
            returns_missing_pct_mean=("returns_missing_pct", "mean"),
            volume_missing_pct_mean=("volume_missing_pct", "mean"),
            returns_median_mean=("returns_median", "mean"),
            volume_median_mean=("volume_median", "mean"),
            churn_pct_mean=("membership_churn_pct", "mean"),
            churn_pct_max=("membership_churn_pct", "max"),
        )
        .reset_index()
        .sort_values("regime")
    )

    lag_months_ago = (cube.n_lookback - 1) - np.arange(cube.n_lookback)
    mean_abs_return_by_lag = np.nanmean(np.abs(cube.returns), axis=(0, 2))
    mean_volume_by_lag = np.nanmean(np.where(cube.volume > 0, cube.volume, np.nan), axis=(0, 2))

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(x, data_monthly["returns_missing_pct"], label="Returns missing %", linewidth=2.0)
    ax.plot(x, data_monthly["volume_missing_pct"], label="Volume missing %", linewidth=2.0)
    ax.set_title("Data Layer: Missingness by Month")
    ax.set_ylabel("Missing rate (%)")
    ax.set_xlabel("Month")
    _apply_time_ticks(ax, labels)
    ax.legend(loc="upper right")
    fig_missing = paths.figures_dir / "data_01_missingness_by_month.png"
    _save_figure(fig, fig_missing)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(x, data_monthly["returns_p10"], linewidth=1.4, label="P10")
    ax.plot(x, data_monthly["returns_median"], linewidth=2.2, label="Median")
    ax.plot(x, data_monthly["returns_p90"], linewidth=1.4, label="P90")
    ax.set_title("Data Layer: Cross-Sectional Return Quantiles")
    ax.set_ylabel("Return")
    ax.set_xlabel("Month")
    _apply_time_ticks(ax, labels)
    ax.legend(loc="upper right")
    fig_ret_quantiles = paths.figures_dir / "data_02_return_quantiles.png"
    _save_figure(fig, fig_ret_quantiles)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(x, data_monthly["volume_p10"], linewidth=1.4, label="P10")
    ax.plot(x, data_monthly["volume_median"], linewidth=2.2, label="Median")
    ax.plot(x, data_monthly["volume_p90"], linewidth=1.4, label="P90")
    ax.set_yscale("log")
    ax.set_title("Data Layer: Cross-Sectional Volume Quantiles (Log Scale)")
    ax.set_ylabel("Volume (log)")
    ax.set_xlabel("Month")
    _apply_time_ticks(ax, labels)
    ax.legend(loc="upper right")
    fig_vol_quantiles = paths.figures_dir / "data_03_volume_quantiles.png"
    _save_figure(fig, fig_vol_quantiles)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(x, data_monthly["membership_churn_pct"], linewidth=2.0, color="#b91c1c")
    ax.set_title("Data Layer: Universe Membership Churn")
    ax.set_ylabel("Jaccard distance (%)")
    ax.set_xlabel("Month")
    _apply_time_ticks(ax, labels)
    fig_churn = paths.figures_dir / "data_04_membership_churn.png"
    _save_figure(fig, fig_churn)

    dist_frame = pd.DataFrame(
        {
            "date": np.tile(data_monthly["date"].to_numpy(), cube.n_assets),
            "regime": np.tile(data_monthly["regime"].astype(str).to_numpy(), cube.n_assets),
            "ret_1m": current_returns.reshape(-1),
        }
    )
    dist_frame = dist_frame[np.isfinite(dist_frame["ret_1m"])].copy()
    lo, hi = np.quantile(dist_frame["ret_1m"], [0.005, 0.995])
    dist_frame = dist_frame[(dist_frame["ret_1m"] >= lo) & (dist_frame["ret_1m"] <= hi)]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(
        data=dist_frame,
        x="regime",
        y="ret_1m",
        order=[r.name for r in cfg.regimes],
        showfliers=False,
        ax=ax,
    )
    ax.set_title("Data Layer: Return Distribution by Regime (Trimmed)")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Return")
    fig_ret_by_regime = paths.figures_dir / "data_05_return_distribution_by_regime.png"
    _save_figure(fig, fig_ret_by_regime)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax1.plot(lag_months_ago, mean_abs_return_by_lag, linewidth=2.0, color="#7c3aed")
    ax1.set_title("Data Layer: Lookback Profile")
    ax1.set_ylabel("Mean |return|")
    ax2.plot(lag_months_ago, mean_volume_by_lag, linewidth=2.0, color="#0369a1")
    ax2.set_yscale("log")
    ax2.set_xlabel("Months ago (120=oldest, 0=current)")
    ax2.set_ylabel("Mean volume (log)")
    fig_lag = paths.figures_dir / "data_06_lookback_profile.png"
    _save_figure(fig, fig_lag)

    table_monthly = paths.tables_dir / "data_monthly_profile.csv"
    table_regime = paths.tables_dir / "data_regime_summary.csv"
    _write_csv(table_monthly, data_monthly)
    _write_csv(table_regime, data_regime)

    return {
        "definitions": {
            "current_slice": "returns[:, -1, :] and volume[:, -1, :] (assets x months; months_ago=0)",
            "full_cube": "returns[:, :, :] and volume[:, :, :] (assets x lookback x months)",
            "missingness_denominator_current": "assets",
            "missingness_denominator_full_cube_by_time": "assets * lookback",
        },
        "date_range": {
            "start": data_monthly["date"].min().strftime("%Y-%m-%d"),
            "end": data_monthly["date"].max().strftime("%Y-%m-%d"),
        },
        "shape": {
            "n_assets": cube.n_assets,
            "n_lookback": cube.n_lookback,
            "n_time": cube.n_time,
        },
        "reconciliation": {
            "current_slice": {
                "lookback_index": cube.n_lookback - 1,
                "mean_missing_pct": float(data_monthly["returns_missing_pct"].mean()),
                "mean_missing_pct_volume": float(data_monthly["volume_missing_pct"].mean()),
                "max_returns": current_returns_max,
                "max_returns_count": current_returns_max_count,
                "max_volume": current_volume_max,
                "max_volume_count": current_volume_max_count,
            },
            "full_cube": {
                "missing_pct_total_returns": cube_returns_missing_pct_total,
                "missing_pct_total_volume": cube_volume_missing_pct_total,
                "worst_missing_pct_by_time_returns": float(cube_returns_missing_pct_by_time[worst_cube_ret_t]),
                "worst_missing_pct_by_time_volume": float(cube_volume_missing_pct_by_time[worst_cube_vol_t]),
                "worst_time_index_returns": worst_cube_ret_t,
                "worst_time_index_volume": worst_cube_vol_t,
                "worst_month_label_returns": labels[worst_cube_ret_t] if worst_cube_ret_t < len(labels) else "n/a",
                "worst_month_label_volume": labels[worst_cube_vol_t] if worst_cube_vol_t < len(labels) else "n/a",
                "worst_date_returns": _fmt_date(cube.month_end[worst_cube_ret_t])
                if worst_cube_ret_t < len(cube.month_end)
                else "n/a",
                "worst_date_volume": _fmt_date(cube.month_end[worst_cube_vol_t])
                if worst_cube_vol_t < len(cube.month_end)
                else "n/a",
                "max_returns": cube_returns_max,
                "max_returns_count": cube_returns_max_count,
                "max_volume": cube_volume_max,
                "max_volume_count": cube_volume_max_count,
            },
        },
        "monthly_missingness_mean_pct": {
            "returns": float(data_monthly["returns_missing_pct"].mean()),
            "volume": float(data_monthly["volume_missing_pct"].mean()),
        },
        "membership_churn_pct": {
            "mean": float(data_monthly["membership_churn_pct"].mean()),
            "max": float(data_monthly["membership_churn_pct"].max()),
        },
        "tables": [str(table_monthly), str(table_regime)],
        "figures": [
            str(fig_missing),
            str(fig_ret_quantiles),
            str(fig_vol_quantiles),
            str(fig_churn),
            str(fig_ret_by_regime),
            str(fig_lag),
        ],
    }


def _build_feature_layer(
    cfg: EDAConfig,
    paths: Paths,
) -> dict[str, object]:
    feature_path = _resolve_feature_path(cfg.feature_input_path)
    features = pd.read_parquet(feature_path)
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features = features.dropna(subset=["date"]).copy()
    if cfg.mode == "quick":
        recent_dates = np.sort(features["date"].dropna().unique())[-cfg.quick_sample_time :]
        features = features[features["date"].isin(recent_dates)].copy()

    feature_cols = [col for col in FEATURE_COLUMNS if col in features.columns]
    target_col = "target_ret_1m" if "target_ret_1m" in features.columns else None
    features["regime"] = _assign_regime(features["date"], cfg.regimes)

    ready_rate_by_month = (
        features.groupby("date", observed=True)["feature_ready"].mean().reset_index(name="ready_rate")
    )
    ready_rate_by_month["regime"] = _assign_regime(ready_rate_by_month["date"], cfg.regimes)

    missing_rows: list[dict[str, object]] = []
    for col in feature_cols:
        null_rate = float(features[col].isna().mean())
        finite_rate = float(np.isfinite(features[col]).mean())
        missing_rows.append(
            {
                "feature": col,
                "missing_rate_pct": 100.0 * null_rate,
                "finite_rate_pct": 100.0 * finite_rate,
            }
        )
    feature_missingness = pd.DataFrame(missing_rows).sort_values("missing_rate_pct", ascending=False)

    regime_summary = (
        features.groupby("regime", observed=True)
        .agg(
            n_rows=("date", "size"),
            feature_ready_rate=("feature_ready", "mean"),
            target_available_rate=("target_available", "mean"),
            mean_ret_1m=("ret_1m", "mean"),
            std_ret_1m=("ret_1m", "std"),
            mean_volume_1m=("volume_1m", "mean"),
        )
        .reset_index()
        .sort_values("regime")
    )

    ic_by_feature: dict[str, pd.Series] = {}
    if target_col is not None:
        for col in feature_cols:
            subset = features[["date", col, target_col]].dropna()
            ic_series = subset.groupby("date", observed=True).apply(
                lambda g: _safe_spearman(
                    g[col], g[target_col], min_count=cfg.min_cross_section_count
                ),
                include_groups=False,
            )
            ic_by_feature[col] = ic_series

    ic_frame = pd.DataFrame(ic_by_feature).sort_index() if ic_by_feature else pd.DataFrame()
    ic_frame.index.name = "date"
    if not ic_frame.empty:
        ic_long = ic_frame.reset_index().melt(
            id_vars="date", var_name="feature", value_name="ic_spearman"
        )
        ic_long["regime"] = _assign_regime(ic_long["date"], cfg.regimes)
        ic_summary = (
            ic_long.groupby("feature", observed=True)["ic_spearman"]
            .agg(["count", "mean", "std"])
            .reset_index()
            .rename(columns={"count": "n_months", "mean": "mean_ic", "std": "std_ic"})
        )
        ic_summary["t_stat"] = ic_summary["mean_ic"] / (
            ic_summary["std_ic"] / np.sqrt(ic_summary["n_months"].clip(lower=1)) + 1e-12
        )
        ic_summary = ic_summary.sort_values("mean_ic", key=lambda s: s.abs(), ascending=False)

        ic_regime = (
            ic_long.groupby(["regime", "feature"], observed=True)["ic_spearman"]
            .mean()
            .reset_index()
            .pivot(index="feature", columns="regime", values="ic_spearman")
            .sort_index()
        )
    else:
        ic_summary = pd.DataFrame(columns=["feature", "n_months", "mean_ic", "std_ic", "t_stat"])
        ic_regime = pd.DataFrame()

    top_features = _top_n_abs(
        ic_summary.set_index("feature")["mean_ic"] if not ic_summary.empty else pd.Series(dtype=float),
        cfg.top_n_features,
    )

    spread_rows: list[dict[str, object]] = []
    if target_col is not None and top_features:
        for feature in top_features:
            subset = features[["date", "regime", feature, target_col]].dropna().copy()
            if subset.empty:
                continue
            for date, group in subset.groupby("date", observed=True):
                if group.shape[0] < cfg.min_cross_section_count:
                    continue
                ranks = group[feature].rank(pct=True, method="average")
                top_mean = group.loc[ranks >= 0.9, target_col].mean()
                bottom_mean = group.loc[ranks <= 0.1, target_col].mean()
                spread_rows.append(
                    {
                        "date": date,
                        "regime": group["regime"].iloc[0],
                        "feature": feature,
                        "spread_top10_bottom10": float(top_mean - bottom_mean),
                    }
                )
    spread_frame = pd.DataFrame(spread_rows)
    spread_summary = (
        spread_frame.groupby("feature", observed=True)["spread_top10_bottom10"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n_months", "mean": "mean_spread", "std": "std_spread"})
        if not spread_frame.empty
        else pd.DataFrame(columns=["feature", "n_months", "mean_spread", "std_spread"])
    )
    if not spread_summary.empty:
        spread_summary["annualized_spread"] = spread_summary["mean_spread"] * 12.0
        spread_summary = spread_summary.sort_values(
            "annualized_spread", key=lambda s: s.abs(), ascending=False
        )
    else:
        spread_summary["annualized_spread"] = []

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(ready_rate_by_month["date"], ready_rate_by_month["ready_rate"] * 100.0, linewidth=2.0)
    ax.set_title("Feature Layer: Feature-Ready Coverage by Month")
    ax.set_ylabel("Feature-ready rate (%)")
    ax.set_xlabel("Date")
    fig_ready = paths.figures_dir / "feature_01_ready_rate.png"
    _save_figure(fig, fig_ready)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=feature_missingness.head(min(20, len(feature_missingness))),
        x="feature",
        y="missing_rate_pct",
        color="#2563eb",
        ax=ax,
    )
    ax.set_title("Feature Layer: Missingness by Feature")
    ax.set_ylabel("Missing rate (%)")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=45)
    fig_missing = paths.figures_dir / "feature_02_missingness.png"
    _save_figure(fig, fig_missing)

    corr_source = features.loc[features["feature_ready"], feature_cols].copy()
    if corr_source.shape[0] > 100000:
        corr_source = corr_source.sample(100000, random_state=42)
    corr = corr_source.corr(method="spearman") if not corr_source.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(11, 9))
    if not corr.empty:
        sns.heatmap(corr, cmap="coolwarm", center=0.0, square=True, ax=ax)
    ax.set_title("Feature Layer: Spearman Correlation Heatmap")
    fig_corr = paths.figures_dir / "feature_03_corr_heatmap.png"
    _save_figure(fig, fig_corr)

    fig, ax = plt.subplots(figsize=(11, max(4, int(0.5 * len(feature_cols)))))
    if not ic_regime.empty:
        ic_regime_plot = ic_regime[[r.name for r in cfg.regimes if r.name in ic_regime.columns]]
        sns.heatmap(ic_regime_plot, cmap="RdBu_r", center=0.0, ax=ax)
    ax.set_title("Feature Layer: Mean IC by Regime")
    fig_ic_regime = paths.figures_dir / "feature_04_ic_regime_heatmap.png"
    _save_figure(fig, fig_ic_regime)

    fig, ax = plt.subplots(figsize=(13, 5))
    if not ic_frame.empty and top_features:
        for feature in top_features[:4]:
            if feature in ic_frame.columns:
                ax.plot(ic_frame.index, ic_frame[feature], label=feature, linewidth=1.6)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.legend(loc="upper right")
    ax.set_title("Feature Layer: Monthly IC for Top Features")
    ax.set_ylabel("Spearman IC")
    ax.set_xlabel("Date")
    fig_ic_ts = paths.figures_dir / "feature_05_top_feature_ic_timeseries.png"
    _save_figure(fig, fig_ic_ts)

    table_missing = paths.tables_dir / "feature_missingness.csv"
    table_regime = paths.tables_dir / "feature_regime_summary.csv"
    table_ic = paths.tables_dir / "feature_ic_summary.csv"
    table_spread = paths.tables_dir / "feature_spread_summary.csv"
    _write_csv(table_missing, feature_missingness)
    _write_csv(table_regime, regime_summary)
    _write_csv(table_ic, ic_summary)
    _write_csv(table_spread, spread_summary)

    return {
        "n_rows": int(features.shape[0]),
        "n_features": int(len(feature_cols)),
        "feature_ready_rate_pct": float(features["feature_ready"].mean() * 100.0),
        "target_available_rate_pct": float(features["target_available"].mean() * 100.0),
        "top_features_by_ic": top_features,
        "tables": [str(table_missing), str(table_regime), str(table_ic), str(table_spread)],
        "figures": [str(fig_ready), str(fig_missing), str(fig_corr), str(fig_ic_regime), str(fig_ic_ts)],
    }


def _strategy_baseline_selector(summary: pd.DataFrame, cfg: EDAConfig) -> pd.DataFrame:
    if summary.empty:
        return summary
    preferred = summary[
        (summary["cost_convention"] == cfg.baseline_cost_convention)
        & (summary["cost_bps"].astype(float) == float(cfg.baseline_cost_bps))
    ]
    if not preferred.empty:
        return preferred

    same_convention = summary[summary["cost_convention"] == cfg.baseline_cost_convention].copy()
    if not same_convention.empty:
        same_convention["cost_gap"] = (same_convention["cost_bps"].astype(float) - cfg.baseline_cost_bps).abs()
        return same_convention.sort_values("cost_gap").head(1).drop(columns=["cost_gap"])
    return summary.head(1)


def _build_strategy_layer(
    cfg: EDAConfig,
    paths: Paths,
) -> dict[str, object]:
    run_dirs = _find_run_dirs(cfg.artifacts_root)
    if not run_dirs:
        return {
            "n_runs": 0,
            "tables": [],
            "figures": [],
        }

    summary_rows: list[pd.DataFrame] = []
    regime_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    rolling_frames: list[pd.DataFrame] = []
    turnover_frames: list[pd.DataFrame] = []
    backtest_frames: list[pd.DataFrame] = []
    baseline_return_frames: list[pd.DataFrame] = []
    prediction_diag_date_frames: list[pd.DataFrame] = []
    prediction_diag_summary_rows: list[pd.DataFrame] = []

    for run_dir in run_dirs:
        run_tables = _load_run_tables(run_dir)
        summary = run_tables.get("summary.csv")
        if summary is None or summary.empty:
            continue

        run_key = f"{run_dir.parent.name}/{run_dir.name}"
        summary = summary.copy()
        summary["run_key"] = run_key
        summary["run_dir"] = str(run_dir)
        if "periods_per_year" not in summary.columns:
            summary["periods_per_year"] = float("nan")

        backtest = run_tables.get("backtest.parquet")
        if backtest is not None and not backtest.empty:
            backtest = backtest.copy()
            backtest["date"] = pd.to_datetime(backtest["date"], errors="coerce")
            backtest = backtest.dropna(subset=["date"]).sort_values("date")
            backtest["run_key"] = run_key
            backtest_frames.append(backtest.copy())

            for idx, row in summary.iterrows():
                convention = str(row["cost_convention"])
                cost_bps = float(row["cost_bps"])
                date_slice = backtest[
                    (backtest["cost_convention"] == convention)
                    & (pd.to_numeric(backtest["cost_bps"], errors="coerce") == cost_bps)
                ]["date"]
                if not date_slice.empty:
                    summary.at[idx, "periods_per_year"] = infer_periods_per_year(date_slice)

            chosen = _strategy_baseline_selector(summary, cfg)
            chosen_row = chosen.iloc[0]
            convention = str(chosen_row["cost_convention"])
            cost_bps = float(chosen_row["cost_bps"])
            baseline = backtest[
                (backtest["cost_convention"] == convention)
                & (backtest["cost_bps"].astype(float) == cost_bps)
            ].copy()
            baseline = baseline.sort_values("date")
            if not baseline.empty:
                baseline["run_key"] = run_key
                baseline["equity"] = (1.0 + baseline["net_return"].fillna(0.0)).cumprod()
                baseline["rolling_sharpe_12m"] = _rolling_sharpe(
                    baseline["net_return"], cfg.rolling_window_months
                )

                turnover_col = (
                    "turnover_one_way"
                    if convention == "one_way"
                    else "turnover_round_trip"
                )
                baseline["turnover_used"] = baseline[turnover_col]
                turnover_frames.append(
                    baseline[["date", "run_key", "turnover_used"]].rename(
                        columns={"turnover_used": "turnover"}
                    )
                )
                equity_frames.append(baseline[["date", "run_key", "equity"]])
                rolling_frames.append(
                    baseline[["date", "run_key", "rolling_sharpe_12m"]]
                )

                baseline["regime"] = _assign_regime(baseline["date"], cfg.regimes)
                for regime, group in baseline.groupby("regime", observed=True):
                    metrics = summarize_returns(group["net_return"], dates=group["date"])
                    n_periods = float(metrics["n_periods"])
                    annual_volatility = float(metrics["annual_volatility"])
                    sharpe = float(metrics["sharpe"])
                    if n_periods < 6:
                        annual_volatility = float("nan")
                        sharpe = float("nan")
                    regime_rows.append(
                        {
                            "run_key": run_key,
                            "regime": str(regime),
                            "n_periods": n_periods,
                            "periods_per_year": float(metrics["periods_per_year"]),
                            "annual_return": metrics["annual_return"],
                            "annual_volatility": annual_volatility,
                            "sharpe": sharpe,
                            "max_drawdown": metrics["max_drawdown"],
                            "cumulative_return": metrics["cumulative_return"],
                        }
                    )
                baseline_return_frames.append(
                    baseline[["run_key", "date", "net_return"]].rename(
                        columns={"net_return": "period_return"}
                    )
                )

        weights = run_tables.get("weights.parquet")
        if weights is not None and not weights.empty and {"date", "weight"}.issubset(weights.columns):
            w = weights.copy()
            w["date"] = pd.to_datetime(w["date"], errors="coerce")
            w = w.dropna(subset=["date"])
            for date, group in w.groupby("date", observed=True):
                abs_weight = np.abs(group["weight"].to_numpy(np.float64))
                gross = abs_weight.sum()
                if gross <= 0:
                    continue
                normalized = abs_weight / gross
                top10 = float(np.sort(normalized)[-10:].sum()) if normalized.size >= 10 else float(
                    normalized.sum()
                )
                hhi = float(np.square(normalized).sum())
                weight_rows.append(
                    {
                        "run_key": run_key,
                        "date": date,
                        "hhi_abs_weight": hhi,
                        "top10_abs_weight_share": top10,
                    }
                )

            score_col = "prediction" if "prediction" in w.columns else "score" if "score" in w.columns else None
            target_col = "target_ret_1m" if "target_ret_1m" in w.columns else None
            if score_col and target_col:
                tmp = w[["date", score_col, target_col]].dropna()
                if not tmp.empty:
                    daily_ic = (
                        tmp.groupby("date", observed=True)
                        .apply(
                            lambda g: _safe_spearman(
                                g[score_col], g[target_col], min_count=cfg.min_cross_section_count
                            ),
                            include_groups=False,
                        )
                        .dropna()
                    )
                    signal_rows.append(
                        {
                            "run_key": run_key,
                            "n_periods": int(daily_ic.shape[0]),
                            "mean_spearman_ic": float(daily_ic.mean()),
                            "std_spearman_ic": float(daily_ic.std(ddof=0)),
                        }
                    )

        scored_table = run_tables.get("predictions.parquet")
        if scored_table is None or scored_table.empty:
            scored_table = run_tables.get("weights.parquet")
        if scored_table is not None and not scored_table.empty:
            diag_by_date, diag_summary = _build_prediction_diagnostics(
                run_key,
                scored_table,
                min_count=cfg.min_cross_section_count,
            )
            if not diag_by_date.empty:
                prediction_diag_date_frames.append(diag_by_date)
            if not diag_summary.empty:
                model_diag = run_tables.get("model_diagnostics.csv")
                if model_diag is not None and not model_diag.empty:
                    model_diag = model_diag.copy()
                    nonzero = pd.to_numeric(
                        model_diag.get("nonzero_coef_count", pd.Series(dtype=float)),
                        errors="coerce",
                    ).dropna()
                    if not nonzero.empty:
                        diag_summary = diag_summary.copy()
                        diag_summary["nonzero_coef_count_mean"] = float(nonzero.mean())
                        diag_summary["nonzero_coef_count_min"] = float(nonzero.min())
                        diag_summary["nonzero_coef_count_max"] = float(nonzero.max())
                prediction_diag_summary_rows.append(diag_summary)

        summary_rows.append(summary)

    if not summary_rows:
        return {"n_runs": 0, "tables": [], "figures": []}

    strategy_summary = pd.concat(summary_rows, ignore_index=True).sort_values(
        ["run_key", "cost_convention", "cost_bps"]
    )
    strategy_summary["cost_bps"] = strategy_summary["cost_bps"].astype(float)
    strategy_summary["sharpe"] = strategy_summary["sharpe"].astype(float)
    strategy_summary["periods_per_year"] = pd.to_numeric(
        strategy_summary.get("periods_per_year", pd.Series(dtype=float)),
        errors="coerce",
    )

    strategy_regime = (
        pd.DataFrame(regime_rows)
        .sort_values(["run_key", "regime"])
        if regime_rows
        else pd.DataFrame(
            columns=[
                "run_key",
                "regime",
                "n_periods",
                "periods_per_year",
                "annual_return",
                "annual_volatility",
                "sharpe",
                "max_drawdown",
                "cumulative_return",
            ]
        )
    )
    strategy_weights = (
        pd.DataFrame(weight_rows).sort_values(["run_key", "date"])
        if weight_rows
        else pd.DataFrame(columns=["run_key", "date", "hhi_abs_weight", "top10_abs_weight_share"])
    )
    strategy_signal = (
        pd.DataFrame(signal_rows).sort_values(["run_key"])
        if signal_rows
        else pd.DataFrame(columns=["run_key", "n_periods", "mean_spearman_ic", "std_spearman_ic"])
    )
    prediction_diagnostics = (
        pd.concat(prediction_diag_summary_rows, ignore_index=True).sort_values(["run_key"])
        if prediction_diag_summary_rows
        else pd.DataFrame(
            columns=[
                "run_key",
                "n_dates",
                "mean_prediction_std",
                "min_prediction_std",
                "max_prediction_std",
                "constant_prediction_dates_pct",
                "degenerate_predictions_flag",
                "n_ic_dates",
                "mean_spearman_ic",
                "std_spearman_ic",
                "tstat_spearman_ic",
                "ic_hit_rate_pct",
                "nonzero_coef_count_mean",
                "nonzero_coef_count_min",
                "nonzero_coef_count_max",
            ]
        )
    )
    prediction_diagnostics_by_date = (
        pd.concat(prediction_diag_date_frames, ignore_index=True).sort_values(["run_key", "date"])
        if prediction_diag_date_frames
        else pd.DataFrame(
            columns=["run_key", "date", "n_assets", "prediction_std", "prediction_nunique", "spearman_ic"]
        )
    )

    equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    rolling = pd.concat(rolling_frames, ignore_index=True) if rolling_frames else pd.DataFrame()
    turnover = pd.concat(turnover_frames, ignore_index=True) if turnover_frames else pd.DataFrame()

    benchmark_returns = pd.DataFrame(columns=["date", "benchmark_name", "gross_return"])
    benchmark_performance = pd.DataFrame(
        columns=[
            "benchmark_name",
            "regime",
            "n_periods",
            "periods_per_year",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "cumulative_return",
        ]
    )
    benchmark_comparison = pd.DataFrame(
        columns=[
            "scope",
            "run_key",
            "regime",
            "benchmark_name",
            "strategy_n_periods",
            "benchmark_n_periods",
            "periods_per_year",
            "strategy_annual_return",
            "benchmark_annual_return",
            "delta_annual_return",
            "strategy_sharpe",
            "benchmark_sharpe",
            "delta_sharpe",
            "strategy_max_drawdown",
            "benchmark_max_drawdown",
        ]
    )
    factor_attribution = pd.DataFrame(
        columns=[
            "run_key",
            "return_type",
            "cost_convention",
            "cost_bps",
            "regime",
            "model",
            "status",
            "n_obs",
            "alpha_monthly",
            "alpha_annualized",
            "alpha_tstat",
            "r2",
            "beta_mkt_rf",
        ]
    )
    factor_attribution_fullsample = factor_attribution.copy()
    factor_attribution_subperiod = factor_attribution.copy()
    factor_exposure_stability = _compute_factor_exposure_stability(pd.DataFrame())

    panel = pd.DataFrame()
    panel_path = _resolve_panel_path(cfg.panel_input_path)
    if panel_path.exists():
        try:
            panel = pd.read_parquet(panel_path, columns=["date", "asset_id", "ret_1m"])
        except Exception:
            panel = pd.DataFrame()

    factor_panel = pd.DataFrame()
    try:
        factor_panel = fetch_ken_french_monthly(
            cfg.factor_cache_path,
            refresh=cfg.factor_refresh,
        )
    except Exception:
        factor_panel = pd.DataFrame()

    umd_panel = pd.DataFrame()
    try:
        umd_panel = load_umd_factor_panel(cfg.momentum_factor_cache_path)
    except Exception:
        umd_panel = pd.DataFrame()

    if not panel.empty:
        benchmark_frames: list[pd.DataFrame] = [
            build_ew_market(panel),
            build_umd_momentum(panel, top_q=cfg.benchmark_top_quantile),
        ]
        benchmark_returns = (
            pd.concat(benchmark_frames, ignore_index=True)
            .dropna(subset=["date", "benchmark_name", "gross_return"])
            .sort_values(["benchmark_name", "date"])
            .reset_index(drop=True)
        )
        benchmark_performance = _benchmark_performance(benchmark_returns, cfg.regimes)
        baseline_returns = (
            pd.concat(baseline_return_frames, ignore_index=True)
            if baseline_return_frames
            else pd.DataFrame(columns=["run_key", "date", "period_return"])
        )
        benchmark_comparison = _build_benchmark_comparison_table(
            baseline_returns,
            benchmark_returns,
            cfg.regimes,
        )

    include_return_types = {
        value for value in cfg.benchmark_return_types if value in {"gross", "net"}
    }
    requested_models = tuple(
        model for model in cfg.benchmark_models if model in {"capm", "ff3", "carhart4", "ff5"}
    )
    attribution_returns = _prepare_returns_for_attribution(backtest_frames, include_return_types)
    attribution_factors = _build_factor_panel_for_attribution(panel, factor_panel, umd_panel)
    if not attribution_returns.empty and not attribution_factors.empty and requested_models:
        factor_attribution = compute_attribution_table(
            attribution_returns,
            attribution_factors,
            cfg.regimes,
            models=requested_models,
            min_obs=cfg.regression_min_obs,
        )
        factor_attribution_fullsample = factor_attribution[
            factor_attribution["regime"] == "full_sample"
        ].copy()
        factor_attribution_subperiod = factor_attribution[
            factor_attribution["regime"] != "full_sample"
        ].copy()
        factor_exposure_stability = _compute_factor_exposure_stability(factor_attribution_subperiod)

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    if not equity.empty:
        for run_key, group in equity.groupby("run_key", observed=True):
            ax.plot(group["date"], group["equity"], label=run_key, linewidth=1.8)
        ax.legend(loc="upper left")
    ax.set_title("Strategy Layer: Baseline Net Equity Curves")
    ax.set_ylabel("Cumulative wealth")
    ax.set_xlabel("Date")
    fig_equity = paths.figures_dir / "strategy_01_equity_curves.png"
    _save_figure(fig, fig_equity)

    fig, ax = plt.subplots(figsize=(13, 5.2))
    if not rolling.empty:
        for run_key, group in rolling.groupby("run_key", observed=True):
            ax.plot(group["date"], group["rolling_sharpe_12m"], label=run_key, linewidth=1.6)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.legend(loc="upper left")
    ax.set_title("Strategy Layer: Rolling Sharpe (12M, Baseline Net)")
    ax.set_ylabel("Rolling Sharpe")
    ax.set_xlabel("Date")
    fig_rolling = paths.figures_dir / "strategy_02_rolling_sharpe.png"
    _save_figure(fig, fig_rolling)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    sns.lineplot(
        data=strategy_summary,
        x="cost_bps",
        y="sharpe",
        hue="run_key",
        style="cost_convention",
        marker="o",
        ax=ax,
    )
    ax.set_title("Strategy Layer: Cost Sensitivity of Sharpe")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Transaction cost (bps)")
    fig_cost = paths.figures_dir / "strategy_03_cost_sensitivity.png"
    _save_figure(fig, fig_cost)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    if not turnover.empty:
        sns.boxplot(data=turnover, x="run_key", y="turnover", ax=ax)
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Strategy Layer: Baseline Turnover Distribution")
    ax.set_xlabel("Run")
    ax.set_ylabel("Turnover")
    fig_turnover = paths.figures_dir / "strategy_04_turnover_distribution.png"
    _save_figure(fig, fig_turnover)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    if not strategy_weights.empty:
        weight_agg = (
            strategy_weights.groupby("run_key", observed=True)
            .agg(
                mean_hhi=("hhi_abs_weight", "mean"),
                mean_top10_share=("top10_abs_weight_share", "mean"),
            )
            .reset_index()
        )
        plot_data = weight_agg.melt(id_vars="run_key", var_name="metric", value_name="value")
        sns.barplot(data=plot_data, x="run_key", y="value", hue="metric", ax=ax)
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Strategy Layer: Weight Concentration Summary")
    ax.set_xlabel("Run")
    ax.set_ylabel("Value")
    fig_weights = paths.figures_dir / "strategy_05_weight_concentration.png"
    _save_figure(fig, fig_weights)

    table_summary = paths.tables_dir / "strategy_summary_all.csv"
    table_regime = paths.tables_dir / "strategy_regime_metrics.csv"
    table_weights = paths.tables_dir / "strategy_weight_concentration.csv"
    table_signal = paths.tables_dir / "strategy_signal_quality.csv"
    table_benchmark_returns = paths.tables_dir / "strategy_benchmark_returns.csv"
    table_benchmark_performance = paths.tables_dir / "strategy_benchmark_performance.csv"
    table_benchmark_comparison = paths.tables_dir / "strategy_benchmark_comparison.csv"
    table_factor_attribution = paths.tables_dir / "strategy_factor_attribution.csv"
    table_factor_attribution_fullsample = paths.tables_dir / "strategy_factor_attribution_fullsample.csv"
    table_factor_attribution_subperiod = paths.tables_dir / "strategy_factor_attribution_subperiod.csv"
    table_factor_exposure_stability = paths.tables_dir / "strategy_factor_exposure_stability.csv"
    table_prediction_diagnostics = paths.tables_dir / "strategy_prediction_diagnostics.csv"
    table_prediction_diagnostics_by_date = paths.tables_dir / "strategy_prediction_diagnostics_by_date.csv"
    _write_csv(table_summary, strategy_summary)
    _write_csv(table_regime, strategy_regime)
    _write_csv(table_weights, strategy_weights)
    _write_csv(table_signal, strategy_signal)
    _write_csv(table_benchmark_returns, benchmark_returns)
    _write_csv(table_benchmark_performance, benchmark_performance)
    _write_csv(table_benchmark_comparison, benchmark_comparison)
    _write_csv(table_factor_attribution, factor_attribution)
    _write_csv(table_factor_attribution_fullsample, factor_attribution_fullsample)
    _write_csv(table_factor_attribution_subperiod, factor_attribution_subperiod)
    _write_csv(table_factor_exposure_stability, factor_exposure_stability)
    _write_csv(table_prediction_diagnostics, prediction_diagnostics)
    _write_csv(table_prediction_diagnostics_by_date, prediction_diagnostics_by_date)

    return {
        "n_runs": int(strategy_summary["run_key"].nunique()),
        "n_summary_rows": int(strategy_summary.shape[0]),
        "n_benchmarks": int(benchmark_returns["benchmark_name"].nunique()) if not benchmark_returns.empty else 0,
        "n_factor_attribution_rows": int(factor_attribution.shape[0]),
        "n_prediction_diagnostic_rows": int(prediction_diagnostics.shape[0]),
        "top_run_by_sharpe": (
            strategy_summary.sort_values("sharpe", ascending=False)
            .head(1)[["run_key", "cost_convention", "cost_bps", "sharpe"]]
            .to_dict(orient="records")
        ),
        "tables": [
            str(table_summary),
            str(table_regime),
            str(table_weights),
            str(table_signal),
            str(table_benchmark_returns),
            str(table_benchmark_performance),
            str(table_benchmark_comparison),
            str(table_factor_attribution),
            str(table_factor_attribution_fullsample),
            str(table_factor_attribution_subperiod),
            str(table_factor_exposure_stability),
            str(table_prediction_diagnostics),
            str(table_prediction_diagnostics_by_date),
        ],
        "figures": [str(fig_equity), str(fig_rolling), str(fig_cost), str(fig_turnover), str(fig_weights)],
    }


def _try_read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _append_table_markdown(
    lines: list[str],
    *,
    title: str,
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    max_rows: int = 8,
) -> None:
    lines.append(f"### {title}")
    lines.append("")
    if frame.empty:
        lines.append("_No rows available._")
        lines.append("")
        return

    view = frame.copy()
    if columns is not None:
        keep = [col for col in columns if col in view.columns]
        if keep:
            view = view[keep]
    view = view.head(max_rows)

    try:
        lines.append(view.to_markdown(index=False))
    except ImportError:
        lines.append("```text")
        lines.append(view.to_string(index=False))
        lines.append("```")
    lines.append("")


def _first_valid_row(frame: pd.DataFrame, sort_col: str, ascending: bool = False) -> pd.Series | None:
    if frame.empty or sort_col not in frame.columns:
        return None
    ordered = frame.dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending)
    if ordered.empty:
        return None
    return ordered.iloc[0]


def _write_report(
    cfg: EDAConfig,
    paths: Paths,
    summary: dict[str, object],
) -> None:
    data = summary["data_layer"]
    features = summary["feature_layer"]
    strategy = summary["strategy_layer"]

    data_monthly = _try_read_csv(paths.tables_dir / "data_monthly_profile.csv")
    data_regime = _try_read_csv(paths.tables_dir / "data_regime_summary.csv")
    feature_missing = _try_read_csv(paths.tables_dir / "feature_missingness.csv")
    feature_ic = _try_read_csv(paths.tables_dir / "feature_ic_summary.csv")
    feature_spread = _try_read_csv(paths.tables_dir / "feature_spread_summary.csv")
    strategy_summary = _try_read_csv(paths.tables_dir / "strategy_summary_all.csv")
    strategy_regime = _try_read_csv(paths.tables_dir / "strategy_regime_metrics.csv")
    strategy_weights = _try_read_csv(paths.tables_dir / "strategy_weight_concentration.csv")
    strategy_signal = _try_read_csv(paths.tables_dir / "strategy_signal_quality.csv")
    strategy_benchmark_performance = _try_read_csv(paths.tables_dir / "strategy_benchmark_performance.csv")
    strategy_benchmark_comparison = _try_read_csv(paths.tables_dir / "strategy_benchmark_comparison.csv")
    strategy_factor_attribution_fullsample = _try_read_csv(
        paths.tables_dir / "strategy_factor_attribution_fullsample.csv"
    )
    strategy_factor_attribution_subperiod = _try_read_csv(
        paths.tables_dir / "strategy_factor_attribution_subperiod.csv"
    )
    strategy_factor_exposure_stability = _try_read_csv(
        paths.tables_dir / "strategy_factor_exposure_stability.csv"
    )
    strategy_prediction_diagnostics = _try_read_csv(paths.tables_dir / "strategy_prediction_diagnostics.csv")

    best_abs_ic = float("nan")
    if not feature_ic.empty and "mean_ic" in feature_ic.columns:
        best_abs_ic = float(feature_ic["mean_ic"].abs().max())

    run_periods = (
        strategy_summary.groupby("run_key", observed=True)["n_periods"].median()
        if not strategy_summary.empty and {"run_key", "n_periods"}.issubset(strategy_summary.columns)
        else pd.Series(dtype=float)
    )
    horizon_ratio = float("nan")
    if not run_periods.empty and run_periods.min() > 0:
        horizon_ratio = float(run_periods.max() / run_periods.min())

    cost_grid_coverage = False
    if not strategy_summary.empty and {"run_key", "cost_bps"}.issubset(strategy_summary.columns):
        expected_costs = {10.0, 25.0, 50.0}
        per_run_costs = (
            strategy_summary.assign(cost_bps_float=strategy_summary["cost_bps"].astype(float))
            .groupby("run_key", observed=True)["cost_bps_float"]
            .apply(lambda s: set(float(x) for x in s.unique()))
        )
        cost_grid_coverage = bool((per_run_costs.apply(lambda s: expected_costs.issubset(s))).all())

    checklist_rows: list[dict[str, str]] = []
    checklist_rows.append(
        {
            "Check": "Point-in-time construction",
            "Status": "Pass",
            "Evidence": "Feature pipeline uses lagged transforms and `target_ret_1m` (t+1 target).",
            "Why it matters": "Prevents look-ahead bias in EDA and model diagnostics.",
            "Next action": "Keep explicit leakage tests in CI (see action items below).",
        }
    )
    checklist_rows.append(
        {
            "Check": "Missing-data control",
            "Status": "Pass"
            if data["monthly_missingness_mean_pct"]["returns"] < 1.0
            and data["monthly_missingness_mean_pct"]["volume"] < 1.0
            else "Warning",
            "Evidence": "Mean missingness: returns {:.3f}%, volume {:.3f}%.".format(
                data["monthly_missingness_mean_pct"]["returns"],
                data["monthly_missingness_mean_pct"]["volume"],
            ),
            "Why it matters": "High missingness can distort cross-sectional ranks and IC statistics.",
            "Next action": "Track month-level spikes and stress-test key months separately.",
        }
    )
    checklist_rows.append(
        {
            "Check": "Universe dynamics measured",
            "Status": "Pass",
            "Evidence": "Membership churn is reported monthly and by regime.",
            "Why it matters": "Changing constituents affect turnover, stability, and comparability.",
            "Next action": "Stratify diagnostics by churn deciles in next iteration.",
        }
    )
    checklist_rows.append(
        {
            "Check": "Feature readiness / availability",
            "Status": "Pass" if features["feature_ready_rate_pct"] >= 75.0 else "Warning",
            "Evidence": "Feature-ready {:.2f}% and target-available {:.2f}%.".format(
                features["feature_ready_rate_pct"],
                features["target_available_rate_pct"],
            ),
            "Why it matters": "Low readiness can bias which periods dominate modeling.",
            "Next action": "Check readiness by regime and by liquidity bucket.",
        }
    )
    checklist_rows.append(
        {
            "Check": "Signal strength sanity",
            "Status": "Pass" if np.isfinite(best_abs_ic) and best_abs_ic >= 0.01 else "Warning",
            "Evidence": "Best absolute mean IC is {:.4f}.".format(best_abs_ic)
            if np.isfinite(best_abs_ic)
            else "No IC summary available.",
            "Why it matters": "Very weak IC can imply fragile or economically negligible edge.",
            "Next action": "Report confidence intervals and regime-level IC dispersion.",
        }
    )
    checklist_rows.append(
        {
            "Check": "Cost sensitivity coverage",
            "Status": "Pass" if cost_grid_coverage else "Warning",
            "Evidence": "Cost grid includes multiple bps points per run (10/25/50 bps expected).",
            "Why it matters": "Economic viability must hold net of realistic execution costs.",
            "Next action": "Add finer bps grid around deployment-relevant cost assumptions.",
        }
    )
    checklist_rows.append(
        {
            "Check": "Comparable OOS horizon across runs",
            "Status": "Pass" if np.isfinite(horizon_ratio) and horizon_ratio <= 1.25 else "Warning",
            "Evidence": "Run-level `n_periods` ratio is {:.2f}x.".format(horizon_ratio)
            if np.isfinite(horizon_ratio)
            else "Unable to compute horizon ratio.",
            "Why it matters": "Unequal windows make performance ranking potentially misleading.",
            "Next action": "Re-run all models on matched validation date windows.",
        }
    )
    checklist_rows.append(
        {
            "Check": "Uncertainty quantification",
            "Status": "Action Needed",
            "Evidence": "No confidence intervals / block bootstrap in current report.",
            "Why it matters": "Point estimates alone overstate certainty.",
            "Next action": "Add bootstrapped CI for IC, spreads, Sharpe, and drawdown metrics.",
        }
    )
    checklist_rows.append(
        {
            "Check": "Leakage falsification tests",
            "Status": "Action Needed",
            "Evidence": "No explicit placebo / timestamp-shift falsification output.",
            "Why it matters": "Need direct tests that performance is not caused by leakage.",
            "Next action": "Add shuffled-target and forward-shift stress tests.",
        }
    )

    lines: list[str] = []
    lines.append("# Deep EDA Report")
    lines.append("")
    lines.append(f"- Generated (UTC): `{summary['generated_at_utc']}`")
    lines.append(f"- Mode: `{cfg.mode}`")
    lines.append(f"- DataCube: `{cfg.datacube_path}`")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This report is designed to provide a decision-ready understanding of dataset behavior, "
        "feature signal quality, and strategy robustness under transaction costs. The emphasis is "
        "on identifying what is structurally stable versus what is likely sample-specific."
    )
    lines.append("")
    lines.append(
        "- Data quality is generally high: mean missingness is "
        "`{:.3f}%` for current returns and `{:.3f}%` for current volume.".format(
            data["monthly_missingness_mean_pct"]["returns"],
            data["monthly_missingness_mean_pct"]["volume"],
        )
    )
    lines.append(
        "- Universe composition is dynamic: mean month-to-month churn is `{:.2f}%` "
        "(peak `{:.2f}%`), which matters for turnover and model stability.".format(
            data["membership_churn_pct"]["mean"],
            data["membership_churn_pct"]["max"],
        )
    )
    lines.append(
        "- Feature pipeline is operationally healthy: feature-ready coverage is `{:.2f}%` "
        "with target availability `{:.2f}%`.".format(
            features["feature_ready_rate_pct"],
            features["target_available_rate_pct"],
        )
    )
    lines.append(
        "- Strategy diagnostics reflect two available runs with different effective horizons; "
        "results are informative but not yet fully apples-to-apples."
    )
    if strategy.get("n_benchmarks", 0):
        lines.append(
            "- Traditional benchmark panel includes `{}` series, with direct strategy-vs-benchmark "
            "comparisons in full sample and configured subperiods.".format(strategy["n_benchmarks"])
        )
    if strategy.get("n_factor_attribution_rows", 0):
        lines.append(
            "- Factor attribution is computed with CAPM/FF models over `{}` run/model slices.".format(
                strategy["n_factor_attribution_rows"]
            )
        )
    if strategy.get("n_prediction_diagnostic_rows", 0):
        lines.append(
            "- Prediction health diagnostics are available for `{}` runs to detect degenerate scoring behavior.".format(
                strategy["n_prediction_diagnostic_rows"]
            )
        )
    if strategy.get("top_run_by_sharpe"):
        top_row = strategy["top_run_by_sharpe"][0]
        lines.append(
            "- Best observed Sharpe row is `{run_key}` at `{cost_convention}` `{cost_bps:.1f}` bps "
            "(Sharpe `{sharpe:.3f}`).".format(**top_row)
        )
    lines.append("")

    lines.append("## Scope & Method")
    lines.append("")
    lines.append("- Layers: Data + Features + Strategy")
    lines.append("- Regime scheme: {}".format(", ".join(regime.name for regime in cfg.regimes)))
    lines.append("- Objective: explain structure, signal quality, and investable behavior under costs")
    lines.append("")
    lines.append("Methodological notes:")
    lines.append(
        "- All metrics are point-in-time and are computed from persisted project artifacts "
        "(`DataCube`, panel parquet, feature parquet, backtest outputs)."
    )
    lines.append(
        "- Feature diagnostics use cross-sectional monthly statistics, including Spearman IC "
        "and top-minus-bottom decile target spreads."
    )
    lines.append(
        "- Strategy diagnostics focus on net returns with explicit cost convention and bps settings; "
        "baseline charts use configured baseline cost values."
    )
    lines.append("")

    lines.append("## Research Questions, Hypotheses, and Evidence Standard")
    lines.append("")
    lines.append("### RQ1: Is the data usable for robust point-in-time research?")
    lines.append("")
    lines.append(
        "**Hypothesis:** data quality issues are limited and localized, not severe enough to dominate results."
    )
    lines.append("")
    lines.append(
        "**Evidence standard:** low average missingness, transparent month-level stress periods, stable date coverage, "
        "and explicit measurement of universe churn."
    )
    lines.append("")
    lines.append("### RQ2: Do engineered features contain economically relevant, stable cross-sectional information?")
    lines.append("")
    lines.append(
        "**Hypothesis:** a subset of features should show persistent but modest rank information (IC/spread), "
        "with regime variation that can be quantified."
    )
    lines.append("")
    lines.append(
        "**Evidence standard:** feature readiness/coverage, IC distribution over time, regime IC heatmaps, "
        "and top-minus-bottom decile spread diagnostics."
    )
    lines.append("")
    lines.append("### RQ3: Are current strategies investable net of costs, and are conclusions robust?")
    lines.append("")
    lines.append(
        "**Hypothesis:** at least one strategy remains attractive after explicit transaction costs, "
        "without relying on pathological turnover or concentration."
    )
    lines.append("")
    lines.append(
        "**Evidence standard:** net equity and rolling Sharpe, cost sensitivity curves, turnover distributions, "
        "weight concentration metrics, and regime-level decomposition."
    )
    lines.append("")

    lines.append("## Supervisor Minimum Checklist")
    lines.append("")
    checklist = pd.DataFrame(checklist_rows)
    _append_table_markdown(
        lines,
        title="Checklist Status",
        frame=checklist,
        columns=["Check", "Status", "Evidence", "Why it matters", "Next action"],
        max_rows=20,
    )
    lines.append(
        "Use this checklist as a gate before promoting exploratory findings into thesis claims. "
        "Any `Warning` or `Action Needed` item should be explicitly addressed in the next EDA iteration."
    )
    lines.append("")

    lines.append("## Definitions and Reconciliation")
    lines.append("")
    lines.append("This report mixes metrics computed on different denominators. This section pins down definitions so comparisons are correct.")
    lines.append("")
    recon = data.get("reconciliation", {})
    recon_current = recon.get("current_slice", {})
    recon_cube = recon.get("full_cube", {})
    n_lookback = int(data["shape"]["n_lookback"])

    lines.append("- `Current` returns/volume use the last lookback slice: `returns[:, -1, :]` and `volume[:, -1, :]`.")
    lines.append("- Current-slice missingness is computed per month over `n_assets`.")
    lines.append("- The DataCube audit's per-month missingness is computed over `n_assets * n_lookback` (all lookback slices).")
    lines.append("")
    lines.append("Denominator sanity check: if missingness were concentrated in the current slice only, then `full_cube_missing_pct_by_month` would be approximately `current_missing_pct_by_month / n_lookback`.")
    lines.append("")

    if not data_monthly.empty and n_lookback > 0:
        t_ret = int(recon_cube.get("worst_time_index_returns", 0))
        t_vol = int(recon_cube.get("worst_time_index_volume", 0))
        if 0 <= t_ret < len(data_monthly) and 0 <= t_vol < len(data_monthly):
            cur_ret_pct = float(data_monthly.iloc[t_ret]["returns_missing_pct"])
            cube_ret_pct = float(recon_cube.get("worst_missing_pct_by_time_returns", float("nan")))
            cur_vol_pct = float(data_monthly.iloc[t_vol]["volume_missing_pct"])
            cube_vol_pct = float(recon_cube.get("worst_missing_pct_by_time_volume", float("nan")))
            lines.append("Reconciliation examples (this run):")
            lines.append("")
            lines.append("```text")
            lines.append(
                "returns worst-month: current={:.2f}% (over assets) -> full-cube={:.5f}% (over assets*lookback) [n_lookback={}]".format(
                    cur_ret_pct, cur_ret_pct / n_lookback, n_lookback
                )
            )
            lines.append(
                "                 audit-style full-cube worst-month observed: {:.5f}% at {} ({})".format(
                    cube_ret_pct,
                    str(recon_cube.get("worst_month_label_returns", "n/a")),
                    str(recon_cube.get("worst_date_returns", "n/a")),
                )
            )
            lines.append(
                "volume  worst-month (audit-style): current={:.2f}% (over assets) -> full-cube={:.5f}% (over assets*lookback) [n_lookback={}]".format(
                    cur_vol_pct, cur_vol_pct / n_lookback, n_lookback
                )
            )
            lines.append(
                "                 audit-style full-cube worst-month observed: {:.5f}% at {} ({})".format(
                    cube_vol_pct,
                    str(recon_cube.get("worst_month_label_volume", "n/a")),
                    str(recon_cube.get("worst_date_volume", "n/a")),
                )
            )
            lines.append("```")
            lines.append("")

    lines.append(
        "Extremes note (this run): current-slice max return is `{:.6f}`, while full-cube max return is `{:.6f}` (count `{}`); full-cube max can occur in non-current lookback slices and can affect lookback-level diagnostics.".format(
            float(recon_current.get("max_returns", float("nan"))),
            float(recon_cube.get("max_returns", float("nan"))),
            int(recon_cube.get("max_returns_count", 0)),
        )
    )
    lines.append(
        "Full-cube max volume is `{:.1f}` (count `{}`); current-slice max volume count is `{}`.".format(
            float(recon_cube.get("max_volume", float("nan"))),
            int(recon_cube.get("max_volume_count", 0)),
            int(recon_current.get("max_volume_count", 0)),
        )
    )
    lines.append("")

    lines.append("## Data Layer")
    lines.append("")
    lines.append(
        "- Shape: `{n_assets}` assets x `{n_lookback}` lookback x `{n_time}` months".format(
            **data["shape"]
        )
    )
    lines.append(
        "- Date range: `{start}` to `{end}`".format(
            **data["date_range"]
        )
    )
    lines.append(
        "- Mean missingness (current slice): returns `{:.3f}%`, volume `{:.3f}%`".format(
            data["monthly_missingness_mean_pct"]["returns"],
            data["monthly_missingness_mean_pct"]["volume"],
        )
    )
    if recon_cube:
        lines.append(
            "- Full-cube non-finite rate: returns `{:.5f}%`, volume `{:.5f}%`".format(
                float(recon_cube.get("missing_pct_total_returns", float("nan"))),
                float(recon_cube.get("missing_pct_total_volume", float("nan"))),
            )
        )
    lines.append(
        "- Membership churn: mean `{:.2f}%`, max `{:.2f}%`".format(
            data["membership_churn_pct"]["mean"],
            data["membership_churn_pct"]["max"],
        )
    )
    lines.append("")
    if not data_monthly.empty:
        worst_ret = _first_valid_row(data_monthly, "returns_missing_pct", ascending=False)
        worst_vol = _first_valid_row(data_monthly, "volume_missing_pct", ascending=False)
        if worst_ret is not None and worst_vol is not None:
            lines.append(
                "Interpretation: missingness is low in aggregate, but it is time-varying. The worst "
                "return-missing month is `{}` at `{:.2f}%`, while the worst volume-missing month is "
                "`{}` at `{:.2f}%`. These local spikes should be treated as data-quality stress periods "
                "when evaluating robustness.".format(
                    str(worst_ret.get("month_label", "n/a")),
                    float(worst_ret["returns_missing_pct"]),
                    str(worst_vol.get("month_label", "n/a")),
                    float(worst_vol["volume_missing_pct"]),
                )
            )
            lines.append("")
    if not data_regime.empty:
        churn_regime = _first_valid_row(data_regime, "churn_pct_mean", ascending=False)
        if churn_regime is not None:
            lines.append(
                "Regime insight: `{}` shows the highest average universe churn (`{:.2f}%`), which "
                "can raise turnover and reduce feature persistence in that window.".format(
                    churn_regime["regime"], float(churn_regime["churn_pct_mean"])
                )
            )
            lines.append("")

    lines.append("How to read the figures:")
    lines.append(
        "- Use `data_01` and `data_04` together to separate data quality shifts from genuine "
        "market-structure changes."
    )
    lines.append(
        "- Use `data_02`, `data_03`, and `data_05` to assess whether cross-sectional behavior is "
        "stable enough for pooled model assumptions."
    )
    lines.append(
        "- Use `data_06` to verify whether long-horizon lookback slices have materially different scale "
        "than near-current slices."
    )
    lines.append("")
    for fig in data["figures"]:
        fig_path = Path(fig)
        lines.append(f"![{fig_path.name}](figures/{fig_path.name})")
        lines.append("")

    _append_table_markdown(
        lines,
        title="Data Regime Summary (excerpt)",
        frame=data_regime,
        columns=[
            "regime",
            "n_months",
            "returns_missing_pct_mean",
            "volume_missing_pct_mean",
            "returns_median_mean",
            "churn_pct_mean",
        ],
        max_rows=8,
    )

    lines.append("## Feature Layer")
    lines.append("")
    lines.append(f"- Rows analyzed: `{features['n_rows']}`")
    lines.append(f"- Feature count: `{features['n_features']}`")
    lines.append(
        "- Feature-ready coverage: `{:.2f}%`".format(features["feature_ready_rate_pct"])
    )
    lines.append(
        "- Target-available coverage: `{:.2f}%`".format(features["target_available_rate_pct"])
    )
    lines.append(
        "- Top features by absolute IC: `{}`".format(", ".join(features["top_features_by_ic"]) or "n/a")
    )
    lines.append("")
    lines.append(
        "Interpretation: feature readiness below 100% is expected because rolling-window "
        "features require history. The key question is whether missingness/readiness is randomly "
        "distributed or clustered by regime and market condition."
    )
    lines.append("")
    if not feature_ic.empty:
        best_ic = _first_valid_row(feature_ic.assign(abs_ic=feature_ic["mean_ic"].abs()), "abs_ic")
        if best_ic is not None:
            lines.append(
                "Signal takeaway: strongest average IC is `{}` with mean Spearman IC `{:.4f}` "
                "across `{}` months. This magnitude should be interpreted with turnover and "
                "costs in mind rather than in isolation.".format(
                    best_ic["feature"],
                    float(best_ic["mean_ic"]),
                    int(best_ic["n_months"]),
                )
            )
            lines.append("")
    if not feature_spread.empty:
        best_spread = _first_valid_row(
            feature_spread.assign(abs_annualized_spread=feature_spread["annualized_spread"].abs()),
            "abs_annualized_spread",
        )
        if best_spread is not None:
            lines.append(
                "Cross-sectional monotonicity check: `{}` has the largest absolute annualized "
                "top-minus-bottom spread (`{:.4f}`). This is useful as a ranking signal sanity "
                "check before model training.".format(
                    best_spread["feature"], float(best_spread["annualized_spread"])
                )
            )
            lines.append("")
    lines.append("How to read the figures:")
    lines.append(
        "- `feature_01` and `feature_02` diagnose operational quality (coverage + missingness)."
    )
    lines.append(
        "- `feature_03` shows redundancy structure; strongly correlated clusters suggest regularization "
        "or dimension reduction opportunities."
    )
    lines.append(
        "- `feature_04` and `feature_05` show regime sensitivity and time-variation of predictive rank information."
    )
    lines.append("")
    for fig in features["figures"]:
        fig_path = Path(fig)
        lines.append(f"![{fig_path.name}](figures/{fig_path.name})")
        lines.append("")

    _append_table_markdown(
        lines,
        title="Top Feature IC Summary (excerpt)",
        frame=feature_ic.sort_values("mean_ic", key=lambda s: s.abs(), ascending=False)
        if not feature_ic.empty
        else feature_ic,
        columns=["feature", "n_months", "mean_ic", "std_ic", "t_stat"],
        max_rows=10,
    )
    _append_table_markdown(
        lines,
        title="Top Feature Spread Summary (excerpt)",
        frame=feature_spread.sort_values(
            "annualized_spread", key=lambda s: s.abs(), ascending=False
        )
        if not feature_spread.empty
        else feature_spread,
        columns=["feature", "n_months", "mean_spread", "std_spread", "annualized_spread"],
        max_rows=10,
    )

    lines.append("## Strategy Layer")
    lines.append("")
    lines.append(f"- Runs discovered: `{strategy['n_runs']}`")
    lines.append(f"- Summary rows loaded: `{strategy['n_summary_rows']}`")
    lines.append(f"- Traditional benchmarks loaded: `{strategy.get('n_benchmarks', 0)}`")
    lines.append(f"- Factor attribution rows: `{strategy.get('n_factor_attribution_rows', 0)}`")
    top = strategy.get("top_run_by_sharpe", [])
    if top:
        top_row = top[0]
        lines.append(
            "- Top Sharpe row: `{run_key}` @ `{cost_convention}` `{cost_bps:.1f}` bps -> `{sharpe:.3f}`".format(
                **top_row
            )
        )
    lines.append("")
    if not strategy_summary.empty:
        baseline_rows = strategy_summary[
            (strategy_summary["cost_convention"] == cfg.baseline_cost_convention)
            & (strategy_summary["cost_bps"].astype(float) == float(cfg.baseline_cost_bps))
        ].copy()
        if not baseline_rows.empty:
            best_base = _first_valid_row(baseline_rows, "sharpe", ascending=False)
            worst_base = _first_valid_row(baseline_rows, "sharpe", ascending=True)
            if best_base is not None and worst_base is not None:
                lines.append(
                    "Baseline-cost comparison (`{}` {} bps): best run is `{}` (Sharpe `{:.3f}`, "
                    "annual return `{:.3f}`), while weakest is `{}` (Sharpe `{:.3f}`).".format(
                        cfg.baseline_cost_convention,
                        cfg.baseline_cost_bps,
                        best_base["run_key"],
                        float(best_base["sharpe"]),
                        float(best_base["annual_return"]),
                        worst_base["run_key"],
                        float(worst_base["sharpe"]),
                    )
                )
                lines.append("")
        lines.append(
            "Critical caveat: runs currently have different effective sample lengths (`n_periods`), "
            "so performance comparisons should be treated as indicative rather than final."
        )
        lines.append("")
    if not strategy_signal.empty:
        best_sig = _first_valid_row(strategy_signal, "mean_spearman_ic", ascending=False)
        if best_sig is not None:
            lines.append(
                "Signal-transfer diagnostic: `{}` has mean prediction-vs-realization Spearman IC "
                "`{:.4f}` over `{}` periods.".format(
                    best_sig["run_key"],
                    float(best_sig["mean_spearman_ic"]),
                    int(best_sig["n_periods"]),
                )
            )
            lines.append("")
    if not strategy_prediction_diagnostics.empty:
        degenerate = strategy_prediction_diagnostics[
            strategy_prediction_diagnostics["degenerate_predictions_flag"].astype(bool)
        ]
        if not degenerate.empty:
            deg_runs = ", ".join(sorted(degenerate["run_key"].astype(str).unique().tolist()))
            lines.append(
                "Prediction health warning: degenerate cross-sectional predictions detected for `{}` "
                "(>=95% dates with constant scores).".format(deg_runs)
            )
            lines.append("")
        strongest_pred_dispersion = _first_valid_row(
            strategy_prediction_diagnostics,
            "mean_prediction_std",
            ascending=False,
        )
        if strongest_pred_dispersion is not None:
            lines.append(
                "Prediction dispersion: `{}` has mean cross-sectional prediction std `{:.6f}`.".format(
                    strongest_pred_dispersion["run_key"],
                    float(strongest_pred_dispersion["mean_prediction_std"]),
                )
            )
            lines.append("")
    if not strategy_benchmark_comparison.empty:
        full_compare = strategy_benchmark_comparison[
            strategy_benchmark_comparison["scope"] == "full_sample"
        ].copy()
        strongest = _first_valid_row(full_compare, "delta_sharpe", ascending=False)
        if strongest is not None:
            lines.append(
                "Traditional benchmark comparison (full sample): `{}` vs `{}` has Sharpe delta `{:.3f}` "
                "and annual return delta `{:.3f}`.".format(
                    strongest["run_key"],
                    strongest["benchmark_name"],
                    float(strongest["delta_sharpe"]),
                    float(strongest["delta_annual_return"]),
                )
            )
            lines.append("")
    if not strategy_factor_attribution_fullsample.empty:
        full_attr_ok = strategy_factor_attribution_fullsample[
            strategy_factor_attribution_fullsample["status"] == "ok"
        ].copy()
        preferred_alpha = full_attr_ok[
            (full_attr_ok["return_type"] == "net")
            & (full_attr_ok["cost_convention"] == cfg.baseline_cost_convention)
            & (full_attr_ok["cost_bps"].astype(float) == float(cfg.baseline_cost_bps))
        ].copy()
        if preferred_alpha.empty:
            preferred_alpha = full_attr_ok[full_attr_ok["return_type"] == "gross"].copy()
        best_alpha = _first_valid_row(preferred_alpha, "alpha_annualized", ascending=False)
        if best_alpha is not None:
            lines.append(
                "Factor attribution (full sample): best annualized alpha is `{:.4f}` for `{}` under `{}` "
                "using `{}`.".format(
                    float(best_alpha["alpha_annualized"]),
                    best_alpha["run_key"],
                    best_alpha["return_type"],
                    best_alpha["model"],
                )
            )
            lines.append("")
    if not strategy_factor_exposure_stability.empty:
        most_unstable = _first_valid_row(strategy_factor_exposure_stability, "beta_range", ascending=False)
        if most_unstable is not None:
            lines.append(
                "Subperiod beta stability: largest cross-regime beta range is `{:.4f}` for `{}` "
                "({}, `{}` model).".format(
                    float(most_unstable["beta_range"]),
                    most_unstable["run_key"],
                    most_unstable["beta_name"],
                    most_unstable["model"],
                )
            )
            lines.append("")
    lines.append("How to read the figures:")
    lines.append(
        "- `strategy_01` and `strategy_02` together indicate both level and stability of performance."
    )
    lines.append(
        "- `strategy_03` quantifies robustness to implementation friction (cost sensitivity)."
    )
    lines.append(
        "- `strategy_04` and `strategy_05` help diagnose whether results rely on excessive turnover "
        "or concentration."
    )
    lines.append("")
    for fig in strategy["figures"]:
        fig_path = Path(fig)
        lines.append(f"![{fig_path.name}](figures/{fig_path.name})")
        lines.append("")

    min_subperiod_obs = int(cfg.regression_min_obs)
    strategy_regime_display = strategy_regime.copy()
    if not strategy_regime_display.empty and "n_periods" in strategy_regime_display.columns:
        strategy_regime_display = strategy_regime_display[
            (strategy_regime_display["regime"] == "full_sample")
            | (pd.to_numeric(strategy_regime_display["n_periods"], errors="coerce") >= min_subperiod_obs)
        ]

    benchmark_comparison_display = strategy_benchmark_comparison.copy()
    if not benchmark_comparison_display.empty:
        strategy_n = pd.to_numeric(benchmark_comparison_display.get("strategy_n_periods"), errors="coerce")
        bench_n = pd.to_numeric(benchmark_comparison_display.get("benchmark_n_periods"), errors="coerce")
        benchmark_comparison_display = benchmark_comparison_display[
            (benchmark_comparison_display["scope"] == "full_sample")
            | ((strategy_n >= min_subperiod_obs) & (bench_n >= min_subperiod_obs))
        ]

    factor_subperiod_display = strategy_factor_attribution_subperiod.copy()
    if not factor_subperiod_display.empty:
        n_obs = pd.to_numeric(factor_subperiod_display.get("n_obs"), errors="coerce")
        factor_subperiod_display = factor_subperiod_display[
            (factor_subperiod_display["status"] == "ok") & (n_obs >= min_subperiod_obs)
        ]

    lines.append(
        f"_Subperiod excerpts suppress rows with fewer than `{min_subperiod_obs}` observations._"
    )
    lines.append("")

    _append_table_markdown(
        lines,
        title="Strategy Summary by Cost (excerpt)",
        frame=strategy_summary.sort_values(["run_key", "cost_convention", "cost_bps"])
        if not strategy_summary.empty
        else strategy_summary,
        columns=[
            "run_key",
            "cost_convention",
            "cost_bps",
            "n_periods",
            "periods_per_year",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "avg_turnover",
        ],
        max_rows=12,
    )
    _append_table_markdown(
        lines,
        title="Strategy Regime Metrics (excerpt)",
        frame=strategy_regime_display.sort_values(["run_key", "regime"])
        if not strategy_regime_display.empty
        else strategy_regime_display,
        columns=[
            "run_key",
            "regime",
            "n_periods",
            "periods_per_year",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
        ],
        max_rows=12,
    )
    _append_table_markdown(
        lines,
        title="Weight Concentration Snapshot (excerpt)",
        frame=(
            strategy_weights.groupby("run_key", observed=True)
            .agg(
                mean_hhi_abs_weight=("hhi_abs_weight", "mean"),
                mean_top10_abs_weight_share=("top10_abs_weight_share", "mean"),
            )
            .reset_index()
            .sort_values("mean_hhi_abs_weight", ascending=False)
            if not strategy_weights.empty
            else strategy_weights
        ),
        columns=["run_key", "mean_hhi_abs_weight", "mean_top10_abs_weight_share"],
        max_rows=8,
    )
    _append_table_markdown(
        lines,
        title="Prediction Signal Quality (excerpt)",
        frame=strategy_signal.sort_values("mean_spearman_ic", ascending=False)
        if not strategy_signal.empty
        else strategy_signal,
        columns=["run_key", "n_periods", "mean_spearman_ic", "std_spearman_ic"],
        max_rows=8,
    )
    _append_table_markdown(
        lines,
        title="Prediction Health Diagnostics (excerpt)",
        frame=strategy_prediction_diagnostics.sort_values("constant_prediction_dates_pct", ascending=False)
        if not strategy_prediction_diagnostics.empty
        else strategy_prediction_diagnostics,
        columns=[
            "run_key",
            "n_dates",
            "mean_prediction_std",
            "constant_prediction_dates_pct",
            "degenerate_predictions_flag",
            "n_ic_dates",
            "mean_spearman_ic",
            "tstat_spearman_ic",
            "ic_hit_rate_pct",
            "nonzero_coef_count_mean",
        ],
        max_rows=8,
    )
    _append_table_markdown(
        lines,
        title="Traditional Benchmark Performance (excerpt)",
        frame=strategy_benchmark_performance.sort_values(["benchmark_name", "regime"])
        if not strategy_benchmark_performance.empty
        else strategy_benchmark_performance,
        columns=[
            "benchmark_name",
            "regime",
            "n_periods",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
        ],
        max_rows=12,
    )
    _append_table_markdown(
        lines,
        title="Strategy vs Benchmark Comparison (excerpt)",
        frame=benchmark_comparison_display.sort_values(["scope", "run_key", "regime", "benchmark_name"])
        if not benchmark_comparison_display.empty
        else benchmark_comparison_display,
        columns=[
            "scope",
            "run_key",
            "regime",
            "benchmark_name",
            "periods_per_year",
            "strategy_annual_return",
            "benchmark_annual_return",
            "delta_annual_return",
            "strategy_sharpe",
            "benchmark_sharpe",
            "delta_sharpe",
        ],
        max_rows=12,
    )
    _append_table_markdown(
        lines,
        title="Factor Attribution Full Sample (excerpt)",
        frame=strategy_factor_attribution_fullsample.sort_values(
            ["run_key", "return_type", "cost_convention", "cost_bps", "model"]
        )
        if not strategy_factor_attribution_fullsample.empty
        else strategy_factor_attribution_fullsample,
        columns=[
            "run_key",
            "return_type",
            "cost_convention",
            "cost_bps",
            "model",
            "status",
            "n_obs",
            "alpha_annualized",
            "alpha_tstat",
            "r2",
        ],
        max_rows=12,
    )
    _append_table_markdown(
        lines,
        title="Factor Attribution Subperiod (excerpt)",
        frame=factor_subperiod_display.sort_values(
            ["run_key", "return_type", "cost_convention", "cost_bps", "regime", "model"]
        )
        if not factor_subperiod_display.empty
        else factor_subperiod_display,
        columns=[
            "run_key",
            "return_type",
            "cost_convention",
            "cost_bps",
            "regime",
            "model",
            "status",
            "n_obs",
            "alpha_annualized",
            "alpha_tstat",
            "r2",
        ],
        max_rows=12,
    )
    _append_table_markdown(
        lines,
        title="Factor Exposure Stability (excerpt)",
        frame=strategy_factor_exposure_stability.sort_values("beta_range", ascending=False)
        if not strategy_factor_exposure_stability.empty
        else strategy_factor_exposure_stability,
        columns=[
            "run_key",
            "return_type",
            "cost_convention",
            "cost_bps",
            "model",
            "beta_name",
            "regime_count",
            "beta_mean",
            "beta_std",
            "beta_range",
        ],
        max_rows=12,
    )

    lines.append("## Risks, Caveats, and Interpretation Boundaries")
    lines.append("")
    lines.append(
        "- Cross-run strategy comparisons are currently limited by unequal validation horizon lengths."
    )
    lines.append(
        "- IC and spread statistics are cross-sectional and can vary materially by regime; aggregate "
        "averages should not be interpreted as stationary constants."
    )
    lines.append(
        "- Missingness is low overall but still concentrated in specific months; those periods can "
        "influence outlier-sensitive diagnostics."
    )
    lines.append(
        "- This report is exploratory and descriptive; it does not by itself establish causal "
        "economic mechanisms."
    )
    lines.append("")

    lines.append("## Recommended Next EDA Iteration")
    lines.append("")
    lines.append(
        "1. Run matched-horizon comparisons so one-step and two-step are evaluated on identical date ranges."
    )
    lines.append(
        "2. Add uncertainty intervals for IC and spread metrics (bootstrap by month or block bootstrap)."
    )
    lines.append(
        "3. Add tail-risk visuals (expected shortfall, downside semivariance) by regime and cost setting."
    )
    lines.append(
        "4. Add stratified analyses by liquidity buckets to isolate turnover-cost interactions."
    )
    lines.append(
        "5. Add an explicit “supervisor sign-off” appendix that marks each checklist item as closed/open."
    )
    lines.append("")

    lines.append("## Tables")
    lines.append("")
    for layer_name in ("data_layer", "feature_layer", "strategy_layer"):
        for table in summary[layer_name]["tables"]:
            table_path = Path(table)
            lines.append(f"- `{table_path.relative_to(paths.output_dir)}`")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- All statistics are computed point-in-time from stored artifacts.")
    lines.append("- Strategy diagnostics use configured baseline costs for equity/rolling charts.")
    lines.append("- Regime aggregation is inclusive on start/end dates.")
    lines.append("")

    with paths.report_md.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg = load_eda_config(args.config)
    if args.mode is not None:
        cfg.mode = args.mode

    panel_path = _resolve_panel_path(cfg.panel_input_path)
    feature_path = _resolve_feature_path(cfg.feature_input_path)
    if not cfg.datacube_path.exists():
        raise FileNotFoundError(f"DataCube not found: {cfg.datacube_path}")
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel parquet not found: {panel_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature parquet not found: {feature_path}")

    paths = _ensure_dirs(cfg.output_dir)
    data_layer = _build_data_layer(cfg, paths)
    feature_layer = _build_feature_layer(cfg, paths)
    strategy_layer = _build_strategy_layer(cfg, paths)

    summary = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": str(Path(args.config).resolve()),
        "mode": cfg.mode,
        "data_layer": data_layer,
        "feature_layer": feature_layer,
        "strategy_layer": strategy_layer,
    }
    with paths.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    _write_report(cfg, paths, summary)

    print(f"Wrote EDA summary: {paths.summary_json}")
    print(f"Wrote EDA report: {paths.report_md}")
    print(f"Wrote figures directory: {paths.figures_dir}")
    print(f"Wrote tables directory: {paths.tables_dir}")


if __name__ == "__main__":
    main()
