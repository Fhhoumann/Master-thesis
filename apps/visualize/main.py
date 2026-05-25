from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.io import load_datacube


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exploratory visualizations for DataCube.")
    parser.add_argument(
        "--datacube",
        default="DataCube_1990_2024.mat",
        help="Path to MAT v7.3 DataCube file.",
    )
    parser.add_argument(
        "--outdir",
        default="reports/figures",
        help="Output directory for generated figures and index markdown.",
    )
    parser.add_argument(
        "--sample-assets",
        type=int,
        default=None,
        help="Optional cap on number of assets to load.",
    )
    parser.add_argument(
        "--sample-time",
        type=int,
        default=None,
        help="Optional cap on number of time points to load.",
    )
    parser.add_argument(
        "--crsp-explorer",
        action="store_true",
        help="Generate interactive per-CRSP ID returns explorer HTML.",
    )
    parser.add_argument(
        "--crsp-explorer-out",
        default=None,
        help="Optional path to CRSP explorer HTML. Defaults to <outdir>/08_crsp_id_returns_explorer.html.",
    )
    parser.add_argument(
        "--crsp-min-months",
        type=int,
        default=24,
        help="Minimum months with finite returns for a CRSP ID to be included.",
    )
    parser.add_argument(
        "--crsp-max-ids",
        type=int,
        default=500,
        help="Maximum number of CRSP IDs to include, ranked by month coverage.",
    )
    parser.add_argument(
        "--crsp-agg",
        choices=("median", "mean"),
        default="median",
        help="Cross-row monthly return aggregation for selected CRSP ID.",
    )
    parser.add_argument(
        "--crsp-lookup-csv",
        default=None,
        help=(
            "Optional CSV with CRSP ID lookup columns. "
            "Expected columns: `asset_id` and optional `ticker`, `company_name`."
        ),
    )
    parser.add_argument(
        "--crsp-lookup-template-out",
        default=None,
        help=(
            "Optional output CSV path for eligible CRSP IDs with ticker/company_name placeholders. "
            "Defaults to <outdir>/08_crsp_id_lookup_template.csv when --crsp-explorer is enabled."
        ),
    )
    return parser.parse_args()


def _time_labels(month_end: Iterable[pd.Timestamp | None]) -> list[str]:
    labels: list[str] = []
    for idx, value in enumerate(month_end):
        if value is None:
            labels.append(f"t{idx}")
        else:
            labels.append(value.strftime("%Y-%m"))
    return labels


def _apply_time_ticks(ax: plt.Axes, labels: list[str]) -> None:
    n = len(labels)
    if n == 0:
        return
    step = max(1, n // 12)
    tick_idx = np.arange(0, n, step)
    ax.set_xticks(tick_idx, [labels[i] for i in tick_idx], rotation=45, ha="right")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _safe_quantiles(array: np.ndarray, probs: tuple[float, ...], axis: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanquantile(array, probs, axis=axis)


def _normalize_asset_ids(values: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    output[finite] = np.rint(values[finite])
    return output


def _normalize_lookup_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _load_crsp_lookup(path: Path | None) -> dict[int, dict[str, str | None]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"CRSP lookup CSV not found: {path}")

    frame = pd.read_csv(path)
    if "asset_id" not in frame.columns:
        raise ValueError(f"CRSP lookup CSV must include `asset_id` column: {path}")

    asset_id = pd.to_numeric(frame["asset_id"], errors="coerce")
    ticker = frame["ticker"] if "ticker" in frame.columns else pd.Series([None] * len(frame))
    company_name = (
        frame["company_name"] if "company_name" in frame.columns else pd.Series([None] * len(frame))
    )

    lookup: dict[int, dict[str, str | None]] = {}
    for aid, tic, cname in zip(asset_id, ticker, company_name, strict=False):
        if not np.isfinite(aid):
            continue
        lookup[int(round(float(aid)))] = {
            "ticker": _normalize_lookup_text(tic),
            "company_name": _normalize_lookup_text(cname),
        }
    return lookup


def _asset_label(asset_id: int, lookup: dict[int, dict[str, str | None]]) -> str:
    row = lookup.get(asset_id)
    if row is None:
        return str(asset_id)
    ticker = row.get("ticker")
    company_name = row.get("company_name")
    if ticker and company_name:
        return f"{asset_id} | {ticker} | {company_name}"
    if ticker:
        return f"{asset_id} | {ticker}"
    if company_name:
        return f"{asset_id} | {company_name}"
    return str(asset_id)


def _write_crsp_lookup_template(
    output_path: Path,
    coverage: pd.DataFrame,
    lookup: dict[int, dict[str, str | None]],
) -> None:
    rows: list[dict[str, object]] = []
    for _, row in coverage.iterrows():
        asset_id = int(row["asset_id"])
        months_observed = int(row["months_observed"])
        existing = lookup.get(asset_id, {})
        rows.append(
            {
                "asset_id": asset_id,
                "months_observed": months_observed,
                "ticker": existing.get("ticker"),
                "company_name": existing.get("company_name"),
            }
        )

    output = pd.DataFrame(rows, columns=["asset_id", "months_observed", "ticker", "company_name"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


def _rank_eligible_crsp_ids(
    permno: np.ndarray,
    returns_current: np.ndarray,
    month_end: list[pd.Timestamp | None],
    *,
    min_months: int,
    max_ids: int,
) -> pd.DataFrame:
    if min_months <= 0:
        raise ValueError(f"`crsp_min_months` must be positive, got {min_months}")
    if max_ids <= 0:
        raise ValueError(f"`crsp_max_ids` must be positive, got {max_ids}")

    dates = pd.to_datetime(np.array(month_end, dtype="datetime64[ns]"), errors="coerce")
    asset_ids = _normalize_asset_ids(permno)

    id_chunks: list[np.ndarray] = []
    for time_index, date_value in enumerate(dates):
        if pd.isna(date_value):
            continue
        mask = np.isfinite(asset_ids[:, time_index]) & np.isfinite(returns_current[:, time_index])
        if not np.any(mask):
            continue
        unique_ids = np.unique(asset_ids[mask, time_index].astype(np.int64))
        if unique_ids.size > 0:
            id_chunks.append(unique_ids)

    columns = ["asset_id", "months_observed"]
    if not id_chunks:
        return pd.DataFrame(columns=columns)

    all_ids = np.concatenate(id_chunks)
    unique_ids, counts = np.unique(all_ids, return_counts=True)
    coverage = pd.DataFrame(
        {"asset_id": unique_ids.astype(np.int64), "months_observed": counts.astype(np.int32)}
    )
    coverage = coverage.loc[coverage["months_observed"] >= min_months].copy()
    if coverage.empty:
        return pd.DataFrame(columns=columns)

    coverage.sort_values(
        by=["months_observed", "asset_id"],
        ascending=[False, True],
        inplace=True,
        ignore_index=True,
    )
    return coverage.head(max_ids).copy()


def _aggregate_id_returns(
    asset_ids: np.ndarray,
    returns: np.ndarray,
    *,
    aggregator: Literal["median", "mean"],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "asset_id": asset_ids.astype(np.int64, copy=False),
            "ret_1m": returns.astype(np.float64, copy=False),
        }
    )
    grouped = frame.groupby("asset_id", sort=True)["ret_1m"]
    if aggregator == "median":
        ret_agg = grouped.median()
    else:
        ret_agg = grouped.mean()

    output = ret_agg.rename("ret_agg").to_frame()
    output["ret_std"] = grouped.std(ddof=0)
    output["row_count"] = grouped.size().astype(np.int32)
    output.reset_index(inplace=True)
    return output


def _build_crsp_monthly_series(
    permno: np.ndarray,
    returns_current: np.ndarray,
    month_end: list[pd.Timestamp | None],
    selected_ids: np.ndarray,
    *,
    aggregator: Literal["median", "mean"],
) -> pd.DataFrame:
    columns = ["asset_id", "date", "time_index", "ret_agg", "ret_std", "row_count", "cumret"]
    if selected_ids.size == 0:
        return pd.DataFrame(columns=columns)

    dates = pd.to_datetime(np.array(month_end, dtype="datetime64[ns]"), errors="coerce")
    normalized_ids = _normalize_asset_ids(permno)
    selected_ids = selected_ids.astype(np.int64, copy=False)

    monthly_frames: list[pd.DataFrame] = []
    for time_index, date_value in enumerate(dates):
        if pd.isna(date_value):
            continue
        mask = np.isfinite(normalized_ids[:, time_index]) & np.isfinite(returns_current[:, time_index])
        if not np.any(mask):
            continue

        ids = normalized_ids[mask, time_index].astype(np.int64, copy=False)
        rets = returns_current[mask, time_index].astype(np.float64, copy=False)
        selected_mask = np.isin(ids, selected_ids)
        if not np.any(selected_mask):
            continue

        grouped = _aggregate_id_returns(ids[selected_mask], rets[selected_mask], aggregator=aggregator)
        grouped["date"] = pd.Timestamp(date_value)
        grouped["time_index"] = int(time_index)
        monthly_frames.append(grouped)

    if not monthly_frames:
        return pd.DataFrame(columns=columns)

    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly.sort_values(by=["asset_id", "date"], inplace=True, ignore_index=True)
    monthly["cumret"] = monthly.groupby("asset_id", sort=False)["ret_agg"].transform(
        lambda s: (1.0 + s).cumprod() - 1.0
    )
    return monthly.loc[:, columns]


def _write_crsp_explorer_html(
    output_path: Path,
    crsp_monthly: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    aggregator: Literal["median", "mean"],
    lookup: dict[int, dict[str, str | None]],
) -> None:
    if crsp_monthly.empty or coverage.empty:
        raise ValueError("Cannot write CRSP explorer with empty data.")

    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    coverage_sorted = coverage.copy()
    coverage_sorted["asset_id"] = coverage_sorted["asset_id"].astype(np.int64)
    coverage_sorted.sort_values(
        by=["months_observed", "asset_id"],
        ascending=[False, True],
        inplace=True,
        ignore_index=True,
    )

    months_observed_map = {
        int(row["asset_id"]): int(row["months_observed"]) for _, row in coverage_sorted.iterrows()
    }
    ordered_ids = coverage_sorted["asset_id"].astype(np.int64).tolist()
    total_traces = len(ordered_ids) * 2
    if total_traces == 0:
        raise ValueError("No eligible CRSP IDs found for explorer.")

    first_id = ordered_ids[0]
    first_months = months_observed_map[first_id]
    first_label = _asset_label(first_id, lookup)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(f"Monthly Return ({aggregator})", "Cumulative Return"),
    )

    for index, asset_id in enumerate(ordered_ids):
        id_series = crsp_monthly.loc[crsp_monthly["asset_id"] == asset_id].copy()
        id_series.sort_values("date", inplace=True)
        visible = index == 0
        asset_label = _asset_label(asset_id, lookup)
        lookup_row = lookup.get(asset_id, {})
        ticker = lookup_row.get("ticker") or "n/a"
        company_name = lookup_row.get("company_name") or "n/a"

        monthly_custom = np.column_stack(
            [
                id_series["row_count"].to_numpy(dtype=np.float64, copy=False),
                id_series["ret_std"].to_numpy(dtype=np.float64, copy=False),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=id_series["date"],
                y=id_series["ret_agg"],
                mode="lines+markers",
                name=f"{asset_label} monthly",
                visible=visible,
                customdata=monthly_custom,
                hovertemplate=(
                    "CRSP ID: "
                    f"{asset_id}"
                    f"<br>Ticker: {ticker}"
                    f"<br>Company: {company_name}"
                    "<br>Month: %{x|%Y-%m}"
                    "<br>Return: %{y:.5f}"
                    "<br>Rows: %{customdata[0]:.0f}"
                    "<br>Std: %{customdata[1]:.5f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=id_series["date"],
                y=id_series["cumret"],
                mode="lines",
                name=f"{asset_label} cumulative",
                visible=visible,
                hovertemplate=(
                    "CRSP ID: "
                    f"{asset_id}"
                    f"<br>Ticker: {ticker}"
                    f"<br>Company: {company_name}"
                    "<br>Month: %{x|%Y-%m}"
                    "<br>Cumulative return: %{y:.5f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    buttons: list[dict[str, object]] = []
    for index, asset_id in enumerate(ordered_ids):
        visible = [False] * total_traces
        visible[2 * index] = True
        visible[2 * index + 1] = True
        months_observed = months_observed_map[asset_id]
        asset_label = _asset_label(asset_id, lookup)
        title = (
            "CRSP ID Returns Explorer"
            f" | {asset_label} | {months_observed} months"
            f" | aggregator={aggregator}"
        )
        buttons.append(
            {
                "label": asset_label,
                "method": "update",
                "args": [{"visible": visible}, {"title": title}],
            }
        )

    fig.update_layout(
        title=(
            "CRSP ID Returns Explorer"
            f" | {first_label} | {first_months} months"
            f" | aggregator={aggregator}"
        ),
        template="plotly_white",
        height=760,
        showlegend=False,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.20,
                "yanchor": "top",
            }
        ],
        margin={"l": 60, "r": 20, "t": 120, "b": 60},
    )
    fig.update_yaxes(title_text="Return", row=1, col=1, zeroline=True, zerolinewidth=1)
    fig.update_yaxes(title_text="Cumulative Return", row=2, col=1, zeroline=True, zerolinewidth=1)
    fig.update_xaxes(title_text="Month End", row=2, col=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True, auto_open=False)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.crsp_min_months <= 0:
        raise ValueError(f"`--crsp-min-months` must be positive, got {args.crsp_min_months}")
    if args.crsp_max_ids <= 0:
        raise ValueError(f"`--crsp-max-ids` must be positive, got {args.crsp_max_ids}")

    sns.set_theme(style="whitegrid", context="talk")

    cube = load_datacube(
        args.datacube,
        sample_assets=args.sample_assets,
        sample_time=args.sample_time,
    )

    n_assets = cube.n_assets
    n_lookback = cube.n_lookback
    n_time = cube.n_time
    x = np.arange(n_time)
    labels = _time_labels(cube.month_end)
    date_start = next((d for d in cube.month_end if d is not None), None)
    date_end = next((d for d in reversed(cube.month_end) if d is not None), None)

    returns_current = cube.current_returns
    volume_current = cube.current_volume

    returns_missing_rate = np.mean(~np.isfinite(returns_current), axis=0)
    volume_missing_rate = np.mean(~np.isfinite(volume_current), axis=0)

    figure_paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(x, 100.0 * returns_missing_rate, label="Returns missing %", linewidth=2.0)
    ax.plot(x, 100.0 * volume_missing_rate, label="Volume missing %", linewidth=2.0)
    ax.set_title("Monthly Missingness (Current Slice)")
    ax.set_ylabel("Missing rate (%)")
    ax.set_xlabel("Month")
    _apply_time_ticks(ax, labels)
    ax.legend(loc="upper right")
    path_missingness = outdir / "01_missingness_by_month.png"
    _save(fig, path_missingness)
    figure_paths.append(path_missingness)

    returns_q05, returns_q50, returns_q95 = _safe_quantiles(
        returns_current, (0.05, 0.50, 0.95), axis=0
    )
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(x, returns_q05, label="P05", linewidth=1.5)
    ax.plot(x, returns_q50, label="Median", linewidth=2.2)
    ax.plot(x, returns_q95, label="P95", linewidth=1.5)
    ax.set_title("Cross-Sectional Return Quantiles by Month (Current Slice)")
    ax.set_ylabel("Return")
    ax.set_xlabel("Month")
    _apply_time_ticks(ax, labels)
    ax.legend(loc="upper right")
    path_ret_q = outdir / "02_returns_quantiles_by_month.png"
    _save(fig, path_ret_q)
    figure_paths.append(path_ret_q)

    positive_volume = np.where(volume_current > 0, volume_current, np.nan)
    vol_q10, vol_q50, vol_q90 = _safe_quantiles(positive_volume, (0.10, 0.50, 0.90), axis=0)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(x, vol_q10, label="P10", linewidth=1.5)
    ax.plot(x, vol_q50, label="Median", linewidth=2.2)
    ax.plot(x, vol_q90, label="P90", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_title("Cross-Sectional Volume Quantiles by Month (Current Slice, Log Scale)")
    ax.set_ylabel("Volume (log scale)")
    ax.set_xlabel("Month")
    _apply_time_ticks(ax, labels)
    ax.legend(loc="upper right")
    path_vol_q = outdir / "03_volume_quantiles_by_month.png"
    _save(fig, path_vol_q)
    figure_paths.append(path_vol_q)

    returns_values = returns_current[np.isfinite(returns_current)]
    ret_lo, ret_hi = np.quantile(returns_values, [0.001, 0.999])
    returns_trimmed = returns_values[(returns_values >= ret_lo) & (returns_values <= ret_hi)]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.histplot(returns_trimmed, bins=100, stat="density", color="#0f766e", ax=ax)
    ax.axvline(np.nanmedian(returns_values), color="black", linestyle="--", linewidth=1.4, label="Median")
    ax.set_title("Distribution of Current Returns (Trimmed 0.1% Tails)")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    path_ret_hist = outdir / "04_current_return_distribution.png"
    _save(fig, path_ret_hist)
    figure_paths.append(path_ret_hist)

    volume_values = volume_current[np.isfinite(volume_current) & (volume_current > 0)]
    volume_log10 = np.log10(volume_values)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.histplot(volume_log10, bins=100, stat="density", color="#1d4ed8", ax=ax)
    ax.axvline(
        np.nanmedian(volume_log10),
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Median log10(volume)",
    )
    ax.set_title("Distribution of Current Volume (log10)")
    ax.set_xlabel("log10(volume)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    path_vol_hist = outdir / "05_current_volume_distribution_log10.png"
    _save(fig, path_vol_hist)
    figure_paths.append(path_vol_hist)

    churn = np.full(n_time, np.nan, dtype=np.float64)
    for t in range(1, n_time):
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

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(x, 100.0 * churn, color="#b91c1c", linewidth=2.0)
    ax.set_title("Universe Membership Churn by Month")
    ax.set_ylabel("Jaccard distance (%)")
    ax.set_xlabel("Month")
    _apply_time_ticks(ax, labels)
    path_churn = outdir / "06_membership_churn.png"
    _save(fig, path_churn)
    figure_paths.append(path_churn)

    lag_months_ago = (n_lookback - 1) - np.arange(n_lookback)
    mean_abs_return_by_lag = np.nanmean(np.abs(cube.returns), axis=(0, 2))
    mean_volume_by_lag = np.nanmean(np.where(cube.volume > 0, cube.volume, np.nan), axis=(0, 2))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax1.plot(lag_months_ago, mean_abs_return_by_lag, color="#7c3aed", linewidth=2.0)
    ax1.set_title("Average Magnitude by Lookback Lag")
    ax1.set_ylabel("Mean |return|")

    ax2.plot(lag_months_ago, mean_volume_by_lag, color="#0369a1", linewidth=2.0)
    ax2.set_yscale("log")
    ax2.set_ylabel("Mean volume (log scale)")
    ax2.set_xlabel("Months ago (120 = oldest, 0 = current)")
    path_lag = outdir / "07_lookback_profile.png"
    _save(fig, path_lag)
    figure_paths.append(path_lag)

    crsp_explorer_output = (
        Path(args.crsp_explorer_out)
        if args.crsp_explorer_out is not None
        else outdir / "08_crsp_id_returns_explorer.html"
    )
    lookup_csv_path = Path(args.crsp_lookup_csv) if args.crsp_lookup_csv is not None else None
    lookup_template_path = (
        Path(args.crsp_lookup_template_out)
        if args.crsp_lookup_template_out is not None
        else outdir / "08_crsp_id_lookup_template.csv"
    )
    crsp_lookup = _load_crsp_lookup(lookup_csv_path)
    crsp_explorer_summary: dict[str, object] = {
        "enabled": False,
        "output_path": str(crsp_explorer_output),
        "included_ids": 0,
        "min_months": int(args.crsp_min_months),
        "max_ids": int(args.crsp_max_ids),
        "aggregator": args.crsp_agg,
        "lookup_csv_path": str(lookup_csv_path) if lookup_csv_path is not None else None,
        "lookup_entries": int(len(crsp_lookup)),
    }

    if args.crsp_explorer:
        eligible_ids = _rank_eligible_crsp_ids(
            cube.permno,
            returns_current,
            cube.month_end,
            min_months=args.crsp_min_months,
            max_ids=args.crsp_max_ids,
        )
        if eligible_ids.empty:
            crsp_explorer_summary["reason"] = "no_eligible_ids"
            print("Skipped CRSP explorer: no eligible CRSP IDs under current filters.")
        else:
            _write_crsp_lookup_template(lookup_template_path, eligible_ids, crsp_lookup)
            crsp_explorer_summary["lookup_template_path"] = str(lookup_template_path)
            selected_ids = eligible_ids["asset_id"].to_numpy(dtype=np.int64, copy=False)
            mapped_ids = 0
            for aid in selected_ids:
                row = crsp_lookup.get(int(aid))
                if row is not None and (row.get("ticker") or row.get("company_name")):
                    mapped_ids += 1
            crsp_explorer_summary["lookup_mapped_ids"] = int(mapped_ids)
            crsp_monthly = _build_crsp_monthly_series(
                cube.permno,
                returns_current,
                cube.month_end,
                selected_ids,
                aggregator=args.crsp_agg,
            )
            if crsp_monthly.empty:
                crsp_explorer_summary["reason"] = "no_aggregated_points"
                print("Skipped CRSP explorer: no aggregated points after filtering.")
            else:
                _write_crsp_explorer_html(
                    crsp_explorer_output,
                    crsp_monthly,
                    eligible_ids,
                    aggregator=args.crsp_agg,
                    lookup=crsp_lookup,
                )
                crsp_explorer_summary["enabled"] = True
                crsp_explorer_summary["included_ids"] = int(eligible_ids.shape[0])
                print(f"Wrote CRSP explorer: {crsp_explorer_output}")
                print(f"Wrote CRSP lookup template: {lookup_template_path}")

    summary = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datacube_path": str(Path(args.datacube).resolve()),
        "shape": {"n_assets": n_assets, "n_lookback": n_lookback, "n_time": n_time},
        "date_range": {
            "start": date_start.strftime("%Y-%m-%d") if date_start is not None else None,
            "end": date_end.strftime("%Y-%m-%d") if date_end is not None else None,
        },
        "current_slice_quality": {
            "returns_missing_mean_pct": float(np.nanmean(returns_missing_rate) * 100.0),
            "volume_missing_mean_pct": float(np.nanmean(volume_missing_rate) * 100.0),
        },
        "membership_churn": {
            "mean_pct": float(np.nanmean(churn) * 100.0),
            "max_pct": float(np.nanmax(churn) * 100.0),
        },
        "crsp_explorer": crsp_explorer_summary,
        "figures": [str(p) for p in figure_paths],
    }

    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    md_path = outdir / "README.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# DataCube Visualization Pack\n\n")
        handle.write(f"- Generated (UTC): `{summary['generated_at_utc']}`\n")
        handle.write(f"- DataCube: `{summary['datacube_path']}`\n")
        handle.write(
            "- Shape: `{n_assets}` assets, `{n_lookback}` lookback, `{n_time}` months\n".format(
                **summary["shape"]
            )
        )
        handle.write(
            "- Date range: `{start}` to `{end}`\n\n".format(
                **summary["date_range"]
            )
        )
        if bool(summary["crsp_explorer"]["enabled"]):
            handle.write("## 08_crsp_id_returns_explorer.html\n\n")
            explorer_output = Path(str(summary["crsp_explorer"]["output_path"]))
            if explorer_output.parent.resolve() == outdir.resolve():
                explorer_link = explorer_output.name
            else:
                explorer_link = str(explorer_output)
            handle.write(f"[Open interactive explorer]({explorer_link})\n\n")
            if summary["crsp_explorer"].get("lookup_template_path") is not None:
                template_path = Path(str(summary["crsp_explorer"]["lookup_template_path"]))
                if template_path.parent.resolve() == outdir.resolve():
                    template_link = template_path.name
                else:
                    template_link = str(template_path)
                handle.write(f"[CRSP ID lookup template]({template_link})\n\n")
        for figure in figure_paths:
            handle.write(f"## {figure.name}\n\n")
            handle.write(f"![{figure.name}]({figure.name})\n\n")

    print(f"Wrote visualization summary: {summary_path}")
    print(f"Wrote visualization index: {md_path}")
    for figure in figure_paths:
        print(f"Wrote figure: {figure}")


if __name__ == "__main__":
    main()
