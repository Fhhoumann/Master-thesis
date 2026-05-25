from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.io import fetch_ken_french_monthly
from core.metrics import summarize_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs using long-only top-decile equal-weight portfolio returns."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Model output parquet files. Use twostep predictions.parquet or onestep weights.parquet "
            "(must contain date/asset_id/target_ret_1m + prediction|score)."
        ),
    )
    parser.add_argument(
        "--top-quantile",
        type=float,
        default=0.10,
        help="Top quantile selected each month for long-only equal-weight portfolio.",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/evaluation/top_decile_eval",
        help="Output directory for top-decile evaluation artifacts.",
    )
    return parser.parse_args()


def _infer_score_column(frame: pd.DataFrame) -> str:
    if "prediction" in frame.columns:
        return "prediction"
    if "score" in frame.columns:
        return "score"
    raise ValueError("Input frame must contain either `prediction` or `score`.")


def _infer_run_key(path: Path) -> str:
    parts = path.parts
    # Expected pattern: artifacts/<experiment>/<pipeline>/<file>
    if len(parts) >= 4 and parts[-2] in {"twostep", "onestep"}:
        return f"{parts[-3]}/{parts[-2]}"
    return path.stem


def _top_decile_returns(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_quantile: float,
) -> pd.DataFrame:
    if not (0.0 < top_quantile < 1.0):
        raise ValueError(f"`top_quantile` must be in (0, 1), got {top_quantile}")

    required = {"date", "asset_id", "target_ret_1m", score_column}
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Input frame missing required columns: {missing}")

    work = frame[["date", "asset_id", "target_ret_1m", score_column]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["target_ret_1m"] = pd.to_numeric(work["target_ret_1m"], errors="coerce")
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values(["date", "asset_id"], kind="stable")

    rows: list[dict[str, object]] = []
    for date, group in work.groupby("date", observed=True):
        valid = group.dropna(subset=["target_ret_1m", score_column]).copy()
        if valid.empty:
            continue
        k = max(1, int(np.floor(valid.shape[0] * top_quantile)))
        # Exact top-k selection avoids percentile tie inflation (e.g., selecting >10%).
        selected = valid.nlargest(k, columns=score_column, keep="first")
        if selected.empty:
            continue
        port_ret = float(selected["target_ret_1m"].mean())
        rows.append(
            {
                "date": date,
                "n_assets_total": int(valid.shape[0]),
                "n_selected": int(selected.shape[0]),
                "selection_rate": float(selected.shape[0] / valid.shape[0]),
                "portfolio_return": port_ret,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["date", "n_assets_total", "n_selected", "selection_rate", "portfolio_return"]
        )
    return out.sort_values("date").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rf_frame = pd.DataFrame()
    try:
        rf_frame = fetch_ken_french_monthly(Path("artifacts/factors/ken_french_monthly.parquet"))
    except Exception:
        rf_frame = pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    all_returns: list[pd.DataFrame] = []

    for raw_input in args.inputs:
        path = Path(raw_input)
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        frame = pd.read_parquet(path)
        score_col = _infer_score_column(frame)
        run_key = _infer_run_key(path)
        returns = _top_decile_returns(frame, score_column=score_col, top_quantile=args.top_quantile)
        if returns.empty:
            continue

        metrics = summarize_returns(
            returns["portfolio_return"],
            dates=returns["date"],
            risk_free_frame=rf_frame,
        )
        summary_rows.append(
            {
                "run_key": run_key,
                "source_file": str(path),
                "score_column": score_col,
                "top_quantile": float(args.top_quantile),
                "mean_selected": float(returns["n_selected"].mean()),
                "mean_selection_rate": float(returns["selection_rate"].mean()),
                **metrics,
            }
        )

        tmp = returns.copy()
        tmp["run_key"] = run_key
        tmp["source_file"] = str(path)
        all_returns.append(tmp)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("sharpe", ascending=False).reset_index(drop=True)
    all_returns_df = pd.concat(all_returns, ignore_index=True) if all_returns else pd.DataFrame()

    summary_path = outdir / "top_decile_summary.csv"
    returns_path = outdir / "top_decile_returns_by_month.csv"
    summary.to_csv(summary_path, index=False)
    all_returns_df.to_csv(returns_path, index=False)

    print(f"Wrote top-decile summary: {summary_path}")
    print(f"Wrote top-decile monthly returns: {returns_path}")


if __name__ == "__main__":
    main()
