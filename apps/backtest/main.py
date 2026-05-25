from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.config import load_backtest_config
from core.io import fetch_ken_french_monthly
from core.metrics import summarize_backtest_frame
from core.portfolio import missing_returns_by_month, run_backtest_with_costs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backtest from weights and realized returns.")
    parser.add_argument("--weights", required=True, help="Path to weights parquet.")
    parser.add_argument("--realized", required=True, help="Path to realized returns parquet.")
    parser.add_argument(
        "--config",
        default="configs/backtests/default.yaml",
        help="Path to backtest YAML config.",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/backtests/manual",
        help="Output directory for backtest artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_backtest_config(args.config)

    weights = pd.read_parquet(args.weights)
    realized = pd.read_parquet(args.realized)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    backtest = run_backtest_with_costs(
        weights=weights,
        realized_returns=realized,
        cost_bps=cfg.transaction_cost_bps,
        conventions=cfg.conventions,
        missing_return_policy=cfg.missing_return_policy,
    )
    rf_frame = pd.DataFrame()
    try:
        rf_frame = fetch_ken_french_monthly(Path("artifacts/factors/ken_french_monthly.parquet"))
    except Exception:
        rf_frame = pd.DataFrame()

    summary = summarize_backtest_frame(backtest, risk_free_frame=rf_frame)
    missing_by_month = missing_returns_by_month(backtest)

    backtest_path = outdir / "backtest.parquet"
    summary_path = outdir / "summary.csv"
    missing_path = outdir / "missing_oos_returns_by_month.csv"
    backtest.to_parquet(backtest_path, index=False)
    summary.to_csv(summary_path, index=False)
    missing_by_month.to_csv(missing_path, index=False)

    print(f"Wrote backtest: {backtest_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote missing-return diagnostics: {missing_path}")


if __name__ == "__main__":
    main()
