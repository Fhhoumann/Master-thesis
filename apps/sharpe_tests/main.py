from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.benchmark import build_pairwise_sharpe_test_table
from core.io import fetch_ken_french_monthly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step 3 pairwise Sharpe-difference tests.")
    parser.add_argument("--enet-backtest", required=True, help="Path to Elastic Net backtest.parquet")
    parser.add_argument("--rf-backtest", required=True, help="Path to Random Forest backtest.parquet")
    parser.add_argument("--torch-backtest", required=True, help="Path to Torch backtest.parquet")
    parser.add_argument(
        "--outdir",
        default="artifacts/step3_sharpe_tests",
        help="Output directory for step 3 Sharpe-difference tables.",
    )
    parser.add_argument(
        "--factor-cache",
        default="artifacts/factors/ken_french_monthly.parquet",
        help="Path to Ken French monthly factor cache containing rf.",
    )
    parser.add_argument("--cost-convention", default="one_way", choices=["one_way", "round_trip"])
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--bootstrap-resamples", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibrate-block-size",
        action="store_true",
        help="Use Ledoit-Wolf Algorithm 3.1 style block-size calibration before the final test.",
    )
    parser.add_argument(
        "--candidate-block-sizes",
        default="1,2,4,6,8,10",
        help="Comma-separated candidate block sizes for Algorithm 3.1 calibration.",
    )
    parser.add_argument(
        "--calibration-pseudo-sequences",
        type=int,
        default=199,
        help="Number of pseudo sequences K for Algorithm 3.1 calibration.",
    )
    parser.add_argument(
        "--calibration-bootstrap-resamples",
        type=int,
        default=199,
        help="Bootstrap resamples M used inside each calibration confidence interval.",
    )
    parser.add_argument(
        "--residual-bootstrap-avg-block-size",
        type=float,
        default=5.0,
        help="Average block size for the stationary bootstrap of VAR(1) residuals during calibration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategies = {
        "Elastic Net Step 2 constrained": pd.read_parquet(args.enet_backtest),
        "Random Forest Step 2 constrained": pd.read_parquet(args.rf_backtest),
        "Torch gamma 5.0": pd.read_parquet(args.torch_backtest),
    }
    risk_free_frame = fetch_ken_french_monthly(Path(args.factor_cache))
    candidate_block_sizes = [
        int(token.strip()) for token in str(args.candidate_block_sizes).split(",") if token.strip()
    ]

    table = build_pairwise_sharpe_test_table(
        strategies,
        risk_free_frame,
        strategy_order=list(strategies.keys()),
        cost_convention=args.cost_convention,
        cost_bps=args.cost_bps,
        block_size=args.block_size,
        bootstrap_resamples=args.bootstrap_resamples,
        alpha=args.alpha,
        seed=args.seed,
        calibrate_block_size=args.calibrate_block_size,
        candidate_block_sizes=candidate_block_sizes,
        calibration_pseudo_sequences=args.calibration_pseudo_sequences,
        calibration_bootstrap_resamples=args.calibration_bootstrap_resamples,
        residual_bootstrap_avg_block_size=args.residual_bootstrap_avg_block_size,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "pairwise_sharpe_tests.csv"
    md_path = outdir / "pairwise_sharpe_tests.md"
    table.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Step 3 Pairwise Sharpe-Difference Tests\n\n")
        try:
            handle.write(table.to_markdown(index=False))
        except ImportError:
            handle.write("```text\n")
            handle.write(table.to_string(index=False))
            handle.write("\n```")
        handle.write("\n")

    print(f"Wrote Sharpe-difference table: {csv_path}")
    print(f"Wrote Sharpe-difference markdown: {md_path}")


if __name__ == "__main__":
    main()
