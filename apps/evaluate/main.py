from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and rank backtest summary files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Summary CSV file paths produced by train or backtest apps.",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/evaluation",
        help="Output directory for aggregated evaluation tables.",
    )
    parser.add_argument(
        "--include-experiments",
        nargs="*",
        default=None,
        help="Optional experiment names to keep in the final comparison table.",
    )
    parser.add_argument(
        "--include-pipelines",
        nargs="*",
        default=None,
        help="Optional pipeline names to keep in the final comparison table.",
    )
    parser.add_argument(
        "--preferred-cost-bps",
        type=float,
        default=0.0,
        help="Preferred transaction-cost level for final comparison row selection.",
    )
    parser.add_argument(
        "--preferred-convention",
        default="one_way",
        choices=["one_way", "round_trip"],
        help="Preferred cost convention for final comparison row selection.",
    )
    return parser.parse_args()


def _build_final_comparison(
    merged: pd.DataFrame,
    *,
    include_experiments: list[str] | None,
    include_pipelines: list[str] | None,
    preferred_cost_bps: float,
    preferred_convention: str,
) -> pd.DataFrame:
    work = merged.copy()
    if include_experiments:
        work = work[work["experiment"].isin(include_experiments)].copy()
    if include_pipelines:
        work = work[work["pipeline"].isin(include_pipelines)].copy()
    if work.empty:
        return pd.DataFrame()

    work["cost_priority"] = (work["cost_bps"] - float(preferred_cost_bps)).abs()
    work["convention_priority"] = (work["cost_convention"] != preferred_convention).astype(int)

    sort_cols = [
        "experiment",
        "convention_priority",
        "cost_priority",
        "cost_bps",
        "sharpe",
        "annual_return",
    ]
    ascending = [True, True, True, True, False, False]
    ranked = work.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    comparison = ranked.groupby("experiment", as_index=False, sort=False).head(1).copy()
    comparison["preferred_cost_bps"] = float(preferred_cost_bps)
    comparison["preferred_convention"] = preferred_convention
    comparison["selected_for_comparison"] = True
    return comparison.sort_values(["pipeline", "model_type", "experiment"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        frame["source_file"] = path
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        raise RuntimeError("No evaluation rows were loaded.")

    sort_columns = [col for col in ["sharpe", "annual_return"] if col in merged.columns]
    leaderboard = merged.sort_values(sort_columns, ascending=False).reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = outdir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    comparison = _build_final_comparison(
        merged,
        include_experiments=args.include_experiments,
        include_pipelines=args.include_pipelines,
        preferred_cost_bps=args.preferred_cost_bps,
        preferred_convention=args.preferred_convention,
    )
    comparison_path = outdir / "final_comparison.csv"
    if not comparison.empty:
        comparison.to_csv(comparison_path, index=False)

    markdown_path = outdir / "leaderboard.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Experiment Leaderboard\n\n")
        try:
            handle.write(leaderboard.head(20).to_markdown(index=False))
        except ImportError:
            handle.write("```text\n")
            handle.write(leaderboard.head(20).to_string(index=False))
            handle.write("\n```")
        if not comparison.empty:
            handle.write("\n\n## Final Comparison\n\n")
            try:
                handle.write(comparison.to_markdown(index=False))
            except ImportError:
                handle.write("```text\n")
                handle.write(comparison.to_string(index=False))
                handle.write("\n```")
        handle.write("\n")

    print(f"Wrote leaderboard: {leaderboard_path}")
    if not comparison.empty:
        print(f"Wrote final comparison: {comparison_path}")
    print(f"Wrote markdown leaderboard: {markdown_path}")


if __name__ == "__main__":
    main()
