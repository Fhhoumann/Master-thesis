#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact recovery summary from artifact runs.")
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Root directory containing run outputs.",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/evaluation",
        help="Directory where recovery summary files will be written.",
    )
    parser.add_argument(
        "--cost-convention",
        default="one_way",
        help="Preferred cost convention for selecting a summary row per run.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=25.0,
        help="Preferred cost bps for selecting a summary row per run.",
    )
    parser.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help="Optional substring filter on run key. Repeatable.",
    )
    return parser.parse_args()


def _find_run_dirs(artifacts_root: Path) -> list[Path]:
    if not artifacts_root.exists():
        return []
    return sorted(path.parent for path in artifacts_root.glob("*/*/summary.csv"))


def _pick_summary_row(
    frame: pd.DataFrame,
    *,
    cost_convention: str,
    cost_bps: float,
) -> tuple[pd.Series, bool]:
    preferred = frame[
        (frame["cost_convention"] == cost_convention)
        & (pd.to_numeric(frame["cost_bps"], errors="coerce") == float(cost_bps))
    ]
    if not preferred.empty:
        return preferred.iloc[0], True
    best = frame.sort_values("sharpe", ascending=False).iloc[0]
    return best, False


def _diag_summary(diag_path: Path) -> dict[str, float | int | bool]:
    if not diag_path.exists():
        return {
            "diag_n_folds": 0,
            "diag_prediction_std_mean": float("nan"),
            "diag_prediction_std_min": float("nan"),
            "diag_prediction_std_max": float("nan"),
            "diag_constant_prediction_folds_pct": float("nan"),
            "diag_nonzero_coef_mean": float("nan"),
            "diag_nonzero_coef_min": float("nan"),
            "diag_nonzero_coef_max": float("nan"),
            "diag_degenerate_flag": False,
        }

    diag = pd.read_csv(diag_path)
    if diag.empty:
        return {
            "diag_n_folds": 0,
            "diag_prediction_std_mean": float("nan"),
            "diag_prediction_std_min": float("nan"),
            "diag_prediction_std_max": float("nan"),
            "diag_constant_prediction_folds_pct": float("nan"),
            "diag_nonzero_coef_mean": float("nan"),
            "diag_nonzero_coef_min": float("nan"),
            "diag_nonzero_coef_max": float("nan"),
            "diag_degenerate_flag": False,
        }

    pred_std = pd.to_numeric(diag.get("prediction_std", pd.Series(dtype=float)), errors="coerce")
    nonzero = pd.to_numeric(diag.get("nonzero_coef_count", pd.Series(dtype=float)), errors="coerce")

    constant_folds_pct = float((pred_std.fillna(0.0) <= 1e-12).mean() * 100.0) if not pred_std.empty else float("nan")
    nonzero_mean = float(nonzero.mean()) if not nonzero.dropna().empty else float("nan")
    degenerate_flag = bool(
        (np.isfinite(constant_folds_pct) and constant_folds_pct >= 95.0)
        or (np.isfinite(nonzero_mean) and nonzero_mean <= 0.0)
    )

    return {
        "diag_n_folds": int(diag.shape[0]),
        "diag_prediction_std_mean": float(pred_std.mean()) if not pred_std.dropna().empty else float("nan"),
        "diag_prediction_std_min": float(pred_std.min()) if not pred_std.dropna().empty else float("nan"),
        "diag_prediction_std_max": float(pred_std.max()) if not pred_std.dropna().empty else float("nan"),
        "diag_constant_prediction_folds_pct": constant_folds_pct,
        "diag_nonzero_coef_mean": nonzero_mean,
        "diag_nonzero_coef_min": float(nonzero.min()) if not nonzero.dropna().empty else float("nan"),
        "diag_nonzero_coef_max": float(nonzero.max()) if not nonzero.dropna().empty else float("nan"),
        "diag_degenerate_flag": degenerate_flag,
    }


def main() -> None:
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = _find_run_dirs(artifacts_root)
    rows: list[dict[str, object]] = []
    filters = [token.lower() for token in args.name_contains]

    for run_dir in run_dirs:
        run_key = f"{run_dir.parent.name}/{run_dir.name}"
        if filters and not any(token in run_key.lower() for token in filters):
            continue

        summary_path = run_dir / "summary.csv"
        summary = pd.read_csv(summary_path)
        if summary.empty:
            continue

        chosen, exact_cost_match = _pick_summary_row(
            summary,
            cost_convention=args.cost_convention,
            cost_bps=args.cost_bps,
        )
        diag = _diag_summary(run_dir / "model_diagnostics.csv")
        row = {
            "run_key": run_key,
            "experiment": chosen.get("experiment"),
            "pipeline": chosen.get("pipeline"),
            "model_type": chosen.get("model_type"),
            "selected_cost_convention": chosen.get("cost_convention"),
            "selected_cost_bps": float(chosen.get("cost_bps")),
            "selected_cost_match": bool(exact_cost_match),
            "n_periods": float(chosen.get("n_periods")),
            "periods_per_year": float(chosen.get("periods_per_year"))
            if pd.notna(chosen.get("periods_per_year"))
            else float("nan"),
            "annual_return": float(chosen.get("annual_return")),
            "annual_volatility": float(chosen.get("annual_volatility")),
            "sharpe": float(chosen.get("sharpe")),
            "max_drawdown": float(chosen.get("max_drawdown")),
            "avg_turnover": float(chosen.get("avg_turnover")),
            "cumulative_return": float(chosen.get("cumulative_return")),
            **diag,
            "summary_path": str(summary_path),
            "diagnostics_path": str(run_dir / "model_diagnostics.csv"),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No runs found for recovery summary.")

    out = out.sort_values(["diag_degenerate_flag", "sharpe"], ascending=[True, False]).reset_index(drop=True)
    csv_path = outdir / "recovery_summary.csv"
    md_path = outdir / "recovery_summary.md"
    out.to_csv(csv_path, index=False)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Recovery Summary\n\n")
        handle.write(
            f"- Preferred cost row: `{args.cost_convention}` `{args.cost_bps:.1f}` bps\n"
        )
        if filters:
            handle.write(f"- Name filter: `{', '.join(args.name_contains)}`\n")
        handle.write("\n")
        display_cols = [
            "run_key",
            "pipeline",
            "model_type",
            "selected_cost_convention",
            "selected_cost_bps",
            "n_periods",
            "periods_per_year",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "avg_turnover",
            "diag_degenerate_flag",
            "diag_prediction_std_mean",
            "diag_nonzero_coef_mean",
        ]
        display = out[[col for col in display_cols if col in out.columns]].copy()
        try:
            handle.write(display.to_markdown(index=False))
        except ImportError:
            handle.write("```text\n")
            handle.write(display.to_string(index=False))
            handle.write("\n```")
        handle.write("\n")

    print(f"Wrote recovery summary CSV: {csv_path}")
    print(f"Wrote recovery summary markdown: {md_path}")


if __name__ == "__main__":
    main()
