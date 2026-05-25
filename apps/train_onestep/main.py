from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.config import load_experiment_config
from core.cv import get_time_splits
from core.features import (
    build_feature_audit,
    resolve_feature_columns,
    validate_feature_audit,
)
from core.io import fetch_ken_french_monthly
from core.metrics import summarize_backtest_frame
from core.models import OneStepTrainer
from core.portfolio import (
    missing_returns_by_month,
    run_backtest_with_costs,
)
from core.tracking import log_json_artifact, log_metrics, log_params, start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one-step policy model and run backtest.")
    parser.add_argument(
        "--config",
        default="configs/experiments/onestep_torch.yaml",
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)

    feature_path = Path(cfg.features.feature_output_dir)
    if feature_path.is_dir():
        feature_path = feature_path / "features.parquet"
    feature_frame = pd.read_parquet(feature_path)
    feature_frame["date"] = pd.to_datetime(feature_frame["date"], errors="coerce")
    feature_frame = feature_frame.dropna(subset=["date"])

    feature_columns = resolve_feature_columns(cfg, "onestep")
    cfg.onestep.feature_columns = feature_columns
    print(f"Using {len(feature_columns)} features: {feature_columns[:5]}...")

    outdir = Path("artifacts") / cfg.name / "onestep"
    outdir.mkdir(parents=True, exist_ok=True)

    feature_audit = build_feature_audit(
        feature_frame,
        cfg=cfg,
        pipeline_type="onestep",
        feature_columns=feature_columns,
    )
    validate_feature_audit(feature_audit)
    feature_audit_path = outdir / "feature_audit.json"
    with feature_audit_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_audit.to_dict(), handle, indent=2)
        handle.write("\n")

    splits = get_time_splits(feature_frame, "date", cfg.cv)
    trainer = OneStepTrainer(cfg.onestep)
    weights = trainer.fit_predict_weights(feature_frame, splits)
    if weights.empty:
        raise RuntimeError("One-step training produced no validation weights.")
    # One-step trainer already maps scores into a capped long-only simplex.
    # Avoid post-normalization here because it can rescale clipped weights above the cap.
    by_date = weights.groupby("date", observed=True)["weight"]
    sum_w = by_date.sum()
    max_w = by_date.max()
    sum_tol = 1e-6
    cap_tol = 1e-6
    if not ((sum_w - 1.0).abs() <= sum_tol).all():
        worst_sum_gap = float((sum_w - 1.0).abs().max())
        raise RuntimeError(
            f"One-step weights do not sum to 1 within tolerance. max_gap={worst_sum_gap:.3e}"
        )
    if (max_w > float(cfg.onestep.weight_clip) + cap_tol).any():
        worst_cap_excess = float((max_w - float(cfg.onestep.weight_clip)).max())
        raise RuntimeError(
            "One-step weights violate weight_clip constraint. "
            f"max_excess={worst_cap_excess:.3e}"
        )

    realized = weights.rename(columns={"target_ret_1m": "ret_1m"})[["date", "asset_id", "ret_1m"]].copy()
    backtest = run_backtest_with_costs(
        weights=weights[["date", "asset_id", "weight"]],
        realized_returns=realized,
        cost_bps=cfg.backtest.transaction_cost_bps,
        conventions=cfg.backtest.conventions,
        initial_turnover_from_zero=cfg.backtest.initial_turnover_from_zero,
        turnover_use_effective_weights=cfg.backtest.turnover_use_effective_weights,
        missing_return_policy=cfg.backtest.missing_return_policy,
    )
    rf_frame = pd.DataFrame()
    try:
        rf_frame = fetch_ken_french_monthly(Path("artifacts/factors/ken_french_monthly.parquet"))
    except Exception:
        rf_frame = pd.DataFrame()

    summary = summarize_backtest_frame(backtest, risk_free_frame=rf_frame)
    missing_by_month = missing_returns_by_month(backtest)
    summary["experiment"] = cfg.name
    summary["pipeline"] = "onestep"
    summary["model_type"] = "torch_linear_policy"

    weights_path = outdir / "weights.parquet"
    backtest_path = outdir / "backtest.parquet"
    summary_path = outdir / "summary.csv"
    missing_path = outdir / "missing_oos_returns_by_month.csv"
    config_snapshot_path = outdir / "resolved_config.json"

    weights.to_parquet(weights_path, index=False)
    backtest.to_parquet(backtest_path, index=False)
    summary.to_csv(summary_path, index=False)
    missing_by_month.to_csv(missing_path, index=False)
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(cfg.model_dump(mode="json"), handle, indent=2)
        handle.write("\n")

    with start_run(cfg.tracking, run_name=f"{cfg.name}-onestep") as run:
        log_params(
            run,
            {
                "experiment": cfg.name,
                "pipeline": "onestep",
                "cv_scheme": cfg.cv.scheme,
                "n_splits": len(splits),
                "epochs": cfg.onestep.epochs,
                "learning_rate": cfg.onestep.learning_rate,
            },
        )
        if not summary.empty:
            best_idx = summary["sharpe"].idxmax()
            best = summary.loc[best_idx]
            log_metrics(
                run,
                {
                    "best_sharpe": float(best["sharpe"]),
                    "best_annual_return": float(best["annual_return"]),
                    "best_max_drawdown": float(best["max_drawdown"]),
                },
            )
        log_json_artifact(
            run,
            {"rows": int(weights.shape[0]), "weights_path": str(weights_path)},
            "onestep_run_info.json",
        )

    print(f"Wrote weights: {weights_path}")
    print(f"Wrote backtest: {backtest_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote missing-return diagnostics: {missing_path}")
    print(f"Wrote feature audit: {feature_audit_path}")
    print(f"Wrote config snapshot: {config_snapshot_path}")


if __name__ == "__main__":
    main()
