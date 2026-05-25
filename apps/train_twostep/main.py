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
from core.models import TwoStepTrainer
from core.portfolio import (
    construct_constrained_long_only_from_scores,
    construct_long_only_from_scores,
    missing_returns_by_month,
    normalize_weights_by_date,
    run_backtest_with_costs,
)
from core.tracking import log_json_artifact, log_metrics, log_params, start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two-step model and run backtest.")
    parser.add_argument(
        "--config",
        default="configs/experiments/twostep_elasticnet.yaml",
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

    feature_columns = resolve_feature_columns(cfg, "twostep")
    cfg.twostep.feature_columns = feature_columns
    print(f"Using {len(feature_columns)} features: {feature_columns[:5]}...")

    outdir = Path("artifacts") / cfg.name / "twostep"
    outdir.mkdir(parents=True, exist_ok=True)

    feature_audit = build_feature_audit(
        feature_frame,
        cfg=cfg,
        pipeline_type="twostep",
        feature_columns=feature_columns,
    )
    validate_feature_audit(feature_audit)
    feature_audit_path = outdir / "feature_audit.json"
    with feature_audit_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_audit.to_dict(), handle, indent=2)
        handle.write("\n")

    splits = get_time_splits(feature_frame, "date", cfg.cv)
    trainer = TwoStepTrainer(cfg.twostep)
    predictions, model_diagnostics = trainer.fit_predict_with_diagnostics(feature_frame, splits)
    if predictions.empty:
        raise RuntimeError("Two-step training produced no validation predictions.")

    if cfg.backtest.two_step_portfolio_rule == "constrained_mean_variance":
        weights = construct_constrained_long_only_from_scores(
            predictions,
            historical_returns=feature_frame[["date", "asset_id", "ret_1m"]],
            splits=splits,
            score_column="prediction",
            top_quantile=cfg.backtest.top_quantile,
            weight_cap=cfg.backtest.two_step_weight_cap,
            covariance_shrinkage=cfg.backtest.two_step_covariance_shrinkage,
            risk_aversion=cfg.backtest.two_step_mv_risk_aversion,
            optimizer_steps=cfg.backtest.two_step_optimizer_steps,
            optimizer_step_size=cfg.backtest.two_step_optimizer_step_size,
        )
    else:
        weights = construct_long_only_from_scores(
            predictions,
            score_column="prediction",
            top_quantile=cfg.backtest.top_quantile,
        )
        weights = normalize_weights_by_date(weights, weight_column="weight", dollar_neutral=False)

    realized = predictions.rename(columns={cfg.twostep.target_column: "ret_1m"})[
        ["date", "asset_id", "ret_1m"]
    ].copy()
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
    summary["pipeline"] = "twostep"
    summary["model_type"] = cfg.twostep.model_type

    prediction_path = outdir / "predictions.parquet"
    weights_path = outdir / "weights.parquet"
    backtest_path = outdir / "backtest.parquet"
    summary_path = outdir / "summary.csv"
    missing_path = outdir / "missing_oos_returns_by_month.csv"
    diagnostics_path = outdir / "model_diagnostics.csv"
    config_snapshot_path = outdir / "resolved_config.json"

    predictions.to_parquet(prediction_path, index=False)
    weights.to_parquet(weights_path, index=False)
    backtest.to_parquet(backtest_path, index=False)
    summary.to_csv(summary_path, index=False)
    missing_by_month.to_csv(missing_path, index=False)
    model_diagnostics.to_csv(diagnostics_path, index=False)
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(cfg.model_dump(mode="json"), handle, indent=2)
        handle.write("\n")

    with start_run(cfg.tracking, run_name=f"{cfg.name}-twostep") as run:
        log_params(
            run,
            {
                "experiment": cfg.name,
                "pipeline": "twostep",
                "model_type": cfg.twostep.model_type,
                "cv_scheme": cfg.cv.scheme,
                "n_splits": len(splits),
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
            {"rows": int(predictions.shape[0]), "prediction_path": str(prediction_path)},
            "twostep_run_info.json",
        )

    print(f"Wrote predictions: {prediction_path}")
    print(f"Wrote weights: {weights_path}")
    print(f"Wrote backtest: {backtest_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote missing-return diagnostics: {missing_path}")
    print(f"Wrote model diagnostics: {diagnostics_path}")
    print(f"Wrote feature audit: {feature_audit_path}")
    print(f"Wrote config snapshot: {config_snapshot_path}")


if __name__ == "__main__":
    main()
