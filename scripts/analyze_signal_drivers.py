from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from core.config import load_experiment_config
from core.cv import get_time_splits
from core.features import resolve_feature_columns
from core.models.onestep import (
    _build_batches,
    _fit_policy,
    _scores_to_weights,
)
from core.models.twostep import (
    _build_model,
    _select_elasticnet_alpha,
    _valid_mask as twostep_valid_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute appendix-style signal driver summaries for final pre-MA models."
    )
    parser.add_argument(
        "--elasticnet-config",
        default="configs/experiments/macroext_twostep_elasticnet_supervisor_constrained_final_prema_v1.yaml",
    )
    parser.add_argument(
        "--random-forest-config",
        default="configs/experiments/macroext_twostep_random_forest_supervisor_constrained_final_prema_v1.yaml",
    )
    parser.add_argument(
        "--torch-config",
        default="configs/experiments/macroext_onestep_torch_supervisor_ra500_final_prema_v1.yaml",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/appendix_signal_drivers_final_prema",
        help="Directory for appendix signal-driver outputs.",
    )
    parser.add_argument(
        "--limit-folds",
        type=int,
        default=None,
        help="Optional number of earliest folds to analyze for quick validation.",
    )
    parser.add_argument(
        "--torch-permutation-repeats",
        type=int,
        default=3,
        help="Number of permutation repeats per feature for the Torch appendix analysis.",
    )
    return parser.parse_args()


def _load_feature_frame(cfg) -> pd.DataFrame:
    feature_path = Path(cfg.features.feature_output_dir)
    if feature_path.is_dir():
        feature_path = feature_path / "features.parquet"
    frame = pd.read_parquet(feature_path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    return frame


def _limit_splits(splits, limit_folds: int | None):
    if limit_folds is None:
        return splits
    return splits[: max(0, int(limit_folds))]


def _aggregate_with_rank(frame: pd.DataFrame, value_col: str, ascending: bool = False) -> pd.DataFrame:
    out = frame.copy()
    out["rank"] = out[value_col].rank(method="dense", ascending=ascending).astype(int)
    out = out.sort_values(["rank", "feature"], ascending=[True, True]).reset_index(drop=True)
    return out


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"

    columns = [str(col) for col in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy()]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(columns))
    ]

    def _fmt_row(values: list[str]) -> str:
        padded = [values[idx].ljust(widths[idx]) for idx in range(len(values))]
        return "| " + " | ".join(padded) + " |"

    header = _fmt_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [_fmt_row(row) for row in rows]
    return "\n".join([header, separator, *body])


def analyze_elasticnet(config_path: str, outdir: Path, limit_folds: int | None) -> dict[str, object]:
    cfg = load_experiment_config(config_path)
    feature_frame = _load_feature_frame(cfg)
    feature_columns = resolve_feature_columns(cfg, "twostep")
    cfg.twostep.feature_columns = feature_columns
    splits = _limit_splits(get_time_splits(feature_frame, "date", cfg.cv), limit_folds)

    target_column = cfg.twostep.target_column
    available = twostep_valid_mask(feature_frame, feature_columns, target_column)
    work = feature_frame.loc[
        available, ["date", "asset_id", target_column, *feature_columns]
    ].copy()

    records: list[dict[str, object]] = []
    for split in splits:
        train = work.loc[work["date"].isin(split.train_dates)]
        if train.empty:
            continue

        model_config = cfg.twostep
        selected_alpha = float("nan")
        chosen = _select_elasticnet_alpha(
            train,
            feature_columns=feature_columns,
            target_column=target_column,
            params=cfg.twostep.params,
        )
        if chosen is not None:
            selected_alpha = float(chosen)
            tuned_params = dict(cfg.twostep.params)
            tuned_params["alpha"] = selected_alpha
            tuned_params.pop("alpha_grid", None)
            model_config = cfg.twostep.model_copy(
                update={"params": tuned_params}
            )

        model = _build_model(model_config)
        model.fit(train[feature_columns], train[target_column])
        if not isinstance(model, Pipeline):
            raise TypeError("Elastic Net appendix analysis expected a sklearn Pipeline.")
        estimator = model.named_steps["model"]
        coef = np.asarray(estimator.coef_, dtype=np.float64)
        for feature_name, coef_value in zip(feature_columns, coef, strict=True):
            records.append(
                {
                    "fold_id": int(split.fold_id),
                    "feature": feature_name,
                    "coefficient": float(coef_value),
                    "abs_coefficient": float(abs(coef_value)),
                    "is_nonzero": int(abs(coef_value) > 1e-12),
                    "selected_alpha": selected_alpha,
                }
            )

    per_fold = pd.DataFrame.from_records(records)
    aggregated = (
        per_fold.groupby("feature", as_index=False)
        .agg(
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            nonzero_frequency=("is_nonzero", "mean"),
            nonzero_count=("is_nonzero", "sum"),
            fold_count=("fold_id", "count"),
        )
    )
    aggregated = _aggregate_with_rank(aggregated, "mean_abs_coefficient", ascending=False)

    per_fold.to_csv(outdir / "elasticnet_coefficients_by_fold.csv", index=False)
    aggregated.to_csv(outdir / "elasticnet_coefficients_aggregated.csv", index=False)
    return {
        "experiment": cfg.name,
        "feature_count": len(feature_columns),
        "folds_analyzed": int(per_fold["fold_id"].nunique()) if not per_fold.empty else 0,
        "output": "elasticnet_coefficients_aggregated.csv",
    }


def analyze_random_forest(config_path: str, outdir: Path, limit_folds: int | None) -> dict[str, object]:
    cfg = load_experiment_config(config_path)
    feature_frame = _load_feature_frame(cfg)
    feature_columns = resolve_feature_columns(cfg, "twostep")
    cfg.twostep.feature_columns = feature_columns
    splits = _limit_splits(get_time_splits(feature_frame, "date", cfg.cv), limit_folds)

    target_column = cfg.twostep.target_column
    available = twostep_valid_mask(feature_frame, feature_columns, target_column)
    work = feature_frame.loc[
        available, ["date", "asset_id", target_column, *feature_columns]
    ].copy()

    records: list[dict[str, object]] = []
    for split in splits:
        train = work.loc[work["date"].isin(split.train_dates)]
        if train.empty:
            continue

        model = _build_model(cfg.twostep)
        model.fit(train[feature_columns], train[target_column])
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
        for feature_name, imp_value in zip(feature_columns, importances, strict=True):
            records.append(
                {
                    "fold_id": int(split.fold_id),
                    "feature": feature_name,
                    "importance": float(imp_value),
                }
            )

    per_fold = pd.DataFrame.from_records(records)
    aggregated = (
        per_fold.groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            fold_count=("fold_id", "count"),
        )
    )
    aggregated["std_importance"] = aggregated["std_importance"].fillna(0.0)
    aggregated = _aggregate_with_rank(aggregated, "mean_importance", ascending=False)

    per_fold.to_csv(outdir / "random_forest_importance_by_fold.csv", index=False)
    aggregated.to_csv(outdir / "random_forest_importance_aggregated.csv", index=False)
    return {
        "experiment": cfg.name,
        "feature_count": len(feature_columns),
        "folds_analyzed": int(per_fold["fold_id"].nunique()) if not per_fold.empty else 0,
        "output": "random_forest_importance_aggregated.csv",
    }


def _portfolio_return_from_batch(
    model,
    batch,
    config,
    permuted_feature_idx: int | None = None,
    seed: int | None = None,
) -> float:
    x = batch.x
    if permuted_feature_idx is not None:
        perm_x = x.clone()
        rng = np.random.default_rng(seed)
        order = rng.permutation(perm_x.shape[0])
        order_tensor = torch.tensor(order, device=perm_x.device, dtype=torch.long)
        perm_x[:, permuted_feature_idx] = perm_x[order_tensor, permuted_feature_idx]
        x = perm_x

    with torch.no_grad():
        scores = model(x)
        weights = _scores_to_weights(
            scores,
            config.weight_clip,
            config.selection_quantile,
        )
        realized_return = torch.sum(weights * batch.y_raw)
    return float(realized_return.cpu().item())


def analyze_torch(
    config_path: str,
    outdir: Path,
    limit_folds: int | None,
    permutation_repeats: int,
) -> dict[str, object]:
    cfg = load_experiment_config(config_path)
    feature_frame = _load_feature_frame(cfg)
    feature_columns = resolve_feature_columns(cfg, "onestep")
    cfg.onestep.feature_columns = feature_columns
    splits = _limit_splits(get_time_splits(feature_frame, "date", cfg.cv), limit_folds)

    target_column = cfg.onestep.target_column
    finite_features = np.isfinite(feature_frame[feature_columns]).all(axis=1)
    finite_target = np.isfinite(feature_frame[target_column])
    available = (
        finite_features
        & finite_target
        & feature_frame["asset_id"].notna()
        & feature_frame["date"].notna()
    )
    work = feature_frame.loc[
        available, ["date", "asset_id", target_column, *feature_columns]
    ].copy()
    work["asset_id"] = work["asset_id"].astype("Int64")
    work = work.dropna(subset=["asset_id"])
    work["asset_id"] = work["asset_id"].astype(np.int64)

    device = torch.device("cpu")
    records: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []

    for split in splits:
        train_batches = _build_batches(
            work,
            feature_columns=feature_columns,
            target_column=target_column,
            dates=split.train_dates,
            device=device,
        )
        valid_batches = _build_batches(
            work,
            feature_columns=feature_columns,
            target_column=target_column,
            dates=split.validation_dates,
            device=device,
        )
        if not train_batches or not valid_batches:
            continue

        model = _fit_policy(
            train_batches,
            config=cfg.onestep,
            n_features=len(feature_columns),
            device=device,
        )
        model.eval()

        baseline_returns = [
            _portfolio_return_from_batch(model, batch, cfg.onestep)
            for batch in valid_batches
        ]
        baseline_mean = float(np.mean(baseline_returns))
        fold_summaries.append(
            {
                "fold_id": int(split.fold_id),
                "baseline_mean_validation_return": baseline_mean,
                "validation_months": len(valid_batches),
            }
        )

        for feature_idx, feature_name in enumerate(feature_columns):
            for repeat_idx in range(permutation_repeats):
                permuted_returns = [
                    _portfolio_return_from_batch(
                        model,
                        batch,
                        cfg.onestep,
                        permuted_feature_idx=feature_idx,
                        seed=(cfg.onestep.random_seed + (1000 * split.fold_id) + (100 * feature_idx) + repeat_idx),
                    )
                    for batch in valid_batches
                ]
                permuted_mean = float(np.mean(permuted_returns))
                records.append(
                    {
                        "fold_id": int(split.fold_id),
                        "feature": feature_name,
                        "repeat_id": int(repeat_idx),
                        "baseline_mean_validation_return": baseline_mean,
                        "permuted_mean_validation_return": permuted_mean,
                        "importance_drop": baseline_mean - permuted_mean,
                    }
                )

    per_feature_repeat = pd.DataFrame.from_records(records)
    per_fold_summary = pd.DataFrame.from_records(fold_summaries)
    aggregated = (
        per_feature_repeat.groupby("feature", as_index=False)
        .agg(
            mean_importance_drop=("importance_drop", "mean"),
            std_importance_drop=("importance_drop", "std"),
            mean_baseline_return=("baseline_mean_validation_return", "mean"),
            mean_permuted_return=("permuted_mean_validation_return", "mean"),
            observation_count=("importance_drop", "count"),
        )
    )
    aggregated["std_importance_drop"] = aggregated["std_importance_drop"].fillna(0.0)
    aggregated = _aggregate_with_rank(aggregated, "mean_importance_drop", ascending=False)

    per_feature_repeat.to_csv(outdir / "torch_permutation_importance_by_fold.csv", index=False)
    per_fold_summary.to_csv(outdir / "torch_validation_baseline_by_fold.csv", index=False)
    aggregated.to_csv(outdir / "torch_permutation_importance_aggregated.csv", index=False)
    return {
        "experiment": cfg.name,
        "feature_count": len(feature_columns),
        "folds_analyzed": int(per_fold_summary["fold_id"].nunique()) if not per_fold_summary.empty else 0,
        "permutation_repeats": int(permutation_repeats),
        "output": "torch_permutation_importance_aggregated.csv",
    }


def build_markdown_summary(outdir: Path) -> None:
    specs = [
        ("Elastic Net", "elasticnet_coefficients_aggregated.csv", "mean_abs_coefficient"),
        ("Random Forest", "random_forest_importance_aggregated.csv", "mean_importance"),
        ("Torch", "torch_permutation_importance_aggregated.csv", "mean_importance_drop"),
    ]
    lines = [
        "# Appendix Signal Drivers",
        "",
        "The tables below summarize the signals that appear most influential in the final pre-MA specifications.",
        "",
    ]
    for title, filename, metric in specs:
        frame = pd.read_csv(outdir / filename).head(10)
        lines.append(f"## {title}")
        lines.append("")
        lines.append(_frame_to_markdown(frame))
        lines.append("")
    (outdir / "signal_drivers_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "elasticnet": analyze_elasticnet(args.elasticnet_config, outdir, args.limit_folds),
        "random_forest": analyze_random_forest(args.random_forest_config, outdir, args.limit_folds),
        "torch": analyze_torch(
            args.torch_config,
            outdir,
            args.limit_folds,
            args.torch_permutation_repeats,
        ),
        "limit_folds": args.limit_folds,
        "torch_permutation_repeats": args.torch_permutation_repeats,
    }
    with (outdir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    build_markdown_summary(outdir)
    print(f"Wrote appendix signal-driver outputs to {outdir}")


if __name__ == "__main__":
    main()
