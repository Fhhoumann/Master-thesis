from __future__ import annotations

import numpy as np
import pandas as pd


def _project_capped_simplex(weights: np.ndarray, cap: float) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.size == 0:
        return values
    if cap * values.size < 1.0 - 1e-12:
        raise ValueError("Capped simplex is infeasible for the requested cap and portfolio size.")

    lo = float(values.min() - cap)
    hi = float(values.max())
    for _ in range(100):
        mid = (lo + hi) / 2.0
        projected = np.clip(values - mid, 0.0, cap)
        total = projected.sum()
        if abs(total - 1.0) <= 1e-10:
            return projected
        if total > 1.0:
            lo = mid
        else:
            hi = mid

    projected = np.clip(values - hi, 0.0, cap)
    total = projected.sum()
    if total <= 0.0:
        return np.full(values.shape[0], 1.0 / float(values.shape[0]), dtype=float)

    residual = 1.0 - total
    if residual > 0.0:
        slack = np.clip(cap - projected, 0.0, None)
        slack_total = float(slack.sum())
        if slack_total > 0.0:
            projected = projected + residual * (slack / slack_total)

    projected = np.clip(projected, 0.0, cap)
    total = projected.sum()
    if total <= 0.0:
        return np.full(values.shape[0], 1.0 / float(values.shape[0]), dtype=float)
    if abs(total - 1.0) > 1e-8:
        projected = projected / total
        projected = np.clip(projected, 0.0, cap)
    return projected


def construct_long_short_from_scores(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prediction",
    top_quantile: float = 0.10,
) -> pd.DataFrame:
    if not (0.0 < top_quantile < 0.5):
        raise ValueError(f"`top_quantile` must be in (0, 0.5), got {top_quantile}")

    required = {"date", "asset_id", score_column}
    missing = [name for name in required if name not in predictions.columns]
    if missing:
        raise ValueError(f"predictions is missing required columns: {missing}")

    out = predictions.copy()
    out["weight"] = 0.0

    for _, date_index in out.groupby("date", observed=True).groups.items():
        valid = out.loc[date_index].dropna(subset=[score_column]).sort_values(score_column)
        if valid.empty:
            continue
        n = max(1, int(np.floor(len(valid) * top_quantile)))
        short_idx = valid.index[:n]
        long_idx = valid.index[-n:]
        out.loc[short_idx, "weight"] = -1.0 / n
        out.loc[long_idx, "weight"] = 1.0 / n
    return out


def construct_long_only_from_scores(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prediction",
    top_quantile: float = 0.10,
) -> pd.DataFrame:
    if not (0.0 < top_quantile <= 1.0):
        raise ValueError(f"`top_quantile` must be in (0, 1], got {top_quantile}")

    required = {"date", "asset_id", score_column}
    missing = [name for name in required if name not in predictions.columns]
    if missing:
        raise ValueError(f"predictions is missing required columns: {missing}")

    out = predictions.copy()
    out["weight"] = 0.0

    for _, date_index in out.groupby("date", observed=True).groups.items():
        valid = out.loc[date_index].dropna(subset=[score_column]).sort_values(score_column)
        if valid.empty:
            continue
        n = max(1, int(np.floor(len(valid) * top_quantile)))
        long_idx = valid.index[-n:]
        out.loc[long_idx, "weight"] = 1.0 / n
    return out


def construct_constrained_long_only_from_scores(
    predictions: pd.DataFrame,
    *,
    historical_returns: pd.DataFrame,
    splits: list,
    score_column: str = "prediction",
    top_quantile: float = 0.10,
    weight_cap: float = 0.10,
    covariance_shrinkage: float = 0.50,
    risk_aversion: float = 5.0,
    optimizer_steps: int = 250,
    optimizer_step_size: float = 0.05,
) -> pd.DataFrame:
    if not (0.0 < top_quantile <= 1.0):
        raise ValueError(f"`top_quantile` must be in (0, 1], got {top_quantile}")
    if not (0.0 < weight_cap <= 1.0):
        raise ValueError(f"`weight_cap` must be in (0, 1], got {weight_cap}")
    if not (0.0 <= covariance_shrinkage <= 1.0):
        raise ValueError(
            f"`covariance_shrinkage` must be in [0, 1], got {covariance_shrinkage}"
        )

    required_pred = {"date", "asset_id", score_column, "fold_id"}
    missing_pred = [name for name in required_pred if name not in predictions.columns]
    if missing_pred:
        raise ValueError(f"predictions is missing required columns: {missing_pred}")

    required_hist = {"date", "asset_id", "ret_1m"}
    missing_hist = [name for name in required_hist if name not in historical_returns.columns]
    if missing_hist:
        raise ValueError(f"historical_returns is missing required columns: {missing_hist}")

    split_map = {int(split.fold_id): split for split in splits}
    hist = historical_returns.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["ret_1m"] = pd.to_numeric(hist["ret_1m"], errors="coerce")
    hist = hist.dropna(subset=["date", "asset_id"])

    outputs: list[pd.DataFrame] = []
    for fold_id, fold_frame in predictions.groupby("fold_id", observed=True):
        split = split_map.get(int(fold_id))
        if split is None:
            raise ValueError(f"No split metadata found for fold_id={fold_id}")

        train_hist = hist[hist["date"].isin(split.train_dates)].copy()
        for _, date_index in fold_frame.groupby("date", observed=True).groups.items():
            valid = fold_frame.loc[date_index].dropna(subset=[score_column]).sort_values(score_column)
            if valid.empty:
                continue

            n = max(1, int(np.floor(len(valid) * top_quantile)))
            selected = valid.iloc[-n:].copy()
            asset_ids = selected["asset_id"].to_numpy()
            score_vec = selected[score_column].to_numpy(dtype=float)

            min_k_for_cap = int(np.ceil(1.0 / weight_cap))
            if len(asset_ids) < min_k_for_cap:
                raise ValueError(
                    "Selected top-quantile subset is too small to satisfy the requested cap."
                )

            asset_hist = train_hist[train_hist["asset_id"].isin(asset_ids)][["date", "asset_id", "ret_1m"]]
            pivot = asset_hist.pivot(index="date", columns="asset_id", values="ret_1m").reindex(columns=asset_ids)
            if pivot.empty or pivot.shape[0] < 2:
                weights = np.full(len(asset_ids), 1.0 / float(len(asset_ids)))
            else:
                centered = pivot.apply(lambda col: col.fillna(col.mean()), axis=0).fillna(0.0)
                returns_matrix = centered.to_numpy(dtype=float)
                cov = np.cov(returns_matrix, rowvar=False, ddof=0)
                if np.ndim(cov) == 0:
                    cov = np.array([[float(cov)]], dtype=float)
                diag = np.diag(np.diag(cov))
                sigma = (1.0 - covariance_shrinkage) * cov + covariance_shrinkage * diag

                # Optimize over the selected set only; this is the robustness layer
                # suggested by the supervisor to make two-step portfolios less mechanical.
                weights = np.full(len(asset_ids), 1.0 / float(len(asset_ids)), dtype=float)
                mu = score_vec
                for _ in range(int(optimizer_steps)):
                    grad = mu - (2.0 * risk_aversion * (sigma @ weights))
                    candidate = weights + float(optimizer_step_size) * grad
                    weights = _project_capped_simplex(candidate, weight_cap)

            selected["weight"] = weights
            outputs.append(selected)

    if not outputs:
        return predictions.assign(weight=0.0)

    out = pd.concat(outputs, ignore_index=True)
    all_rows = predictions[["date", "asset_id", score_column, "fold_id"]].copy()
    out = all_rows.merge(
        out[["date", "asset_id", "fold_id", "weight"]],
        on=["date", "asset_id", "fold_id"],
        how="left",
    )
    out["weight"] = out["weight"].fillna(0.0)
    return out


def normalize_weights_by_date(
    weights: pd.DataFrame,
    *,
    weight_column: str = "weight",
    clip: float | None = None,
    gross_target: float = 1.0,
    dollar_neutral: bool = True,
) -> pd.DataFrame:
    if weight_column not in weights.columns:
        raise ValueError(f"weights is missing column `{weight_column}`")

    out = weights.copy()
    values = out[weight_column].astype(float)
    if clip is not None:
        values = values.clip(lower=-clip, upper=clip)
    if dollar_neutral:
        group_mean = values.groupby(out["date"], observed=True).transform("mean")
        values = values - group_mean

    gross_by_date = values.abs().groupby(out["date"], observed=True).transform("sum")
    scale = (gross_target / gross_by_date).where(gross_by_date > 0.0, 0.0)
    out[weight_column] = values * scale
    return out
