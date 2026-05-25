from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from core.config.schemas import OneStepModelConfig
from core.cv.splitters import TimeSeriesSplit


def _valid_mask(frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> pd.Series:
    finite_features = np.isfinite(frame[feature_columns]).all(axis=1)
    finite_target = np.isfinite(frame[target_column])
    return finite_features & finite_target & frame["asset_id"].notna() & frame["date"].notna()


class LinearPolicy(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _scores_to_weights(
    scores: torch.Tensor,
    weight_clip: float,
    selection_quantile: float,
) -> torch.Tensor:
    # Long-only direct policy with active concentration:
    # select top-q names by score and map into capped simplex (sum=1, w_i in [0, clip]).
    def _project_capped_simplex(v: torch.Tensor, cap: float) -> torch.Tensor:
        work = v.to(dtype=torch.float64)
        lo = float((work.min() - cap).item())
        hi = float(work.max().item())
        tol = 1e-10

        for _ in range(80):
            mid = (lo + hi) / 2.0
            projected = torch.clamp(work - mid, min=0.0, max=cap)
            total = float(projected.sum().item())
            if abs(total - 1.0) <= tol:
                return projected.to(dtype=v.dtype)
            if total > 1.0:
                lo = mid
            else:
                hi = mid

        projected = torch.clamp(work - hi, min=0.0, max=cap)
        total = float(projected.sum().item())
        if total <= 0.0:
            return torch.full_like(v, 1.0 / float(v.shape[0]))

        # Due to floating-point arithmetic, the final bisection iterate can land
        # slightly below the unit-sum target. Fill only across remaining slack so
        # we preserve both the simplex sum and the hard cap.
        residual = 1.0 - total
        if residual > 0.0:
            slack = torch.clamp(cap - projected, min=0.0)
            slack_total = float(slack.sum().item())
            if slack_total > 0.0:
                projected = projected + residual * (slack / slack_total)

        projected = torch.clamp(projected, min=0.0, max=cap)
        total = float(projected.sum().item())
        if total <= 0.0:
            return torch.full_like(v, 1.0 / float(v.shape[0]))
        if abs(total - 1.0) > 1e-8:
            projected = projected / total
            projected = torch.clamp(projected, min=0.0, max=cap)
        return projected.to(dtype=v.dtype)

    n_assets = int(scores.shape[0])
    k = max(1, int(np.floor(n_assets * selection_quantile)))
    min_k_for_cap = int(np.ceil(1.0 / weight_clip))
    if k < min_k_for_cap:
        k = min(n_assets, min_k_for_cap)
    top_vals, top_idx = torch.topk(scores, k=k, largest=True, sorted=False)
    sel = torch.softmax(top_vals, dim=0)
    sel = _project_capped_simplex(sel, weight_clip)

    out = torch.zeros_like(scores)
    out[top_idx] = sel
    return out


@dataclass(slots=True)
class _DateBatch:
    date: pd.Timestamp
    asset_ids: np.ndarray
    x: torch.Tensor
    y_raw: torch.Tensor
    y_active: torch.Tensor


def _build_batches(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    dates: np.ndarray,
    device: torch.device,
) -> list[_DateBatch]:
    batches: list[_DateBatch] = []
    for date in dates:
        subset = frame.loc[frame["date"] == date]
        if subset.empty:
            continue
        x = torch.tensor(subset[feature_columns].to_numpy(np.float32), device=device)
        y_raw = torch.tensor(subset[target_column].to_numpy(np.float32), device=device)
        # Train on cross-sectional active returns so policy learns stock selection
        # relative to the universe benchmark, not just market direction.
        # Keep returns in native units (no per-month std scaling) for cleaner
        # alignment between training objective and realized return evaluation.
        y_centered = y_raw - y_raw.mean()
        y_active = y_centered
        if x.shape[0] < 5:
            continue
        batches.append(
            _DateBatch(
                date=pd.Timestamp(date),
                asset_ids=subset["asset_id"].astype("int64").to_numpy(),
                x=x,
                y_raw=y_raw,
                y_active=y_active,
            )
        )
    return batches


def _turnover_penalty(
    prev_ids: np.ndarray | None,
    prev_weights: torch.Tensor | None,
    current_ids: np.ndarray,
    current_weights: torch.Tensor,
) -> torch.Tensor:
    if prev_ids is None or prev_weights is None:
        return torch.zeros((), device=current_weights.device)

    common, prev_idx, curr_idx = np.intersect1d(prev_ids, current_ids, return_indices=True)
    penalty = torch.zeros((), device=current_weights.device)

    if common.size:
        prev_shared = prev_weights[torch.tensor(prev_idx, device=current_weights.device)]
        curr_shared = current_weights[torch.tensor(curr_idx, device=current_weights.device)]
        penalty = penalty + torch.mean(torch.abs(curr_shared - prev_shared))

    if common.size < len(prev_ids):
        missing_prev = np.setdiff1d(np.arange(len(prev_ids)), prev_idx)
        if missing_prev.size:
            penalty = penalty + torch.mean(
                torch.abs(prev_weights[torch.tensor(missing_prev, device=current_weights.device)])
            )

    if common.size < len(current_ids):
        missing_curr = np.setdiff1d(np.arange(len(current_ids)), curr_idx)
        if missing_curr.size:
            penalty = penalty + torch.mean(
                torch.abs(current_weights[torch.tensor(missing_curr, device=current_weights.device)])
            )
    return penalty


def _fit_policy(
    batches: list[_DateBatch],
    config: OneStepModelConfig,
    n_features: int,
    device: torch.device,
) -> LinearPolicy:
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    model = LinearPolicy(n_features).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.l2_penalty,
    )

    for _ in range(config.epochs):
        port_returns: list[torch.Tensor] = []
        penalties: list[torch.Tensor] = []
        prev_ids: np.ndarray | None = None
        prev_w: torch.Tensor | None = None

        for batch in batches:
            scores = model(batch.x)
            weights = _scores_to_weights(
                scores,
                config.weight_clip,
                config.selection_quantile,
            )
            port_returns.append(torch.sum(weights * batch.y_active))
            penalties.append(_turnover_penalty(prev_ids, prev_w, batch.asset_ids, weights))
            prev_ids = batch.asset_ids
            prev_w = weights

        if not port_returns:
            break

        returns_tensor = torch.stack(port_returns)
        penalty_tensor = torch.stack(penalties).mean() if penalties else torch.zeros((), device=device)

        mean_ret = returns_tensor.mean()
        var_ret = returns_tensor.var(unbiased=False)
        utility = mean_ret - (config.risk_aversion * var_ret) - (
            config.turnover_penalty * penalty_tensor
        )
        loss = -utility

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


@dataclass(slots=True)
class OneStepTrainer:
    config: OneStepModelConfig

    def fit_predict_weights(self, frame: pd.DataFrame, splits: list[TimeSeriesSplit]) -> pd.DataFrame:
        feature_columns = self.config.feature_columns
        if not feature_columns:
            raise ValueError("OneStepModelConfig.feature_columns must be provided.")

        target_column = self.config.target_column
        available = _valid_mask(frame, feature_columns, target_column)
        work = frame.loc[available, ["date", "asset_id", target_column, *feature_columns]].copy()
        work["asset_id"] = work["asset_id"].astype("Int64")
        work = work.dropna(subset=["asset_id"])
        work["asset_id"] = work["asset_id"].astype(np.int64)

        device = torch.device("cpu")
        outputs: list[pd.DataFrame] = []
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
                config=self.config,
                n_features=len(feature_columns),
                device=device,
            )
            model.eval()

            for batch in valid_batches:
                with torch.no_grad():
                    scores = model(batch.x)
                    weights = _scores_to_weights(
                        scores,
                        self.config.weight_clip,
                        self.config.selection_quantile,
                    )
                outputs.append(
                    pd.DataFrame(
                        {
                            "date": batch.date,
                            "asset_id": batch.asset_ids,
                            "target_ret_1m": batch.y_raw.cpu().numpy(),
                            "score": scores.cpu().numpy(),
                            "weight": weights.cpu().numpy(),
                            "fold_id": split.fold_id,
                        }
                    )
                )

        if not outputs:
            return pd.DataFrame(
                columns=["date", "asset_id", "target_ret_1m", "score", "weight", "fold_id"]
            )
        return pd.concat(outputs, ignore_index=True)
