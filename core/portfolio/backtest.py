from __future__ import annotations

import numpy as np
import pandas as pd

from core.costs.transaction import apply_transaction_costs, compute_turnover


def _prepare_effective_weights_and_gross(
    weights: pd.DataFrame,
    realized_returns: pd.DataFrame,
    *,
    date_column: str,
    asset_column: str,
    weight_column: str,
    return_column: str,
    missing_return_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = weights.merge(
        realized_returns[[date_column, asset_column, return_column]],
        on=[date_column, asset_column],
        how="left",
    ).copy()
    returns = pd.to_numeric(merged[return_column], errors="coerce")
    valid_returns = np.isfinite(returns.to_numpy(dtype=float, copy=False))
    merged["missing_return_flag"] = ~valid_returns

    gross_before = merged[weight_column].abs().groupby(merged[date_column], observed=True).transform("sum")
    gross_valid = (
        merged[weight_column]
        .where(valid_returns, 0.0)
        .abs()
        .groupby(merged[date_column], observed=True)
        .transform("sum")
    )

    if missing_return_policy == "exclude_renorm":
        # Exclude missing/invalid realized returns and re-scale remaining names
        # so each date keeps its original gross exposure.
        scale = (gross_before / gross_valid).where(gross_valid > 0.0, 0.0)
        effective_weight = merged[weight_column].where(valid_returns, 0.0) * scale
        effective_returns = returns.where(valid_returns, 0.0)
    elif missing_return_policy == "fill_zero":
        scale = pd.Series(1.0, index=merged.index, dtype=float)
        effective_weight = merged[weight_column]
        effective_returns = returns.fillna(0.0)
    else:
        raise ValueError(
            "Unsupported missing_return_policy: "
            f"{missing_return_policy}. Expected one of: exclude_renorm, fill_zero."
        )

    merged["effective_weight"] = pd.to_numeric(effective_weight, errors="coerce").fillna(0.0)
    merged["effective_return"] = pd.to_numeric(effective_returns, errors="coerce").fillna(0.0)
    merged["weighted_return"] = merged["effective_weight"] * merged["effective_return"]

    gross = (
        merged.groupby(date_column, observed=True)["weighted_return"].sum().rename("gross_return")
    ).reset_index()
    diagnostics = (
        merged.groupby(date_column, observed=True)
        .agg(
            n_assets_total=(asset_column, "size"),
            n_missing_return=("missing_return_flag", "sum"),
            n_valid_return=("missing_return_flag", lambda s: int((~s).sum())),
            gross_weight_before=(weight_column, lambda s: float(np.abs(s).sum())),
            gross_weight_valid=(
                weight_column,
                lambda s: float(np.abs(s[~merged.loc[s.index, "missing_return_flag"]]).sum()),
            ),
            dropped_weight_gross=(
                weight_column,
                lambda s: float(np.abs(s[merged.loc[s.index, "missing_return_flag"]]).sum()),
            ),
            gross_weight_effective=("effective_weight", lambda s: float(np.abs(s).sum())),
            renorm_scale=("weighted_return", lambda s: float(scale.loc[s.index].iloc[0]) if not s.empty else 0.0),
        )
        .reset_index()
    )
    diagnostics["missing_return_rate"] = (
        diagnostics["n_missing_return"] / diagnostics["n_assets_total"].where(diagnostics["n_assets_total"] > 0, np.nan)
    ).fillna(0.0)
    diagnostics["missing_return_policy"] = missing_return_policy

    gross = gross.merge(diagnostics, on=date_column, how="left")
    gross = gross.sort_values(date_column).reset_index(drop=True)
    return merged, gross


def portfolio_returns_from_weights(
    weights: pd.DataFrame,
    realized_returns: pd.DataFrame,
    *,
    date_column: str = "date",
    asset_column: str = "asset_id",
    weight_column: str = "weight",
    return_column: str = "ret_1m",
    missing_return_policy: str = "exclude_renorm",
) -> pd.DataFrame:
    required_w = {date_column, asset_column, weight_column}
    required_r = {date_column, asset_column, return_column}
    missing_w = [name for name in required_w if name not in weights.columns]
    missing_r = [name for name in required_r if name not in realized_returns.columns]
    if missing_w:
        raise ValueError(f"weights missing required columns: {missing_w}")
    if missing_r:
        raise ValueError(f"realized_returns missing required columns: {missing_r}")

    _, gross = _prepare_effective_weights_and_gross(
        weights=weights,
        realized_returns=realized_returns,
        date_column=date_column,
        asset_column=asset_column,
        weight_column=weight_column,
        return_column=return_column,
        missing_return_policy=missing_return_policy,
    )
    return gross


def run_backtest_with_costs(
    weights: pd.DataFrame,
    realized_returns: pd.DataFrame,
    *,
    cost_bps: list[float],
    conventions: list[str],
    initial_turnover_from_zero: bool = False,
    turnover_use_effective_weights: bool = True,
    date_column: str = "date",
    asset_column: str = "asset_id",
    weight_column: str = "weight",
    return_column: str = "ret_1m",
    missing_return_policy: str = "exclude_renorm",
) -> pd.DataFrame:
    effective_frame, gross = _prepare_effective_weights_and_gross(
        weights,
        realized_returns,
        date_column=date_column,
        asset_column=asset_column,
        weight_column=weight_column,
        return_column=return_column,
        missing_return_policy=missing_return_policy,
    )

    turnover_weights = weights
    if turnover_use_effective_weights:
        turnover_weights = effective_frame[[date_column, asset_column, "effective_weight"]].rename(
            columns={"effective_weight": weight_column}
        )

    turnover = compute_turnover(
        turnover_weights,
        date_column=date_column,
        asset_column=asset_column,
        weight_column=weight_column,
        initial_turnover_from_zero=initial_turnover_from_zero,
    )

    frames: list[pd.DataFrame] = []
    for bps in cost_bps:
        for convention in conventions:
            net = apply_transaction_costs(
                returns=gross,
                turnover=turnover,
                cost_bps=bps,
                convention=convention,
            )
            frames.append(net)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def missing_returns_by_month(backtest_frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "n_assets_total",
        "n_missing_return",
        "n_valid_return",
        "missing_return_rate",
        "gross_weight_before",
        "gross_weight_valid",
        "gross_weight_effective",
        "dropped_weight_gross",
        "renorm_scale",
        "missing_return_policy",
    }
    missing = [name for name in required if name not in backtest_frame.columns]
    if missing:
        raise ValueError(f"backtest frame missing required diagnostics columns: {missing}")

    out = (
        backtest_frame[
            [
                "date",
                "n_assets_total",
                "n_missing_return",
                "n_valid_return",
                "missing_return_rate",
                "gross_weight_before",
                "gross_weight_valid",
                "gross_weight_effective",
                "dropped_weight_gross",
                "renorm_scale",
                "missing_return_policy",
            ]
        ]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["missing_return_rate_pct"] = 100.0 * out["missing_return_rate"]
    return out
