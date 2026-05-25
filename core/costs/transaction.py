from __future__ import annotations

import numpy as np
import pandas as pd


def compute_turnover(
    weights: pd.DataFrame,
    *,
    date_column: str = "date",
    asset_column: str = "asset_id",
    weight_column: str = "weight",
    initial_turnover_from_zero: bool = False,
) -> pd.DataFrame:
    required = {date_column, asset_column, weight_column}
    missing = [name for name in required if name not in weights.columns]
    if missing:
        raise ValueError(f"weights is missing required columns: {missing}")

    pivot = weights.pivot_table(
        index=date_column,
        columns=asset_column,
        values=weight_column,
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    diff = pivot.diff().fillna(0.0)
    if initial_turnover_from_zero and not pivot.empty:
        # If portfolio is initiated from cash, first rebalance trades from 0 -> w_0.
        diff.iloc[0, :] = pivot.iloc[0, :]
    gross_weight_change = diff.abs().sum(axis=1)
    one_way_turnover = 0.5 * gross_weight_change
    round_trip_turnover = gross_weight_change

    return pd.DataFrame(
        {
            "date": pivot.index,
            "turnover_one_way": one_way_turnover.to_numpy(),
            "turnover_round_trip": round_trip_turnover.to_numpy(),
        }
    )


def apply_transaction_costs(
    returns: pd.DataFrame,
    turnover: pd.DataFrame,
    *,
    cost_bps: float,
    convention: str,
) -> pd.DataFrame:
    if convention not in {"one_way", "round_trip"}:
        raise ValueError(f"Unsupported convention: {convention}")
    if "date" not in returns.columns or "gross_return" not in returns.columns:
        raise ValueError("returns must contain `date` and `gross_return`.")

    turnover_column = "turnover_one_way" if convention == "one_way" else "turnover_round_trip"
    merged = returns.merge(turnover[["date", turnover_column]], on="date", how="left").copy()
    merged[turnover_column] = merged[turnover_column].fillna(0.0)

    cost_rate = float(cost_bps) / 10_000.0
    merged["cost_rate"] = cost_rate
    merged["cost"] = cost_rate * merged[turnover_column]
    merged["net_return"] = merged["gross_return"] - merged["cost"]
    merged["cost_convention"] = convention
    merged["cost_bps"] = float(cost_bps)
    return merged
