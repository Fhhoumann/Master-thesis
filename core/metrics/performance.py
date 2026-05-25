from __future__ import annotations

import numpy as np
import pandas as pd


def infer_periods_per_year(
    dates: pd.Series | pd.Index | None,
    *,
    default_periods_per_year: float = 12.0,
) -> float:
    if dates is None:
        return float(default_periods_per_year)

    parsed = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if parsed.shape[0] < 2:
        return float(default_periods_per_year)

    unique_dates = np.sort(parsed.unique())
    if unique_dates.shape[0] < 2:
        return float(default_periods_per_year)

    deltas = np.diff(unique_dates).astype("timedelta64[D]").astype(np.int64)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return float(default_periods_per_year)

    median_days = float(np.median(deltas))
    if median_days >= 250:
        return 1.0
    if median_days >= 150:
        return 2.0
    if median_days >= 60:
        return 4.0
    if median_days >= 35:
        return 6.0
    if median_days >= 20:
        return 12.0
    if median_days >= 10:
        return 24.0
    if median_days >= 4:
        return 52.0
    return 252.0


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _align_excess_returns(
    returns: pd.Series,
    dates: pd.Series | pd.Index | None,
    risk_free_frame: pd.DataFrame | None,
) -> pd.Series | None:
    if risk_free_frame is None or dates is None:
        return None
    if "date" not in risk_free_frame.columns or "rf" not in risk_free_frame.columns:
        return None

    ret_dates = pd.to_datetime(pd.Series(dates), errors="coerce")
    if ret_dates.shape[0] != returns.shape[0]:
        return None

    rf = risk_free_frame[["date", "rf"]].copy()
    rf["date"] = pd.to_datetime(rf["date"], errors="coerce")
    rf["rf"] = pd.to_numeric(rf["rf"], errors="coerce")
    rf = rf.dropna(subset=["date", "rf"]).copy()
    if rf.empty:
        return None
    rf["month_key"] = rf["date"].dt.to_period("M")
    rf = rf.sort_values("date").drop_duplicates("month_key", keep="last")

    frame = pd.DataFrame(
        {
            "ret": pd.to_numeric(returns, errors="coerce"),
            "date": ret_dates,
        }
    )
    frame = frame.dropna(subset=["ret", "date"]).copy()
    if frame.empty:
        return None
    frame["month_key"] = frame["date"].dt.to_period("M")
    merged = frame.merge(rf[["month_key", "rf"]], on="month_key", how="left")
    excess = merged["ret"] - merged["rf"]
    excess = excess.dropna()
    return excess if not excess.empty else None


def summarize_returns(
    returns: pd.Series,
    *,
    periods_per_year: int | float = 12,
    dates: pd.Series | pd.Index | None = None,
    risk_free_frame: pd.DataFrame | None = None,
) -> dict[str, float]:
    resolved_periods_per_year = infer_periods_per_year(
        dates,
        default_periods_per_year=float(periods_per_year),
    )
    series = returns.dropna()
    if series.empty:
        return {
            "n_periods": 0.0,
            "periods_per_year": float(resolved_periods_per_year),
            "mean_period_return": 0.0,
            "std_period_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "cumulative_return": 0.0,
        }

    n_periods = float(series.shape[0])
    mean_period_return = float(series.mean())
    std_period_return = float(series.std(ddof=0))
    annual_return = float((1.0 + series).prod() ** (resolved_periods_per_year / n_periods) - 1.0)
    annual_volatility = float(std_period_return * np.sqrt(resolved_periods_per_year))
    excess_series = _align_excess_returns(series, dates, risk_free_frame)
    if excess_series is not None:
        excess_mean = float(excess_series.mean())
        excess_std = float(excess_series.std(ddof=0))
        sharpe = float(np.sqrt(resolved_periods_per_year) * excess_mean / (excess_std + 1e-12))
    else:
        sharpe = float(np.sqrt(resolved_periods_per_year) * mean_period_return / (std_period_return + 1e-12))
    cumulative_return = float((1.0 + series).prod() - 1.0)

    return {
        "n_periods": n_periods,
        "periods_per_year": float(resolved_periods_per_year),
        "mean_period_return": mean_period_return,
        "std_period_return": std_period_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(series),
        "cumulative_return": cumulative_return,
    }


def summarize_backtest_frame(
    frame: pd.DataFrame,
    *,
    risk_free_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"cost_convention", "cost_bps", "net_return"}
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"backtest frame missing required columns: {missing}")

    rows: list[dict[str, float | str]] = []
    for (convention, cost_bps), group in frame.groupby(
        ["cost_convention", "cost_bps"], observed=True
    ):
        date_series = group["date"] if "date" in group.columns else None
        metrics = summarize_returns(group["net_return"], dates=date_series, risk_free_frame=risk_free_frame)
        avg_turnover = float(group.filter(like="turnover_").mean(axis=1).mean())
        rows.append(
            {
                "cost_convention": str(convention),
                "cost_bps": float(cost_bps),
                "avg_turnover": avg_turnover,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["cost_convention", "cost_bps"]).reset_index(drop=True)
