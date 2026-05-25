from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from core.config import RegimeConfig

FactorModel = Literal["capm", "ff3", "carhart4", "ff5"]

MODEL_FACTORS: dict[FactorModel, tuple[str, ...]] = {
    # CAPM market factor built from thesis universe (top-500 panel), not Ken French market factor.
    "capm": ("mkt_rf",),
    "ff3": ("mkt_rf", "smb", "hml"),
    "carhart4": ("mkt_rf", "smb", "hml", "umd"),
    "ff5": ("mkt_rf", "smb", "hml", "rmw", "cma"),
}

ALL_BETA_COLUMNS = (
    "beta_mkt_rf",
    "beta_smb",
    "beta_hml",
    "beta_umd",
    "beta_rmw",
    "beta_cma",
)


def _assign_regime(dates: pd.Series, regimes: list[RegimeConfig]) -> pd.Series:
    labels = pd.Series(data="outside_regimes", index=dates.index, dtype="object")
    parsed = pd.to_datetime(dates, errors="coerce")
    for regime in regimes:
        start = pd.Timestamp(regime.start)
        end = pd.Timestamp(regime.end)
        mask = parsed.between(start, end, inclusive="both")
        labels.loc[mask] = regime.name
    return labels


def _empty_result(status: str, n_obs: int) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "status": status,
        "n_obs": int(n_obs),
        "alpha_monthly": float("nan"),
        "alpha_annualized": float("nan"),
        "alpha_tstat": float("nan"),
        "r2": float("nan"),
    }
    for beta_col in ALL_BETA_COLUMNS:
        out[beta_col] = float("nan")
    return out


def run_factor_regressions(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: FactorModel,
    min_obs: int = 24,
) -> dict[str, float | int | str]:
    if model not in MODEL_FACTORS:
        raise ValueError(f"Unsupported model: {model}")

    factor_cols = MODEL_FACTORS[model]
    required = {"date", "rf", *factor_cols}
    missing = [col for col in required if col not in factors.columns]
    if missing:
        raise ValueError(f"factors is missing required columns: {missing}")

    ret_frame = returns.dropna().rename("period_return").to_frame().reset_index()
    if ret_frame.empty:
        return _empty_result("insufficient_obs", 0)

    date_col = ret_frame.columns[0]
    ret_frame = ret_frame.rename(columns={date_col: "date"})
    ret_frame["date"] = pd.to_datetime(ret_frame["date"], errors="coerce")
    ret_frame = ret_frame.dropna(subset=["date"]).copy()
    ret_frame["month_key"] = ret_frame["date"].dt.to_period("M")

    fac = factors[["date", "rf", *factor_cols]].copy()
    fac["date"] = pd.to_datetime(fac["date"], errors="coerce")
    fac["month_key"] = fac["date"].dt.to_period("M")
    for col in ("rf", *factor_cols):
        fac[col] = pd.to_numeric(fac[col], errors="coerce")
    fac = fac.dropna(subset=["month_key", "rf", *factor_cols]).drop_duplicates(
        "month_key", keep="last"
    )

    merged = ret_frame.merge(
        fac[["month_key", "rf", *factor_cols]],
        on="month_key",
        how="inner",
    ).dropna()
    n_obs = int(merged.shape[0])
    if n_obs < min_obs:
        return _empty_result("insufficient_obs", n_obs)

    y = (merged["period_return"] - merged["rf"]).to_numpy(np.float64)
    x = merged[list(factor_cols)].to_numpy(np.float64)
    x = np.column_stack([np.ones(n_obs, dtype=np.float64), x])

    k = x.shape[1]
    if n_obs <= k:
        return _empty_result("singular_design", n_obs)

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta

    # HC1 robust covariance
    xe = x * residual[:, None]
    meat = xe.T @ xe
    cov = (n_obs / (n_obs - k)) * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    alpha_monthly = float(beta[0])
    alpha_annualized = float((1.0 + alpha_monthly) ** 12 - 1.0)
    alpha_tstat = float(alpha_monthly / (se[0] + 1e-12))

    ssr = float(np.sum(np.square(residual)))
    centered = y - float(np.mean(y))
    sst = float(np.sum(np.square(centered)))
    r2 = float(1.0 - ssr / sst) if sst > 0 else float("nan")

    out: dict[str, float | int | str] = {
        "status": "ok",
        "n_obs": n_obs,
        "alpha_monthly": alpha_monthly,
        "alpha_annualized": alpha_annualized,
        "alpha_tstat": alpha_tstat,
        "r2": r2,
        "beta_mkt_rf": float("nan"),
    }
    for idx, name in enumerate(factor_cols, start=1):
        out[f"beta_{name}"] = float(beta[idx])
    return out


def compute_attribution_table(
    returns_frame: pd.DataFrame,
    factors: pd.DataFrame,
    regimes: list[RegimeConfig],
    models: tuple[FactorModel, ...] = ("capm",),
    min_obs: int = 24,
) -> pd.DataFrame:
    required = {"date", "run_key", "return_type", "period_return"}
    missing = [col for col in required if col not in returns_frame.columns]
    if missing:
        raise ValueError(f"returns_frame is missing required columns: {missing}")

    frame = returns_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["period_return"] = pd.to_numeric(frame["period_return"], errors="coerce")
    frame = frame.dropna(subset=["date", "run_key", "return_type", "period_return"]).copy()
    if "cost_convention" not in frame.columns:
        frame["cost_convention"] = "none"
    if "cost_bps" not in frame.columns:
        frame["cost_bps"] = 0.0
    frame["cost_bps"] = pd.to_numeric(frame["cost_bps"], errors="coerce").fillna(0.0)
    frame["regime"] = _assign_regime(frame["date"], regimes)

    rows: list[dict[str, float | int | str]] = []
    group_cols = ["run_key", "return_type", "cost_convention", "cost_bps"]
    for keys, group in frame.groupby(group_cols, observed=True):
        run_key, return_type, cost_convention, cost_bps = keys
        samples: dict[str, pd.Series] = {"full_sample": group.set_index("date")["period_return"]}
        for regime in regimes:
            regime_slice = group[group["regime"] == regime.name]
            samples[regime.name] = regime_slice.set_index("date")["period_return"]

        for regime_name, series in samples.items():
            for model in models:
                stats = run_factor_regressions(series, factors, model=model, min_obs=min_obs)
                rows.append(
                    {
                        "run_key": str(run_key),
                        "return_type": str(return_type),
                        "cost_convention": str(cost_convention),
                        "cost_bps": float(cost_bps),
                        "regime": str(regime_name),
                        "model": str(model),
                        **stats,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "run_key",
                "return_type",
                "cost_convention",
                "cost_bps",
                "regime",
                "model",
                "status",
                "n_obs",
                "alpha_monthly",
                "alpha_annualized",
                "alpha_tstat",
                "r2",
                *ALL_BETA_COLUMNS,
            ]
        )
    return out.sort_values(
        ["run_key", "return_type", "cost_convention", "cost_bps", "regime", "model"]
    ).reset_index(drop=True)
