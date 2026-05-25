from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SharpeDifferenceTestResult:
    strategy_a: str
    strategy_b: str
    n_obs: int
    block_size: int
    bootstrap_resamples: int
    delta_sharpe: float
    se_delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str


@dataclass(slots=True)
class BlockSizeCalibrationResult:
    selected_block_size: int
    target_coverage: float
    residual_bootstrap_avg_block_size: float
    pseudo_sequences: int
    bootstrap_resamples: int
    calibration_table: pd.DataFrame


def prepare_pairwise_excess_returns(
    strategy_a: pd.DataFrame,
    strategy_b: pd.DataFrame,
    risk_free_frame: pd.DataFrame,
    *,
    return_column: str = "net_return",
    cost_convention: str = "one_way",
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    required = {"date", return_column, "cost_convention", "cost_bps"}
    for name, frame in {"strategy_a": strategy_a, "strategy_b": strategy_b}.items():
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    rf_required = {"date", "rf"}
    rf_missing = [col for col in rf_required if col not in risk_free_frame.columns]
    if rf_missing:
        raise ValueError(f"risk_free_frame is missing required columns: {rf_missing}")

    def _prep(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["cost_bps"] = pd.to_numeric(out["cost_bps"], errors="coerce")
        out[return_column] = pd.to_numeric(out[return_column], errors="coerce")
        out = out[
            (out["cost_convention"] == cost_convention)
            & np.isclose(out["cost_bps"], float(cost_bps))
        ].copy()
        out = out.dropna(subset=["date", return_column]).sort_values("date")
        out["month_key"] = out["date"].dt.to_period("M")
        out = out.drop_duplicates(subset=["month_key"], keep="last")
        return out[["date", "month_key", return_column]].rename(columns={return_column: value_name})

    rf = risk_free_frame.copy()
    rf["date"] = pd.to_datetime(rf["date"], errors="coerce")
    rf["rf"] = pd.to_numeric(rf["rf"], errors="coerce")
    rf = rf.dropna(subset=["date", "rf"]).sort_values("date")
    rf["month_key"] = rf["date"].dt.to_period("M")
    rf = rf.drop_duplicates(subset=["month_key"], keep="last")

    a = _prep(strategy_a, "ret_a")
    b = _prep(strategy_b, "ret_b")
    merged = a.merge(b[["month_key", "ret_b"]], on="month_key", how="inner")
    merged = merged.merge(rf[["month_key", "rf"]], on="month_key", how="inner")
    merged = merged.dropna(subset=["ret_a", "ret_b", "rf"]).copy()
    merged["excess_a"] = merged["ret_a"] - merged["rf"]
    merged["excess_b"] = merged["ret_b"] - merged["rf"]
    return merged[["date", "month_key", "excess_a", "excess_b"]].sort_values("date").reset_index(drop=True)


def sharpe_ratio(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(values.mean() / (values.std(ddof=0) + 1e-12))


def sharpe_difference(excess_returns: np.ndarray) -> float:
    data = np.asarray(excess_returns, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("excess_returns must have shape (T, 2)")
    return float(sharpe_ratio(data[:, 0]) - sharpe_ratio(data[:, 1]))


def _gradient_of_delta(mu_i: float, mu_n: float, gamma_i: float, gamma_n: float) -> np.ndarray:
    sigma_i_sq = max(gamma_i - mu_i * mu_i, 1e-12)
    sigma_n_sq = max(gamma_n - mu_n * mu_n, 1e-12)
    sigma_i = np.sqrt(sigma_i_sq)
    sigma_n = np.sqrt(sigma_n_sq)
    return np.array(
        [
            gamma_i / (sigma_i_sq ** 1.5),
            -gamma_n / (sigma_n_sq ** 1.5),
            -0.5 * mu_i / (sigma_i_sq ** 1.5),
            0.5 * mu_n / (sigma_n_sq ** 1.5),
        ],
        dtype=np.float64,
    )


def _sample_u(data: np.ndarray) -> np.ndarray:
    return np.array(
        [
            data[:, 0].mean(),
            data[:, 1].mean(),
            np.mean(np.square(data[:, 0])),
            np.mean(np.square(data[:, 1])),
        ],
        dtype=np.float64,
    )


def _y_vectors(data: np.ndarray, u_hat: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            data[:, 0] - u_hat[0],
            data[:, 1] - u_hat[1],
            np.square(data[:, 0]) - u_hat[2],
            np.square(data[:, 1]) - u_hat[3],
        ]
    ).astype(np.float64)


def _fit_var1(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, k = values.shape
    if t < 3:
        return np.zeros((k,), dtype=np.float64), np.zeros((k, k), dtype=np.float64)

    y = values[1:]
    x = values[:-1]
    x_design = np.column_stack([np.ones(t - 1, dtype=np.float64), x])
    beta = np.linalg.pinv(x_design.T @ x_design) @ x_design.T @ y
    intercept = beta[0]
    phi = beta[1:].T
    return intercept.astype(np.float64), phi.astype(np.float64)


def _var1_residuals(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=np.float64)
    t, k = data.shape
    if t < 3:
        intercept = np.zeros((k,), dtype=np.float64)
        phi = np.zeros((k, k), dtype=np.float64)
        residuals = data[1:] - data[:-1] if t > 1 else np.zeros((0, k), dtype=np.float64)
        return intercept, phi, residuals

    intercept, phi = _fit_var1(data)
    fitted = intercept + data[:-1] @ phi.T
    residuals = data[1:] - fitted
    return intercept, phi, residuals.astype(np.float64)


def _quadratic_spectral_kernel(x: float) -> float:
    z = abs(float(x))
    if z < 1e-12:
        return 1.0
    scale = 6.0 * np.pi * z / 5.0
    return float(25.0 / (12.0 * np.pi * np.pi * z * z) * ((np.sin(scale) / scale) - np.cos(scale)))


def _hac_long_run_covariance(
    y: np.ndarray,
    *,
    prewhiten: bool = True,
    bandwidth: float | None = None,
) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64)
    t, k = values.shape
    if t < 2:
        return np.eye(k, dtype=np.float64) * 1e-12

    phi = np.zeros((k, k), dtype=np.float64)
    work = values.copy()
    if prewhiten and t >= 3:
        _, phi = _fit_var1(values)
        work = values[1:] - values[:-1] @ phi.T
        t = work.shape[0]

    if bandwidth is None:
        bandwidth = max(1.0, 1.3221 * (t ** 0.2))

    gamma0 = (work.T @ work) / t
    omega = gamma0.copy()
    max_lag = t - 1
    for lag in range(1, max_lag + 1):
        weight = _quadratic_spectral_kernel(lag / bandwidth)
        if abs(weight) < 1e-12:
            continue
        gamma = (work[lag:].T @ work[:-lag]) / t
        omega = omega + weight * (gamma + gamma.T)

    if prewhiten and t >= 2:
        eye = np.eye(phi.shape[0], dtype=np.float64)
        adjust = np.linalg.pinv(eye - phi)
        omega = adjust @ omega @ adjust.T

    omega = 0.5 * (omega + omega.T)
    return omega


def estimate_sharpe_difference_standard_error(
    excess_returns: np.ndarray,
    *,
    prewhiten: bool = True,
) -> float:
    data = np.asarray(excess_returns, dtype=np.float64)
    t = data.shape[0]
    if t < 2:
        return float("nan")
    u_hat = _sample_u(data)
    y = _y_vectors(data, u_hat)
    psi_hat = _hac_long_run_covariance(y, prewhiten=prewhiten)
    grad = _gradient_of_delta(*u_hat)
    variance = float(grad @ psi_hat @ grad / t)
    return float(np.sqrt(max(variance, 1e-16)))


def circular_block_bootstrap_sample(
    data: np.ndarray,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    t = values.shape[0]
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n_blocks = int(np.ceil(t / block_size))
    starts = rng.integers(0, t, size=n_blocks)
    pieces: list[np.ndarray] = []
    for start in starts:
        idx = (start + np.arange(block_size)) % t
        pieces.append(values[idx])
    out = np.vstack(pieces)
    return out[:t]


def stationary_bootstrap_indices(
    n_obs: int,
    *,
    avg_block_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if avg_block_size <= 0:
        raise ValueError("avg_block_size must be positive")

    p = min(1.0, 1.0 / float(avg_block_size))
    idx = np.empty(n_obs, dtype=np.int64)
    idx[0] = int(rng.integers(0, n_obs))
    for t in range(1, n_obs):
        if float(rng.random()) < p:
            idx[t] = int(rng.integers(0, n_obs))
        else:
            idx[t] = (idx[t - 1] + 1) % n_obs
    return idx


def generate_var1_pseudo_sequence(
    values: np.ndarray,
    *,
    residual_bootstrap_avg_block_size: float = 5.0,
    rng: np.random.Generator,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    t, k = data.shape
    if t < 3:
        return data.copy()

    intercept, phi, residuals = _var1_residuals(data)
    if residuals.shape[0] == 0:
        return data.copy()

    idx = stationary_bootstrap_indices(
        residuals.shape[0],
        avg_block_size=residual_bootstrap_avg_block_size,
        rng=rng,
    )
    boot_resid = residuals[idx]

    pseudo = np.empty_like(data)
    pseudo[0] = data[0]
    for i in range(1, t):
        pseudo[i] = intercept + phi @ pseudo[i - 1] + boot_resid[i - 1]
    return pseudo


def bootstrap_standard_error_from_blocks(
    excess_returns_bootstrap: np.ndarray,
    *,
    block_size: int,
) -> float:
    data = np.asarray(excess_returns_bootstrap, dtype=np.float64)
    t = data.shape[0]
    l = t // block_size
    if l <= 0:
        return float("nan")
    trimmed = data[: l * block_size]
    u_hat = _sample_u(trimmed)
    y = _y_vectors(trimmed, u_hat)
    blocks = y.reshape(l, block_size, y.shape[1])
    f = blocks.sum(axis=1) / np.sqrt(float(block_size))
    psi_hat = (f.T @ f) / l
    grad = _gradient_of_delta(*u_hat)
    variance = float(grad @ psi_hat @ grad / trimmed.shape[0])
    return float(np.sqrt(max(variance, 1e-16)))


def ledoit_wolf_sharpe_difference_test(
    excess_returns: pd.DataFrame | np.ndarray,
    *,
    strategy_a: str,
    strategy_b: str,
    block_size: int = 6,
    bootstrap_resamples: int = 999,
    alpha: float = 0.05,
    seed: int = 42,
) -> SharpeDifferenceTestResult:
    if isinstance(excess_returns, pd.DataFrame):
        required = {"excess_a", "excess_b"}
        missing = [col for col in required if col not in excess_returns.columns]
        if missing:
            raise ValueError(f"excess_returns is missing required columns: {missing}")
        data = excess_returns[["excess_a", "excess_b"]].to_numpy(np.float64)
    else:
        data = np.asarray(excess_returns, dtype=np.float64)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("excess_returns must have shape (T, 2)")
    if data.shape[0] < max(24, block_size * 2):
        raise ValueError("Not enough observations for robust Sharpe-difference testing.")

    delta_hat = sharpe_difference(data)
    se_hat = estimate_sharpe_difference_standard_error(data, prewhiten=True)
    d_stat = abs(delta_hat) / (se_hat + 1e-12)

    rng = np.random.default_rng(seed)
    centered_stats = np.empty(int(bootstrap_resamples), dtype=np.float64)
    for m in range(int(bootstrap_resamples)):
        boot = circular_block_bootstrap_sample(data, block_size=block_size, rng=rng)
        delta_boot = sharpe_difference(boot)
        se_boot = bootstrap_standard_error_from_blocks(boot, block_size=block_size)
        centered_stats[m] = abs(delta_boot - delta_hat) / (se_boot + 1e-12)

    z_star = float(np.quantile(centered_stats, 1.0 - float(alpha)))
    p_value = float((np.count_nonzero(centered_stats >= d_stat) + 1.0) / (bootstrap_resamples + 1.0))
    ci_lower = float(delta_hat - z_star * se_hat)
    ci_upper = float(delta_hat + z_star * se_hat)

    return SharpeDifferenceTestResult(
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        n_obs=int(data.shape[0]),
        block_size=int(block_size),
        bootstrap_resamples=int(bootstrap_resamples),
        delta_sharpe=float(delta_hat),
        se_delta=float(se_hat),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        method="Ledoit-Wolf style studentized circular block bootstrap",
    )


def calibrate_block_size_algorithm_3_1(
    excess_returns: pd.DataFrame | np.ndarray,
    *,
    candidate_block_sizes: list[int] | tuple[int, ...] = (1, 2, 4, 6, 8, 10),
    alpha: float = 0.05,
    pseudo_sequences: int = 199,
    bootstrap_resamples: int = 199,
    residual_bootstrap_avg_block_size: float = 5.0,
    seed: int = 42,
) -> BlockSizeCalibrationResult:
    if isinstance(excess_returns, pd.DataFrame):
        required = {"excess_a", "excess_b"}
        missing = [col for col in required if col not in excess_returns.columns]
        if missing:
            raise ValueError(f"excess_returns is missing required columns: {missing}")
        data = excess_returns[["excess_a", "excess_b"]].to_numpy(np.float64)
    else:
        data = np.asarray(excess_returns, dtype=np.float64)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("excess_returns must have shape (T, 2)")
    if data.shape[0] < 24:
        raise ValueError("Not enough observations for block-size calibration.")

    delta_hat = sharpe_difference(data)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    target_coverage = 1.0 - float(alpha)

    for block_size in candidate_block_sizes:
        contains = 0
        for k in range(int(pseudo_sequences)):
            pseudo = generate_var1_pseudo_sequence(
                data,
                residual_bootstrap_avg_block_size=residual_bootstrap_avg_block_size,
                rng=rng,
            )
            pseudo_delta = sharpe_difference(pseudo)
            result = ledoit_wolf_sharpe_difference_test(
                pseudo,
                strategy_a="pseudo_a",
                strategy_b="pseudo_b",
                block_size=int(block_size),
                bootstrap_resamples=int(bootstrap_resamples),
                alpha=float(alpha),
                seed=seed + 1000 * (k + 1) + int(block_size),
            )
            if result.ci_lower <= pseudo_delta <= result.ci_upper:
                contains += 1

        g_hat = float(contains / float(pseudo_sequences))
        rows.append(
            {
                "block_size": int(block_size),
                "g_hat": g_hat,
                "target_coverage": target_coverage,
                "abs_gap": abs(g_hat - target_coverage),
                "original_delta_hat": float(delta_hat),
            }
        )

    table = pd.DataFrame(rows).sort_values(["abs_gap", "block_size"]).reset_index(drop=True)
    selected = int(table.loc[0, "block_size"])
    return BlockSizeCalibrationResult(
        selected_block_size=selected,
        target_coverage=target_coverage,
        residual_bootstrap_avg_block_size=float(residual_bootstrap_avg_block_size),
        pseudo_sequences=int(pseudo_sequences),
        bootstrap_resamples=int(bootstrap_resamples),
        calibration_table=table,
    )


def build_pairwise_sharpe_test_table(
    strategies: dict[str, pd.DataFrame],
    risk_free_frame: pd.DataFrame,
    *,
    strategy_order: list[str],
    cost_convention: str = "one_way",
    cost_bps: float = 0.0,
    block_size: int = 6,
    bootstrap_resamples: int = 999,
    alpha: float = 0.05,
    seed: int = 42,
    calibrate_block_size: bool = False,
    candidate_block_sizes: list[int] | tuple[int, ...] = (1, 2, 4, 6, 8, 10),
    calibration_pseudo_sequences: int = 199,
    calibration_bootstrap_resamples: int = 199,
    residual_bootstrap_avg_block_size: float = 5.0,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for i, strategy_a in enumerate(strategy_order):
        for strategy_b in strategy_order[i + 1 :]:
            paired = prepare_pairwise_excess_returns(
                strategies[strategy_a],
                strategies[strategy_b],
                risk_free_frame,
                cost_convention=cost_convention,
                cost_bps=cost_bps,
            )
            resolved_block_size = int(block_size)
            calibration_result: BlockSizeCalibrationResult | None = None
            if calibrate_block_size:
                calibration_result = calibrate_block_size_algorithm_3_1(
                    paired,
                    candidate_block_sizes=candidate_block_sizes,
                    alpha=alpha,
                    pseudo_sequences=calibration_pseudo_sequences,
                    bootstrap_resamples=calibration_bootstrap_resamples,
                    residual_bootstrap_avg_block_size=residual_bootstrap_avg_block_size,
                    seed=seed + i,
                )
                resolved_block_size = int(calibration_result.selected_block_size)
            result = ledoit_wolf_sharpe_difference_test(
                paired,
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                block_size=resolved_block_size,
                bootstrap_resamples=bootstrap_resamples,
                alpha=alpha,
                seed=seed + i,
            )
            row = asdict(result)
            row["block_size_calibrated"] = bool(calibrate_block_size)
            row["candidate_block_sizes"] = ",".join(str(int(b)) for b in candidate_block_sizes)
            row["calibration_target_coverage"] = float(1.0 - alpha) if calibrate_block_size else float("nan")
            row["calibration_pseudo_sequences"] = (
                int(calibration_pseudo_sequences) if calibrate_block_size else int(0)
            )
            row["calibration_bootstrap_resamples"] = (
                int(calibration_bootstrap_resamples) if calibrate_block_size else int(0)
            )
            row["residual_bootstrap_avg_block_size"] = (
                float(residual_bootstrap_avg_block_size) if calibrate_block_size else float("nan")
            )
            if calibration_result is not None:
                best_row = calibration_result.calibration_table.iloc[0]
                row["calibration_g_hat"] = float(best_row["g_hat"])
                row["calibration_abs_gap"] = float(best_row["abs_gap"])
            else:
                row["calibration_g_hat"] = float("nan")
                row["calibration_abs_gap"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)
