from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.config.schemas import TwoStepModelConfig
from core.cv.splitters import TimeSeriesSplit


def _build_model(config: TwoStepModelConfig):
    params = dict(config.params)
    if config.model_type == "elasticnet":
        params.pop("alpha_grid", None)
        defaults = {"alpha": 0.01, "l1_ratio": 0.5, "random_state": 42, "max_iter": 10000}
        defaults.update(params)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(**defaults)),
            ]
        )
    if config.model_type == "random_forest":
        defaults = {"n_estimators": 400, "max_depth": 6, "random_state": 42, "n_jobs": -1}
        defaults.update(params)
        return RandomForestRegressor(**defaults)

    defaults = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    defaults.update(params)
    return LGBMRegressor(**defaults)


def _valid_mask(frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> pd.Series:
    finite_features = np.isfinite(frame[feature_columns]).all(axis=1)
    finite_target = np.isfinite(frame[target_column])
    return finite_features & finite_target & frame["asset_id"].notna() & frame["date"].notna()


def _select_elasticnet_alpha(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    params: dict[str, object],
) -> float | None:
    alpha_grid = params.get("alpha_grid")
    if not isinstance(alpha_grid, list) or not alpha_grid:
        return None

    candidates: list[float] = []
    for value in alpha_grid:
        try:
            alpha = float(value)
        except (TypeError, ValueError):
            continue
        if alpha > 0:
            candidates.append(alpha)
    if not candidates:
        return None

    unique_dates = np.sort(train["date"].dropna().unique())
    if unique_dates.size < 24:
        return min(candidates)

    holdout_months = min(12, max(6, unique_dates.size // 5))
    split_idx = unique_dates.size - holdout_months
    if split_idx <= 0:
        return min(candidates)

    fit_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:]
    fit_data = train[train["date"].isin(fit_dates)]
    val_data = train[train["date"].isin(val_dates)]
    if fit_data.empty or val_data.empty:
        return min(candidates)

    best_alpha: float | None = None
    best_mse = float("inf")
    for alpha in candidates:
        trial_params = dict(params)
        trial_params["alpha"] = alpha
        trial_params.pop("alpha_grid", None)
        trial_cfg = TwoStepModelConfig(
            model_type="elasticnet",
            params=trial_params,
            feature_columns=feature_columns,
            target_column=target_column,
        )
        model = _build_model(trial_cfg)
        model.fit(fit_data[feature_columns], fit_data[target_column])
        pred = model.predict(val_data[feature_columns])
        mse = float(np.mean(np.square(pred - val_data[target_column].to_numpy(np.float64))))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    return best_alpha


@dataclass(slots=True)
class TwoStepTrainer:
    config: TwoStepModelConfig

    def fit_predict(self, frame: pd.DataFrame, splits: list[TimeSeriesSplit]) -> pd.DataFrame:
        predictions, _ = self.fit_predict_with_diagnostics(frame, splits)
        return predictions

    def fit_predict_with_diagnostics(
        self,
        frame: pd.DataFrame,
        splits: list[TimeSeriesSplit],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_columns = self.config.feature_columns
        if not feature_columns:
            raise ValueError("TwoStepModelConfig.feature_columns must be provided.")

        target_column = self.config.target_column
        available = _valid_mask(frame, feature_columns, target_column)
        work = frame.loc[available, ["date", "asset_id", target_column, *feature_columns]].copy()

        predictions: list[pd.DataFrame] = []
        diagnostics: list[dict[str, float | int | str]] = []
        for split in splits:
            train_mask = work["date"].isin(split.train_dates)
            valid_mask = work["date"].isin(split.validation_dates)
            train = work.loc[train_mask]
            valid = work.loc[valid_mask]
            if train.empty or valid.empty:
                continue

            model_config = self.config
            selected_alpha = float("nan")
            if self.config.model_type == "elasticnet":
                chosen = _select_elasticnet_alpha(
                    train,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    params=self.config.params,
                )
                if chosen is not None:
                    selected_alpha = float(chosen)
                    tuned_params = dict(self.config.params)
                    tuned_params["alpha"] = selected_alpha
                    tuned_params.pop("alpha_grid", None)
                    model_config = TwoStepModelConfig(
                        model_type="elasticnet",
                        params=tuned_params,
                        feature_columns=feature_columns,
                        target_column=target_column,
                    )

            model = _build_model(model_config)
            model.fit(train[feature_columns], train[target_column])
            pred_values = model.predict(valid[feature_columns])
            pred_std = float(np.std(pred_values))

            nonzero_coef_count = float("nan")
            coef_l1_norm = float("nan")
            intercept = float("nan")
            if self.config.model_type == "elasticnet" and isinstance(model, Pipeline):
                estimator = model.named_steps.get("model")
                if estimator is not None and hasattr(estimator, "coef_"):
                    coef = np.asarray(estimator.coef_, dtype=np.float64)
                    nonzero_coef_count = float(np.sum(np.abs(coef) > 1e-12))
                    coef_l1_norm = float(np.abs(coef).sum())
                    intercept = float(getattr(estimator, "intercept_", float("nan")))

            diagnostics.append(
                {
                    "fold_id": int(split.fold_id),
                    "model_type": self.config.model_type,
                    "train_rows": int(train.shape[0]),
                    "valid_rows": int(valid.shape[0]),
                    "train_dates": int(len(split.train_dates)),
                    "valid_dates": int(len(split.validation_dates)),
                    "prediction_std": pred_std,
                    "nonzero_coef_count": nonzero_coef_count,
                    "coef_l1_norm": coef_l1_norm,
                    "intercept": intercept,
                    "selected_alpha": selected_alpha,
                }
            )

            fold = valid[["date", "asset_id", target_column]].copy()
            fold["prediction"] = pred_values
            fold["fold_id"] = split.fold_id
            fold["model_type"] = self.config.model_type
            predictions.append(fold)

        if not predictions:
            return pd.DataFrame(
                columns=["date", "asset_id", target_column, "prediction", "fold_id", "model_type"]
            ), pd.DataFrame(
                columns=[
                    "fold_id",
                    "model_type",
                    "train_rows",
                    "valid_rows",
                    "train_dates",
                    "valid_dates",
                    "prediction_std",
                    "nonzero_coef_count",
                    "coef_l1_norm",
                    "intercept",
                    "selected_alpha",
                ]
            )
        return pd.concat(predictions, ignore_index=True), pd.DataFrame(diagnostics)
