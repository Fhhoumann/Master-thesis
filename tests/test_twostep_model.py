from __future__ import annotations

import numpy as np
import pandas as pd

from core.config.schemas import TwoStepModelConfig
from core.cv.splitters import TimeSeriesSplit
from core.models.twostep import TwoStepTrainer


def test_twostep_trainer_returns_fold_diagnostics() -> None:
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-01-31",
            "2020-02-29",
            "2020-02-29",
            "2020-03-31",
            "2020-03-31",
        ]
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "asset_id": [1, 2, 1, 2, 1, 2],
            "target_ret_1m": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03],
            "feature_a": [1.0, -1.0, 1.1, -1.1, 1.2, -1.2],
            "feature_b": [0.5, -0.5, 0.6, -0.6, 0.7, -0.7],
        }
    )
    split = TimeSeriesSplit(
        train_dates=np.array(pd.to_datetime(["2020-01-31", "2020-02-29"])),
        validation_dates=np.array(pd.to_datetime(["2020-03-31"])),
        fold_id=0,
    )
    cfg = TwoStepModelConfig(
        model_type="elasticnet",
        feature_columns=["feature_a", "feature_b"],
        target_column="target_ret_1m",
        params={"alpha": 0.0001, "l1_ratio": 0.1, "max_iter": 10000},
    )

    trainer = TwoStepTrainer(cfg)
    predictions, diagnostics = trainer.fit_predict_with_diagnostics(frame, [split])

    assert not predictions.empty
    assert predictions["fold_id"].nunique() == 1
    assert not diagnostics.empty
    assert diagnostics.iloc[0]["fold_id"] == 0
    assert diagnostics.iloc[0]["model_type"] == "elasticnet"
    assert float(diagnostics.iloc[0]["prediction_std"]) >= 0.0
    assert float(diagnostics.iloc[0]["nonzero_coef_count"]) >= 0.0
