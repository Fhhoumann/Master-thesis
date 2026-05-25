from __future__ import annotations

import pandas as pd

from apps.evaluate.main import _build_final_comparison


def test_build_final_comparison_prefers_requested_cost_and_convention() -> None:
    frame = pd.DataFrame(
        {
            "experiment": ["a", "a", "b", "b"],
            "pipeline": ["one", "one", "one", "one"],
            "model_type": ["m1", "m1", "m2", "m2"],
            "cost_convention": ["one_way", "round_trip", "one_way", "one_way"],
            "cost_bps": [0.0, 0.0, 25.0, 0.0],
            "sharpe": [0.1, 0.2, 0.3, 0.2],
            "annual_return": [0.01, 0.02, 0.03, 0.02],
        }
    )

    out = _build_final_comparison(
        frame,
        include_experiments=None,
        include_pipelines=None,
        preferred_cost_bps=0.0,
        preferred_convention="one_way",
    )

    assert out.shape[0] == 2
    assert out.loc[out["experiment"] == "a", "cost_convention"].iloc[0] == "one_way"
    assert float(out.loc[out["experiment"] == "a", "cost_bps"].iloc[0]) == 0.0
    assert float(out.loc[out["experiment"] == "b", "cost_bps"].iloc[0]) == 0.0
