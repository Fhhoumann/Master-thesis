from __future__ import annotations

import numpy as np
import torch

from core.models.onestep import _scores_to_weights


def test_scores_to_weights_respects_cap_and_sum_for_concentrated_scores() -> None:
    scores = torch.tensor([10.0] + [0.0] * 16, dtype=torch.float32)
    weights = _scores_to_weights(scores, weight_clip=0.10, selection_quantile=0.10)

    assert weights.shape == scores.shape
    assert np.isclose(float(weights.sum().item()), 1.0, atol=1e-6)
    assert float(weights.max().item()) <= 0.10 + 1e-6
    assert float(weights.min().item()) >= -1e-9
