from __future__ import annotations

import numpy as np
import pandas as pd

from core.costs import apply_transaction_costs, compute_turnover


def test_turnover_and_cost_application() -> None:
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29"), pd.Timestamp("2020-02-29")],
            "asset_id": [1, 2, 1, 2],
            "weight": [0.5, -0.5, 0.2, -0.2],
        }
    )
    turnover = compute_turnover(weights)
    feb = turnover.loc[turnover["date"] == pd.Timestamp("2020-02-29")].iloc[0]
    assert np.isclose(feb["turnover_one_way"], 0.3)
    assert np.isclose(feb["turnover_round_trip"], 0.6)

    returns = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "gross_return": [0.01, 0.02],
        }
    )
    net = apply_transaction_costs(returns, turnover, cost_bps=10.0, convention="one_way")
    feb_net = net.loc[net["date"] == pd.Timestamp("2020-02-29"), "net_return"].iloc[0]
    assert np.isclose(feb_net, 0.0197)

