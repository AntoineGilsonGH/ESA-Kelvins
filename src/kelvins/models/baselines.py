"""Non-learned baselines from the challenge paper (Uriot et al., 2020).

Official test scores to reproduce:
    ConstantRisk(-5)  L = 2.5
    LatestRisk        L = 0.694  (MSE_HR 0.513, F2 0.739)
    SescRules         L = 0.569  (MSE_HR 0.414, F2 0.728)  steps 0-6 of the winning team

Each sesc step taken alone reproduces Table 4 of the paper to the third decimal; the full
combination gives 0.582 here instead of 0.569 because the paper does not fully specify
how overlapping rules are combined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CLIP_LOW, HIGH_RISK
from ..data import EventDataset


class ConstantRisk:
    """CRP: always predict the same risk (-5 by default)."""

    def __init__(self, value: float = -5.0):
        self.value = value

    def fit(self, ds: EventDataset):
        return self

    def predict(self, ds: EventDataset) -> pd.Series:
        return pd.Series(self.value, index=ds.event_ids, name="risk")


class LatestRisk:
    """LRP: naive forecast, the final risk equals the latest known risk r_-2.

    This is optimal for a random walk, and hard to beat here: only 12 of 97 teams did.
    """

    def fit(self, ds: EventDataset):
        return self

    def predict(self, ds: EventDataset) -> pd.Series:
        r = ds.latest_risk.to_numpy()
        return pd.Series(np.where(r >= HIGH_RISK, r, CLIP_LOW), index=ds.event_ids, name="risk")


class SescRules:
    """Hand-made rules of the winning team ``sesc`` (paper, eq. 2, steps 0-6).

    Not machine learning: thresholds were tuned against the metric and the leaderboard.
    1. promote borderline events (slightly below -6) to high risk: F2 rewards recall;
    2. demote events that look safe (payload chaser, small target, large miss distance);
    3. lower very high risks, since risk tends to decrease as TCA approaches.
    """

    def __init__(self, steps=(0, 1, 2, 3, 4, 5, 6)):
        self.steps = set(steps)

    def fit(self, ds: EventDataset):
        return self

    def predict(self, ds: EventDataset) -> pd.Series:
        last = ds.last
        r2 = last["risk"].to_numpy()
        out = np.where(r2 >= HIGH_RISK, r2, CLIP_LOW)
        s = self.steps
        if 0 in s:
            out = np.where((r2 >= -6.04) & (r2 < -6.00), -5.95, out)
        if 1 in s:
            out = np.where((r2 >= -6.40) & (r2 < -6.04), -5.60, out)
        if 2 in s:
            out = np.where((r2 >= -7.30) & (r2 < -6.40), -5.00, out)
        if 3 in s:
            out = np.where(last["c_object_type"].to_numpy() == "PAYLOAD", -6.00001, out)
        if 4 in s:
            out = np.where(last["t_span"].to_numpy() < 0.5, -6.00001, out)
        if 5 in s:
            out = np.where(last["miss_distance"].to_numpy() > 30000, -6.00001, out)
        if 6 in s:
            out = np.where((r2 >= -4.0) & (r2 < -3.5), -4.0, out)
            out = np.where(r2 >= -3.5, -3.5, out)
        return pd.Series(out, index=ds.event_ids, name="risk")
