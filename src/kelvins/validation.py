"""Local validation on the training events.

Why it matters here: competitors reported that their local scores did not predict the
leaderboard (paper, section 6). Two reasons: (1) only 66 eligible high-risk training
events, so any fold contains ~13 of them; (2) the test set has a much higher share of
high-risk events (6.9 % vs 0.8 %) and a different mission mix.
Use repeated stratified K-fold and always compare to the latest-risk baseline *on the
same folds*: the relative gain is far more stable than the raw score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from .data import EventDataset
from .metrics import report
from .models.baselines import LatestRisk


def cv_splits(ds: EventDataset, n_splits: int = 5, n_repeats: int = 3, seed: int = 0):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    ids = ds.event_ids.to_numpy()
    for tr, va in rskf.split(ids, ds.is_high.to_numpy()):
        yield ids[tr], ids[va]


def cross_validate(make_model, ds: EventDataset, **kw) -> pd.DataFrame:
    """Fit ``make_model()`` on each training fold; report the model and the baseline per fold."""
    rows = []
    for k, (tr, va) in enumerate(cv_splits(ds, **kw)):
        d_tr, d_va = ds.subset(tr), ds.subset(va)
        pred = make_model().fit(d_tr).predict(d_va)
        base = LatestRisk().fit(d_tr).predict(d_va)
        r, b = report(d_va.y, pred), report(d_va.y, base)
        rows.append({"fold": k, **r, "baseline_score": b["score"],
                     "gain_vs_lrp": 1 - r["score"] / b["score"]})
    return pd.DataFrame(rows)


def summarize_cv(df: pd.DataFrame) -> dict:
    return {
        "cv_score": df.score.mean(),
        "cv_score_std": df.score.std(),
        "cv_gain_vs_lrp": df.gain_vs_lrp.mean(),
        "cv_gain_std": df.gain_vs_lrp.std(),
        "cv_frac_folds_better": float((df.gain_vs_lrp > 0).mean()),
    }
