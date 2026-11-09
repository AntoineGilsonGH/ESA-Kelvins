"""Official challenge metric:  L = MSE_HR / F2   (lower is better).

* F2   : F-beta score (beta = 2, recall weighs 4x precision) of the binary task
         "final risk >= -6", computed on all events.
* MSE_HR: mean squared error of the predicted log-risk, computed only on the events
         whose *true* final risk is >= -6. Predictions below -6 are first set to -6.001.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLIP_LOW, HIGH_RISK


def clip_predictions(y_pred) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=float)
    return np.where(y_pred < HIGH_RISK, CLIP_LOW, y_pred)


def confusion(y_true, y_pred) -> dict:
    t = np.asarray(y_true) >= HIGH_RISK
    p = np.asarray(y_pred) >= HIGH_RISK
    return {"tp": int((t & p).sum()), "fp": int((~t & p).sum()),
            "fn": int((t & ~p).sum()), "tn": int((~t & ~p).sum())}


def fbeta(y_true, y_pred, beta: float = 2.0) -> float:
    c = confusion(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def mse_hr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    mask = y_true >= HIGH_RISK
    if not mask.any():
        return float("nan")
    return float(np.mean((y_true[mask] - clip_predictions(y_pred)[mask]) ** 2))


def score(y_true, y_pred) -> float:
    """The leaderboard loss L = MSE_HR / F2."""
    f2 = fbeta(y_true, y_pred)
    return float("inf") if f2 == 0 else mse_hr(y_true, y_pred) / f2


def report(y_true, y_pred) -> dict:
    y_true, y_pred = _align(y_true, y_pred)
    c = confusion(y_true, y_pred)
    return {
        "score": score(y_true, y_pred),
        "mse_hr": mse_hr(y_true, y_pred),
        "f2": fbeta(y_true, y_pred),
        "precision": c["tp"] / max(c["tp"] + c["fp"], 1),
        "recall": c["tp"] / max(c["tp"] + c["fn"], 1),
        **c,
    }


def bootstrap_score(y_true, y_pred, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % bootstrap interval of the score (resampling events).

    With only ~150 high-risk test events, differences of a few percent between models
    are often within noise: always look at this interval.
    """
    y_true, y_pred = _align(y_true, y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    s = [score(y_true[i], y_pred[i]) for i in idx]
    return tuple(np.percentile(s, [2.5, 97.5]))


def _align(y_true, y_pred):
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        y_pred = y_pred.reindex(y_true.index)
        if y_pred.isna().any():
            raise ValueError("predictions missing for some events")
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
