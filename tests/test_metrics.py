import numpy as np
import pytest

from kelvins.metrics import clip_predictions, fbeta, mse_hr, report, score


def test_clipping():
    assert np.allclose(clip_predictions([-7, -6, -5.5, -30]), [-6.001, -6, -5.5, -6.001])


def test_perfect_prediction():
    y = np.array([-30, -10, -6, -5, -4.2])
    r = report(y, y)
    assert r["f2"] == 1 and r["mse_hr"] == 0 and r["score"] == 0


def test_f2_weighs_recall_more_than_precision():
    y = np.array([-5, -5, -30, -30])
    only_fn = np.array([-5, -30, -30, -30])   # precision 1, recall 0.5
    only_fp = np.array([-5, -5, -5, -30])     # precision 2/3, recall 1
    assert fbeta(y, only_fp) > fbeta(y, only_fn)


def test_mse_only_on_true_high_risk_and_uses_clip():
    y = np.array([-5.0, -30.0])
    pred = np.array([-20.0, -4.0])  # FN predicted -20 -> clipped to -6.001
    assert mse_hr(y, pred) == pytest.approx((-5 + 6.001) ** 2)


def test_no_positive_prediction_gives_infinite_score():
    assert score(np.array([-5.0, -30.0]), np.array([-30.0, -30.0])) == float("inf")
