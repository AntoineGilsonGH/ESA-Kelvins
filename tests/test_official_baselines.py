"""Integration tests: reproduce the numbers published in Uriot et al. (2020).

Skipped when the data has not been downloaded (python scripts/download_data.py).
"""
import pytest

from kelvins.config import DATA_RAW

pytestmark = pytest.mark.skipif(not (DATA_RAW / "train_data.csv").exists(), reason="data not downloaded")


@pytest.fixture(scope="module")
def data():
    from kelvins.data import make_test, make_train
    return make_train(), make_test()


def test_dataset_sizes_match_paper(data):
    train, test = data
    assert len(train) == 8293 and int(train.is_high.sum()) == 66
    assert len(test) == 2167 and int(test.is_high.sum()) == 150
    assert (train.cdms.time_to_tca >= 2).all() and (test.cdms.time_to_tca >= 2).all()


def test_constant_risk_score(data):
    from kelvins.metrics import score
    from kelvins.models import ConstantRisk
    train, test = data
    assert score(test.y, ConstantRisk(-5).fit(train).predict(test)) == pytest.approx(2.5, abs=0.01)


def test_latest_risk_score(data):
    from kelvins.metrics import report
    from kelvins.models import LatestRisk
    train, test = data
    r = report(test.y, LatestRisk().fit(train).predict(test))
    assert r["score"] == pytest.approx(0.694, abs=0.001)
    assert r["mse_hr"] == pytest.approx(0.513, abs=0.001)
    assert r["f2"] == pytest.approx(0.739, abs=0.001)


@pytest.mark.parametrize("steps,expected", [((0,), 0.680), ((0, 1), 0.653), ((0, 1, 2), 0.603),
                                            ((0, 1, 2, 6), 0.575)])
def test_sesc_steps_match_table_4(data, steps, expected):
    from kelvins.metrics import score
    from kelvins.models import SescRules
    train, test = data
    assert score(test.y, SescRules(steps).fit(train).predict(test)) == pytest.approx(expected, abs=0.002)
