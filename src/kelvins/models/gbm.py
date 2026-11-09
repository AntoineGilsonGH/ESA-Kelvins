"""First learned model: LightGBM predicting the *correction* to the latest risk.

Reproduces the paper's post-competition experiment (section 6.1): the target is
h = r - r_-2 (how much the risk will still move), so the model starts from the naive
forecast and only has to learn deviations. h is mapped to a uniform distribution with a
quantile transformer, then mapped back.

This is a starting point to improve, not a tuned solution: in the paper, only 1 of 5,000
such models beat the latest-risk baseline on both train and test.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from ..data import EventDataset
from ..features import build_features


class LGBMResidual:
    def __init__(self, history: bool = False, drop_ids: bool = True, n_models: int = 10,
                 seed: int = 0, **lgb_params):
        self.history = history
        # The paper drops mission_id and object type to force rules that transfer across missions.
        self.drop_ids = drop_ids
        self.n_models = n_models
        self.seed = seed
        self.params = {"n_estimators": 25, "learning_rate": 0.05, "colsample_bytree": 0.25,
                       "verbose": -1, **lgb_params}

    def _X(self, ds: EventDataset) -> pd.DataFrame:
        X = build_features(ds, history=self.history)
        if self.drop_ids:
            X = X.drop(columns=[c for c in X.columns
                                if c == "mission_id" or c.startswith("c_object_type=")])
        return X

    def fit(self, ds: EventDataset):
        X = self._X(ds)
        self.columns_ = X.columns
        h = (ds.y - ds.latest_risk).to_numpy().reshape(-1, 1)
        self.qt_ = QuantileTransformer(n_quantiles=min(1000, len(h)), output_distribution="uniform")
        target = self.qt_.fit_transform(h).ravel()
        self.models_ = [lgb.LGBMRegressor(random_state=self.seed + i, **self.params).fit(X, target)
                        for i in range(self.n_models)]
        return self

    def predict(self, ds: EventDataset) -> pd.Series:
        X = self._X(ds).reindex(columns=self.columns_)
        u = np.mean([m.predict(X) for m in self.models_], axis=0)
        h = self.qt_.inverse_transform(np.clip(u, 0, 1).reshape(-1, 1)).ravel()
        return pd.Series(ds.latest_risk.to_numpy() + h, index=ds.event_ids, name="risk")

    def feature_importance(self) -> pd.Series:
        imp = np.mean([m.booster_.feature_importance("gain") for m in self.models_], axis=0)
        return pd.Series(imp, index=self.columns_).sort_values(ascending=False)
