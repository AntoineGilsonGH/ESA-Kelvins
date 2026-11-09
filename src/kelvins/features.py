"""Event-level feature tables built from the input CDMs.

Two families, to be combined freely:
* ``last_cdm_features``: the latest available CDM (the "static" view used by 65 % of
  the competitors and by the paper's post-competition experiments);
* ``history_features``: summaries of the CDM time series (how many CDMs, how the risk,
  miss distance and uncertainties evolved).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import EventDataset

OBJECT_TYPES = ["DEBRIS", "UNKNOWN", "PAYLOAD", "ROCKET BODY", "TBA"]
DROP = {"event_id", "true_risk"}


def numeric_columns(cdms: pd.DataFrame) -> list[str]:
    return [c for c in cdms.columns if c not in DROP and pd.api.types.is_numeric_dtype(cdms[c])]


def last_cdm_features(ds: EventDataset, one_hot_object_type: bool = True) -> pd.DataFrame:
    last = ds.last
    X = last[numeric_columns(ds.cdms)].copy()
    if one_hot_object_type:
        for t in OBJECT_TYPES:
            X[f"c_object_type={t}"] = (last["c_object_type"] == t).astype(float)
    X["log_miss_distance"] = np.log10(1 + last["miss_distance"])
    X["log_mahalanobis"] = np.log10(1e-3 + last["mahalanobis_distance"])
    return X


def _slope(t: np.ndarray, v: np.ndarray) -> float:
    if len(t) < 2 or np.ptp(t) == 0:
        return np.nan
    return float(np.polyfit(t, v, 1)[0])


def history_features(ds: EventDataset, cols=("risk", "max_risk_estimate", "miss_distance",
                                            "mahalanobis_distance", "t_sigma_r", "c_sigma_r",
                                            "c_sigma_t")) -> pd.DataFrame:
    g = ds.cdms.groupby("event_id")
    out = pd.DataFrame(index=ds.event_ids)
    out["n_cdm"] = g.size()
    out["ttca_first"] = g.time_to_tca.max()
    out["ttca_last"] = g.time_to_tca.min()
    for c in cols:
        out[f"{c}_mean"] = g[c].mean()
        out[f"{c}_std"] = g[c].std()
        out[f"{c}_min"] = g[c].min()
        out[f"{c}_max"] = g[c].max()
        out[f"{c}_first"] = g[c].first()
        out[f"{c}_delta_last"] = g[c].last() - g[c].nth(-2).groupby(
            ds.cdms.loc[g[c].nth(-2).index, "event_id"]).first()
    # risk trend per day (negative time_to_tca so that time flows forward)
    out["risk_slope"] = g.apply(lambda d: _slope(-d.time_to_tca.values, d.risk.values),
                                include_groups=False)
    return out.loc[ds.event_ids]


def build_features(ds: EventDataset, history: bool = True) -> pd.DataFrame:
    X = last_cdm_features(ds)
    if history:
        X = X.join(history_features(ds))
    return X
