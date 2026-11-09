"""Load the CDM tables and turn them into event-level datasets.

Vocabulary
----------
CDM    : one row = one Conjunction Data Message (a snapshot of a close approach).
event  : one close approach = a time series of CDMs sharing an ``event_id``.
r_-2   : risk of the latest CDM available at least 2 days before TCA (model input).
r      : risk of the final CDM, within 1 day of TCA (target).

The official test set was built with three rules (paper, section 4.2):
  (i) at least 2 CDMs, (ii) last CDM with time_to_tca < 1,
  (iii) first CDM with time_to_tca >= 2; CDMs with time_to_tca < 2 are hidden.
``make_train`` applies the same rules to the training events so that train and test
are built the same way (8,293 eligible training events, 66 of them high risk).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CUTOFF_DAYS, DATA_INTERIM, DATA_RAW, HIGH_RISK, TARGET_MAX_TTCA

FILES = {
    "train": "train_data.csv",
    "test": "test_data.csv",
    "test_private": "test_data_private.csv",
}


def load_raw(name: str, cache: bool = True) -> pd.DataFrame:
    """Read one of the raw CSVs (cached as parquet after the first read)."""
    csv = DATA_RAW / FILES[name]
    pq = DATA_INTERIM / f"{name}.parquet"
    if cache and pq.exists() and pq.stat().st_mtime >= csv.stat().st_mtime:
        return pd.read_parquet(pq)
    if not csv.exists():
        raise FileNotFoundError(f"{csv} is missing: run `python scripts/download_data.py` first.")
    df = pd.read_csv(csv)
    if cache:
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq)
    return df


def sort_chronologically(cdms: pd.DataFrame) -> pd.DataFrame:
    """Within each event, oldest CDM first (time_to_tca decreasing)."""
    return cdms.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort")


@dataclass
class EventDataset:
    """Input CDMs (only those available >= 2 days before TCA) and the final risk per event."""

    cdms: pd.DataFrame  # long format, chronological within each event
    y: pd.Series        # final risk r, indexed by event_id

    @property
    def event_ids(self) -> pd.Index:
        return self.y.index

    @property
    def last(self) -> pd.DataFrame:
        """Latest available CDM of each event (one row per event, indexed by event_id)."""
        return self.cdms.groupby("event_id").tail(1).set_index("event_id").loc[self.event_ids]

    @property
    def latest_risk(self) -> pd.Series:
        """r_-2: the risk of the latest available CDM."""
        return self.last["risk"]

    @property
    def is_high(self) -> pd.Series:
        return self.y >= HIGH_RISK

    def subset(self, event_ids) -> "EventDataset":
        event_ids = pd.Index(event_ids)
        return EventDataset(self.cdms[self.cdms.event_id.isin(event_ids)], self.y.loc[event_ids])

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:
        return (f"EventDataset({len(self)} events, {len(self.cdms)} CDMs, "
                f"{int(self.is_high.sum())} high risk = {self.is_high.mean():.2%})")


def eligibility(cdms: pd.DataFrame) -> pd.DataFrame:
    """Per-event flags of the three test-set rules."""
    g = cdms.groupby("event_id").time_to_tca
    out = pd.DataFrame({
        "n_cdm": g.size(),
        "first_ttca": g.max(),
        "last_ttca": g.min(),
    })
    out["eligible"] = (out.n_cdm >= 2) & (out.last_ttca < TARGET_MAX_TTCA) & (out.first_ttca >= CUTOFF_DAYS)
    return out


def make_train(eligible_only: bool = True) -> EventDataset:
    """Training events cropped like the test set.

    With ``eligible_only=False`` every event keeping at least one CDM before the cutoff is
    returned (its target may then be far from TCA: useful for pre-training, not for scoring).
    """
    raw = sort_chronologically(load_raw("train"))
    y = raw.groupby("event_id").risk.last()  # chronological order -> last row is the final CDM
    elig = eligibility(raw)
    inputs = raw[raw.time_to_tca >= CUTOFF_DAYS]
    keep = elig.index[elig.eligible] if eligible_only else inputs.event_id.unique()
    keep = pd.Index(keep).intersection(inputs.event_id.unique())
    # an event whose target is itself an input CDM is useless
    keep = keep[(raw[raw.event_id.isin(keep)].groupby("event_id").time_to_tca.min() < CUTOFF_DAYS).values]
    return EventDataset(inputs[inputs.event_id.isin(keep)].reset_index(drop=True), y.loc[keep].rename("risk"))


def make_test() -> EventDataset:
    """Official test set: published CDMs + private final risks."""
    cdms = sort_chronologically(load_raw("test")).reset_index(drop=True)
    private = load_raw("test_private").set_index("event_id")["true_risk"]
    y = private.loc[np.sort(cdms.event_id.unique())].rename("risk")
    return EventDataset(cdms, y)
