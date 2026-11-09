"""Paths and challenge constants (Uriot et al., 2020, sections 4.1-4.3)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ZENODO_RECORD = "4463683"

# log10 collision probability. An event is "high risk" when its final risk >= -6.
HIGH_RISK = -6.0
# Any prediction below the threshold is replaced by -6.001: for a true negative the exact
# value does not matter, for a false negative it minimises the squared error.
CLIP_LOW = -6.001
# Inputs: only CDMs issued at least 2 days before TCA (latest moment to plan a manoeuvre).
CUTOFF_DAYS = 2.0
# Target: the last CDM must be within 1 day of TCA to be a good proxy of the final risk.
TARGET_MAX_TTCA = 1.0
# Floor value used in the data for "negligible" risk.
RISK_FLOOR = -30.0
