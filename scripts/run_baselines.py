"""Evaluate every model on local cross-validation and on the official test set.

Writes results/leaderboard.md and results/leaderboard.csv.
Add your own model to MODELS to compare it with the baselines.
"""

import _bootstrap  # noqa: F401
import pandas as pd

from kelvins.config import RESULTS
from kelvins.data import make_test, make_train
from kelvins.metrics import bootstrap_score, report
from kelvins.models import ConstantRisk, LatestRisk, LGBMResidual, SescRules
from kelvins.validation import cross_validate, summarize_cv

MODELS = {
    "Constant risk −5 (CRP)": lambda: ConstantRisk(-5.0),
    "Latest risk (LRP)": LatestRisk,
    "Règles sesc (1er, à la main)": SescRules,
    "LightGBM résidu, dernier CDM": lambda: LGBMResidual(history=False),
    "LightGBM résidu, + historique": lambda: LGBMResidual(history=True),
}
PAPER = {"Constant risk −5 (CRP)": 2.5, "Latest risk (LRP)": 0.694,
         "Règles sesc (1er, à la main)": 0.569}


def main():
    train, test = make_train(), make_test()
    print(train, test, sep="\n")
    rows = []
    for name, make in MODELS.items():
        cv = summarize_cv(cross_validate(make, train, n_splits=5, n_repeats=3))
        pred = make().fit(train).predict(test)
        r = report(test.y, pred)
        lo, hi = bootstrap_score(test.y, pred)
        rows.append({"modèle": name, "test_score": r["score"], "IC95_bas": lo, "IC95_haut": hi,
                     "mse_hr": r["mse_hr"], "f2": r["f2"], "fn": r["fn"], "fp": r["fp"],
                     "cv_score": cv["cv_score"], "cv_gain_vs_lrp": cv["cv_gain_vs_lrp"],
                     "article": PAPER.get(name)})
        print(f"{name:34s} test L={r['score']:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"CV L={cv['cv_score']:.3f}  gain CV vs LRP={cv['cv_gain_vs_lrp']:+.1%}")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "leaderboard.csv", index=False)
    fmt = df.copy()
    for c in ["test_score", "IC95_bas", "IC95_haut", "mse_hr", "f2", "cv_score"]:
        fmt[c] = fmt[c].map("{:.3f}".format)
    fmt["cv_gain_vs_lrp"] = df.cv_gain_vs_lrp.map("{:+.1%}".format)
    fmt["article"] = df.article.map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    (RESULTS / "leaderboard.md").write_text(fmt.to_markdown(index=False) + "\n")
    print("\n" + fmt.to_markdown(index=False))


if __name__ == "__main__":
    main()
