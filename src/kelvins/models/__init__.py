"""Every model follows the same tiny interface:

    model = SomeModel(...).fit(train: EventDataset)
    pred  = model.predict(test: EventDataset)  # pd.Series of log10 risk, indexed by event_id

Predictions are raw log-risks; the metric applies the -6.001 clipping itself.
"""

from .baselines import ConstantRisk, LatestRisk, SescRules  # noqa: F401
from .gbm import LGBMResidual  # noqa: F401
