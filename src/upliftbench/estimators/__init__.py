"""Estimator registry."""

from __future__ import annotations

from collections.abc import Callable

from upliftbench.estimators.base import BaseUpliftEstimator
from upliftbench.estimators.dml import LinearDMLEstimator
from upliftbench.estimators.dr_learner import DRLearner
from upliftbench.estimators.s_learner import SLearner
from upliftbench.estimators.t_learner import TLearner
from upliftbench.estimators.x_learner import XLearner

ESTIMATOR_REGISTRY: dict[str, Callable[[], BaseUpliftEstimator]] = {
    SLearner.name: SLearner,
    TLearner.name: TLearner,
    XLearner.name: XLearner,
    DRLearner.name: DRLearner,
    LinearDMLEstimator.name: LinearDMLEstimator,
}


def get_estimator(name: str) -> BaseUpliftEstimator:
    if name not in ESTIMATOR_REGISTRY:
        raise KeyError(f"Unknown estimator {name!r}. Available: {sorted(ESTIMATOR_REGISTRY)}")
    return ESTIMATOR_REGISTRY[name]()
