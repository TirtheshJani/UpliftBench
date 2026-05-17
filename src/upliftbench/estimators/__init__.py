"""Estimator registry. Adds entries as estimator modules are imported."""

from __future__ import annotations

from collections.abc import Callable

from upliftbench.estimators.base import BaseUpliftEstimator
from upliftbench.estimators.s_learner import SLearner
from upliftbench.estimators.t_learner import TLearner

ESTIMATOR_REGISTRY: dict[str, Callable[[], BaseUpliftEstimator]] = {
    SLearner.name: SLearner,
    TLearner.name: TLearner,
}


def get_estimator(name: str) -> BaseUpliftEstimator:
    if name not in ESTIMATOR_REGISTRY:
        raise KeyError(f"Unknown estimator {name!r}. Available: {sorted(ESTIMATOR_REGISTRY)}")
    return ESTIMATOR_REGISTRY[name]()
