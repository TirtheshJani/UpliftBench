"""Estimator registry for upliftbench.

Populated by Phase 2 (S, T) and Phase 3 (X, DR, DML).
"""

from collections.abc import Callable

ESTIMATOR_REGISTRY: dict[str, Callable[[], object]] = {}
