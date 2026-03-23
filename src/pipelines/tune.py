"""Hyperparameter tuning pipeline with Ray Tune."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run_tuning(cfg: DictConfig) -> None:
    """Run hyperparameter tuning with Ray Tune.

    Implementation deferred to Phase 7.
    """
    log.info("Starting hyperparameter tuning")
    raise NotImplementedError("run_tuning will be implemented in Phase 7")
