"""Evaluation pipeline for trained AE and VAE models."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run_evaluation(cfg: DictConfig) -> None:
    """Run evaluation pipeline for a trained AE or VAE.

    Loads a trained checkpoint, runs prediction on background validation +
    signal events, computes anomaly scores, and generates evaluation plots.

    Implementation deferred to Phase 5.
    """
    model_name = cfg.model.name
    log.info("Starting %s evaluation", model_name.upper())
    raise NotImplementedError("run_evaluation will be implemented in Phase 5")
