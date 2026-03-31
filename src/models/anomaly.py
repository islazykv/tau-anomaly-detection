"""Anomaly scoring based on reconstruction error."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from torch import Tensor

log = logging.getLogger(__name__)


def reconstruction_error(x: Tensor, x_hat: Tensor) -> Tensor:
    """Compute per-event mean squared error across features as the anomaly score."""
    return (x - x_hat).pow(2).mean(dim=1)


def per_feature_error(x: Tensor, x_hat: Tensor) -> Tensor:
    """Compute per-event, per-feature squared error for interpretability."""
    return (x - x_hat).pow(2)


def elbo_score(
    x: Tensor,
    x_hat: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """Compute per-event negative ELBO as an alternative anomaly score (VAE only)."""
    recon = (x - x_hat).pow(2).mean(dim=1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return recon + beta * kl


def compute_threshold(
    scores: np.ndarray,
    strategy: str = "percentile",
    percentile: float = 95.0,
    n_std: float = 3.0,
) -> float:
    """Compute anomaly threshold from background scores.

    Strategies:
        percentile -- threshold at the given percentile of the score distribution.
        std_dev    -- threshold at mean + n_std standard deviations.
    """
    if strategy == "percentile":
        return float(np.percentile(scores, percentile))
    if strategy == "std_dev":
        return float(scores.mean() + n_std * scores.std())
    msg = f"Unknown threshold strategy: {strategy}"
    raise ValueError(msg)


def build_scores_frame(
    scores: np.ndarray,
    labels: np.ndarray,
    origins: np.ndarray,
) -> pd.DataFrame:
    """Build a tidy DataFrame of anomaly scores with sample_type and eventOrigin columns."""
    return pd.DataFrame(
        {
            "anomaly_score": scores,
            "sample_type": np.where(labels == 0, "background", "signal"),
            "eventOrigin": origins,
        }
    )
