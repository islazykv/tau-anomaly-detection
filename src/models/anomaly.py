"""Anomaly scoring based on reconstruction error."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from torch import Tensor

log = logging.getLogger(__name__)


def reconstruction_error(x: Tensor, x_hat: Tensor) -> Tensor:
    """Per-event mean squared error across features.

    Higher values indicate more anomalous events.

    Args:
        x: Original input features, shape ``(batch, features)``.
        x_hat: Reconstructed features, shape ``(batch, features)``.

    Returns:
        Per-event reconstruction error, shape ``(batch,)``.
    """
    return (x - x_hat).pow(2).mean(dim=1)


def per_feature_error(x: Tensor, x_hat: Tensor) -> Tensor:
    """Per-event, per-feature squared error.

    Useful for interpretability — identifies which features contribute
    most to the anomaly score (replaces SHAP).

    Args:
        x: Original input features, shape ``(batch, features)``.
        x_hat: Reconstructed features, shape ``(batch, features)``.

    Returns:
        Per-event, per-feature error, shape ``(batch, features)``.
    """
    return (x - x_hat).pow(2)


def elbo_score(
    x: Tensor,
    x_hat: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """Per-event negative ELBO as anomaly score (VAE only).

    Args:
        x: Original input features, shape ``(batch, features)``.
        x_hat: Reconstructed features, shape ``(batch, features)``.
        mu: Latent mean, shape ``(batch, latent_dim)``.
        logvar: Latent log-variance, shape ``(batch, latent_dim)``.
        beta: KL weight.

    Returns:
        Per-event negative ELBO, shape ``(batch,)``.
    """
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

    Args:
        scores: Anomaly scores from background events.
        strategy: ``"percentile"`` or ``"std_dev"``.
        percentile: Percentile value (used if strategy is ``"percentile"``).
        n_std: Number of standard deviations above mean (used if ``"std_dev"``).

    Returns:
        Threshold value.
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
    """Build tidy DataFrame of anomaly scores for evaluation.

    Column layout::

        anomaly_score | sample_type | eventOrigin

    Args:
        scores: Per-event anomaly scores.
        labels: Binary labels (0 = background, 1 = signal).
        origins: Per-event ``eventOrigin`` strings.

    Returns:
        Scores DataFrame.
    """
    return pd.DataFrame(
        {
            "anomaly_score": scores,
            "sample_type": np.where(labels == 0, "background", "signal"),
            "eventOrigin": origins,
        }
    )
