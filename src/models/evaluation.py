"""Anomaly detection evaluation metrics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

log = logging.getLogger(__name__)


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC for anomaly detection (bkg=0, sig=1)."""
    auc = roc_auc_score(labels, scores)
    log.info("ROC AUC: %.4f", auc)
    return float(auc)


def compute_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve arrays returning (fpr, tpr, thresholds)."""
    return roc_curve(labels, scores)


def compute_sic_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute the SIC curve as tpr / sqrt(fpr)."""
    return tpr / np.sqrt(np.maximum(fpr, epsilon))


def compute_roc_per_sample_type(
    scores_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute ROC AUC per sample_type.

    Signal: ROC AUC vs all background.
    Background types: ROC AUC vs remaining background (one-vs-rest).
    """
    is_signal = scores_df["sample_type"] == "signal"
    bkg = scores_df[~is_signal]
    sig = scores_df[is_signal]

    results: dict[str, float] = {}

    # Signal vs all background
    if len(sig) > 0:
        combined_scores = np.concatenate(
            [bkg["anomaly_score"].to_numpy(), sig["anomaly_score"].to_numpy()]
        )
        combined_labels = np.concatenate([np.zeros(len(bkg)), np.ones(len(sig))])
        results["signal"] = float(roc_auc_score(combined_labels, combined_scores))

    # Each background type vs remaining background
    for st in sorted(bkg["sample_type"].unique()):
        st_mask = bkg["sample_type"] == st
        st_scores = bkg.loc[st_mask, "anomaly_score"].to_numpy()
        rest_scores = bkg.loc[~st_mask, "anomaly_score"].to_numpy()
        if len(st_scores) == 0 or len(rest_scores) == 0:
            continue
        combined_scores = np.concatenate([rest_scores, st_scores])
        combined_labels = np.concatenate(
            [np.zeros(len(rest_scores)), np.ones(len(st_scores))]
        )
        results[st] = float(roc_auc_score(combined_labels, combined_scores))

    log.info("Computed ROC AUC for %d sample types", len(results))
    return results


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    scores_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute all evaluation metrics (ROC AUC, max SIC, optional per-origin ROC)."""
    fpr, tpr, _ = compute_roc_curve(labels, scores)
    sic = compute_sic_curve(fpr, tpr)

    metrics: dict[str, Any] = {
        "roc_auc": compute_roc_auc(labels, scores),
        "max_sic": float(np.nanmax(sic)),
    }

    if scores_df is not None:
        metrics["roc_per_sample_type"] = compute_roc_per_sample_type(scores_df)

    log.info(
        "Metrics: ROC AUC=%.4f, max SIC=%.4f", metrics["roc_auc"], metrics["max_sic"]
    )
    return metrics
