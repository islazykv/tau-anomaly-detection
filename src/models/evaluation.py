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


def compute_roc_per_origin(
    scores_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute ROC AUC per signal eventOrigin against the full background."""
    bkg = scores_df[scores_df["sample_type"] == "background"]
    sig = scores_df[scores_df["sample_type"] == "signal"]

    results: dict[str, float] = {}
    for origin in sorted(sig["eventOrigin"].unique()):
        sig_origin = sig[sig["eventOrigin"] == origin]
        combined_scores = np.concatenate(
            [bkg["anomaly_score"].to_numpy(), sig_origin["anomaly_score"].to_numpy()]
        )
        combined_labels = np.concatenate([np.zeros(len(bkg)), np.ones(len(sig_origin))])
        auc = float(roc_auc_score(combined_labels, combined_scores))
        results[origin] = auc

    log.info("Computed ROC AUC for %d signal origins", len(results))
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
        metrics["roc_per_origin"] = compute_roc_per_origin(scores_df)

    log.info(
        "Metrics: ROC AUC=%.4f, max SIC=%.4f", metrics["roc_auc"], metrics["max_sic"]
    )
    return metrics
