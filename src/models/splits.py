"""Background/signal splitting and feature preparation for anomaly detection."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split as _sklearn_split

from src.processing.validation import METADATA_COLUMNS

log = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract training features and per-event weights from the MC DataFrame.

    Unlike the supervised predecessor, no class labels are returned — the
    autoencoder trains on features only.

    Args:
        df: MC DataFrame with metadata and feature columns.

    Returns:
        Tuple of (features, weights).
    """
    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    features = df[feature_cols].copy()
    weights = df["weight"].copy()
    return features, weights


def split_background_signal(
    df: pd.DataFrame,
    background_origins: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate background from signal events using eventOrigin.

    Args:
        df: MC DataFrame containing both background and signal events.
        background_origins: Set of ``eventOrigin`` values that identify
            background samples.

    Returns:
        Tuple of (background_df, signal_df).
    """
    is_background = df["eventOrigin"].isin(background_origins)
    bkg_df = df.loc[is_background].copy()
    sig_df = df.loc[~is_background].copy()
    log.info(
        "Split MC: %d background, %d signal events",
        len(bkg_df),
        len(sig_df),
    )
    return bkg_df, sig_df


def train_val_split(
    features: pd.DataFrame,
    weights: pd.Series,
    val_fraction: float = 0.2,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Random train/val split on background-only data.

    No stratification is needed because the training set contains only
    background events.

    Args:
        features: Background-only input features.
        weights: Per-event sample weights (aligned with *features*).
        val_fraction: Fraction of events reserved for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, w_train, w_val).
    """
    X_train, X_val, w_train, w_val = _sklearn_split(
        features,
        weights,
        test_size=val_fraction,
        random_state=seed,
    )
    log.info(
        "Train/val split: %d train, %d val events",
        len(X_train),
        len(X_val),
    )
    return X_train, X_val, w_train, w_val
