"""LightningDataModule for background-only autoencoder training."""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.splits import prepare_features, split_background_signal, train_val_split

log = logging.getLogger(__name__)


class AnomalyDataModule(L.LightningDataModule):
    """DataModule that trains on background only and holds out signal for evaluation.

    Scaler parameters are stored in the Lightning checkpoint via
    ``state_dict`` / ``load_state_dict`` — no external joblib files needed.

    Each batch yields ``(features, weights)`` tensors.
    """

    def __init__(
        self,
        mc_path: str,
        background_origins: set[str],
        normalization: str = "z_score",
        val_fraction: float = 0.2,
        batch_size: int = 2048,
        seed: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.mc_path = mc_path
        self.background_origins = background_origins
        self.normalization = normalization
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        # Populated in setup()
        self.train_dataset: TensorDataset
        self.val_dataset: TensorDataset
        self.predict_dataset: TensorDataset
        self.predict_labels: np.ndarray  # 0=bkg, 1=sig per predict event
        self.predict_origins: np.ndarray  # eventOrigin per predict event

        # Scaler params (saved in checkpoint)
        self.scaler_mean_: np.ndarray
        self.scaler_scale_: np.ndarray
        self.feature_names_: list[str]
        self.n_features: int

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Read parquet, split bkg/sig, fit scaler, build TensorDatasets."""
        df = pd.read_parquet(self.mc_path)
        bkg_df, sig_df = split_background_signal(df, self.background_origins)

        # Extract features and weights
        bkg_X, bkg_w = prepare_features(bkg_df)
        sig_X, sig_w = prepare_features(sig_df)

        self.feature_names_ = list(bkg_X.columns)
        self.n_features = len(self.feature_names_)

        # Train/val split on background only
        X_train, X_val, w_train, w_val = train_val_split(
            bkg_X, bkg_w, val_fraction=self.val_fraction, seed=self.seed
        )

        # Fit scaler on training background
        self._fit_scaler(X_train)

        # Transform all sets
        X_train_s = self._transform(X_train)
        X_val_s = self._transform(X_val)
        X_sig_s = self._transform(sig_X)

        # Build TensorDatasets (features + weights shuffle together)
        self.train_dataset = _to_tensor_dataset(X_train_s, w_train)
        self.val_dataset = _to_tensor_dataset(X_val_s, w_val)

        # Predict set: bkg_val + signal (for evaluation)
        X_pred = np.concatenate([X_val_s, X_sig_s], axis=0)
        w_pred = np.concatenate(
            [w_val.to_numpy(dtype=np.float32), sig_w.to_numpy(dtype=np.float32)],
            axis=0,
        )
        self.predict_dataset = _to_tensor_dataset(X_pred, w_pred)

        # Labels and origins for evaluation
        self.predict_labels = np.concatenate(
            [np.zeros(len(X_val_s)), np.ones(len(X_sig_s))]
        )
        self.predict_origins = np.concatenate(
            [
                bkg_df["eventOrigin"].loc[X_val.index].to_numpy(),
                sig_df["eventOrigin"].to_numpy(),
            ]
        )
        log.info(
            "DataModule ready: %d train, %d val, %d signal, %d features",
            len(X_train),
            len(X_val),
            len(sig_X),
            self.n_features,
        )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Scaler (stored in Lightning checkpoint)
    # ------------------------------------------------------------------

    def _fit_scaler(self, X: pd.DataFrame) -> None:
        """Compute scaler parameters from training background."""
        values = X.to_numpy(dtype=np.float64)
        if self.normalization == "z_score":
            self.scaler_mean_ = values.mean(axis=0)
            self.scaler_scale_ = values.std(axis=0)
            # Prevent division by zero for constant features
            self.scaler_scale_[self.scaler_scale_ == 0] = 1.0
        elif self.normalization == "min_max":
            self.scaler_mean_ = values.min(axis=0)
            scale = values.max(axis=0) - values.min(axis=0)
            scale[scale == 0] = 1.0
            self.scaler_scale_ = scale
        else:
            msg = f"Unknown normalization: {self.normalization}"
            raise ValueError(msg)

    def _transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply fitted scaler to features."""
        values = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else X
        result = (
            values - self.scaler_mean_.astype(np.float32)
        ) / self.scaler_scale_.astype(np.float32)
        if self.normalization == "min_max":
            result = np.clip(result, 0.0, 1.0)
        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaler transformation."""
        return X * self.scaler_scale_.astype(np.float32) + self.scaler_mean_.astype(
            np.float32
        )

    def state_dict(self) -> dict[str, Any]:
        """Save scaler params into the Lightning checkpoint."""
        return {
            "scaler_mean_": self.scaler_mean_,
            "scaler_scale_": self.scaler_scale_,
            "normalization": self.normalization,
            "feature_names_": self.feature_names_,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scaler params from a Lightning checkpoint."""
        self.scaler_mean_ = state_dict["scaler_mean_"]
        self.scaler_scale_ = state_dict["scaler_scale_"]
        self.normalization = state_dict["normalization"]
        self.feature_names_ = state_dict["feature_names_"]
        self.n_features = len(self.feature_names_)


def _to_tensor_dataset(
    X: np.ndarray | pd.DataFrame,
    w: np.ndarray | pd.Series,
) -> TensorDataset:
    """Convert arrays to a TensorDataset of (features, weights)."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=np.float32)
    elif X.dtype != np.float32:
        X = X.astype(np.float32)
    if isinstance(w, pd.Series):
        w = w.to_numpy(dtype=np.float32)
    elif w.dtype != np.float32:
        w = w.astype(np.float32)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(w))
