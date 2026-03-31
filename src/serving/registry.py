"""Model registry for loading trained AE/VAE checkpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.models.ae import Autoencoder
from src.models.anomaly import per_feature_error, reconstruction_error
from src.models.config import AEConfig, VAEConfig
from src.models.vae import VariationalAutoencoder

log = logging.getLogger(__name__)


@dataclass
class _ScalerState:
    """Lightweight scaler state extracted from a checkpoint."""

    mean: np.ndarray
    scale: np.ndarray
    normalization: str
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = (X - self.mean.astype(np.float32)) / self.scale.astype(np.float32)
        if self.normalization == "min_max":
            result = np.clip(result, 0.0, 1.0)
        return result


class ModelRegistry:
    """Holds a trained model and scaler loaded from a Lightning checkpoint."""

    def __init__(
        self,
        model: Autoencoder | VariationalAutoencoder,
        scaler: _ScalerState,
        threshold: float,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        model_name: str,
        model_cfg: dict,
        threshold: float = 0.0,
    ) -> ModelRegistry:
        """Load model weights and scaler state from a Lightning checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Scaler from DataModule state
        dm_state = ckpt.get("datamodule_state_dict", {})
        scaler = _ScalerState(
            mean=dm_state["scaler_mean_"],
            scale=dm_state["scaler_scale_"],
            normalization=dm_state["normalization"],
            feature_names=dm_state["feature_names_"],
        )

        n_features = scaler.n_features

        # Model
        model: Autoencoder | VariationalAutoencoder
        if model_name == "ae":
            cfg = AEConfig(**model_cfg)
            model = Autoencoder.load_from_checkpoint(
                ckpt_path, cfg=cfg, n_features=n_features, map_location="cpu"
            )
        elif model_name == "vae":
            cfg = VAEConfig(**model_cfg)
            model = VariationalAutoencoder.load_from_checkpoint(
                ckpt_path, cfg=cfg, n_features=n_features, map_location="cpu"
            )
        else:
            msg = f"Unknown model: {model_name}"
            raise ValueError(msg)

        model.eval()
        log.info(
            "Loaded %s from %s (%d features)",
            model_name.upper(),
            ckpt_path,
            n_features,
        )
        return cls(model=model, scaler=scaler, threshold=threshold)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on raw unnormalized features and return anomaly scores."""
        x_scaled = self.scaler.transform(features)
        x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)

        if isinstance(self.model, VariationalAutoencoder):
            x_hat, _mu, _logvar = self.model(x_tensor)
        else:
            x_hat = self.model(x_tensor)

        scores = reconstruction_error(x_tensor, x_hat).numpy()
        pf_err = per_feature_error(x_tensor, x_hat).numpy()
        return scores, pf_err

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.model.cfg.name

    @property
    def n_features(self) -> int:
        return self.scaler.n_features

    @property
    def feature_names(self) -> list[str]:
        return self.scaler.feature_names
