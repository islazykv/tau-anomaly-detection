"""Autoencoder LightningModule for anomaly detection."""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from src.models.config import AEConfig

log = logging.getLogger(__name__)

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
}


class Autoencoder(L.LightningModule):
    """Symmetric autoencoder trained on background-only events for anomaly detection."""

    def __init__(self, cfg: AEConfig, n_features: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features
        self.save_hyperparameters({"model": cfg.__dict__, "n_features": n_features})

        activation_cls = ACTIVATIONS[cfg.activation]
        self.encoder = _build_stack(
            [n_features, *cfg.hidden_sizes, cfg.latent_dim],
            activation_cls,
            cfg.dropout,
        )
        self.decoder = _build_stack(
            [cfg.latent_dim, *reversed(cfg.hidden_sizes), n_features],
            activation_cls,
            cfg.dropout,
            final_activation=False,
        )
        self.loss_fn = _get_loss_fn(cfg.loss)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Encode and decode the input, returning the reconstruction."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: Tensor) -> Tensor:
        """Return the latent-space representation of the input."""
        return self.encoder(x)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, x: Tensor, x_hat: Tensor, weights: Tensor) -> Tensor:
        """Compute per-event reconstruction loss weighted by sample weights."""
        raw = self.loss_fn(x_hat, x)  # (batch, features)
        per_event = raw.mean(dim=1)  # (batch,)
        return (per_event * weights).mean()

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute weighted reconstruction loss on one training batch."""
        x, w = batch
        x_hat = self(x)
        loss = self._compute_loss(x, x_hat, w)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute weighted reconstruction loss on one validation batch."""
        x, w = batch
        x_hat = self(x)
        loss = self._compute_loss(x, x_hat, w)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Return reconstructed features for downstream anomaly scoring."""
        x, _w = batch
        return self(x)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Configure AdamW optimizer with optional LR scheduling."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        config: dict[str, Any] = {"optimizer": optimizer}

        scheduler = _build_scheduler(optimizer, self.cfg)
        if scheduler is not None:
            config["lr_scheduler"] = scheduler

        return config


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_stack(
    dims: list[int],
    activation_cls: type[nn.Module],
    dropout: float,
    *,
    final_activation: bool = True,
) -> nn.Sequential:
    """Build a fully-connected layer stack with activation and optional dropout."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if is_last and not final_activation:
            break
        layers.append(activation_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _get_loss_fn(name: str) -> nn.Module:
    """Return a loss module with reduction='none'.

    Losses:
        mse       -- mean squared error.
        smooth_l1 -- Huber-like smooth L1 loss.
        bce       -- binary cross-entropy with logits.
    """
    if name == "mse":
        return nn.MSELoss(reduction="none")
    if name == "smooth_l1":
        return nn.SmoothL1Loss(reduction="none")
    if name == "bce":
        return nn.BCEWithLogitsLoss(reduction="none")
    msg = f"Unknown loss function: {name}"
    raise ValueError(msg)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: AEConfig,
) -> dict[str, Any] | None:
    """Build a Lightning LR scheduler dict, or None if disabled.

    Schedulers:
        none               -- no scheduling.
        reduce_on_plateau   -- reduce LR when val_loss plateaus.
        cosine_annealing    -- cosine decay to lr_min over n_epochs.
    """
    if cfg.lr_scheduler == "none":
        return None
    if cfg.lr_scheduler == "reduce_on_plateau":
        return {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=cfg.lr_patience,
                factor=cfg.lr_factor,
                min_lr=cfg.lr_min,
            ),
            "monitor": "val_loss",
        }
    if cfg.lr_scheduler == "cosine_annealing":
        return {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.n_epochs,
                eta_min=cfg.lr_min,
            ),
        }
    msg = f"Unknown lr_scheduler: {cfg.lr_scheduler}"
    raise ValueError(msg)
