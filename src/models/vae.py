"""Variational Autoencoder LightningModule for anomaly detection."""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from src.models.ae import ACTIVATIONS, _build_stack, _get_loss_fn, _build_scheduler
from src.models.config import VAEConfig

log = logging.getLogger(__name__)

LOGVAR_CLAMP_MIN = -10.0
LOGVAR_CLAMP_MAX = 10.0


class VariationalAutoencoder(L.LightningModule):
    """Variational autoencoder for unsupervised anomaly detection.

    Trained on background-only events. Anomaly scoring can use either
    reconstruction error or the full ELBO.

    Args:
        cfg: Typed VAE model configuration.
        n_features: Number of input features (set by DataModule).
    """

    def __init__(self, cfg: VAEConfig, n_features: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features
        self.save_hyperparameters({"model": cfg.__dict__, "n_features": n_features})

        activation_cls = ACTIVATIONS[cfg.activation]

        # Encoder: input -> hidden layers (no latent yet)
        encoder_dims = [n_features, *cfg.hidden_sizes]
        self.encoder = _build_stack(
            encoder_dims,
            activation_cls,
            cfg.dropout,
            final_activation=True,
        )

        # Latent projections from last hidden size
        last_hidden = cfg.hidden_sizes[-1]
        self.fc_mu = nn.Linear(last_hidden, cfg.latent_dim)
        self.fc_logvar = nn.Linear(last_hidden, cfg.latent_dim)

        # Decoder: latent -> reversed hidden layers -> output
        self.decoder = _build_stack(
            [cfg.latent_dim, *reversed(cfg.hidden_sizes), n_features],
            activation_cls,
            cfg.dropout,
            final_activation=False,
        )

        self.recon_loss_fn = _get_loss_fn(cfg.reconstruction_loss)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (mu, logvar) of the latent distribution."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z from q(z|x) using the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _get_beta(self) -> float:
        """Return the current KL weight, applying warmup if configured."""
        if self.cfg.beta_schedule == "warmup":
            progress = min(
                self.current_epoch / max(self.cfg.beta_warmup_epochs, 1), 1.0
            )
            return self.cfg.beta * progress
        return self.cfg.beta

    def _compute_loss(
        self,
        x: Tensor,
        x_hat: Tensor,
        mu: Tensor,
        logvar: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute weighted ELBO loss.

        Returns:
            Tuple of (total_loss, recon_loss, kl_loss) — all scalars.
        """
        # Reconstruction: reduction="none" -> (batch, features)
        raw_recon = self.recon_loss_fn(x_hat, x)
        per_event_recon = raw_recon.mean(dim=1)
        recon_loss = (per_event_recon * weights).mean()

        # KL divergence: sum over latent dims, weighted mean over batch
        kl_per_event = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        kl_loss = (kl_per_event * weights).mean()

        beta = self._get_beta()
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, w = batch
        x_hat, mu, logvar = self(x)
        total_loss, recon_loss, kl_loss = self._compute_loss(x, x_hat, mu, logvar, w)

        self._log_metrics("train", total_loss, recon_loss, kl_loss, mu, logvar)
        return total_loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, w = batch
        x_hat, mu, logvar = self(x)
        total_loss, recon_loss, kl_loss = self._compute_loss(x, x_hat, mu, logvar, w)

        self._log_metrics("val", total_loss, recon_loss, kl_loss, mu, logvar)
        return total_loss

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Return reconstructed features for anomaly scoring."""
        x, _w = batch
        x_hat, _mu, _logvar = self(x)
        return x_hat

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_metrics(
        self,
        prefix: str,
        total_loss: Tensor,
        recon_loss: Tensor,
        kl_loss: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> None:
        """Log loss components and latent diagnostics for collapse monitoring."""
        beta = self._get_beta()
        self.log(
            f"{prefix}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(f"{prefix}_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log(f"{prefix}_kl_loss", kl_loss, on_step=False, on_epoch=True)
        self.log(f"{prefix}_beta", beta, on_step=False, on_epoch=True)
        self.log(f"{prefix}_mu_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log(f"{prefix}_mu_var", mu.var(), on_step=False, on_epoch=True)
        self.log(f"{prefix}_logvar_mean", logvar.mean(), on_step=False, on_epoch=True)

        # Collapse warnings (logged, not raised)
        if mu.var().item() < 0.1:
            log.warning(
                "Potential posterior collapse: mu.var() = %.4f", mu.var().item()
            )
        if logvar.mean().item() < -5.0:
            log.warning(
                "Potential posterior collapse: logvar.mean() = %.4f",
                logvar.mean().item(),
            )

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
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
