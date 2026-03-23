from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class AEConfig:
    """Typed config for the Autoencoder model."""

    name: str = "ae"

    # Architecture
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 64, 32])
    latent_dim: int = 16
    dropout: float = 0.1
    activation: str = "relu"

    # Loss
    loss: str = "mse"

    # Normalization
    normalization: str = "z_score"

    # Training
    n_epochs: int = 200
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # LR scheduler
    lr_scheduler: str = "reduce_on_plateau"
    lr_patience: int = 10
    lr_factor: float = 0.5
    lr_min: float = 1e-6

    # AMP
    amp: bool = False


@dataclass
class VAEConfig(AEConfig):
    """Typed config for the Variational Autoencoder model."""

    name: str = "vae"

    # VAE-specific loss
    reconstruction_loss: str = "mse"

    # KL divergence
    beta: float = 1.0
    beta_schedule: str = "constant"
    beta_warmup_epochs: int = 20


def register_configs() -> None:
    """Register structured configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="model", name="ae", node=AEConfig)
    cs.store(group="model", name="vae", node=VAEConfig)
