"""Ray Tune hyperparameter search for AE and VAE models."""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import DictConfig, OmegaConf
from ray import train as ray_train, tune
from ray.train import RunConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler

from src.models.ae import Autoencoder
from src.models.config import AEConfig, VAEConfig
from src.models.datamodule import AnomalyDataModule
from src.models.vae import VariationalAutoencoder

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Search space
# ------------------------------------------------------------------


def build_search_space(
    tuning_cfg: DictConfig,
    model_name: str,
) -> dict[str, Any]:
    """Convert YAML search_space config to Ray Tune sample distributions.

    Supported types: ``int`` (randint), ``float`` (uniform / loguniform),
    ``categorical`` (choice).

    Args:
        tuning_cfg: The ``cfg.tuning`` sub-config containing ``search_space``.
        model_name: ``"ae"`` or ``"vae"`` — selects the matching key in the
            search_space config.

    Returns:
        Dict mapping parameter names to Ray Tune sample objects.
    """
    space_cfg = tuning_cfg.search_space[model_name]
    space: dict[str, Any] = {}

    for param, spec in space_cfg.items():
        match spec["type"]:
            case "int":
                # randint upper bound is exclusive → +1
                space[param] = tune.randint(spec["low"], spec["high"] + 1)
            case "float":
                if spec.get("log", False):
                    space[param] = tune.loguniform(spec["low"], spec["high"])
                else:
                    space[param] = tune.uniform(spec["low"], spec["high"])
            case "categorical":
                space[param] = tune.choice(list(spec["choices"]))
            case other:
                msg = f"Unknown search_space type: {other}"
                raise ValueError(msg)

    return space


# ------------------------------------------------------------------
# Architecture helpers
# ------------------------------------------------------------------


def build_hidden_sizes(n_layers: int, layer_size: int) -> list[int]:
    """Build a halving encoder architecture from tunable scalars.

    Example: ``n_layers=3, layer_size=128`` → ``[128, 64, 32]``.
    Minimum layer width is clamped to 8.
    """
    return [max(layer_size // (2**i), 8) for i in range(n_layers)]


def make_model_config(
    trial_config: dict[str, Any],
    model_name: str,
    base_cfg: DictConfig,
) -> AEConfig | VAEConfig:
    """Build a typed model config by merging trial params into base defaults.

    Non-tunable parameters (activation, loss, lr_scheduler, …) are inherited
    from the base Hydra config.
    """
    base = dict(OmegaConf.to_container(base_cfg.model, resolve=True))

    # Override with trial hyperparameters
    base["hidden_sizes"] = build_hidden_sizes(
        trial_config["n_layers"], trial_config["layer_size"]
    )
    for key in ("latent_dim", "dropout", "learning_rate", "weight_decay", "batch_size"):
        base[key] = trial_config[key]

    if model_name == "vae":
        base["beta"] = trial_config["beta"]
        return VAEConfig(**base)
    return AEConfig(**base)


# ------------------------------------------------------------------
# Per-trial training
# ------------------------------------------------------------------


class _TuneReportCallback(L.Callback):
    """Lightning callback that reports ``val_loss`` to Ray Tune."""

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            ray_train.report({"val_loss": float(metrics["val_loss"])})


def _train_trial(
    config: dict[str, Any],
    base_cfg: DictConfig,
    dm_kwargs: dict[str, Any],
) -> None:
    """Training function executed by each Ray Tune trial.

    Each trial recreates the DataModule from scratch to guarantee clean
    state and correct batch_size for this trial's config.
    """
    model_name = base_cfg.model.name
    model_cfg = make_model_config(config, model_name, base_cfg)

    # Recreate DataModule per trial with trial-specific batch_size
    trial_dm_kwargs = {**dm_kwargs, "batch_size": model_cfg.batch_size}
    dm = AnomalyDataModule(**trial_dm_kwargs)
    dm.setup()

    # Build model
    if model_name == "vae":
        model = VariationalAutoencoder(model_cfg, n_features=dm.n_features)
    else:
        model = Autoencoder(model_cfg, n_features=dm.n_features)

    callbacks: list[L.Callback] = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=base_cfg.pipeline.early_stopping_patience,
        ),
        _TuneReportCallback(),
    ]

    trainer = L.Trainer(
        max_epochs=model_cfg.n_epochs,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        precision="16-mixed" if model_cfg.amp else "32-true",
    )
    trainer.fit(model, datamodule=dm)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def run_tune(
    cfg: DictConfig,
    dm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run Ray Tune hyperparameter search with ASHA scheduler.

    Args:
        cfg: Full Hydra config (must include ``model``, ``tuning``,
            ``pipeline`` sections).
        dm_kwargs: Constructor kwargs for :class:`AnomalyDataModule`.

    Returns:
        Best trial's config dict (raw search-space values).
    """
    model_name = cfg.model.name
    tuning_cfg = cfg.tuning

    search_space = build_search_space(tuning_cfg, model_name)
    log.info("Search space keys: %s", list(search_space.keys()))

    scheduler = ASHAScheduler(
        max_t=cfg.model.n_epochs,
        grace_period=10,
        reduction_factor=3,
    )

    trainable = tune.with_resources(
        tune.with_parameters(
            _train_trial,
            base_cfg=cfg,
            dm_kwargs=dm_kwargs,
        ),
        resources={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
    )

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=TuneConfig(
            scheduler=scheduler,
            metric="val_loss",
            mode="min",
            num_samples=tuning_cfg.num_samples,
        ),
        run_config=RunConfig(name=tuning_cfg.study_name),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config
    best_metric = best_result.metrics["val_loss"]

    log.info("Best trial config: %s", best_config)
    log.info("Best val_loss: %.6f", best_metric)

    return best_config


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------


def export_best_config(
    best_config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Convert best trial config to a model config dict for YAML export.

    Maps the raw search-space values (``n_layers``, ``layer_size``, …)
    back to model config keys (``hidden_sizes``, …).
    """
    result: dict[str, Any] = {
        "hidden_sizes": build_hidden_sizes(
            best_config["n_layers"], best_config["layer_size"]
        ),
        "latent_dim": best_config["latent_dim"],
        "dropout": round(float(best_config["dropout"]), 6),
        "learning_rate": float(best_config["learning_rate"]),
        "weight_decay": float(best_config["weight_decay"]),
        "batch_size": best_config["batch_size"],
    }
    if model_name == "vae":
        result["beta"] = float(best_config["beta"])
    return result
