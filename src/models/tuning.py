"""Ray Tune hyperparameter search for AE and VAE models."""

from __future__ import annotations

import logging
import os
from typing import Any

# Work around Ray >=2.44 bug where deprecated RunConfig.verbose="DEPRECATED"
# causes AttributeError in tune/experimental/output.py (str has no .value).
os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "0")

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import DictConfig, OmegaConf
import ray
from ray import tune
from ray.tune import RunConfig, TuneConfig
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

    Types:
        int         -- random integer (randint).
        float       -- uniform or log-uniform continuous.
        categorical -- choice from a list.
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
    """Build a halving encoder architecture from tunable scalars."""
    return [max(layer_size // (2**i), 8) for i in range(n_layers)]


def make_model_config(
    trial_config: dict[str, Any],
    model_name: str,
    base_cfg: DictConfig,
) -> AEConfig | VAEConfig:
    """Build a typed model config by merging trial parameters into base defaults."""
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
    """Lightning callback that reports val_loss to Ray Tune."""

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Report validation loss to Ray Tune after each epoch."""
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            tune.report({"val_loss": float(metrics["val_loss"])})


def _train_trial(
    config: dict[str, Any],
    base_cfg: DictConfig,
    dm_kwargs: dict[str, Any],
) -> None:
    """Training function executed by each Ray Tune trial."""
    import warnings

    # Suppress noisy Lightning warnings inside Ray workers
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

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
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        precision="16-mixed" if model_cfg.amp else "32-true",
    )
    trainer.fit(model, datamodule=dm)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def _make_run_config(study_name: str, storage_path: str | None = None) -> RunConfig:
    """Build a RunConfig compatible with Ray >=2.44 deprecation changes."""
    return RunConfig(name=study_name, verbose=1, storage_path=storage_path)


def run_tune(
    cfg: DictConfig,
    dm_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run Ray Tune hyperparameter search with ASHA scheduler."""
    # Ensure Ray workers can import project modules (src.*).
    import pyrootutils

    root = str(pyrootutils.find_root(indicator=[".git", "pyproject.toml"]))
    runtime_env = {"env_vars": {"PYTHONPATH": root}}

    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

    model_name = cfg.model.name
    tuning_cfg = cfg.tuning

    search_space = build_search_space(tuning_cfg, model_name)
    log.info("Search space keys: %s", list(search_space.keys()))

    max_t = cfg.model.n_epochs
    scheduler = ASHAScheduler(
        max_t=max_t,
        grace_period=min(10, max_t),
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

    storage_path = os.path.join(root, "data", "ray_results")
    experiment_path = os.path.join(storage_path, tuning_cfg.study_name)

    logging.getLogger("ray.tune").setLevel(logging.WARNING)

    n_requested = tuning_cfg.num_samples

    if os.path.exists(experiment_path):
        tuner = tune.Tuner.restore(
            experiment_path,
            trainable=trainable,
            param_space=search_space,
            resume_errored=True,
        )
        n_previous = len(tuner.get_results())
        total = n_previous + n_requested
        # Bump total so Ray runs num_samples MORE on top of existing trials.
        tuner._local_tuner._tune_config.num_samples = total
        log.info(
            "Resuming: %d previous + %d new → %d total",
            n_previous,
            n_requested,
            total,
        )
    else:
        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=TuneConfig(
                scheduler=scheduler,
                metric="val_loss",
                mode="min",
                num_samples=n_requested,
            ),
            run_config=_make_run_config(
                tuning_cfg.study_name,
                storage_path=storage_path,
            ),
        )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config
    if best_config is None:
        msg = "No completed trials — best_result.config is None"
        raise RuntimeError(msg)
    best_metric = best_result.metrics["val_loss"]

    log.info("Best trial config: %s", best_config)
    log.info("Best val_loss: %.6f", best_metric)

    trial_df = results.get_dataframe()
    return best_config, trial_df


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------


def export_best_config(
    best_config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Convert best trial config to a model config dict for YAML export."""
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
