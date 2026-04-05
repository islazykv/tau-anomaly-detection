"""Training pipeline for AE and VAE models."""

from __future__ import annotations

import logging
from pathlib import Path

import lightning as L
import pyrootutils
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.models.ae import Autoencoder
from src.models.callbacks import EpochProgressBar, MetricTracker
from src.models.config import AEConfig, VAEConfig
from src.models.datamodule import AnomalyDataModule
from src.models.vae import VariationalAutoencoder
from src.processing.analysis import get_background_origins, get_output_paths

log = logging.getLogger(__name__)


def _build_model(
    cfg: DictConfig,
    n_features: int,
) -> L.LightningModule:
    """Instantiate a model based on cfg.model.name.

    Options: "ae", "vae".
    """
    name = cfg.model.name
    model_params = dict(OmegaConf.to_container(cfg.model, resolve=True))  # type: ignore[arg-type]
    if name == "ae":
        model_cfg = AEConfig(**model_params)
        return Autoencoder(model_cfg, n_features=n_features)
    if name == "vae":
        model_cfg = VAEConfig(**model_params)
        return VariationalAutoencoder(model_cfg, n_features=n_features)
    msg = f"Unknown model: {name}"
    raise ValueError(msg)


def _build_trainer(
    cfg: DictConfig,
    model_name: str,
    models_dir: Path,
    logger: WandbLogger | bool,
) -> tuple[L.Trainer, MetricTracker]:
    """Create a Lightning Trainer with standard callbacks and optional logger."""
    tracker = MetricTracker()
    callbacks = [
        EarlyStopping(
            monitor=cfg.pipeline.monitor_metric,
            mode=cfg.pipeline.monitor_mode,
            patience=cfg.pipeline.early_stopping_patience,
            verbose=False,
        ),
        ModelCheckpoint(
            dirpath=models_dir,
            filename=f"{model_name}-best",
            monitor=cfg.pipeline.monitor_metric,
            mode=cfg.pipeline.monitor_mode,
            save_top_k=1,
            verbose=False,
        ),
        EpochProgressBar(),
        tracker,
    ]

    trainer = L.Trainer(
        max_epochs=cfg.model.n_epochs,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        precision="16-mixed" if cfg.model.amp else "32-true",
        enable_progress_bar=False,
    )
    return trainer, tracker


def train(cfg: DictConfig) -> None:
    """Run the full training pipeline for AE or VAE and save the checkpoint."""
    model_name = cfg.model.name
    log.info("Starting %s training", model_name.upper())
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed, workers=True)

    # Paths
    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])
    output_paths = get_output_paths(cfg)
    dataframes_dir = root / output_paths["dataframes_dir"]
    models_dir = root / output_paths["models_dir"]
    plots_dir = root / output_paths["plots_dir"] / model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # DataModule
    mc_path = dataframes_dir / "mc.parquet"
    background_origins = get_background_origins(cfg)
    log.info("Background origins: %s", background_origins)

    dm = AnomalyDataModule(
        mc_path=str(mc_path),
        background_origins=background_origins,
        normalization=cfg.model.normalization,
        val_fraction=cfg.pipeline.val_fraction,
        batch_size=cfg.model.batch_size,
        seed=cfg.seed,
    )
    dm.setup()

    # Model
    model = _build_model(cfg, n_features=dm.n_features)
    log.info(
        "Model: %s (%d parameters)",
        model_name,
        sum(p.numel() for p in model.parameters()),
    )

    # WandB logger
    wandb_cfg = cfg.pipeline.wandb
    if wandb_cfg.enabled:
        wandb_logger: WandbLogger | bool = WandbLogger(
            project=wandb_cfg.project,
            name=f"{cfg.experiment_name}-{model_name}",
            log_model=wandb_cfg.log_model,
            config=dict(OmegaConf.to_container(cfg.model, resolve=True)),  # type: ignore[arg-type]
        )
    else:
        wandb_logger = False

    # Trainer
    trainer, tracker = _build_trainer(cfg, model_name, models_dir, wandb_logger)

    # Fit
    trainer.fit(model, datamodule=dm)

    import shutil

    import torch

    best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
    ckpt_path = models_dir / f"{model_name}.ckpt"

    if best_model_path:
        shutil.copy2(best_model_path, ckpt_path)
        log.info(
            "Saved best checkpoint (%s) as %s",
            Path(best_model_path).name,
            ckpt_path.relative_to(root),
        )
        best_ckpt = torch.load(best_model_path, weights_only=False, map_location="cpu")
        model.load_state_dict(best_ckpt["state_dict"])
    else:
        trainer.save_checkpoint(ckpt_path)
        log.info("Saved final checkpoint: %s", ckpt_path.relative_to(root))

    # Save loss plot
    from src.models.plots import plot_loss

    fig = plot_loss(
        tracker.history["train_loss"],
        tracker.history["val_loss"],
        title=f"{model_name.upper()} Loss Plot",
        lr=tracker.history.get("lr"),
    )
    fig.savefig(plots_dir / f"{model_name}_loss.png", dpi=150, bbox_inches="tight")
    log.info(
        "Saved loss plot: %s", (plots_dir / f"{model_name}_loss.png").relative_to(root)
    )

    # Compute anomaly scores on predict set and save for downstream evaluation
    import numpy as np

    from src.models.anomaly import build_scores_frame, reconstruction_error

    all_scores: list[np.ndarray] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dm.predict_dataloader():
            x, _w = batch
            x = x.to(device)
            if model_name == "vae":
                x_hat, _mu, _logvar = model(x)
            else:
                x_hat = model(x)
            scores = reconstruction_error(x, x_hat)
            all_scores.append(scores.cpu().numpy())

    scores_array = np.concatenate(all_scores)
    scores_df = build_scores_frame(
        scores=scores_array,
        labels=dm.predict_labels,
        origins=dm.predict_origins,
        sample_types=dm.predict_sample_types,
    )
    scores_path = dataframes_dir / f"{model_name}_scores.parquet"
    scores_df.to_parquet(scores_path)
    log.info(
        "Saved scores: %s (%d events)", scores_path.relative_to(root), len(scores_df)
    )
