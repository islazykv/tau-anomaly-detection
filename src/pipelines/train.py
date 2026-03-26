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
from src.processing.analysis import get_output_paths

log = logging.getLogger(__name__)


def _get_background_origins(cfg: DictConfig) -> set[str]:
    """Extract background sample IDs from the samples config."""
    bg_cfg = cfg.samples.background
    excludes = set(bg_cfg.get("exclude", []))
    return {s["id"] for s in bg_cfg.samples if s["id"] not in excludes}


def _build_model(
    cfg: DictConfig,
    n_features: int,
) -> L.LightningModule:
    """Instantiate AE or VAE based on cfg.model.name."""
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
    """Create a Lightning Trainer with callbacks and logger."""
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
    """Run the full training pipeline for AE or VAE.

    Steps:
        1. Resolve output paths
        2. Setup DataModule (background-only training)
        3. Instantiate model (AE or VAE)
        4. Create Trainer with callbacks and WandB logger
        5. Fit model
        6. Save checkpoint
    """
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
    background_origins = _get_background_origins(cfg)
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

    best_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
    if best_path:
        best_path = Path(best_path).relative_to(root)
    log.info("Training complete — best checkpoint: %s", best_path)

    # Save final checkpoint with scaler state
    ckpt_path = models_dir / f"{model_name}.ckpt"
    trainer.save_checkpoint(ckpt_path)
    log.info("Saved checkpoint: %s", ckpt_path.relative_to(root))

    # Save loss plot
    from src.models.plots import plot_loss

    fig = plot_loss(
        tracker.history["train_loss"],
        tracker.history["val_loss"],
        title=f"{model_name.upper()} Loss Plot",
    )
    fig.savefig(plots_dir / f"{model_name}_loss.png", dpi=150, bbox_inches="tight")
    log.info(
        "Saved loss plot: %s", (plots_dir / f"{model_name}_loss.png").relative_to(root)
    )
