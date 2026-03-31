"""Evaluation pipeline for trained AE and VAE models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pyrootutils
import torch
from omegaconf import DictConfig, OmegaConf

from src.models.ae import Autoencoder
from src.models.anomaly import (
    build_scores_frame,
    compute_threshold,
    reconstruction_error,
)
from src.models.config import AEConfig, VAEConfig
from src.models.datamodule import AnomalyDataModule
from src.models.evaluation import compute_metrics
from src.models.vae import VariationalAutoencoder
from src.processing.analysis import get_background_origins, get_output_paths
from src.processing.io import save_dataframe

log = logging.getLogger(__name__)


def _load_model(
    ckpt_path: Path,
    cfg: DictConfig,
    n_features: int,
) -> Autoencoder | VariationalAutoencoder:
    """Load a trained model from a Lightning checkpoint.

    Options for cfg.model.name: "ae", "vae".
    """
    model_params = dict(OmegaConf.to_container(cfg.model, resolve=True))  # type: ignore[arg-type]
    if cfg.model.name == "ae":
        model_cfg = AEConfig(**model_params)
        return Autoencoder.load_from_checkpoint(
            ckpt_path, cfg=model_cfg, n_features=n_features
        )
    if cfg.model.name == "vae":
        model_cfg = VAEConfig(**model_params)
        return VariationalAutoencoder.load_from_checkpoint(
            ckpt_path, cfg=model_cfg, n_features=n_features
        )
    msg = f"Unknown model: {cfg.model.name}"
    raise ValueError(msg)


def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained AE or VAE and save anomaly scores and metrics."""
    model_name = cfg.model.name
    log.info("Starting %s evaluation", model_name.upper())

    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])
    output_paths = get_output_paths(cfg)
    dataframes_dir = root / output_paths["dataframes_dir"]
    models_dir = root / output_paths["models_dir"]
    metrics_dir = root / output_paths["dataframes_dir"]

    # DataModule (for predict_dataloader, labels, origins)
    mc_path = dataframes_dir / "mc.parquet"
    background_origins = get_background_origins(cfg)

    dm = AnomalyDataModule(
        mc_path=str(mc_path),
        background_origins=background_origins,
        normalization=cfg.model.normalization,
        val_fraction=cfg.pipeline.val_fraction,
        batch_size=cfg.model.batch_size,
        seed=cfg.seed,
    )
    dm.setup()

    # Load model
    ckpt_path = models_dir / f"{model_name}.ckpt"
    model = _load_model(ckpt_path, cfg, n_features=dm.n_features)
    model.eval()
    log.info("Loaded checkpoint: %s", ckpt_path)

    # Predict on bkg_val + signal
    device = next(model.parameters()).device
    all_scores: list[np.ndarray] = []

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
    log.info("Computed anomaly scores for %d events", len(scores_array))

    # Build scores DataFrame
    scores_df = build_scores_frame(
        scores=scores_array,
        labels=dm.predict_labels,
        origins=dm.predict_origins,
    )

    # Threshold
    bkg_scores = scores_df[scores_df["sample_type"] == "background"][
        "anomaly_score"
    ].to_numpy()
    threshold = compute_threshold(
        bkg_scores,
        strategy=cfg.pipeline.threshold_strategy,
        percentile=cfg.pipeline.threshold_percentile,
    )
    log.info(
        "Anomaly threshold (%.1f%% percentile): %.6f",
        cfg.pipeline.threshold_percentile,
        threshold,
    )

    # Metrics
    metrics = compute_metrics(
        labels=dm.predict_labels,
        scores=scores_array,
        scores_df=scores_df,
    )
    metrics["threshold"] = threshold
    metrics["n_background"] = int((dm.predict_labels == 0).sum())
    metrics["n_signal"] = int((dm.predict_labels == 1).sum())

    # Save
    scores_path = dataframes_dir / f"{model_name}_scores.parquet"
    save_dataframe(scores_df, scores_path)

    metrics_path = metrics_dir / f"{model_name}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved metrics to %s", metrics_path)

    log.info(
        "%s evaluation complete — ROC AUC: %.4f, max SIC: %.4f",
        model_name.upper(),
        metrics["roc_auc"],
        metrics["max_sic"],
    )
