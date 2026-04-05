"""Evaluation pipeline for trained AE and VAE models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pyrootutils
import torch
from omegaconf import DictConfig, OmegaConf

from src.models.ae import Autoencoder
from src.models.anomaly import (
    build_scores_frame,
    compute_threshold,
    per_feature_error,
    reconstruction_error,
)
from src.models.config import AEConfig, VAEConfig
from src.models.datamodule import AnomalyDataModule
from src.models.evaluation import compute_metrics, compute_roc_curve, compute_sic_curve
from src.models.plots import (
    plot_feature_histograms,
    plot_latent_histograms,
    plot_latent_pairplot,
    plot_per_feature_importance,
    plot_reconstruction_error,
    plot_reconstruction_performance,
    plot_roc_curve,
    plot_roc_per_sample_type,
    plot_sic_curve,
)
from src.models.vae import VariationalAutoencoder
from src.processing.analysis import get_background_origins, get_output_paths
from src.visualization.plots import save_figure

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
    """Evaluate a trained AE or VAE and save anomaly scores, metrics, and plots."""
    model_name = cfg.model.name
    log.info("Starting %s evaluation", model_name.upper())

    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])
    output_paths = get_output_paths(cfg)
    dataframes_dir = root / output_paths["dataframes_dir"]
    models_dir = root / output_paths["models_dir"]
    plots_dir = root / output_paths["plots_dir"] / f"{model_name}_evaluation"
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    all_x_hat: list[torch.Tensor] = []
    all_mu: list[torch.Tensor] = []
    all_logvar: list[torch.Tensor] = []
    all_x_orig: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dm.predict_dataloader():
            x, _w = batch
            x = x.to(device)
            all_x_orig.append(x.cpu())
            if model_name == "vae":
                x_hat, mu, logvar = model(x)
                all_mu.append(mu.cpu())
                all_logvar.append(logvar.cpu())
            else:
                x_hat = model(x)
            all_x_hat.append(x_hat.cpu())

    x_orig = torch.cat(all_x_orig)
    x_hat = torch.cat(all_x_hat)
    scores_array = reconstruction_error(x_orig, x_hat).numpy()
    log.info("Computed anomaly scores for %d events", len(scores_array))

    # Build scores DataFrame
    scores_df = build_scores_frame(
        scores=scores_array,
        labels=dm.predict_labels,
        origins=dm.predict_origins,
        sample_types=dm.predict_sample_types,
    )

    # Threshold
    bkg_scores = scores_array[dm.predict_labels == 0]
    sig_scores = scores_array[dm.predict_labels == 1]
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

    log.info(
        "%s ROC AUC: %.4f, max SIC: %.4f",
        model_name.upper(),
        metrics["roc_auc"],
        metrics["max_sic"],
    )

    # --- Plots ---

    # Reconstruction error distribution
    fig = plot_reconstruction_error(
        bkg_scores,
        sig_scores,
        threshold=threshold,
        title=f"{model_name.upper()} Reconstruction Error",
    )
    save_figure(fig, plots_dir / "reconstruction_error.png")

    # ROC curve
    fpr, tpr, _ = compute_roc_curve(dm.predict_labels, scores_array)
    fig = plot_roc_curve(
        fpr, tpr, auc=metrics["roc_auc"], title=f"{model_name.upper()} ROC Curve"
    )
    save_figure(fig, plots_dir / "roc_curve.png")

    # SIC curve
    sic = compute_sic_curve(fpr, tpr)
    fig = plot_sic_curve(
        fpr, tpr, sic, title=f"{model_name.upper()} Significance Improvement"
    )
    save_figure(fig, plots_dir / "sic_curve.png")

    # Per-sample-type ROC
    if metrics.get("roc_per_sample_type"):
        display_labels = OmegaConf.to_container(cfg.merge.display_labels, resolve=True)
        fig = plot_roc_per_sample_type(
            metrics["roc_per_sample_type"],
            title=f"{model_name.upper()} ROC AUC per Sample Type",
            display_labels=display_labels,
        )
        save_figure(fig, plots_dir / "roc_per_sample_type.png")

    # Per-feature importance
    feat_errors = per_feature_error(x_orig, x_hat).numpy()
    mean_feat_errors = feat_errors.mean(axis=0)
    fig = plot_per_feature_importance(
        mean_feat_errors,
        dm.feature_names_,
        title=f"{model_name.upper()} Per-Feature Reconstruction Error",
    )
    save_figure(fig, plots_dir / "per_feature_importance.png")

    # Single event reconstruction
    fig = plot_reconstruction_performance(
        x_orig[0].numpy(),
        x_hat[0].numpy(),
        dm.feature_names_,
        event_idx=0,
        title=f"Single Event Reconstruction ({model_name.upper()})",
    )
    save_figure(fig, plots_dir / "single_event.png")

    # Feature histograms
    fig = plot_feature_histograms(
        x_orig.numpy(),
        x_hat.numpy(),
        dm.feature_names_,
        title=f"Feature Distributions ({model_name.upper()})",
    )
    save_figure(fig, plots_dir / "feature_histograms.png")

    # Latent space
    if model_name == "ae":
        with torch.no_grad():
            z = model.encode(x_orig)
            assert isinstance(z, torch.Tensor)
            z_np = z.numpy()
        fig = plot_latent_histograms(
            z_np,
            labels=dm.predict_labels,
            title=f"Latent Dimension Histograms ({model_name.upper()})",
        )
        save_figure(fig, plots_dir / "latent_histograms.png")
        fig = plot_latent_pairplot(
            z_np,
            labels=dm.predict_labels,
            title=f"{model_name.upper()} Latent Pairplot",
        )
        save_figure(fig, plots_dir / "latent_pairplot.png")

    elif model_name == "vae":
        mu_np = torch.cat(all_mu).numpy()
        logvar_np = torch.cat(all_logvar).numpy()

        fig = plot_latent_histograms(mu_np, labels=dm.predict_labels)
        save_figure(fig, plots_dir / "latent_histograms.png")

        fig = plot_latent_pairplot(mu_np, labels=dm.predict_labels)
        save_figure(fig, plots_dir / "latent_pairplot.png")

        from src.models.plots import (
            plot_latent_mean_spread,
            plot_logvar_spread,
            plot_mu_vs_logvar,
        )

        fig = plot_latent_mean_spread(mu_np)
        save_figure(fig, plots_dir / "mu_spread.png")

        fig = plot_logvar_spread(logvar_np)
        save_figure(fig, plots_dir / "logvar_spread.png")

        fig = plot_mu_vs_logvar(mu_np, logvar_np)
        save_figure(fig, plots_dir / "mu_vs_logvar.png")

        from src.models.latent import compute_kl_per_dimension
        from src.models.plots import plot_kl_per_dimension

        kl_per_dim = compute_kl_per_dimension(mu_np, logvar_np)
        fig = plot_kl_per_dimension(kl_per_dim)
        save_figure(fig, plots_dir / "kl_per_dim.png")

    log.info("Saved evaluation plots to %s", plots_dir.relative_to(root))

    # --- Save ---

    scores_path = dataframes_dir / f"{model_name}_scores.parquet"
    scores_df.to_parquet(scores_path)

    metrics_path = dataframes_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved metrics to %s", metrics_path)

    log.info(
        "%s evaluation complete — ROC AUC: %.4f, max SIC: %.4f",
        model_name.upper(),
        metrics["roc_auc"],
        metrics["max_sic"],
    )
