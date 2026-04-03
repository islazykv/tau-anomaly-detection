"""Diagnostic and evaluation plots for AE/VAE anomaly detection."""

from __future__ import annotations

import logging

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 33 — Training diagnostics
# ---------------------------------------------------------------------------


def plot_loss(
    train_loss: list[float],
    val_loss: list[float],
    title: str = "Loss Plot",
    loss_type: str = "MSE",
) -> plt.Figure:
    """Plot training and validation loss curves with the best epoch marked."""
    epochs = range(1, len(train_loss) + 1)
    best_epoch = int(np.argmin(val_loss)) + 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, train_loss, label="Training loss", linewidth=1.5, color="green")
    ax.plot(epochs, val_loss, label="Validation loss", linewidth=1.5, color="red")
    ax.axvline(
        best_epoch,
        color="blue",
        linestyle="--",
        linewidth=1.0,
        label=f"Best epoch: {best_epoch}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss ({loss_type})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ampl.draw_atlas_label(0.02, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


def plot_loss_components(
    recon_loss: list[float],
    kl_loss: list[float],
    beta_values: list[float] | None = None,
    title: str = "VAE Loss Components",
) -> plt.Figure:
    """Plot VAE reconstruction and KL loss components."""
    n_epochs = len(recon_loss)
    epochs = range(1, n_epochs + 1)
    n_panels = 3 if beta_values is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    axes[0].plot(epochs, recon_loss)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction Loss")
    axes[0].set_title("Reconstruction")
    axes[0].set_xlim(0, n_epochs + 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, kl_loss)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL Divergence")
    axes[1].set_title("KL")
    axes[1].set_xlim(0, n_epochs + 1)
    axes[1].grid(True, alpha=0.3)

    if beta_values is not None:
        axes[2].plot(epochs, beta_values)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Beta")
        axes[2].set_title("Beta Schedule")
        axes[2].set_xlim(0, n_epochs + 1)
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=axes[0])
    return fig


def plot_reconstruction_error(
    bkg_scores: np.ndarray,
    sig_scores: np.ndarray,
    threshold: float | None = None,
    title: str = "Reconstruction Error Distribution",
    n_bins: int = 100,
) -> plt.Figure:
    """Plot reconstruction error distributions for background vs signal."""
    fig, ax = plt.subplots()
    ax.hist(bkg_scores, bins=n_bins, alpha=0.6, label="Background", density=True)
    ax.hist(sig_scores, bins=n_bins, alpha=0.6, label="Signal", density=True)
    if threshold is not None:
        ax.axvline(
            threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}"
        )
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    return fig


# ---------------------------------------------------------------------------
# Step 34 — Reconstruction plots
# ---------------------------------------------------------------------------


def plot_reconstruction_performance(
    x_original: np.ndarray,
    x_reconstructed: np.ndarray,
    feature_names: list[str],
    event_idx: int = 0,
    title: str = "Single Event Reconstruction",
) -> plt.Figure:
    """Bar chart comparing original vs reconstructed features for one event."""
    fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 0.4), 5))
    x_pos = np.arange(len(feature_names))
    width = 0.35

    ax.bar(x_pos - width / 2, x_original, width, label="Original", alpha=0.8)
    ax.bar(x_pos + width / 2, x_reconstructed, width, label="Reconstructed", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_xlim(-0.5, len(feature_names) - 0.5)
    ax.set_ylabel("Feature Value")
    ax.set_title(f"{title} (event {event_idx})")
    ax.legend()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


def plot_reconstruction_comparison(
    ae_scores: np.ndarray,
    vae_scores: np.ndarray,
    labels: np.ndarray,
    title: str = "AE vs VAE Reconstruction Error",
    n_bins: int = 80,
) -> plt.Figure:
    """Side-by-side reconstruction error distributions for AE and VAE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, scores, name in zip(axes, [ae_scores, vae_scores], ["AE", "VAE"]):
        bkg = scores[labels == 0]
        sig = scores[labels == 1]
        ax.hist(bkg, bins=n_bins, alpha=0.6, label="Background", density=True)
        ax.hist(sig, bins=n_bins, alpha=0.6, label="Signal", density=True)
        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Density")
        ax.set_title(name)
        ax.legend()
        ax.set_yscale("log")

    fig.suptitle(title)
    fig.tight_layout()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=axes[0])
    return fig


def plot_feature_histograms(
    x_original: np.ndarray,
    x_reconstructed: np.ndarray,
    feature_names: list[str],
    n_cols: int = 3,
    n_bins: int = 50,
    title: str = "Feature Distributions: Original vs Reconstructed",
) -> plt.Figure:
    """Per-feature histograms comparing original and reconstructed values."""
    n_features = len(feature_names)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    axes_flat = axes.flatten() if n_features > 1 else [axes]

    for i, (ax, name) in enumerate(zip(axes_flat, feature_names)):
        ax.hist(
            x_original[:, i],
            bins=n_bins,
            alpha=0.6,
            label="Original",
            density=True,
            histtype="stepfilled",
        )
        ax.hist(
            x_reconstructed[:, i],
            bins=n_bins,
            alpha=0.6,
            label="Reconstructed",
            density=True,
            histtype="stepfilled",
        )
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend()

    for ax in axes_flat[n_features:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.suptitle(title, y=1.01)
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=axes_flat[0])
    return fig


# ---------------------------------------------------------------------------
# Step 35 — VAE latent diagnostics
# ---------------------------------------------------------------------------


def plot_latent_histograms(
    z: np.ndarray,
    labels: np.ndarray | None = None,
    n_cols: int = 4,
    title: str = "Latent Dimension Histograms",
) -> plt.Figure:
    """Histogram of each latent dimension, optionally colored by label."""
    latent_dim = z.shape[1]
    n_rows = (latent_dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes_flat = axes.flatten() if latent_dim > 1 else [axes]

    for i, ax in enumerate(axes_flat[:latent_dim]):
        if labels is not None:
            for lbl, name in [(0, "Background"), (1, "Signal")]:
                ax.hist(
                    z[labels == lbl, i], bins=50, alpha=0.6, label=name, density=True
                )
            ax.legend(fontsize=6)
        else:
            ax.hist(z[:, i], bins=50, alpha=0.7, density=True)
        ax.set_title(f"z[{i}]", fontsize=8)

    for ax in axes_flat[latent_dim:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=axes_flat[0])
    return fig


def plot_latent_space_2d(
    embedding: np.ndarray,
    labels: np.ndarray,
    method: str = "t-SNE",
    title: str | None = None,
) -> plt.Figure:
    """2D scatter plot of latent space embedding colored by label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, name, color in [(0, "Background", "C0"), (1, "Signal", "C1")]:
        mask = labels == lbl
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=2,
            alpha=0.3,
            label=name,
            color=color,
        )
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(title or f"Latent Space ({method})")
    ax.legend(markerscale=5)
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    return fig


def plot_latent_pairplot(
    z: np.ndarray,
    labels: np.ndarray,
    max_dims: int = 6,
    title: str = "Latent Pairplot",
) -> plt.Figure:
    """Pairplot of first ``max_dims`` latent dimensions."""
    dims = min(z.shape[1], max_dims)
    df = pd.DataFrame(z[:, :dims], columns=[f"z[{i}]" for i in range(dims)])
    df["label"] = np.where(labels == 0, "Background", "Signal")

    g = sns.pairplot(df, hue="label", plot_kws={"s": 5, "alpha": 0.3}, corner=True)
    g.figure.suptitle(title, y=1.02)
    return g.figure


def plot_latent_mean_histograms(
    mu: np.ndarray,
    n_cols: int = 4,
    title: str = "Latent Mean (mu) Histograms",
) -> plt.Figure:
    """Histogram of mu per latent dimension."""
    return _plot_per_dim_histograms(mu, n_cols=n_cols, title=title, dim_prefix="mu")


def plot_latent_mean_spread(
    mu: np.ndarray,
    title: str = "Latent Mean Spread per Dimension",
) -> plt.Figure:
    """Variance of mu per dimension. Warns if var < 0.1 (potential collapse)."""
    variances = mu.var(axis=0)
    fig, ax = plt.subplots()
    dims = np.arange(len(variances))
    colors = ["red" if v < 0.1 else "C0" for v in variances]
    ax.bar(dims, variances, color=colors)
    ax.axhline(0.1, color="red", linestyle="--", alpha=0.5, label="Collapse threshold")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Var(mu)")
    ax.set_title(title)
    ax.legend()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)

    n_collapsed = (variances < 0.1).sum()
    if n_collapsed > 0:
        log.warning(
            "%d latent dimensions with mu.var < 0.1 (potential collapse)", n_collapsed
        )
    return fig


def plot_latent_mean_boxplot(
    mu: np.ndarray,
    title: str = "Latent Mean Distribution per Dimension",
) -> plt.Figure:
    """Boxplot of mu per latent dimension. Shows full distribution relative to prior mean=0."""
    n_dims = mu.shape[1]
    fig, ax = plt.subplots(figsize=(max(8, n_dims * 0.8), 5))
    ax.boxplot(mu, patch_artist=True, medianprops={"color": "C1"})
    ax.axhline(0, color="grey", linestyle=":", alpha=0.5, label="Prior mean = 0")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("mu")
    ax.set_title(title)
    ax.set_xticks(range(1, n_dims + 1))
    ax.set_xticklabels([str(i) for i in range(n_dims)])
    ax.legend()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    return fig


def plot_logvar_histograms(
    logvar: np.ndarray,
    n_cols: int = 4,
    title: str = "Log-Variance Histograms",
) -> plt.Figure:
    """Histogram of logvar per latent dimension."""
    return _plot_per_dim_histograms(
        logvar, n_cols=n_cols, title=title, dim_prefix="logvar"
    )


def plot_logvar_spread(
    logvar: np.ndarray,
    title: str = "Log-Variance Spread per Dimension",
) -> plt.Figure:
    """Mean logvar per dimension. Warns if logvar < -5 (potential collapse)."""
    means = logvar.mean(axis=0)
    fig, ax = plt.subplots()
    dims = np.arange(len(means))
    colors = ["red" if m < -8 else "C0" for m in means]
    ax.bar(dims, means, color=colors)
    ax.axhline(-8, color="red", linestyle="--", alpha=0.5, label="Collapse threshold")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Mean(logvar)")
    ax.set_title(title)
    ax.legend()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)

    n_collapsed = (means < -8).sum()
    if n_collapsed > 0:
        log.warning(
            "%d latent dimensions with logvar < -8 (potential collapse)", n_collapsed
        )
    return fig


def plot_logvar_boxplot(
    logvar: np.ndarray,
    title: str = "Log-Variance Distribution per Dimension",
) -> plt.Figure:
    """Boxplot of logvar per latent dimension. Shows full distribution relative to prior logvar=0."""
    n_dims = logvar.shape[1]
    fig, ax = plt.subplots(figsize=(max(8, n_dims * 0.8), 5))
    ax.boxplot(logvar, patch_artist=True, medianprops={"color": "C1"})
    ax.axhline(0, color="grey", linestyle=":", alpha=0.5, label="Prior logvar = 0")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("logvar")
    ax.set_title(title)
    ax.set_xticks(range(1, n_dims + 1))
    ax.set_xticklabels([str(i) for i in range(n_dims)])
    ax.legend()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    return fig


def plot_mu_vs_logvar(
    mu: np.ndarray,
    logvar: np.ndarray,
    title: str = "Mu vs Logvar per Dimension",
) -> plt.Figure:
    """Scatter of mean(mu) vs mean(logvar) per latent dimension."""
    mu_means = mu.mean(axis=0)
    logvar_means = logvar.mean(axis=0)

    fig, ax = plt.subplots()
    ax.scatter(mu_means, logvar_means, s=40)
    for i, (x, y) in enumerate(zip(mu_means, logvar_means)):
        ax.annotate(str(i), (x, y), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Mean(mu)")
    ax.set_ylabel("Mean(logvar)")
    ax.set_title(title)
    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    ax.axvline(0, color="grey", linestyle=":", alpha=0.5)
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    return fig


def plot_kl_per_dimension(
    kl_per_dim: np.ndarray,
    title: str = "KL Divergence per Latent Dimension",
) -> plt.Figure:
    """Bar chart of mean KL per latent dimension."""
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(kl_per_dim)), kl_per_dim)
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Mean KL Divergence")
    ax.set_title(title)
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    return fig


def plot_sampled_latent_space(
    z_sampled: np.ndarray,
    z_encoded: np.ndarray,
    title: str = "Sampled vs Encoded Latent Space",
) -> plt.Figure:
    """Compare sampled (from prior) and encoded latent vectors (first 2 dims)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        z_encoded[:, 0], z_encoded[:, 1], s=2, alpha=0.3, label="Encoded", color="C0"
    )
    ax.scatter(
        z_sampled[:, 0],
        z_sampled[:, 1],
        s=2,
        alpha=0.3,
        label="Sampled (prior)",
        color="C1",
    )
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_title(title)
    ax.legend(markerscale=5)
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    return fig


# ---------------------------------------------------------------------------
# Step 36 — Evaluation plots
# ---------------------------------------------------------------------------


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    title: str = "ROC Curve",
) -> plt.Figure:
    """Plot ROC curve with AUC annotation."""
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Background Efficiency (FPR)")
    ax.set_ylabel("Signal Efficiency (TPR)")
    ax.set_title(title)
    ax.legend()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    return fig


def plot_sic_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    sic: np.ndarray,
    title: str = "Significance Improvement Characteristic",
) -> plt.Figure:
    """Plot SIC curve (signal_eff / sqrt(bkg_eff))."""
    fig, ax = plt.subplots()
    # Plot SIC vs signal efficiency
    ax.plot(tpr, sic)
    ax.set_xlabel("Signal Efficiency (TPR)")
    ax.set_ylabel("SIC = TPR / sqrt(FPR)")
    ax.set_title(title)
    max_idx = np.nanargmax(sic)
    ax.axhline(
        sic[max_idx],
        color="red",
        linestyle="--",
        alpha=0.4,
        label=f"Max SIC = {sic[max_idx]:.2f}",
    )
    ax.legend()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    return fig


def plot_roc_per_origin(
    roc_per_origin: dict[str, float],
    title: str = "ROC AUC per Signal Origin",
) -> plt.Figure:
    """Horizontal bar chart of ROC AUC per signal origin."""
    origins = sorted(roc_per_origin.keys())
    aucs = [roc_per_origin[o] for o in origins]

    fig, ax = plt.subplots(figsize=(8, max(4, len(origins) * 0.3)))
    y_pos = np.arange(len(origins))
    ax.barh(y_pos, aucs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(origins, fontsize=7)
    ax.set_xlabel("ROC AUC")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


def plot_per_feature_importance(
    mean_errors: np.ndarray,
    feature_names: list[str],
    title: str = "Per-Feature Reconstruction Error (Importance)",
) -> plt.Figure:
    """Bar chart of mean per-feature reconstruction error."""
    order = np.argsort(mean_errors)[::-1]
    fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 0.4), 5))
    ax.bar(np.arange(len(feature_names)), mean_errors[order])
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("Mean Squared Error")
    ax.set_title(title)
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tuning analysis plots
# ---------------------------------------------------------------------------

_HP_DISPLAY_NAMES = {
    "config/n_layers": "n_layers",
    "config/layer_size": "layer_size",
    "config/latent_dim": "latent_dim",
    "config/dropout": "dropout",
    "config/learning_rate": "learning_rate",
    "config/weight_decay": "weight_decay",
    "config/batch_size": "batch_size",
    "config/beta": "beta",
}


def _get_hp_columns(trial_df: pd.DataFrame) -> list[str]:
    """Return hyperparameter columns present in the trial DataFrame."""
    return [c for c in _HP_DISPLAY_NAMES if c in trial_df.columns]


def plot_optimization_history(
    trial_df: pd.DataFrame,
    title: str = "Optimization History",
) -> plt.Figure:
    """Scatter of val_loss per trial with a running-best line overlay."""
    df = trial_df.sort_values("trial_id").reset_index(drop=True)
    x = np.arange(1, len(df) + 1)
    vals = df["val_loss"].to_numpy()
    running_best = np.minimum.accumulate(vals)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(x, vals, s=18, alpha=0.6, zorder=2, label="Objective Value")
    ax.plot(x, running_best, color="C1", linewidth=2, zorder=3, label="Best Value")
    margin = max(1, len(df) * 0.03)
    ax.set_xlim(1 - margin, len(df) + margin)
    ax.set_xlabel("Trial")
    ax.ticklabel_format(axis="y", style="plain")
    ax.set_ylabel("Objective Value")
    ax.set_title(title)
    ax.legend()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


def plot_hyperparameter_importance(
    trial_df: pd.DataFrame,
    title: str = "Hyperparameter Importances",
) -> plt.Figure:
    """Absolute Spearman correlation of each HP with val_loss."""
    hp_cols = _get_hp_columns(trial_df)
    correlations = {}
    for col in hp_cols:
        vals = pd.to_numeric(trial_df[col], errors="coerce")
        if vals.nunique() > 1:
            correlations[_HP_DISPLAY_NAMES[col]] = abs(
                vals.corr(trial_df["val_loss"], method="spearman")
            )

    names = list(correlations.keys())
    values = list(correlations.values())
    order = np.argsort(values)[::-1]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(range(len(names)), [values[i] for i in order])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right")
    ax.set_xlabel("Hyperparameter")
    ax.set_ylabel("Hyperparameter Importance")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


def plot_parallel_coordinates(
    trial_df: pd.DataFrame,
    title: str = "Parallel Coordinates",
) -> plt.Figure:
    """Parallel coordinates plot of HPs colored by val_loss."""
    hp_cols = _get_hp_columns(trial_df)
    df = trial_df[hp_cols + ["val_loss"]].copy()
    df.columns = [_HP_DISPLAY_NAMES.get(c, c) for c in df.columns]

    # Normalize each column to [0, 1] for display
    hp_names = [_HP_DISPLAY_NAMES[c] for c in hp_cols]
    df_norm = df.copy()
    for col in hp_names:
        vals = pd.to_numeric(df_norm[col], errors="coerce")
        vmin, vmax = vals.min(), vals.max()
        df_norm[col] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.5

    fig, ax = plt.subplots(figsize=(16, 9))
    cmap = plt.cm.viridis_r
    vmin, vmax = df["val_loss"].min(), df["val_loss"].max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for _, row in df_norm.iterrows():
        vals = [float(row[c]) for c in hp_names]
        color = cmap(norm(row["val_loss"]))
        ax.plot(range(len(hp_names)), vals, alpha=0.6, color=color, linewidth=1.5)

    ax.set_xticks(range(len(hp_names)))
    ax.set_xticklabels(hp_names, rotation=45, ha="right")
    ax.set_ylabel("Normalized value")
    ax.set_title(title)
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="val_loss")
    fig.tight_layout()
    return fig


def plot_hp_vs_objective(
    trial_df: pd.DataFrame,
    n_cols: int = 3,
    title: str = "Hyperparameters vs val_loss",
) -> plt.Figure:
    """Scatter plot of each HP vs val_loss."""
    hp_cols = _get_hp_columns(trial_df)
    n = len(hp_cols)
    n_rows = max(1, (n + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, col in zip(axes_flat, hp_cols):
        vals = pd.to_numeric(trial_df[col], errors="coerce")
        ax.scatter(vals, trial_df["val_loss"], alpha=0.7, edgecolors="k", linewidth=0.3)
        ax.set_xlabel(_HP_DISPLAY_NAMES[col])
        ax.set_ylabel("val_loss")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for ax in axes_flat[:n]:
        ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_hp_contour(
    trial_df: pd.DataFrame,
    params: tuple[str, str] | None = None,
    title: str = "Hyperparameter Contour",
    n_bins: int = 20,
) -> plt.Figure:
    """Pairwise contour/heatmap of two HPs colored by val_loss.

    If *params* is not given, the two HPs with the highest absolute Spearman
    correlation with ``val_loss`` are chosen automatically.
    """
    hp_cols = _get_hp_columns(trial_df)

    if params is not None:
        col_x = f"config/{params[0]}" if f"config/{params[0]}" in hp_cols else params[0]
        col_y = f"config/{params[1]}" if f"config/{params[1]}" in hp_cols else params[1]
    else:
        # Pick the two most important HPs by Spearman correlation
        correlations: list[tuple[str, float]] = []
        for col in hp_cols:
            vals = pd.to_numeric(trial_df[col], errors="coerce")
            if vals.nunique() > 1:
                correlations.append(
                    (col, abs(vals.corr(trial_df["val_loss"], method="spearman")))
                )
        correlations.sort(key=lambda t: t[1], reverse=True)
        col_x = correlations[0][0]
        col_y = correlations[1][0] if len(correlations) > 1 else correlations[0][0]

    from scipy.interpolate import griddata

    x = pd.to_numeric(trial_df[col_x], errors="coerce").to_numpy()
    y = pd.to_numeric(trial_df[col_y], errors="coerce").to_numpy()
    z = trial_df["val_loss"].to_numpy()

    # Build interpolated grid for contour/heatmap
    xi = np.linspace(x.min(), x.max(), n_bins)
    yi = np.linspace(y.min(), y.max(), n_bins)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    zi_grid = griddata((x, y), z, (xi_grid, yi_grid), method="cubic")

    fig, ax = plt.subplots(figsize=(16, 9))
    cmap = plt.cm.viridis_r
    # Heatmap
    im = ax.contourf(xi_grid, yi_grid, zi_grid, levels=20, cmap=cmap, alpha=0.8)
    # Contour lines
    ax.contour(
        xi_grid, yi_grid, zi_grid, levels=20, colors="k", linewidths=0.3, alpha=0.4
    )
    # Scatter overlay
    ax.scatter(x, y, c=z, cmap=cmap, s=30, edgecolors="k", linewidth=0.5, zorder=3)

    ax.set_xlabel(_HP_DISPLAY_NAMES.get(col_x, col_x))
    ax.set_ylabel(_HP_DISPLAY_NAMES.get(col_y, col_y))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Objective Value")
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plot_per_dim_histograms(
    values: np.ndarray,
    n_cols: int = 4,
    title: str = "",
    dim_prefix: str = "dim",
) -> plt.Figure:
    """Generic per-dimension histogram grid."""
    n_dims = values.shape[1]
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes_flat = axes.flatten() if n_dims > 1 else [axes]

    for i, ax in enumerate(axes_flat[:n_dims]):
        ax.hist(values[:, i], bins=50, alpha=0.7, density=True)
        ax.set_title(f"{dim_prefix}[{i}]", fontsize=8)

    for ax in axes_flat[n_dims:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    ampl.draw_atlas_label(0.05, 0.97, simulation=True, status="final", ax=axes_flat[0])
    return fig
