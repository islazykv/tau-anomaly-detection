"""Latent space analysis utilities (t-SNE, UMAP)."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.manifold import TSNE

log = logging.getLogger(__name__)


def compute_tsne(
    z: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    seed: int = 1,
    max_samples: int = 10000,
) -> np.ndarray:
    """Compute t-SNE embedding of latent vectors, subsampling if needed."""
    if len(z) > max_samples:
        log.info("Subsampling %d -> %d for t-SNE", len(z), max_samples)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(z), max_samples, replace=False)
        z = z[idx]

    log.info("Running t-SNE on %d events (latent_dim=%d)", len(z), z.shape[1])
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(z)


def compute_umap(
    z: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 1,
    max_samples: int = 50000,
) -> np.ndarray:
    """Compute UMAP embedding of latent vectors, falling back to t-SNE if umap-learn is missing."""
    try:
        import umap
    except ImportError:
        log.warning("umap-learn not installed, falling back to t-SNE")
        return compute_tsne(
            z, n_components=n_components, seed=seed, max_samples=max_samples
        )

    if len(z) > max_samples:
        log.info("Subsampling %d -> %d for UMAP", len(z), max_samples)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(z), max_samples, replace=False)
        z = z[idx]

    log.info("Running UMAP on %d events (latent_dim=%d)", len(z), z.shape[1])
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(z)


def compute_kl_per_dimension(
    mu: np.ndarray,
    logvar: np.ndarray,
) -> np.ndarray:
    """Compute mean KL divergence per latent dimension for collapse detection."""
    kl = -0.5 * (1 + logvar - mu**2 - np.exp(logvar))
    return kl.mean(axis=0)
