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
    """Compute t-SNE embedding of latent vectors.

    Args:
        z: Latent vectors, shape ``(n_events, latent_dim)``.
        n_components: Number of t-SNE dimensions (2 or 3).
        perplexity: t-SNE perplexity parameter.
        seed: Random seed for reproducibility.
        max_samples: Subsample if more events than this (t-SNE is O(n^2)).

    Returns:
        t-SNE embedding, shape ``(n_events, n_components)`` or
        ``(max_samples, n_components)`` if subsampled.
    """
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
    """Compute UMAP embedding of latent vectors.

    Requires ``umap-learn`` to be installed. Falls back to t-SNE if not
    available.

    Args:
        z: Latent vectors, shape ``(n_events, latent_dim)``.
        n_components: Number of UMAP dimensions.
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance parameter for UMAP.
        seed: Random seed for reproducibility.
        max_samples: Subsample if more events than this.

    Returns:
        UMAP embedding, shape ``(n_events, n_components)`` or subsampled.
    """
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
    """Compute mean KL divergence per latent dimension.

    Useful for identifying collapsed or uninformative latent dimensions.

    Args:
        mu: Latent means, shape ``(n_events, latent_dim)``.
        logvar: Latent log-variances, shape ``(n_events, latent_dim)``.

    Returns:
        Mean KL per dimension, shape ``(latent_dim,)``.
    """
    kl = -0.5 * (1 + logvar - mu**2 - np.exp(logvar))
    return kl.mean(axis=0)
