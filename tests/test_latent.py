"""Tests for src.models.latent."""

from __future__ import annotations

import numpy as np

from src.models.latent import compute_kl_per_dimension, compute_tsne


class TestComputeTsne:
    def test_output_shape_2d(self) -> None:
        rng = np.random.default_rng(42)
        z = rng.normal(size=(200, 8))
        embedding = compute_tsne(z, n_components=2, perplexity=10.0)
        assert embedding.shape == (200, 2)

    def test_subsampling(self) -> None:
        rng = np.random.default_rng(42)
        z = rng.normal(size=(500, 4))
        embedding = compute_tsne(z, max_samples=100, perplexity=10.0)
        assert embedding.shape == (100, 2)

    def test_reproducible(self) -> None:
        rng = np.random.default_rng(42)
        z = rng.normal(size=(100, 4))
        e1 = compute_tsne(z, seed=1, perplexity=10.0)
        e2 = compute_tsne(z, seed=1, perplexity=10.0)
        np.testing.assert_array_equal(e1, e2)


class TestKlPerDimension:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.normal(size=(100, 8))
        logvar = rng.normal(size=(100, 8))
        kl = compute_kl_per_dimension(mu, logvar)
        assert kl.shape == (8,)

    def test_zero_for_standard_normal(self) -> None:
        mu = np.zeros((1000, 4))
        logvar = np.zeros((1000, 4))
        kl = compute_kl_per_dimension(mu, logvar)
        np.testing.assert_allclose(kl, 0.0, atol=1e-7)

    def test_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.normal(size=(1000, 4))
        logvar = rng.normal(size=(1000, 4))
        kl = compute_kl_per_dimension(mu, logvar)
        assert (kl >= 0).all()

    def test_higher_for_offset_mu(self) -> None:
        mu_zero = np.zeros((1000, 4))
        mu_offset = np.full((1000, 4), 5.0)
        logvar = np.zeros((1000, 4))
        kl_zero = compute_kl_per_dimension(mu_zero, logvar)
        kl_offset = compute_kl_per_dimension(mu_offset, logvar)
        assert (kl_offset > kl_zero).all()
