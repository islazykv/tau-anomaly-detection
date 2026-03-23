"""Tests for src.models.anomaly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.anomaly import (
    build_scores_frame,
    compute_threshold,
    elbo_score,
    per_feature_error,
    reconstruction_error,
)

BATCH = 16
FEATURES = 10
LATENT = 4


class TestReconstructionError:
    def test_shape(self) -> None:
        x = torch.randn(BATCH, FEATURES)
        x_hat = torch.randn(BATCH, FEATURES)
        scores = reconstruction_error(x, x_hat)
        assert scores.shape == (BATCH,)

    def test_zero_for_identical(self) -> None:
        x = torch.randn(BATCH, FEATURES)
        scores = reconstruction_error(x, x.clone())
        assert scores.sum().item() == pytest.approx(0.0, abs=1e-7)

    def test_nonnegative(self) -> None:
        x = torch.randn(BATCH, FEATURES)
        x_hat = torch.randn(BATCH, FEATURES)
        scores = reconstruction_error(x, x_hat)
        assert (scores >= 0).all()

    def test_higher_for_worse_reconstruction(self) -> None:
        x = torch.randn(BATCH, FEATURES)
        x_close = x + 0.01 * torch.randn(BATCH, FEATURES)
        x_far = x + 10.0 * torch.randn(BATCH, FEATURES)
        scores_close = reconstruction_error(x, x_close)
        scores_far = reconstruction_error(x, x_far)
        assert scores_far.mean() > scores_close.mean()


class TestPerFeatureError:
    def test_shape(self) -> None:
        x = torch.randn(BATCH, FEATURES)
        x_hat = torch.randn(BATCH, FEATURES)
        errors = per_feature_error(x, x_hat)
        assert errors.shape == (BATCH, FEATURES)

    def test_mean_equals_reconstruction_error(self) -> None:
        x = torch.randn(BATCH, FEATURES)
        x_hat = torch.randn(BATCH, FEATURES)
        pf = per_feature_error(x, x_hat).mean(dim=1)
        re = reconstruction_error(x, x_hat)
        torch.testing.assert_close(pf, re)


class TestElboScore:
    def test_shape(self) -> None:
        x = torch.randn(BATCH, FEATURES)
        x_hat = torch.randn(BATCH, FEATURES)
        mu = torch.randn(BATCH, LATENT)
        logvar = torch.randn(BATCH, LATENT)
        scores = elbo_score(x, x_hat, mu, logvar)
        assert scores.shape == (BATCH,)

    def test_reduces_to_recon_when_kl_zero(self) -> None:
        x = torch.randn(BATCH, FEATURES)
        x_hat = torch.randn(BATCH, FEATURES)
        mu = torch.zeros(BATCH, LATENT)
        logvar = torch.zeros(BATCH, LATENT)
        elbo = elbo_score(x, x_hat, mu, logvar, beta=1.0)
        recon = reconstruction_error(x, x_hat)
        torch.testing.assert_close(elbo, recon)

    def test_beta_scales_kl(self) -> None:
        x = torch.randn(BATCH, FEATURES)
        x_hat = torch.randn(BATCH, FEATURES)
        mu = torch.randn(BATCH, LATENT)
        logvar = torch.randn(BATCH, LATENT)
        score_low = elbo_score(x, x_hat, mu, logvar, beta=0.1)
        score_high = elbo_score(x, x_hat, mu, logvar, beta=10.0)
        assert not torch.allclose(score_low, score_high)


class TestComputeThreshold:
    def test_percentile(self) -> None:
        scores = np.arange(100, dtype=np.float64)
        threshold = compute_threshold(scores, strategy="percentile", percentile=95.0)
        assert threshold == pytest.approx(95.0, abs=0.5)

    def test_std_dev(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.normal(0, 1, size=10000)
        threshold = compute_threshold(scores, strategy="std_dev", n_std=3.0)
        assert threshold == pytest.approx(3.0, abs=0.2)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown threshold strategy"):
            compute_threshold(np.array([1.0]), strategy="unknown")


class TestBuildScoresFrame:
    def test_columns(self) -> None:
        n = 50
        df = build_scores_frame(
            scores=np.random.default_rng(0).normal(size=n),
            labels=np.array([0] * 30 + [1] * 20),
            origins=np.array(["bkg"] * 30 + ["sig"] * 20),
        )
        assert list(df.columns) == ["anomaly_score", "sample_type", "eventOrigin"]
        assert len(df) == n

    def test_sample_type_values(self) -> None:
        labels = np.array([0, 0, 1, 1])
        df = build_scores_frame(
            scores=np.zeros(4),
            labels=labels,
            origins=np.array(["a", "a", "b", "b"]),
        )
        assert list(df["sample_type"]) == [
            "background",
            "background",
            "signal",
            "signal",
        ]
