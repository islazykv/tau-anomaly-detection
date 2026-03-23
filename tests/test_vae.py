"""Tests for src.models.vae."""

from __future__ import annotations

import pytest
import torch

from src.models.config import VAEConfig
from src.models.vae import VariationalAutoencoder

N_FEATURES = 10
BATCH_SIZE = 16


@pytest.fixture()
def cfg() -> VAEConfig:
    return VAEConfig(hidden_sizes=[32, 16], latent_dim=8, dropout=0.1)


@pytest.fixture()
def model(cfg: VAEConfig) -> VariationalAutoencoder:
    return VariationalAutoencoder(cfg, n_features=N_FEATURES)


@pytest.fixture()
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(BATCH_SIZE, N_FEATURES)
    w = torch.ones(BATCH_SIZE)
    return x, w


class TestForward:
    def test_output_shapes(self, model: VariationalAutoencoder, batch: tuple) -> None:
        x, _ = batch
        x_hat, mu, logvar = model(x)
        assert x_hat.shape == x.shape
        assert mu.shape == (BATCH_SIZE, 8)
        assert logvar.shape == (BATCH_SIZE, 8)

    def test_encode_shapes(self, model: VariationalAutoencoder, batch: tuple) -> None:
        x, _ = batch
        mu, logvar = model.encode(x)
        assert mu.shape == (BATCH_SIZE, 8)
        assert logvar.shape == (BATCH_SIZE, 8)

    def test_decode_shape(self, model: VariationalAutoencoder) -> None:
        z = torch.randn(BATCH_SIZE, 8)
        x_hat = model.decode(z)
        assert x_hat.shape == (BATCH_SIZE, N_FEATURES)

    def test_eval_mode_deterministic(
        self, model: VariationalAutoencoder, batch: tuple
    ) -> None:
        x, _ = batch
        model.eval()
        with torch.no_grad():
            out1, _, _ = model(x)
            out2, _, _ = model(x)
        torch.testing.assert_close(out1, out2)

    def test_train_mode_stochastic(
        self, model: VariationalAutoencoder, batch: tuple
    ) -> None:
        x, _ = batch
        model.train()
        torch.manual_seed(0)
        out1, _, _ = model(x)
        torch.manual_seed(1)
        out2, _, _ = model(x)
        assert not torch.allclose(out1, out2)


class TestReparameterize:
    def test_returns_mu_in_eval(self, model: VariationalAutoencoder) -> None:
        mu = torch.randn(BATCH_SIZE, 8)
        logvar = torch.zeros(BATCH_SIZE, 8)
        model.eval()
        z = model.reparameterize(mu, logvar)
        torch.testing.assert_close(z, mu)

    def test_adds_noise_in_train(self, model: VariationalAutoencoder) -> None:
        mu = torch.zeros(BATCH_SIZE, 8)
        logvar = torch.zeros(BATCH_SIZE, 8)
        model.train()
        z = model.reparameterize(mu, logvar)
        assert not torch.allclose(z, mu)


class TestLogvarClamping:
    def test_logvar_clamped(self, model: VariationalAutoencoder) -> None:
        x = torch.randn(BATCH_SIZE, N_FEATURES) * 100  # extreme input
        _, logvar = model.encode(x)
        assert logvar.min().item() >= -10.0
        assert logvar.max().item() <= 10.0


class TestLoss:
    def test_returns_three_scalars(
        self, model: VariationalAutoencoder, batch: tuple
    ) -> None:
        x, w = batch
        x_hat, mu, logvar = model(x)
        total, recon, kl = model._compute_loss(x, x_hat, mu, logvar, w)
        assert total.ndim == 0
        assert recon.ndim == 0
        assert kl.ndim == 0

    def test_kl_nonnegative(self, model: VariationalAutoencoder, batch: tuple) -> None:
        x, w = batch
        x_hat, mu, logvar = model(x)
        _, _, kl = model._compute_loss(x, x_hat, mu, logvar, w)
        assert kl.item() >= 0

    def test_kl_zero_for_standard_normal(self, model: VariationalAutoencoder) -> None:
        x = torch.randn(BATCH_SIZE, N_FEATURES)
        x_hat = torch.randn(BATCH_SIZE, N_FEATURES)
        mu = torch.zeros(BATCH_SIZE, 8)
        logvar = torch.zeros(BATCH_SIZE, 8)
        w = torch.ones(BATCH_SIZE)
        _, _, kl = model._compute_loss(x, x_hat, mu, logvar, w)
        assert kl.item() == pytest.approx(0.0, abs=1e-6)

    def test_weights_affect_loss(
        self, model: VariationalAutoencoder, batch: tuple
    ) -> None:
        x, _ = batch
        x_hat, mu, logvar = model(x)
        w_ones = torch.ones(BATCH_SIZE)
        w_large = torch.full((BATCH_SIZE,), 10.0)
        total_ones, _, _ = model._compute_loss(x, x_hat, mu, logvar, w_ones)
        total_large, _, _ = model._compute_loss(x, x_hat, mu, logvar, w_large)
        assert total_large.item() > total_ones.item()


class TestBetaSchedule:
    def test_constant_beta(self) -> None:
        cfg = VAEConfig(beta=2.0, beta_schedule="constant")
        model = VariationalAutoencoder(cfg, n_features=N_FEATURES)
        assert model._get_beta() == 2.0

    def test_warmup_beta_at_start(self) -> None:
        cfg = VAEConfig(beta=1.0, beta_schedule="warmup", beta_warmup_epochs=10)
        model = VariationalAutoencoder(cfg, n_features=N_FEATURES)
        # current_epoch defaults to 0
        assert model._get_beta() == pytest.approx(0.0)

    def test_warmup_beta_midway(self) -> None:
        cfg = VAEConfig(beta=1.0, beta_schedule="warmup", beta_warmup_epochs=10)
        model = VariationalAutoencoder(cfg, n_features=N_FEATURES)
        model.trainer = type("Trainer", (), {"current_epoch": 5})()  # mock
        # LightningModule.current_epoch reads from trainer
        assert model._get_beta() == pytest.approx(0.5)

    def test_warmup_beta_after_warmup(self) -> None:
        cfg = VAEConfig(beta=1.0, beta_schedule="warmup", beta_warmup_epochs=10)
        model = VariationalAutoencoder(cfg, n_features=N_FEATURES)
        model.trainer = type("Trainer", (), {"current_epoch": 20})()
        assert model._get_beta() == pytest.approx(1.0)


class TestLightningSteps:
    def test_training_step(self, model: VariationalAutoencoder, batch: tuple) -> None:
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_validation_step(self, model: VariationalAutoencoder, batch: tuple) -> None:
        loss = model.validation_step(batch, 0)
        assert loss.ndim == 0

    def test_predict_step(self, model: VariationalAutoencoder, batch: tuple) -> None:
        x_hat = model.predict_step(batch, 0)
        assert x_hat.shape == (BATCH_SIZE, N_FEATURES)


class TestOptimizer:
    def test_configure_optimizers(self, model: VariationalAutoencoder) -> None:
        opt_config = model.configure_optimizers()
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config


class TestReconstructionLosses:
    @pytest.mark.parametrize("loss_name", ["mse", "smooth_l1", "bce"])
    def test_all_reconstruction_losses(self, loss_name: str) -> None:
        cfg = VAEConfig(hidden_sizes=[16], latent_dim=4, reconstruction_loss=loss_name)
        model = VariationalAutoencoder(cfg, n_features=N_FEATURES)
        x = torch.randn(4, N_FEATURES)
        w = torch.ones(4)
        x_hat, mu, logvar = model(x)
        total, recon, kl = model._compute_loss(x, x_hat, mu, logvar, w)
        assert total.ndim == 0
