"""Tests for src.models.ae."""

from __future__ import annotations

import pytest
import torch

from src.models.ae import Autoencoder, _build_stack, _get_loss_fn
from src.models.config import AEConfig

N_FEATURES = 10
BATCH_SIZE = 16


@pytest.fixture()
def cfg() -> AEConfig:
    return AEConfig(hidden_sizes=[32, 16], latent_dim=8, dropout=0.1)


@pytest.fixture()
def model(cfg: AEConfig) -> Autoencoder:
    return Autoencoder(cfg, n_features=N_FEATURES)


@pytest.fixture()
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(BATCH_SIZE, N_FEATURES)
    w = torch.ones(BATCH_SIZE)
    return x, w


class TestForward:
    def test_output_shape(self, model: Autoencoder, batch: tuple) -> None:
        x, _ = batch
        x_hat = model(x)
        assert x_hat.shape == x.shape

    def test_encode_shape(self, model: Autoencoder, batch: tuple) -> None:
        x, _ = batch
        z = model.encode(x)
        assert z.shape == (BATCH_SIZE, 8)

    def test_deterministic(self, model: Autoencoder, batch: tuple) -> None:
        x, _ = batch
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)


class TestLoss:
    def test_scalar_output(self, model: Autoencoder, batch: tuple) -> None:
        x, w = batch
        x_hat = model(x)
        loss = model._compute_loss(x, x_hat, w)
        assert loss.ndim == 0

    def test_nonnegative(self, model: Autoencoder, batch: tuple) -> None:
        x, w = batch
        x_hat = model(x)
        loss = model._compute_loss(x, x_hat, w)
        assert loss.item() >= 0

    def test_zero_for_perfect_reconstruction(self, model: Autoencoder) -> None:
        x = torch.randn(BATCH_SIZE, N_FEATURES)
        w = torch.ones(BATCH_SIZE)
        loss = model._compute_loss(x, x.clone(), w)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_weights_affect_loss(self, model: Autoencoder, batch: tuple) -> None:
        x, _ = batch
        x_hat = model(x)
        w_ones = torch.ones(BATCH_SIZE)
        w_large = torch.full((BATCH_SIZE,), 10.0)
        loss_ones = model._compute_loss(x, x_hat, w_ones)
        loss_large = model._compute_loss(x, x_hat, w_large)
        assert loss_large.item() > loss_ones.item()


class TestLightningSteps:
    def test_training_step(self, model: Autoencoder, batch: tuple) -> None:
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_validation_step(self, model: Autoencoder, batch: tuple) -> None:
        loss = model.validation_step(batch, 0)
        assert loss.ndim == 0

    def test_predict_step(self, model: Autoencoder, batch: tuple) -> None:
        x_hat = model.predict_step(batch, 0)
        assert x_hat.shape == (BATCH_SIZE, N_FEATURES)


class TestOptimizer:
    def test_configure_optimizers_reduce_on_plateau(self, cfg: AEConfig) -> None:
        model = Autoencoder(cfg, n_features=N_FEATURES)
        opt_config = model.configure_optimizers()
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
        assert opt_config["lr_scheduler"]["monitor"] == "val_loss"

    def test_configure_optimizers_cosine(self) -> None:
        cfg = AEConfig(lr_scheduler="cosine_annealing")
        model = Autoencoder(cfg, n_features=N_FEATURES)
        opt_config = model.configure_optimizers()
        assert "lr_scheduler" in opt_config

    def test_configure_optimizers_none(self) -> None:
        cfg = AEConfig(lr_scheduler="none")
        model = Autoencoder(cfg, n_features=N_FEATURES)
        opt_config = model.configure_optimizers()
        assert "lr_scheduler" not in opt_config


class TestActivations:
    @pytest.mark.parametrize("activation", ["relu", "leaky_relu", "elu", "selu"])
    def test_all_activations_work(self, activation: str) -> None:
        cfg = AEConfig(hidden_sizes=[16], latent_dim=4, activation=activation)
        model = Autoencoder(cfg, n_features=N_FEATURES)
        x = torch.randn(4, N_FEATURES)
        x_hat = model(x)
        assert x_hat.shape == x.shape


class TestLossFunctions:
    @pytest.mark.parametrize("loss_name", ["mse", "smooth_l1", "bce"])
    def test_all_losses_work(self, loss_name: str) -> None:
        cfg = AEConfig(hidden_sizes=[16], latent_dim=4, loss=loss_name)
        model = Autoencoder(cfg, n_features=N_FEATURES)
        x = torch.randn(4, N_FEATURES)
        w = torch.ones(4)
        x_hat = model(x)
        loss = model._compute_loss(x, x_hat, w)
        assert loss.ndim == 0

    def test_unknown_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss function"):
            _get_loss_fn("unknown")


class TestBuildStack:
    def test_layer_count(self) -> None:
        stack = _build_stack([10, 8, 4], torch.nn.ReLU, dropout=0.0)
        linear_layers = [m for m in stack if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 2

    def test_no_final_activation(self) -> None:
        stack = _build_stack(
            [10, 4], torch.nn.ReLU, dropout=0.0, final_activation=False
        )
        assert isinstance(stack[-1], torch.nn.Linear)
