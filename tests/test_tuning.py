"""Tests for src.models.tuning."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.models.config import AEConfig, VAEConfig
from src.models.tuning import (
    build_hidden_sizes,
    build_search_space,
    export_best_config,
    make_model_config,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def tuning_cfg():
    """Minimal tuning config matching the YAML structure."""
    return OmegaConf.create(
        {
            "search_space": {
                "ae": {
                    "n_layers": {"type": "int", "low": 2, "high": 5},
                    "layer_size": {"type": "categorical", "choices": [32, 64, 128]},
                    "latent_dim": {"type": "categorical", "choices": [4, 8, 16]},
                    "dropout": {"type": "float", "low": 0.0, "high": 0.5, "log": False},
                    "learning_rate": {
                        "type": "float",
                        "low": 1e-5,
                        "high": 1e-2,
                        "log": True,
                    },
                    "weight_decay": {
                        "type": "float",
                        "low": 1e-6,
                        "high": 1e-3,
                        "log": True,
                    },
                    "batch_size": {"type": "categorical", "choices": [256, 512]},
                },
                "vae": {
                    "n_layers": {"type": "int", "low": 2, "high": 5},
                    "layer_size": {"type": "categorical", "choices": [32, 64, 128]},
                    "latent_dim": {"type": "categorical", "choices": [4, 8, 16]},
                    "dropout": {"type": "float", "low": 0.0, "high": 0.5, "log": False},
                    "learning_rate": {
                        "type": "float",
                        "low": 1e-5,
                        "high": 1e-2,
                        "log": True,
                    },
                    "weight_decay": {
                        "type": "float",
                        "low": 1e-6,
                        "high": 1e-3,
                        "log": True,
                    },
                    "batch_size": {"type": "categorical", "choices": [256, 512]},
                    "beta": {"type": "float", "low": 0.001, "high": 10.0, "log": True},
                },
            },
        }
    )


@pytest.fixture()
def base_ae_cfg():
    """Base Hydra config with AE model defaults."""
    return OmegaConf.create(
        {
            "model": {
                "name": "ae",
                "hidden_sizes": [128, 64, 32],
                "latent_dim": 16,
                "dropout": 0.1,
                "activation": "relu",
                "loss": "mse",
                "normalization": "z_score",
                "n_epochs": 200,
                "batch_size": 2048,
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
                "lr_scheduler": "reduce_on_plateau",
                "lr_patience": 10,
                "lr_factor": 0.5,
                "lr_min": 1e-6,
                "amp": False,
            },
        }
    )


@pytest.fixture()
def base_vae_cfg():
    """Base Hydra config with VAE model defaults."""
    return OmegaConf.create(
        {
            "model": {
                "name": "vae",
                "hidden_sizes": [128, 64, 32],
                "latent_dim": 16,
                "dropout": 0.1,
                "activation": "relu",
                "loss": "mse",
                "reconstruction_loss": "mse",
                "normalization": "z_score",
                "n_epochs": 200,
                "batch_size": 2048,
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
                "lr_scheduler": "reduce_on_plateau",
                "lr_patience": 10,
                "lr_factor": 0.5,
                "lr_min": 1e-6,
                "amp": False,
                "beta": 1.0,
                "beta_schedule": "constant",
                "beta_warmup_epochs": 20,
            },
        }
    )


# ------------------------------------------------------------------
# build_search_space
# ------------------------------------------------------------------


class TestBuildSearchSpace:
    def test_ae_returns_expected_keys(self, tuning_cfg) -> None:
        space = build_search_space(tuning_cfg, "ae")
        assert set(space.keys()) == {
            "n_layers",
            "layer_size",
            "latent_dim",
            "dropout",
            "learning_rate",
            "weight_decay",
            "batch_size",
        }

    def test_vae_includes_beta(self, tuning_cfg) -> None:
        space = build_search_space(tuning_cfg, "vae")
        assert "beta" in space
        assert len(space) == 8  # 7 AE params + beta

    def test_unknown_type_raises(self) -> None:
        cfg = OmegaConf.create({"search_space": {"ae": {"x": {"type": "unknown"}}}})
        with pytest.raises(ValueError, match="Unknown search_space type"):
            build_search_space(cfg, "ae")

    def test_all_returned_values_are_ray_domains(self, tuning_cfg) -> None:
        """Each value should be a Ray Tune sample object (has .sample())."""
        space = build_search_space(tuning_cfg, "ae")
        for value in space.values():
            assert hasattr(value, "sample"), f"{value} is not a Tune domain"


# ------------------------------------------------------------------
# build_hidden_sizes
# ------------------------------------------------------------------


class TestBuildHiddenSizes:
    def test_three_layers(self) -> None:
        assert build_hidden_sizes(3, 128) == [128, 64, 32]

    def test_single_layer(self) -> None:
        assert build_hidden_sizes(1, 64) == [64]

    def test_min_clamp_to_eight(self) -> None:
        sizes = build_hidden_sizes(5, 32)
        assert all(s >= 8 for s in sizes)
        assert sizes == [32, 16, 8, 8, 8]

    def test_two_layers(self) -> None:
        assert build_hidden_sizes(2, 256) == [256, 128]

    def test_large_depth(self) -> None:
        sizes = build_hidden_sizes(4, 64)
        assert sizes == [64, 32, 16, 8]


# ------------------------------------------------------------------
# make_model_config
# ------------------------------------------------------------------


class TestMakeModelConfig:
    def test_ae_config_type(self, base_ae_cfg) -> None:
        trial = {
            "n_layers": 2,
            "layer_size": 64,
            "latent_dim": 8,
            "dropout": 0.2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "batch_size": 512,
        }
        cfg = make_model_config(trial, "ae", base_ae_cfg)
        assert isinstance(cfg, AEConfig)

    def test_ae_hidden_sizes(self, base_ae_cfg) -> None:
        trial = {
            "n_layers": 2,
            "layer_size": 64,
            "latent_dim": 8,
            "dropout": 0.2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "batch_size": 512,
        }
        cfg = make_model_config(trial, "ae", base_ae_cfg)
        assert cfg.hidden_sizes == [64, 32]

    def test_ae_overrides_from_trial(self, base_ae_cfg) -> None:
        trial = {
            "n_layers": 3,
            "layer_size": 128,
            "latent_dim": 16,
            "dropout": 0.3,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "batch_size": 1024,
        }
        cfg = make_model_config(trial, "ae", base_ae_cfg)
        assert cfg.latent_dim == 16
        assert cfg.dropout == 0.3
        assert cfg.learning_rate == 5e-4
        assert cfg.batch_size == 1024

    def test_ae_preserves_base_defaults(self, base_ae_cfg) -> None:
        trial = {
            "n_layers": 2,
            "layer_size": 64,
            "latent_dim": 8,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 512,
        }
        cfg = make_model_config(trial, "ae", base_ae_cfg)
        assert cfg.activation == "relu"
        assert cfg.loss == "mse"
        assert cfg.lr_scheduler == "reduce_on_plateau"

    def test_vae_config_type(self, base_vae_cfg) -> None:
        trial = {
            "n_layers": 3,
            "layer_size": 128,
            "latent_dim": 16,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 1024,
            "beta": 0.5,
        }
        cfg = make_model_config(trial, "vae", base_vae_cfg)
        assert isinstance(cfg, VAEConfig)

    def test_vae_beta_override(self, base_vae_cfg) -> None:
        trial = {
            "n_layers": 2,
            "layer_size": 64,
            "latent_dim": 8,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 512,
            "beta": 2.5,
        }
        cfg = make_model_config(trial, "vae", base_vae_cfg)
        assert cfg.beta == 2.5

    def test_vae_preserves_schedule_defaults(self, base_vae_cfg) -> None:
        trial = {
            "n_layers": 2,
            "layer_size": 64,
            "latent_dim": 8,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 512,
            "beta": 1.0,
        }
        cfg = make_model_config(trial, "vae", base_vae_cfg)
        assert cfg.beta_schedule == "constant"
        assert cfg.beta_warmup_epochs == 20


# ------------------------------------------------------------------
# export_best_config
# ------------------------------------------------------------------


class TestExportBestConfig:
    def test_ae_keys(self) -> None:
        best = {
            "n_layers": 3,
            "layer_size": 128,
            "latent_dim": 16,
            "dropout": 0.15,
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
            "batch_size": 1024,
        }
        result = export_best_config(best, "ae")
        assert set(result.keys()) == {
            "hidden_sizes",
            "latent_dim",
            "dropout",
            "learning_rate",
            "weight_decay",
            "batch_size",
        }

    def test_ae_hidden_sizes(self) -> None:
        best = {
            "n_layers": 3,
            "layer_size": 128,
            "latent_dim": 16,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 512,
        }
        result = export_best_config(best, "ae")
        assert result["hidden_sizes"] == [128, 64, 32]

    def test_ae_no_beta(self) -> None:
        best = {
            "n_layers": 2,
            "layer_size": 64,
            "latent_dim": 8,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 256,
        }
        result = export_best_config(best, "ae")
        assert "beta" not in result

    def test_vae_includes_beta(self) -> None:
        best = {
            "n_layers": 2,
            "layer_size": 64,
            "latent_dim": 8,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "beta": 2.5,
        }
        result = export_best_config(best, "vae")
        assert result["beta"] == 2.5

    def test_dropout_rounded(self) -> None:
        best = {
            "n_layers": 2,
            "layer_size": 64,
            "latent_dim": 8,
            "dropout": 0.123456789,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 512,
        }
        result = export_best_config(best, "ae")
        assert result["dropout"] == pytest.approx(0.123457, abs=1e-6)

    def test_values_are_native_python_types(self) -> None:
        """Ensure all values are JSON-serializable native types."""
        best = {
            "n_layers": 3,
            "layer_size": 128,
            "latent_dim": 16,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 1024,
            "beta": 0.5,
        }
        result = export_best_config(best, "vae")
        assert isinstance(result["learning_rate"], float)
        assert isinstance(result["weight_decay"], float)
        assert isinstance(result["beta"], float)
        assert isinstance(result["hidden_sizes"], list)
