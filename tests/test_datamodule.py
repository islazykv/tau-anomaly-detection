"""Tests for src.models.datamodule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.datamodule import AnomalyDataModule


BKG_ORIGINS = {"bkg_ttbar", "bkg_wjets"}
N_BKG = 140
N_SIG = 60
N_FEATURES = 3


@pytest.fixture()
def mc_parquet(tmp_path: object) -> str:
    """Write a synthetic MC parquet and return its path."""
    import pathlib

    tmp = pathlib.Path(str(tmp_path))
    rng = np.random.default_rng(42)
    n = N_BKG + N_SIG
    df = pd.DataFrame(
        {
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            "feat_c": rng.normal(size=n),
            "weight": rng.uniform(0.5, 2.0, size=n),
            "eventOrigin": (
                ["bkg_ttbar"] * 80 + ["bkg_wjets"] * 60 + ["sig_stau"] * 60
            ),
            "tau_n": rng.integers(1, 4, size=n),
        }
    )
    path = tmp / "mc.parquet"
    df.to_parquet(path)
    return str(path)


def _make_dm(mc_parquet: str, **kwargs) -> AnomalyDataModule:
    """Create a DataModule with test defaults."""
    defaults = dict(
        mc_path=mc_parquet,
        background_origins=BKG_ORIGINS,
        normalization="z_score",
        val_fraction=0.2,
        batch_size=32,
        seed=1,
    )
    defaults.update(kwargs)
    return AnomalyDataModule(**defaults)


class TestSetup:
    def test_dataset_sizes(self, mc_parquet: str) -> None:
        dm = _make_dm(mc_parquet)
        dm.setup()
        n_val = int(N_BKG * 0.2)
        n_train = N_BKG - n_val
        assert len(dm.train_dataset) == pytest.approx(n_train, abs=2)
        assert len(dm.val_dataset) == pytest.approx(n_val, abs=2)
        assert len(dm.predict_dataset) == pytest.approx(n_val + N_SIG, abs=2)

    def test_predict_labels(self, mc_parquet: str) -> None:
        dm = _make_dm(mc_parquet)
        dm.setup()
        n_val = len(dm.val_dataset)
        assert dm.predict_labels[:n_val].sum() == 0  # all bkg
        assert dm.predict_labels[n_val:].sum() == N_SIG  # all sig

    def test_feature_count(self, mc_parquet: str) -> None:
        dm = _make_dm(mc_parquet)
        dm.setup()
        assert dm.n_features == N_FEATURES
        assert len(dm.feature_names_) == N_FEATURES


class TestScaler:
    def test_z_score_produces_unit_variance(self, mc_parquet: str) -> None:
        dm = _make_dm(mc_parquet, normalization="z_score")
        dm.setup()
        X_train = dm.train_dataset.tensors[0].numpy()
        assert np.abs(X_train.mean(axis=0)).max() < 0.15
        assert np.abs(X_train.std(axis=0) - 1.0).max() < 0.15

    def test_min_max_bounded(self, mc_parquet: str) -> None:
        dm = _make_dm(mc_parquet, normalization="min_max")
        dm.setup()
        X_train = dm.train_dataset.tensors[0].numpy()
        assert X_train.min() >= 0.0
        assert X_train.max() <= 1.0

    def test_inverse_transform_round_trip(self, mc_parquet: str) -> None:
        dm = _make_dm(mc_parquet, normalization="z_score")
        dm.setup()
        X_scaled = dm.train_dataset.tensors[0].numpy()
        X_back = dm.inverse_transform(X_scaled)
        X_rescaled = dm._transform(X_back)
        np.testing.assert_allclose(X_scaled, X_rescaled, atol=1e-5)

    def test_unknown_normalization_raises(self, mc_parquet: str) -> None:
        dm = _make_dm(mc_parquet, normalization="unknown")
        with pytest.raises(ValueError, match="Unknown normalization"):
            dm.setup()


class TestStateDict:
    def test_round_trip(self, mc_parquet: str) -> None:
        dm1 = _make_dm(mc_parquet)
        dm1.setup()
        state = dm1.state_dict()

        dm2 = _make_dm(mc_parquet)
        dm2.load_state_dict(state)

        np.testing.assert_array_equal(dm1.scaler_mean_, dm2.scaler_mean_)
        np.testing.assert_array_equal(dm1.scaler_scale_, dm2.scaler_scale_)
        assert dm1.feature_names_ == dm2.feature_names_
        assert dm2.n_features == N_FEATURES


class TestDataloaders:
    def test_train_batch_shape(self, mc_parquet: str) -> None:
        dm = _make_dm(mc_parquet)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        features, weights = batch
        assert features.ndim == 2
        assert features.shape[1] == N_FEATURES
        assert weights.ndim == 1
        assert len(weights) == len(features)

    def test_val_batch_shape(self, mc_parquet: str) -> None:
        dm = _make_dm(mc_parquet)
        dm.setup()
        batch = next(iter(dm.val_dataloader()))
        features, weights = batch
        assert features.shape[1] == N_FEATURES

    def test_predict_batch_shape(self, mc_parquet: str) -> None:
        dm = _make_dm(mc_parquet)
        dm.setup()
        batch = next(iter(dm.predict_dataloader()))
        features, weights = batch
        assert features.shape[1] == N_FEATURES
