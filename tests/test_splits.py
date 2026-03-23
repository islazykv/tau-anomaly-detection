"""Tests for src.models.splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.splits import prepare_features, split_background_signal, train_val_split


@pytest.fixture()
def mc_df() -> pd.DataFrame:
    """Synthetic MC DataFrame with 3 features, 2 background and 1 signal origin."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
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


BKG_ORIGINS = {"bkg_ttbar", "bkg_wjets"}


class TestPrepareFeatures:
    def test_returns_features_and_weights(self, mc_df: pd.DataFrame) -> None:
        features, weights = prepare_features(mc_df)
        assert list(features.columns) == ["feat_a", "feat_b", "feat_c"]
        assert len(weights) == len(mc_df)

    def test_drops_metadata_columns(self, mc_df: pd.DataFrame) -> None:
        features, _ = prepare_features(mc_df)
        for col in ("eventOrigin", "tau_n", "weight"):
            assert col not in features.columns

    def test_returns_copies(self, mc_df: pd.DataFrame) -> None:
        features, weights = prepare_features(mc_df)
        features.iloc[0, 0] = 999.0
        assert mc_df.iloc[0, 0] != 999.0
        weights.iloc[0] = 999.0
        assert mc_df["weight"].iloc[0] != 999.0


class TestSplitBackgroundSignal:
    def test_correct_counts(self, mc_df: pd.DataFrame) -> None:
        bkg, sig = split_background_signal(mc_df, BKG_ORIGINS)
        assert len(bkg) == 140
        assert len(sig) == 60

    def test_all_bkg_origins_are_background(self, mc_df: pd.DataFrame) -> None:
        bkg, _ = split_background_signal(mc_df, BKG_ORIGINS)
        assert set(bkg["eventOrigin"].unique()).issubset(BKG_ORIGINS)

    def test_no_signal_in_background(self, mc_df: pd.DataFrame) -> None:
        bkg, sig = split_background_signal(mc_df, BKG_ORIGINS)
        assert "sig_stau" not in bkg["eventOrigin"].values
        assert "sig_stau" in sig["eventOrigin"].values

    def test_returns_copies(self, mc_df: pd.DataFrame) -> None:
        bkg, sig = split_background_signal(mc_df, BKG_ORIGINS)
        bkg.iloc[0, 0] = 999.0
        assert mc_df.iloc[0, 0] != 999.0


class TestTrainValSplit:
    def test_split_sizes(self, mc_df: pd.DataFrame) -> None:
        features, weights = prepare_features(mc_df)
        X_train, X_val, w_train, w_val = train_val_split(
            features, weights, val_fraction=0.2, seed=1
        )
        assert len(X_train) + len(X_val) == len(features)
        assert len(X_val) == pytest.approx(len(features) * 0.2, abs=2)

    def test_weights_aligned(self, mc_df: pd.DataFrame) -> None:
        features, weights = prepare_features(mc_df)
        X_train, X_val, w_train, w_val = train_val_split(
            features, weights, val_fraction=0.2, seed=1
        )
        assert len(w_train) == len(X_train)
        assert len(w_val) == len(X_val)

    def test_no_index_overlap(self, mc_df: pd.DataFrame) -> None:
        features, weights = prepare_features(mc_df)
        X_train, X_val, _, _ = train_val_split(
            features, weights, val_fraction=0.2, seed=1
        )
        assert set(X_train.index).isdisjoint(set(X_val.index))

    def test_reproducible(self, mc_df: pd.DataFrame) -> None:
        features, weights = prepare_features(mc_df)
        X1, _, _, _ = train_val_split(features, weights, seed=42)
        X2, _, _, _ = train_val_split(features, weights, seed=42)
        pd.testing.assert_frame_equal(X1, X2)
