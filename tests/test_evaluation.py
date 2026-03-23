"""Tests for src.models.evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import (
    compute_metrics,
    compute_roc_auc,
    compute_roc_curve,
    compute_roc_per_origin,
    compute_sic_curve,
)


@pytest.fixture()
def perfect_data() -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores with perfect separation."""
    labels = np.array([0] * 50 + [1] * 50)
    scores = np.array([0.1] * 50 + [0.9] * 50)
    return labels, scores


@pytest.fixture()
def random_data() -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores with no separation (random)."""
    rng = np.random.default_rng(42)
    labels = np.array([0] * 500 + [1] * 500)
    scores = rng.normal(size=1000)
    return labels, scores


@pytest.fixture()
def scores_df() -> pd.DataFrame:
    """Scores DataFrame with multiple signal origins."""
    rng = np.random.default_rng(42)
    n_bkg, n_sig_a, n_sig_b = 100, 30, 20
    return pd.DataFrame(
        {
            "anomaly_score": np.concatenate(
                [
                    rng.normal(0, 1, size=n_bkg),
                    rng.normal(3, 1, size=n_sig_a),
                    rng.normal(2, 1, size=n_sig_b),
                ]
            ),
            "sample_type": (
                ["background"] * n_bkg + ["signal"] * n_sig_a + ["signal"] * n_sig_b
            ),
            "eventOrigin": (
                ["bkg_ttbar"] * n_bkg
                + ["GG_1000_100"] * n_sig_a
                + ["GG_1500_200"] * n_sig_b
            ),
        }
    )


class TestRocAuc:
    def test_perfect_separation(self, perfect_data: tuple) -> None:
        labels, scores = perfect_data
        auc = compute_roc_auc(labels, scores)
        assert auc == pytest.approx(1.0)

    def test_random_near_half(self, random_data: tuple) -> None:
        labels, scores = random_data
        auc = compute_roc_auc(labels, scores)
        assert 0.4 < auc < 0.6


class TestRocCurve:
    def test_output_shapes(self, perfect_data: tuple) -> None:
        labels, scores = perfect_data
        fpr, tpr, thresholds = compute_roc_curve(labels, scores)
        assert len(fpr) == len(tpr) == len(thresholds)
        assert fpr[0] == 0.0
        assert tpr[-1] == 1.0


class TestSicCurve:
    def test_shape(self) -> None:
        fpr = np.linspace(0, 1, 100)
        tpr = np.linspace(0, 1, 100)
        sic = compute_sic_curve(fpr, tpr)
        assert sic.shape == fpr.shape

    def test_no_nan_or_inf(self) -> None:
        fpr = np.linspace(0.01, 1, 100)
        tpr = np.linspace(0.01, 1, 100)
        sic = compute_sic_curve(fpr, tpr)
        assert np.all(np.isfinite(sic))

    def test_handles_zero_fpr(self) -> None:
        fpr = np.array([0.0, 0.1, 0.5, 1.0])
        tpr = np.array([0.0, 0.5, 0.8, 1.0])
        sic = compute_sic_curve(fpr, tpr)
        assert np.all(np.isfinite(sic))


class TestRocPerOrigin:
    def test_returns_all_origins(self, scores_df: pd.DataFrame) -> None:
        results = compute_roc_per_origin(scores_df)
        assert set(results.keys()) == {"GG_1000_100", "GG_1500_200"}

    def test_values_are_valid_auc(self, scores_df: pd.DataFrame) -> None:
        results = compute_roc_per_origin(scores_df)
        for auc in results.values():
            assert 0.0 <= auc <= 1.0

    def test_higher_separation_better_auc(self, scores_df: pd.DataFrame) -> None:
        results = compute_roc_per_origin(scores_df)
        # GG_1000_100 has mean=3 vs bkg mean=0, GG_1500_200 has mean=2
        assert results["GG_1000_100"] > results["GG_1500_200"]


class TestComputeMetrics:
    def test_keys(self, perfect_data: tuple) -> None:
        labels, scores = perfect_data
        metrics = compute_metrics(labels, scores)
        assert "roc_auc" in metrics
        assert "max_sic" in metrics

    def test_with_scores_df(self, perfect_data: tuple, scores_df: pd.DataFrame) -> None:
        labels, scores = perfect_data
        metrics = compute_metrics(labels, scores, scores_df=scores_df)
        assert "roc_per_origin" in metrics

    def test_max_sic_positive(self, perfect_data: tuple) -> None:
        labels, scores = perfect_data
        metrics = compute_metrics(labels, scores)
        assert metrics["max_sic"] > 0
