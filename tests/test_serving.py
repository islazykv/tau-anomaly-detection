"""Tests for the serving layer (schemas, registry, app)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.models.ae import Autoencoder
from src.models.config import AEConfig, VAEConfig
from src.models.vae import VariationalAutoencoder
from src.serving.app import app
from src.serving.registry import ModelRegistry, _ScalerState
from src.serving.schemas import PredictionRequest


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

N_FEATURES = 8


@pytest.fixture()
def ae_model() -> Autoencoder:
    cfg = AEConfig(hidden_sizes=[16, 8], latent_dim=4, dropout=0.0)
    model = Autoencoder(cfg, n_features=N_FEATURES)
    model.eval()
    return model


@pytest.fixture()
def vae_model() -> VariationalAutoencoder:
    cfg = VAEConfig(hidden_sizes=[16, 8], latent_dim=4, dropout=0.0)
    model = VariationalAutoencoder(cfg, n_features=N_FEATURES)
    model.eval()
    return model


@pytest.fixture()
def scaler() -> _ScalerState:
    return _ScalerState(
        mean=np.zeros(N_FEATURES, dtype=np.float64),
        scale=np.ones(N_FEATURES, dtype=np.float64),
        normalization="z_score",
        feature_names=[f"f{i}" for i in range(N_FEATURES)],
    )


@pytest.fixture()
def ae_registry(ae_model: Autoencoder, scaler: _ScalerState) -> ModelRegistry:
    return ModelRegistry(model=ae_model, scaler=scaler, threshold=0.5)


@pytest.fixture()
def vae_registry(
    vae_model: VariationalAutoencoder, scaler: _ScalerState
) -> ModelRegistry:
    return ModelRegistry(model=vae_model, scaler=scaler, threshold=0.5)


# ------------------------------------------------------------------
# Schema tests
# ------------------------------------------------------------------


class TestSchemas:
    def test_prediction_request_valid(self) -> None:
        req = PredictionRequest(features=[[1.0, 2.0, 3.0]])
        assert len(req.features) == 1

    def test_prediction_request_empty_rejects(self) -> None:
        with pytest.raises(Exception):  # noqa: B017, PT011
            PredictionRequest(features=[])


# ------------------------------------------------------------------
# Registry tests
# ------------------------------------------------------------------


class TestScalerState:
    def test_z_score_transform(self, scaler: _ScalerState) -> None:
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        out = scaler.transform(x)
        np.testing.assert_array_almost_equal(out, x)

    def test_min_max_clips(self) -> None:
        s = _ScalerState(
            mean=np.zeros(2),
            scale=np.ones(2),
            normalization="min_max",
            feature_names=["a", "b"],
        )
        x = np.array([[-0.5, 1.5]])
        out = s.transform(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestModelRegistry:
    def test_predict_ae(self, ae_registry: ModelRegistry) -> None:
        features = np.random.randn(5, N_FEATURES).astype(np.float32)
        scores, pf_err = ae_registry.predict(features)
        assert scores.shape == (5,)
        assert pf_err.shape == (5, N_FEATURES)
        assert np.all(scores >= 0)

    def test_predict_vae(self, vae_registry: ModelRegistry) -> None:
        features = np.random.randn(3, N_FEATURES).astype(np.float32)
        scores, pf_err = vae_registry.predict(features)
        assert scores.shape == (3,)
        assert pf_err.shape == (3, N_FEATURES)

    def test_properties(self, ae_registry: ModelRegistry) -> None:
        assert ae_registry.model_name == "ae"
        assert ae_registry.n_features == N_FEATURES
        assert len(ae_registry.feature_names) == N_FEATURES


# ------------------------------------------------------------------
# App endpoint tests
# ------------------------------------------------------------------


@pytest.fixture()
def _set_ae_registry(ae_registry: ModelRegistry):
    """Inject the AE registry into the app module."""
    import src.serving.app as app_mod

    prev = app_mod._registry
    app_mod._registry = ae_registry
    yield
    app_mod._registry = prev


@pytest.mark.usefixtures("_set_ae_registry")
class TestApp:
    def test_health(self) -> None:
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_name"] == "ae"
        assert body["n_features"] == N_FEATURES

    def test_predict(self) -> None:
        features = np.random.randn(2, N_FEATURES).tolist()
        client = TestClient(app)
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["events"]) == 2
        assert "anomaly_score" in body["events"][0]
        assert "flagged" in body["events"][0]
        assert "per_feature_error" in body["events"][0]

    def test_predict_wrong_features(self) -> None:
        client = TestClient(app)
        resp = client.post("/predict", json={"features": [[1.0, 2.0]]})
        assert resp.status_code == 422

    def test_predict_empty_body(self) -> None:
        client = TestClient(app)
        resp = client.post("/predict", json={"features": []})
        assert resp.status_code == 422


class TestAppNoModel:
    def test_health_503_when_no_model(self) -> None:
        import src.serving.app as app_mod

        prev = app_mod._registry
        app_mod._registry = None
        try:
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
            assert resp.status_code == 503
        finally:
            app_mod._registry = prev
