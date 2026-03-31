"""FastAPI application for anomaly-detection inference."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pyrootutils
import uvicorn
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf

from src.serving.registry import ModelRegistry
from src.serving.schemas import (
    EventScore,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger(__name__)

_registry: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    if _registry is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _registry


# ------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Manage model lifecycle on startup and shutdown."""
    yield


app = FastAPI(
    title="Tau Anomaly Detection API",
    description="Serves trained AE / VAE anomaly scores for ATLAS tau data.",
    version="0.1.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return server health status and loaded model metadata."""
    reg = _get_registry()
    return HealthResponse(
        model_name=reg.model_name,
        n_features=reg.n_features,
        feature_names=reg.feature_names,
        threshold=reg.threshold,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Compute anomaly scores for a batch of events and flag anomalies."""
    reg = _get_registry()

    features = np.asarray(request.features, dtype=np.float32)
    if features.shape[1] != reg.n_features:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {reg.n_features} features, got {features.shape[1]}",
        )

    scores, pf_errors = reg.predict(features)

    events = [
        EventScore(
            anomaly_score=float(scores[i]),
            flagged=bool(scores[i] > reg.threshold),
            per_feature_error=pf_errors[i].tolist(),
        )
        for i in range(len(scores))
    ]

    return PredictResponse(
        model_name=reg.model_name,
        threshold=reg.threshold,
        events=events,
    )


# ------------------------------------------------------------------
# Hydra entry point
# ------------------------------------------------------------------


def run_server(cfg: DictConfig) -> None:
    """Load model from checkpoint and start the uvicorn server."""
    global _registry  # noqa: PLW0603

    from src.processing.analysis import get_output_paths

    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])
    output_paths = get_output_paths(cfg)
    models_dir = root / output_paths["models_dir"]
    dataframes_dir = root / output_paths["dataframes_dir"]

    model_name: str = cfg.model.name
    ckpt_path = models_dir / f"{model_name}.ckpt"
    model_cfg: dict[str, Any] = dict(
        OmegaConf.to_container(cfg.model, resolve=True)  # type: ignore[arg-type]
    )

    # Load threshold from metrics if available
    import json

    metrics_path = dataframes_dir / f"{model_name}_metrics.json"
    threshold = 0.0
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        threshold = metrics.get("threshold", 0.0)
        log.info("Loaded threshold from metrics: %.6f", threshold)

    _registry = ModelRegistry.from_checkpoint(
        ckpt_path=ckpt_path,
        model_name=model_name,
        model_cfg=model_cfg,
        threshold=threshold,
    )

    host = cfg.get("serve_host", "0.0.0.0")
    port = cfg.get("serve_port", 8000)
    log.info("Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=int(port))
