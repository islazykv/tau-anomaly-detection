"""Pydantic schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body containing a batch of events for scoring."""

    features: list[list[float]] = Field(
        ...,
        description="Batch of events. Each inner list contains feature values in the "
        "same order as the training features.",
        min_length=1,
    )


class EventScore(BaseModel):
    """Anomaly score and flag for a single event."""

    anomaly_score: float
    flagged: bool
    per_feature_error: list[float]


class PredictionResponse(BaseModel):
    """Response body containing per-event anomaly scores."""

    model_name: str
    threshold: float
    events: list[EventScore]


class HealthResponse(BaseModel):
    """Response body for the health-check endpoint."""

    status: str = "ok"
    model_name: str
    n_features: int
    feature_names: list[str]
    threshold: float
