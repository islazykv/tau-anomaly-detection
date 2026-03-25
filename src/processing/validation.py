"""Pandera schema validation for processed MC DataFrames."""

from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa

log = logging.getLogger(__name__)

METADATA_COLUMNS = {"eventOrigin", "sample_type", "tau_n", "weight"}


def _build_mc_schema(df: pd.DataFrame) -> pa.DataFrameSchema:
    """Build a DataFrameSchema for the MC DataFrame.

    The schema validates fixed metadata columns with strict checks and
    dynamically validates all remaining columns as numeric training features.
    """
    columns: dict[str, pa.Column] = {
        "eventOrigin": pa.Column(nullable=False),
        "sample_type": pa.Column(nullable=False),
        "tau_n": pa.Column(nullable=False),
        "weight": pa.Column(float, nullable=False),
    }

    # Every non-metadata column is a training feature: must be numeric, no nulls.
    feature_cols = sorted(set(df.columns) - METADATA_COLUMNS)
    for col in feature_cols:
        columns[col] = pa.Column(nullable=False)

    checks = [
        # All training features must be numeric (int or float).
        pa.Check(
            lambda df: all(
                pd.api.types.is_numeric_dtype(df[c])
                for c in df.columns
                if c not in METADATA_COLUMNS
            ),
            error="Non-numeric training feature column detected",
        ),
    ]

    return pa.DataFrameSchema(
        columns=columns,
        checks=checks,
        strict=False,
        coerce=False,
    )


def validate_mc(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the MC DataFrame against the expected schema.

    Args:
        df: The MC DataFrame produced by feature engineering.

    Returns:
        The validated DataFrame (unchanged if valid).

    Raises:
        pandera.errors.SchemaError: If validation fails.
    """
    feature_cols = set(df.columns) - METADATA_COLUMNS
    if len(feature_cols) == 0:
        raise pa.errors.SchemaError(
            schema=None,
            data=df,
            message="No training feature columns found",
        )

    schema = _build_mc_schema(df)
    validated = schema.validate(df)
    log.info(
        "MC validation passed: %d rows, %d features",
        len(df),
        len(feature_cols),
    )
    return validated
