from __future__ import annotations

import pandas as pd

_DEFAULT_SIGNAL_TYPE_NAMES: dict[str, str] = {
    "GG": "gluinos",
    "SS": "squarks",
}


def get_sample_labels(
    df: pd.DataFrame,
    display_labels: dict[str, str] | None = None,
) -> list[str]:
    """Derive display labels for each unique eventOrigin in the DataFrame."""
    if display_labels is None:
        display_labels = {}
    origins = sorted(df["eventOrigin"].unique())
    return [display_labels.get(o, o) for o in origins]


def get_signal_type_name(origin: str) -> str:
    """Map a signal eventOrigin prefix to a human-readable name."""
    prefix = origin.split("_")[0]
    return _DEFAULT_SIGNAL_TYPE_NAMES.get(prefix, origin)
