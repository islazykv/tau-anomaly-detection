from __future__ import annotations

import logging
from pathlib import Path

import awkward as ak
import pandas as pd

log = logging.getLogger(__name__)


def save_samples(samples: dict[str, ak.Array], output_dir: Path) -> None:
    """Save each sample array as a parquet file under output_dir.

    Args:
        samples: Dict mapping sample IDs to awkward arrays.
        output_dir: Directory to write parquet files into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_id, array in samples.items():
        path = output_dir / f"{sample_id}.parquet"
        ak.to_parquet(array, str(path))
        log.info("Saved %s (%d events)", path, len(array))


def load_samples(
    input_dir: Path,
    sample_ids: list[str],
) -> dict[str, ak.Array]:
    """Load parquet files from input_dir and return a dict keyed by sample id.

    Args:
        input_dir: Directory containing parquet files.
        sample_ids: List of sample IDs to load.
    """
    input_dir = Path(input_dir)
    out: dict[str, ak.Array] = {}

    for sid in sample_ids:
        path = input_dir / f"{sid}.parquet"
        out[sid] = ak.from_parquet(str(path))
        log.info("Loaded %s (%d events)", path, len(out[sid]))

    return out


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to a parquet file with snappy compression.

    Args:
        df: DataFrame to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    log.info("Saved %s (%d rows, %d columns)", path, len(df), len(df.columns))


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame from a parquet file.

    Args:
        path: Path to the parquet file.
    """
    df = pd.read_parquet(Path(path), engine="pyarrow")
    log.info("Loaded %s (%d rows, %d columns)", path, len(df), len(df.columns))
    return df
