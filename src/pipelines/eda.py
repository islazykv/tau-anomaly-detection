"""Exploratory data analysis pipeline: load → quality checks → plots."""

from __future__ import annotations

import logging
from typing import cast

import pyrootutils
from omegaconf import DictConfig, OmegaConf

from src.eda.checks import (
    summarize_feature_ranges,
    summarize_missing,
)
from src.eda.plots import (
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_sample_balance,
)
from src.eda.utils import get_sample_labels
from src.processing.analysis import get_output_paths
from src.processing.io import load_dataframe
from src.processing.validation import METADATA_COLUMNS
from src.visualization.plots import save_figure

log = logging.getLogger(__name__)


def eda(cfg: DictConfig) -> None:
    """Run the full EDA pipeline: load, check data quality, and save plots."""
    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])
    output_paths = get_output_paths(cfg)
    dataframes_dir = root / output_paths["dataframes_dir"]
    plots_dir = root / output_paths["plots_dir"] / "eda"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_mc = load_dataframe(dataframes_dir / "mc.parquet")
    log.info("Loaded MC: %d rows, %d columns", len(df_mc), len(df_mc.columns))

    sample_labels = get_sample_labels(df_mc)
    log.info("Samples: %s", sample_labels)

    missing = summarize_missing(df_mc)
    if missing.empty:
        log.info("Missing values: none")
    else:
        log.warning("Missing values detected:\n%s", missing.to_string())

    ranges = summarize_feature_ranges(df_mc)
    log.info(
        "Feature ranges computed for %d features",
        len(ranges.columns.get_level_values(0).unique()),
    )

    log.info("Generating sample balance plot...")
    display_labels = cast(
        dict[str, str] | None,
        OmegaConf.to_container(cfg.merge.display_labels, resolve=True),
    )
    fig = plot_sample_balance(df_mc, display_labels=display_labels)
    save_figure(fig, plots_dir / "sample_balance.png")

    log.info("Generating correlation matrix...")
    fig = plot_correlation_matrix(df_mc)
    save_figure(fig, plots_dir / "correlation_matrix.png")

    log.info("Generating feature distributions...")
    training_cols = [
        c
        for c in df_mc.select_dtypes(include="number").columns
        if c not in METADATA_COLUMNS
    ]
    fig = plot_feature_distributions(
        df_mc, features=training_cols[:12], group_col="sample_type"
    )
    save_figure(fig, plots_dir / "feature_distributions.png")

    log.info("EDA complete — plots saved to %s", plots_dir)
