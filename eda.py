import logging

import hydra
import matplotlib
import pyrootutils
from omegaconf import DictConfig

matplotlib.use("Agg")

pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)

from src.eda.checks import (  # noqa: E402
    summarize_feature_ranges,
    summarize_missing,
)
from src.eda.plots import (  # noqa: E402
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_sample_balance,
)
from src.eda.utils import get_sample_labels  # noqa: E402
from src.processing.analysis import get_output_paths  # noqa: E402
from src.processing.io import load_dataframe  # noqa: E402
from src.processing.validation import METADATA_COLUMNS  # noqa: E402
from src.visualization.plots import save_figure  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Run the full EDA pipeline: load, check data quality, and save plots."""
    output_paths = get_output_paths(cfg)
    dataframes_dir = output_paths["dataframes_dir"]
    plots_dir = output_paths["plots_dir"] / "eda"
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
    fig = plot_sample_balance(df_mc)
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
    fig = plot_feature_distributions(df_mc, features=training_cols[:12])
    save_figure(fig, plots_dir / "feature_distributions.png")

    log.info("EDA complete — plots saved to %s", plots_dir)


if __name__ == "__main__":
    main()
