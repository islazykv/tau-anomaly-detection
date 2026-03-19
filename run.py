"""Unified entry point for the tau anomaly detection pipeline.

Usage:
    uv run python run.py stage=preprocess
    uv run python run.py stage=feature_engineer
    uv run python run.py stage=eda
    uv run python run.py stage=train model=ae
    uv run python run.py stage=train model=vae
    uv run python run.py stage=evaluate model=ae
    uv run python run.py stage=tune
    uv run python run.py stage=serve model=ae
"""

from __future__ import annotations

import logging

import hydra
import matplotlib
import pyrootutils
from omegaconf import DictConfig

matplotlib.use("Agg")

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Dispatch to the appropriate pipeline stage."""
    stage = cfg.stage
    log.info("Running stage: %s", stage)

    match stage:
        case "preprocess":
            from src.processing.pipeline import run_preprocessing

            run_preprocessing(cfg)

        case "feature_engineer":
            from src.processing.pipeline import run_feature_engineering

            run_feature_engineering(cfg)

        case "eda":
            from src.eda.pipeline import run_eda

            run_eda(cfg)

        case "train":
            from src.models.pipeline import run_training

            run_training(cfg)

        case "evaluate":
            from src.models.pipeline import run_evaluation

            run_evaluation(cfg)

        case "tune":
            from src.models.pipeline import run_tuning

            run_tuning(cfg)

        case "serve":
            from src.serving.app import run_server

            run_server(cfg)

        case _:
            msg = f"Unknown stage: {stage}"
            raise ValueError(msg)


if __name__ == "__main__":
    main()
