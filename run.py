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
            from src.pipelines.preprocess import preprocess

            preprocess(cfg)

        case "feature_engineer":
            from src.pipelines.feature_engineer import feature_engineer

            feature_engineer(cfg)

        case "eda":
            from src.pipelines.eda import eda

            eda(cfg)

        case "train":
            from src.pipelines.train import train

            train(cfg)

        case "evaluate":
            from src.pipelines.evaluate import evaluate

            evaluate(cfg)

        case "tune":
            from src.pipelines.tune import tune

            tune(cfg)

        case "serve":
            from src.serving.app import run_server

            run_server(cfg)

        case _:
            msg = f"Unknown stage: {stage}"
            raise ValueError(msg)


if __name__ == "__main__":
    main()
