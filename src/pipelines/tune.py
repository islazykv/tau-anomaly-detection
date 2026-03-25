"""Hyperparameter tuning pipeline with Ray Tune."""

from __future__ import annotations

import json
import logging

import pyrootutils
import ray
from omegaconf import DictConfig, OmegaConf

from src.models.tuning import export_best_config, run_tune
from src.processing.analysis import get_output_paths

log = logging.getLogger(__name__)


def _get_background_origins(cfg: DictConfig) -> set[str]:
    """Extract background sample IDs from the samples config."""
    bg_cfg = cfg.samples.background
    excludes = set(bg_cfg.get("exclude", []))
    return {s["id"] for s in bg_cfg.samples if s["id"] not in excludes}


def tune(cfg: DictConfig) -> None:
    """Run hyperparameter tuning with Ray Tune.

    Steps:
        1. Resolve output paths and build DataModule kwargs
        2. Initialize Ray (local cluster)
        3. Run ASHA-scheduled hyperparameter search
        4. Export best config as JSON
        5. Shutdown Ray
    """
    model_name = cfg.model.name
    log.info("Starting %s hyperparameter tuning", model_name.upper())
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Paths
    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])
    output_paths = get_output_paths(cfg)
    dataframes_dir = root / output_paths["dataframes_dir"]

    # DataModule constructor kwargs (serializable for Ray object store)
    mc_path = dataframes_dir / "mc.parquet"
    background_origins = _get_background_origins(cfg)
    log.info("Background origins: %s", background_origins)

    dm_kwargs: dict[str, object] = {
        "mc_path": str(mc_path),
        "background_origins": background_origins,
        "normalization": cfg.model.normalization,
        "val_fraction": cfg.pipeline.val_fraction,
        "batch_size": cfg.model.batch_size,
        "seed": cfg.seed,
    }

    # Ray lifecycle
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    try:
        best_config, _trial_df = run_tune(cfg, dm_kwargs)

        # Convert to model-config format and persist
        best_model_config = export_best_config(best_config, model_name)

        results_dir = dataframes_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"{model_name}_best_tuning.json"
        with open(results_path, "w") as f:
            json.dump(best_model_config, f, indent=2)

        log.info("Best config saved to %s", results_path)
        log.info("Best model config:\n%s", json.dumps(best_model_config, indent=2))
    finally:
        if ray.is_initialized():
            ray.shutdown()
