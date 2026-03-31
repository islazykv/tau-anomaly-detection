"""Hyperparameter tuning pipeline with Ray Tune."""

from __future__ import annotations

import json
import logging

import pyrootutils
import ray
from omegaconf import DictConfig, OmegaConf

from src.models.plots import (
    plot_hp_vs_objective,
    plot_hyperparameter_importance,
    plot_optimization_history,
    plot_parallel_coordinates,
)
from src.models.tuning import export_best_config, run_tune
from src.processing.analysis import get_output_paths
from src.visualization.plots import save_figure

log = logging.getLogger(__name__)


def _get_background_origins(cfg: DictConfig) -> set[str]:
    """Return the set of background sample IDs after applying exclusions."""
    bg_cfg = cfg.samples.background
    excludes = set(bg_cfg.get("exclude", []))
    return {s["id"] for s in bg_cfg.samples if s["id"] not in excludes}


def tune(cfg: DictConfig) -> None:
    """Run ASHA-scheduled hyperparameter tuning with Ray Tune and save the best config."""
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
        best_config, trial_df = run_tune(cfg, dm_kwargs)

        # Convert to model-config format and persist
        best_model_config = export_best_config(best_config, model_name)

        results_dir = dataframes_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"{model_name}_best_tuning.json"
        with open(results_path, "w") as f:
            json.dump(best_model_config, f, indent=2)

        log.info("Best config saved to %s", results_path)
        log.info("Best model config:\n%s", json.dumps(best_model_config, indent=2))

        # Tuning analysis plots
        plots_dir = root / output_paths["plots_dir"] / f"{model_name}_tuning"
        plots_dir.mkdir(parents=True, exist_ok=True)

        save_figure(
            plot_optimization_history(trial_df), plots_dir / "optimization_history.png"
        )
        save_figure(
            plot_hyperparameter_importance(trial_df), plots_dir / "hp_importance.png"
        )
        save_figure(
            plot_parallel_coordinates(trial_df), plots_dir / "parallel_coordinates.png"
        )
        save_figure(plot_hp_vs_objective(trial_df), plots_dir / "hp_vs_objective.png")
        log.info("Tuning plots saved to %s", plots_dir)
    finally:
        if ray.is_initialized():
            ray.shutdown()
