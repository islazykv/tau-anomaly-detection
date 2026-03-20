import logging

import hydra
import matplotlib
import pyrootutils
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)

from src.models.tuning import build_search_space, export_best_config, run_tune  # noqa: E402
from src.processing.analysis import get_output_paths  # noqa: E402
from src.processing.io import load_dataframe  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Ray Tune hyperparameter search for AE or VAE with ASHA scheduling."""
    log.info("Starting hyperparameter tuning:\n%s", OmegaConf.to_yaml(cfg))

    output_paths = get_output_paths(cfg)
    dataframes_dir = root / output_paths["dataframes_dir"]
    models_dir = root / output_paths["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    df_mc = load_dataframe(dataframes_dir / "mc.parquet")
    log.info("Loaded MC: %d events, %d columns", len(df_mc), len(df_mc.columns))

    model_type = cfg.model.name
    search_space = build_search_space(cfg.tuning.search_space[model_type])
    log.info("Search space for %s: %s", model_type, list(search_space.keys()))

    results = run_tune(
        cfg=cfg,
        df_mc=df_mc,
        search_space=search_space,
        num_samples=cfg.tuning.num_samples,
        model_type=model_type,
    )

    best_result = results.get_best_result(metric="val_loss", mode="min")
    log.info(
        "Best trial — val_loss: %.6f",
        best_result.metrics["val_loss"],
    )
    log.info("Best config: %s", best_result.config)

    params_path = models_dir / f"{model_type}_best_params.yaml"
    export_best_config(best_result, params_path)
    log.info("Best params exported to: %s", params_path)

    log.info("Tuning complete — results saved to %s", models_dir)


if __name__ == "__main__":
    main()
