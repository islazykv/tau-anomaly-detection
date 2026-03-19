# CLAUDE.md — Tau Supersymmetry Anomaly Detection

Blueprint for an ATLAS HEP analysis using **autoencoders (AE)** and **variational
autoencoders (VAE)** for anomaly-based signal detection, reusing the preprocessing
pipeline from the existing `tau-supersymmetry-search` project.

---

## 0. Workflow Rules (READ FIRST)

**These rules override everything else in this file.**

1. **Work in small, incremental steps.** One step = one small unit of work.
2. **Wait for user confirmation after every step.** Ask what to do next.
3. **Show what you will do before doing it.** Wait for approval before writing code.
4. **Follow the implementation plan in section 10**, treating each item as a separate step.
5. **The user will copy some files from the predecessor** (`tau-supersymmetry-search`,
   located at `/home/islazyk/tau-supersymmetry`). Do not create files the user plans
   to copy — ask first.
6. **Never auto-commit.** Only commit when the user explicitly asks.
7. **If unsure, ask.** Prefer asking over assuming.

---

## 1. Project Overview

| Field | Value |
|-------|-------|
| Name | `tau-supersymmetry-anomaly` |
| Goal | Detect SUSY signals in ATLAS tau data via unsupervised anomaly detection |
| Predecessor | `tau-supersymmetry-search` (supervised BDT/DNN classification) |
| Key difference | Train on **background only**; signals = anomalies with high reconstruction error |

---

## 2. Tech Stack

**Core**: Python >=3.13, uv, PyTorch >=2.10, PyTorch Lightning >=2.5, torchmetrics,
torcheval, scikit-learn >=1.8, Ray Tune >=2.40

**Data** (reuse from predecessor): uproot, awkward, pandas, Pandera, pyrootutils, scipy, numpy

**Config & tracking**: Hydra >=1.3.2, MLflow >=3.9 (local SQLite backend)

**Viz**: matplotlib, seaborn, atlas-mpl-style

**Serving**: FastAPI, uvicorn

**Notebooks**: ipykernel, ipywidgets, tqdm

**Dev**: pytest, pytest-cov, mypy, ruff, pre-commit, httpx, types-PyYAML

---

## 3. Project Structure

```
tau-supersymmetry-anomaly/
├── src/
│   ├── processing/              # REUSE from predecessor (copy as-is)
│   ├── models/                  # NEW
│   │   ├── ae.py                # Autoencoder (LightningModule)
│   │   ├── vae.py               # VAE (LightningModule)
│   │   ├── datamodule.py        # LightningDataModule (background-only training)
│   │   ├── anomaly.py           # anomaly scoring (reconstruction error, thresholds)
│   │   ├── latent.py            # latent space analysis (t-SNE, UMAP)
│   │   ├── evaluation.py        # anomaly detection metrics (ROC, SIC)
│   │   ├── plots.py             # training curves, reconstruction, latent diagnostics
│   │   ├── splits.py            # background-only train/val split, signal held out
│   │   └── tuning.py            # Ray Tune integration
│   ├── eda/                     # REUSE from predecessor
│   ├── serving/                 # ADAPT for AE/VAE
│   │   ├── app.py, registry.py, schemas.py
│   ├── visualization/           # REUSE
│   └── utils.py                 # REUSE
├── configs/                     # Hydra configuration
│   ├── config.yaml              # root config with defaults
│   ├── model/                   # ae.yaml, vae.yaml (NEW)
│   ├── tuning/                  # Ray Tune search spaces (NEW)
│   ├── pipeline/                # ADAPT (anomaly_detection mode)
│   └── features/, data/, samples/, merge/, run/, paths/, hydra/  # REUSE
├── tests/
├── notebooks/                   # 00–06
├── run.py                       # SINGLE unified entry point (Hydra stage dispatch)
├── Makefile
├── Dockerfile
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## 4. Key Design Decisions

### Unified entry point (`run.py`)

Single Hydra-dispatched entry point replaces 10 separate scripts:
```bash
uv run python run.py stage=preprocess|feature_engineer|eda|train|evaluate|tune model=ae|vae
```
Uses `match cfg.stage` to dispatch with lazy imports.

### Typed dataclass configs

Use Hydra structured configs (`@dataclass`) for `AEConfig` and `VAEConfig` instead
of raw `DictConfig`. Register with `ConfigStore`. This eliminates mypy union-attr issues.

`VAEConfig` extends `AEConfig` adding: `reconstruction_loss`, `beta`, `beta_schedule`,
`beta_warmup_epochs`.

### Training: background only

- Train/val split partitions **background events only**
- Signal events are held out, used only at evaluation time
- DataModule yields `(features, weights)` tuples (weights must be in TensorDataset
  so they shuffle together)

### Normalization

Default `z_score` (StandardScaler). Alternative `min_max` (MinMaxScaler). Config option:
`model.normalization`. Z-score is incompatible with BCE loss. Scaler params saved inside
Lightning checkpoint (no joblib).

### Loss functions

AE: `mse` (default), `smooth_l1`, `bce` (only with min_max). VAE adds KL divergence
weighted by `beta`. Apply event weights via `reduction="none"` then weighted mean.

### LR schedulers

`reduce_on_plateau` (default), `cosine_annealing`, or `none`. Configured in model YAML.

### Anomaly scoring

Per-event reconstruction error (MSE across features). Higher = more anomalous.
For VAE, can also use ELBO. Per-feature reconstruction error replaces SHAP for
interpretability.

### Evaluation metrics

ROC AUC (bkg=0, sig=1), SIC curves (signal_eff / sqrt(bkg_eff)), per-signal-mass-point
ROC, reconstruction error distributions (bkg vs each signal), per-feature reconstruction
error, latent space visualization (t-SNE/UMAP).

---

## 5. Model Architecture Notes

### AE (`src/models/ae.py`)
LightningModule with configurable encoder/decoder layers, latent dim, dropout, activation.
`forward(x) -> x_hat`. Loss supports mse/bce/smooth_l1.

### VAE (`src/models/vae.py`)
Extends AE pattern. Encoder outputs to `fc_mu` and `fc_logvar` layers. Reparameterization
trick. `forward(x) -> (x_hat, mu, logvar)`. Log `recon_loss`, `kl_loss`, `beta`,
`mu_mean`, `mu_var`, `logvar_mean` for collapse monitoring. Clamp logvar to [-10, 10].

### DataModule (`src/models/datamodule.py`)
LightningDataModule. `setup()`: reads MC parquet, splits bkg/sig, fits scaler on
bkg_train, stores signal for eval. Provides `train_dataloader`, `val_dataloader`,
`predict_dataloader` (bkg_test + signal).

### Plot catalog (`src/models/plots.py`)
- **Training**: `plot_loss()`, `plot_loss_components()` (VAE), `plot_reconstruction_error()`
- **Reconstruction**: `plot_reconstruction_performance()` (single event bar chart),
  `plot_reconstruction_comparison()` (AE vs VAE), `plot_feature_histograms()` (original vs reconstructed)
- **VAE latent**: `plot_latent_histograms()`, `plot_latent_space_2d()`, `plot_latent_pairplot()`,
  `plot_latent_mean_histograms()`, `plot_latent_mean_spread()` (warn if var < 0.1),
  `plot_logvar_histograms()`, `plot_logvar_spread()` (warn if logvar < -5),
  `plot_mu_vs_logvar()`, `plot_kl_per_dimension()`, `plot_sampled_latent_space()`

---

## 6. Preprocessing Pipeline (Reuse)

Entire `src/processing/` reused from predecessor. Key adaptation for anomaly detection:

- `splits.py`: `prepare_features(df)` — features + weights only (no class labels for training).
  `split_background_signal(df)` — separate bkg from signal before training.
  Train/val split on background only (no stratification needed).
- Output DataFrame: `anomaly_score | sample_type | eventOrigin` (not `y_true | y_pred | p_<class>`)
- `data.parquet` (real data) not used in training

---

## 7. Integration Notes

**PyTorch Lightning**: Handles device detection, AMP, training loops, early stopping,
checkpointing, metric logging. Use `MLFlowLogger` with SQLite backend.

**Ray Tune**: ASHA scheduler for aggressive early stopping. Native Lightning integration
via `TuneReportCheckpointCallback`. Search space defined in `configs/tuning/default.yaml`.

**MLflow**: Log params (architecture, lr, beta, etc.), per-epoch metrics, final metrics
(ROC AUC, SIC), artifacts (plots, checkpoints, score DataFrames).

---

## 8. Coding Conventions

### Inherited from predecessor
- `from __future__ import annotations` in every file
- `logging` module (not print), Google-style docstrings, type hints everywhere
- Functional style, feature sets via Hydra YAML, parquet intermediate format
- ATLAS publication-style plots (atlas-mpl-style)

### New for this project
- Typed dataclass configs (not raw DictConfig)
- LightningModule for all models (no manual training loops)
- Single entry point (`run.py`)
- Never initialize as `None` then conditionally assign — use early returns
- `torch.save`/`torch.load` with `weights_only=True`
- `torchmetrics` for metrics, Ray Tune for HP search
- `matplotlib.use("Agg")` before any matplotlib import in entry points
- `pyrootutils.setup_root()` in entry points for Hydra config resolution

---

## 9. Data Paths

ROOT ntuples: `/disk/atlas3/data_MC/{analysis_base}/{campaign}/vector_XEplateau/{sample}.root`

Processed outputs under `data/processed/ML/{version}/{run}/{region}/{channel}/`:
- `dataframes/`: `mc.parquet`, `data.parquet`, `ae_scores.parquet`, `vae_scores.parquet`
- `models/`: `ae.ckpt`, `vae.ckpt`
- `plots/`: `eda/`, `ae/`, `vae/`, `ae_evaluation/`, `vae_evaluation/`
- `metrics/`: `ae_metrics.json`, `vae_metrics.json`

---

## 10. Step-by-Step Implementation Plan

Each numbered step is a separate unit of work requiring user confirmation.
Steps marked **USER COPIES** mean the user will copy files from the predecessor.

### Phase 1: Scaffold
1. Initialize git repo, create `.gitignore`
2. Create `pyproject.toml`
3. Create `.pre-commit-config.yaml`
4. Create `Makefile`
5. Create `.github/workflows/ci.yml`
6. Create empty package structure (`src/`, `tests/` `__init__.py` files)
7. Create `run.py` skeleton
8. Run `uv sync` and `pre-commit install`, verify
9. **USER COPIES** `src/processing/` — Claude verifies
10. **USER COPIES** `src/eda/` — Claude verifies
11. **USER COPIES** `src/visualization/` and `src/utils.py` — Claude verifies
12. **USER COPIES** `configs/` (features, data, samples, merge, run, paths, hydra) — Claude verifies

### Phase 2: Configs
13. Create `configs/model/ae.yaml`
14. Create `configs/model/vae.yaml`
15. Create `configs/tuning/default.yaml`
16. Update `configs/config.yaml`
17. Update `configs/pipeline/default.yaml`
18. Create typed `AEConfig` dataclass
19. Create typed `VAEConfig` dataclass

### Phase 3: Data
20. Implement `src/models/splits.py`
21. Implement `src/models/datamodule.py`
22. Write tests for splits and datamodule

### Phase 4: Models
23. Implement `src/models/ae.py`
24. Write tests for AE
25. Implement `src/models/vae.py`
26. Write tests for VAE

### Phase 5: Scoring & Evaluation
27. Implement `src/models/anomaly.py`
28. Write tests for anomaly scoring
29. Implement `src/models/evaluation.py`
30. Write tests for evaluation metrics
31. Implement `src/models/latent.py`
32. Write tests for latent utilities

### Phase 6: Plots
33. `src/models/plots.py` — training diagnostics
34. `src/models/plots.py` — reconstruction plots
35. `src/models/plots.py` — VAE latent diagnostics
36. `src/models/plots.py` — evaluation plots

### Phase 7: Tuning
37. Implement `src/models/tuning.py`
38. Wire up `run.py stage=tune`
39. Write tests for tuning

### Phase 8: Serving
40. `src/serving/schemas.py`
41. `src/serving/registry.py`
42. `src/serving/app.py`
43. Write serving tests

### Phase 9: Integration & Polish
44. Wire all stages into `run.py`
45. Create `Dockerfile`
46. Create notebooks (00–06)
47. Final CI check
48. First real training run on ATLAS data

---

## 11. Known Pitfalls

1. **Event weights in loss**: Use `reduction="none"`, then `(loss.mean(dim=1) * w).mean()`
2. **VAE logvar clamping**: `torch.clamp(logvar, -10, 10)` to prevent NaN
3. **VAE collapse detection**: Warn if `mu.var() < 0.1` or `logvar < -5`
4. **Z-score + BCE incompatible**: Z-score gives unbounded values; use MSE/SmoothL1
5. **MinMaxScaler clipping**: Clip test data to [0, 1] after transform for BCE
6. **SmoothL1 for noisy features**: Less sensitive to outliers (HEP kinematic tails)
7. **KL normalization**: `.sum(dim=1).mean()` — sum over latent dims, mean over batch
8. **Scaler in checkpoint**: Save scaler params inside Lightning checkpoint (no joblib)
9. **Weight shuffling**: Include weights in `TensorDataset` so they shuffle with features
10. **torch.load security**: Always `weights_only=True`
11. **Never init as None then conditionally assign** — mypy can't narrow across blocks
