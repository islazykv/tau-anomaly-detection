"""Custom Lightning callbacks."""

from __future__ import annotations

from collections import defaultdict

import lightning as L
from tqdm.auto import tqdm


class EpochProgressBar(L.Callback):
    """Lightweight epoch-level tqdm progress bar.

    Replaces Lightning's default per-batch progress bar with a single bar
    that tracks epochs and displays key metrics (train_loss, val_loss).
    """

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._bar = tqdm(total=trainer.max_epochs, desc="Training", unit="epoch")

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        postfix = {}
        for key in ("train_loss", "val_loss"):
            if key in metrics:
                postfix[key] = f"{float(metrics[key]):.5f}"
        for key in ("val_mu_var", "val_logvar_mean"):
            if key in metrics:
                postfix[key] = f"{float(metrics[key]):.3f}"
        self._bar.set_postfix(postfix)
        self._bar.update(1)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._bar.close()


class MetricTracker(L.Callback):
    """Record per-epoch metrics for post-training plotting.

    After training, access ``tracker.history`` — a dict mapping metric
    names to lists of per-epoch float values.

    Example::

        tracker = MetricTracker()
        trainer = L.Trainer(callbacks=[..., tracker])
        trainer.fit(model, datamodule=dm)
        plot_loss(tracker.history["train_loss"], tracker.history["val_loss"])
    """

    def __init__(self) -> None:
        super().__init__()
        self.history: dict[str, list[float]] = defaultdict(list)

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        metrics = trainer.callback_metrics
        for key, value in metrics.items():
            if key.startswith("train_"):
                self.history[key].append(float(value))

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        for key, value in metrics.items():
            if not key.startswith("train_"):
                self.history[key].append(float(value))
