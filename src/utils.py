from __future__ import annotations

import logging
import os
import warnings

log = logging.getLogger(__name__)


def suppress_warnings():
    """Suppress Python warnings, Lightning tips, and WandB noise."""
    warnings.filterwarnings("ignore")

    # Suppress Lightning info-level messages (GPU/TPU available, litlogger tip, …)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    # Suppress verbose WandB login / sync output
    os.environ.setdefault("WANDB_SILENT", "true")

    log.info("Unessential warnings suppressed.")
