from __future__ import annotations

import logging

import awkward as ak
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def merge_backgrounds(
    samples: dict[str, ak.Array],
    cfg: DictConfig,
) -> dict[str, ak.Array]:
    """Merge background samples into groups according to the configured strategy.

    Args:
        samples: Dict mapping sample IDs to awkward arrays.
        cfg: Hydra DictConfig with 'merge.background_strategy' key.
    """
    strategy = cfg.merge.background_strategy

    if strategy == "as_one":
        merged = ak.concatenate(list(samples.values()), axis=0)
        return {"background": merged}

    if strategy == "as_primary":
        groups = OmegaConf.to_container(cfg.merge.primary_groups, resolve=True)
        out: dict[str, ak.Array] = {}
        for group_name, member_ids in groups.items():
            arrays = [samples[sid] for sid in member_ids if sid in samples]
            if arrays:
                out[group_name] = (
                    ak.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
                )
        return out

    raise ValueError(
        f"Unsupported background_strategy '{strategy}'. Use 'as_one' or 'as_primary'."
    )


def group_samples(
    samples: dict[str, ak.Array],
    cfg: DictConfig,
) -> dict[str, dict[str, ak.Array]]:
    """Group a flat sample dict into data/background/signal categories.

    Args:
        samples: Dict mapping sample IDs to awkward arrays.
        cfg: Hydra DictConfig with 'samples' section.
    """
    data_ids: set[str] = set()
    if cfg.samples.data.get("enabled", False):
        data_ids = {
            s["id"]
            for s in OmegaConf.to_container(cfg.samples.data.samples, resolve=True)
        }

    signal_patterns: list[str] = []
    if cfg.samples.signal.get("enabled", False):
        signal_patterns = list(cfg.samples.signal.filter_patterns)

    data: dict[str, ak.Array] = {}
    background: dict[str, ak.Array] = {}
    signal: dict[str, ak.Array] = {}

    for sid, array in samples.items():
        if sid in data_ids:
            data[sid] = array
        elif any(pat in sid for pat in signal_patterns):
            signal[sid] = array
        else:
            background[sid] = array

    return {"data": data, "background": background, "signal": signal}


def merge_signals(
    signal: dict[str, ak.Array],
    cfg: DictConfig,
) -> dict[str, ak.Array]:
    """Merge signal samples according to the configured strategy.

    Args:
        signal: Dict mapping signal sample IDs to awkward arrays.
        cfg: Hydra DictConfig with 'merge.signal_strategy' key.
            Supported strategies: 'as_is', 'as_one', 'as_type', 'as_mass'.
    """
    strategy = cfg.merge.signal_strategy

    if strategy == "as_is":
        return dict(signal)

    if strategy == "as_one":
        return {"signal": ak.concatenate(list(signal.values()), axis=0)}

    type_names: dict[str, str] = OmegaConf.to_container(
        cfg.merge.signal_type_names, resolve=True
    )

    if strategy == "as_type":
        out: dict[str, ak.Array] = {}
        for sid, array in signal.items():
            prefix = sid.split("_")[0]
            group = type_names.get(prefix, prefix)
            out[group] = (
                ak.concatenate([out[group], array], axis=0) if group in out else array
            )
        return out

    if strategy == "as_mass":
        thresholds: dict[str, list[int]] = OmegaConf.to_container(
            cfg.merge.mass_thresholds, resolve=True
        )
        out = {}
        for sid, array in signal.items():
            parts = sid.split("_")
            prefix = parts[0]
            mass = int(parts[1])
            type_name = type_names.get(prefix, prefix)
            lo, hi = thresholds.get(type_name, [0, 0])
            if mass < lo:
                group = f"low_mass_{type_name}"
            elif mass < hi:
                group = f"medium_mass_{type_name}"
            else:
                group = f"high_mass_{type_name}"
            out[group] = (
                ak.concatenate([out[group], array], axis=0) if group in out else array
            )
        return out

    raise ValueError(
        f"Unsupported signal_strategy '{strategy}'. "
        "Use 'as_is', 'as_one', 'as_type', or 'as_mass'."
    )


def split_mc_data(
    grouped: dict[str, dict[str, ak.Array]],
) -> tuple[dict[str, ak.Array], dict[str, ak.Array]]:
    """Split grouped samples into MC (background + signal) and data dicts.

    Args:
        grouped: Dict with 'background', 'signal', and 'data' keys.
    """
    samples_mc = {**grouped.get("background", {}), **grouped.get("signal", {})}
    samples_data = dict(grouped.get("data", {}))
    return samples_mc, samples_data


def combine_background_signal(
    background: dict[str, ak.Array],
    signal: dict[str, ak.Array],
) -> dict[str, ak.Array]:
    """Combine background and signal sample dicts into one.

    Args:
        background: Dict mapping background sample IDs to arrays.
        signal: Dict mapping signal sample IDs to arrays.
    """
    return {**background, **signal}


def assign_class(samples: dict[str, ak.Array]) -> None:
    """Add an integer 'class' field to each sample in-place.

    Args:
        samples: Dict mapping sample IDs to awkward arrays.
    """
    for i, key in enumerate(samples):
        samples[key]["class"] = i


def dict_to_array(samples: dict[str, ak.Array]) -> ak.Array:
    """Concatenate all samples into a single awkward array.

    Args:
        samples: Dict mapping sample IDs to awkward arrays.
    """
    return ak.concatenate(list(samples.values()), axis=0)
