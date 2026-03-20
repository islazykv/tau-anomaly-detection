from __future__ import annotations

import awkward as ak
from omegaconf import DictConfig, OmegaConf


def extract_feature_from_array(array_in: ak.Array, feature_name: str) -> ak.Array:
    """Extract a single feature field from an awkward array.

    Args:
        array_in: Input awkward array.
        feature_name: Name of the field to extract.
    """
    return array_in[feature_name]


def drop_features(array_in: ak.Array, feature_list: list[str]) -> ak.Array:
    """Return the array with the specified feature fields removed.

    Args:
        array_in: Input awkward array.
        feature_list: Names of fields to remove.
    """
    return array_in[[f for f in ak.fields(array_in) if f not in feature_list]]


def assign_event_origin(grouped: dict[str, dict[str, ak.Array]]) -> None:
    """Add an 'eventOrigin' field to each array in-place, set to the sample id.

    Args:
        grouped: Nested dict of category -> sample_id -> awkward array.
    """
    for category in grouped.values():
        for sid, array in category.items():
            category[sid] = ak.with_field(array, sid, "eventOrigin")


def resolve_features_to_drop(cfg: DictConfig) -> list[str]:
    """Build the list of features to drop before rectangularization.

    Args:
        cfg: Hydra DictConfig with 'features' section.
    """
    feat_cfg = cfg.features
    drop: list[str] = []
    for group in ("cleaning", "truth", "weights"):
        drop.extend(OmegaConf.to_container(feat_cfg[group], resolve=True))
    drop.extend(OmegaConf.to_container(feat_cfg.drop_extra, resolve=True))
    return drop


def resolve_features(cfg: DictConfig) -> list[str]:
    """Build a flat feature list from Hydra config based on scope and channel.

    Args:
        cfg: Hydra DictConfig with 'features' and 'analysis' sections.
    """
    feat_cfg = cfg.features
    scope = cfg.analysis.scope
    channel = str(cfg.analysis.channel) if cfg.analysis.channel is not None else None

    if scope in ("NTuples", "CC"):
        return list(OmegaConf.to_container(feat_cfg.features, resolve=True))

    features: list[str] = []
    for group in (
        "cleaning",
        "truth",
        "weights",
        "training",
        "tau",
        "jet",
        "kinematic",
    ):
        features.extend(OmegaConf.to_container(feat_cfg[group], resolve=True))

    if channel in ("1had1lep", "1had0lep", "1"):
        features.extend(OmegaConf.to_container(feat_cfg.channel_1, resolve=True))
    elif channel == "2":
        features.extend(OmegaConf.to_container(feat_cfg.channel_2, resolve=True))

    return features
