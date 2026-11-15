"""The sole EXP-023 treatment: partition-local painting centering with a common mask."""

import numpy as np

from . import data
from .caption_data import allowed_captions
from .caption_targets import fit_targets


def centered_targets(captions, fitting, vectors, config):
    # Physically select before aggregation/PCA; held-out caption vectors are never requested.
    selected = allowed_captions(captions, fitting)
    keys = set(fitting[data.KEYS].itertuples(index=False, name=None))
    if not set(selected[data.KEYS].itertuples(index=False, name=None)) <= keys:
        raise ValueError("Caption partition leakage")
    uncentered, _, pca, records, metadata = fit_targets(selected, fitting, vectors, config)
    centered = {}
    valid = {}
    for painting, rows in records.groupby("painting", sort=True):
        pair_keys = list(rows[data.KEYS].itertuples(index=False, name=None))
        eligible = [key for key in pair_keys if key in uncentered]
        # Require >=3 originally eligible languages, all original projected vectors valid
        # for the centering mean. Invalid originals never enter that mean or either loss.
        if len(eligible) < config["minimum_languages"]:
            continue
        mean = np.stack([uncentered[key] for key in eligible]).mean(axis=0)
        for key in eligible:
            delta = uncentered[key] - mean
            norm = np.linalg.norm(delta)
            if np.isfinite(delta).all() and norm > config["normalization_epsilon"]:
                valid[key] = uncentered[key]
                centered[key] = (delta / norm).astype(np.float32)
    counts = {}
    for painting, _ in centered:
        counts[painting] = counts.get(painting, 0) + 1
    weights = {key: 1. / counts[key[0]] for key in centered}
    metadata.update(common_mask_pairs=len(centered), common_mask_paintings=len(counts),
                    all_bce_pairs_retained=len(fitting), centered_epsilon=config["normalization_epsilon"],
                    common_mask_keys=[list(key) for key in sorted(centered)])
    return {"C1": (valid, weights, pca, records, metadata),
            "C2": (centered, weights, pca, records, metadata)}


def prepare_partitions(captions, fitting, inner, vectors, config):
    results = {phase: centered_targets(captions, rows, vectors, config)
               for phase, rows in (("inner", inner), ("full", fitting))}
    return {condition: {phase: result[condition] for phase, result in results.items()}
            for condition in ("C1", "C2")}
