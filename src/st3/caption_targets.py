"""Vote-preserving caption means and partition-local, unwhitened PCA targets."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from . import data
from .caption_data import allowed_captions


def raw_targets(captions, pairs, vectors):
    selected = allowed_captions(captions, pairs)
    means, records = [], []
    for (painting, language), group in selected.groupby(data.KEYS, sort=True):
        values = np.stack([vectors[key] for key in group.caption_id])
        means.append(values.mean(axis=0))
        records.append({"painting": painting, "language": language, "captions": len(group),
                        "positive": int(group.emotion.isin(data.POSITIVE).sum()),
                        **{emotion: int(group.emotion.eq(emotion).sum()) for emotion in sorted(data.POSITIVE | data.NEGATIVE)}})
    if not records:
        raise ValueError("No valid original binary captions in the training partition")
    return pd.DataFrame(records), np.stack(means).astype(np.float32)


def fit_targets(captions, pairs, vectors, config):
    records, values = raw_targets(captions, pairs, vectors)
    width = config["target_dimensions"]
    if min(values.shape) < width:
        raise ValueError("Insufficient training targets for the frozen PCA dimension")
    pca = PCA(n_components=width, whiten=False, svd_solver=config["pca_solver"], random_state=config["pca_seed"])
    with threadpool_limits(limits=4):
        projected = pca.fit_transform(values)
    norms = np.linalg.norm(projected, axis=1)
    valid = np.isfinite(projected).all(axis=1) & (norms > 1e-8)
    normalized = np.zeros_like(projected, dtype=np.float32)
    normalized[valid] = projected[valid] / norms[valid, None]
    keys = list(records[data.KEYS].itertuples(index=False, name=None))
    targets = {key: value for key, value, keep in zip(keys, normalized, valid) if keep}
    counts = records.loc[valid].groupby("painting").size()
    weights = {key: 1. / counts[key[0]] for key in targets}
    variance_ratio = np.nan_to_num(pca.explained_variance_ratio_, nan=0.)
    metadata = {"fit_paintings": sorted(pairs.painting.unique()), "fit_pairs": len(pairs),
                "caption_pairs": len(records), "valid_targets": int(valid.sum()),
                "degenerate_targets": int((~valid).sum()), "captions": int(records.captions.sum()),
                "explained_variance_ratio": variance_ratio.astype(float).tolist(),
                "explained_variance_total": float(variance_ratio.sum()), "whiten": pca.whiten}
    return targets, weights, pca, records, metadata
