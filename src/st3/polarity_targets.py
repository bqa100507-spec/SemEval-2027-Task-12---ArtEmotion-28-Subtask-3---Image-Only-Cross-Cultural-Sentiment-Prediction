"""Shared train-only PCA geometry for pooled and polarity-specific captions."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from . import data
from .caption_data import allowed_captions

KINDS = ("all", "positive", "negative")


def polarity_means(captions, pairs, vectors):
    selected = allowed_captions(captions, pairs)
    records, values, masks = [], [], []
    for (painting, language), group in selected.groupby(data.KEYS, sort=True):
        matrix = np.stack([vectors[key] for key in group.caption_id])
        positive = group.emotion.isin(data.POSITIVE).to_numpy()
        counts = (len(group), int(positive.sum()), int((~positive).sum()))
        means = [matrix.mean(axis=0)]
        for keep in (positive, ~positive):
            means.append(matrix[keep].mean(axis=0) if keep.any() else np.zeros(matrix.shape[1], dtype=np.float32))
        records.append({"painting": painting, "language": language,
                        **dict(zip(("n_all", "n_positive", "n_negative"), counts))})
        values.append(means)
        masks.append([count > 0 for count in counts])
    if not records:
        raise ValueError("No valid polarity captions in this training partition")
    records = pd.DataFrame(records)
    expected = records.merge(pairs[[*data.KEYS, "k", "n"]], on=data.KEYS, validate="one_to_one")
    if (not expected.n_all.le(expected.n).all() or not expected.n_positive.le(expected.k).all()
            or not expected.n_negative.le(expected.n - expected.k).all()):
        raise ValueError("Caption polarity counts exceed the original votes")
    return records, np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=bool)


def fit_polarity_targets(captions, pairs, vectors, config):
    records, means, available = polarity_means(captions, pairs, vectors)
    return fit_shared_geometry(records, means, available, config)


def fit_shared_geometry(records, means, available, config):
    width = config["target_dimensions"]
    if min(means[:, 0].shape) < width:
        raise ValueError("Insufficient training pairs for the fixed PCA dimension")
    pca = PCA(n_components=width, whiten=False, svd_solver=config["pca_solver"], random_state=config["pca_seed"])
    with threadpool_limits(limits=4):
        pca.fit(means[:, 0])
        projected = pca.transform(means.reshape(-1, means.shape[-1])).reshape(len(means), 3, width)
    norms = np.linalg.norm(projected, axis=-1)
    valid = available & np.isfinite(projected).all(axis=-1) & (norms > 1e-8)
    targets = np.zeros_like(projected, dtype=np.float32)
    targets[valid] = projected[valid] / norms[valid, None]
    metadata = {"fit_paintings": sorted(records.painting.unique()), "fit_pairs": len(records),
                "caption_pairs": len(records), "whiten": False, "fit_kind": "all",
                "variance_ratio": np.nan_to_num(pca.explained_variance_ratio_, nan=0.).tolist(),
                "valid_channels": dict(zip(KINDS, valid.sum(axis=0).astype(int).tolist())),
                "degenerate_channels": dict(zip(KINDS, (available & ~valid).sum(axis=0).astype(int).tolist()))}
    return {"records": records, "targets": targets, "valid": valid, "pca": pca, "metadata": metadata}


def condition_targets(bundle, condition):
    if condition not in ("M2", "M3"):
        raise ValueError("Only M2 and M3 have semantic targets")
    if condition == "M2":
        values = np.repeat(bundle["targets"][:, :1], 2, axis=1)
        mask = np.repeat(bundle["valid"][:, :1], 2, axis=1)
    else:
        values, mask = bundle["targets"][:, 1:], bundle["valid"][:, 1:]
    keys = list(bundle["records"][data.KEYS].itertuples(index=False, name=None))
    present = mask.any(axis=1)
    counts = bundle["records"].loc[present].groupby("painting").size()
    targets = {key: (value, keep) for key, value, keep in zip(keys, values, mask)}
    weights = {key: 1. / counts[key[0]] if keep else 0. for key, keep in zip(keys, present)}
    return targets, weights


def average_pairwise_cosine(values):
    values = values[np.linalg.norm(values, axis=1) > 1e-8]
    if len(values) < 2:
        return None
    unit = values / np.linalg.norm(values, axis=1, keepdims=True).clip(1e-12)
    value = (np.square(unit.sum(axis=0)).sum() - len(unit)) / (len(unit) * (len(unit) - 1))
    return float(np.clip(value, -1., 1.))


def polarity_audit(captions, pairs, vectors):
    records, means, mask = polarity_means(captions, pairs, vectors)
    both = mask[:, 1] & mask[:, 2]
    scored = both & (np.linalg.norm(means[:, 1], axis=1) > 1e-8) & (np.linalg.norm(means[:, 2], axis=1) > 1e-8)
    similarities = (means[scored, 1] * means[scored, 2]).sum(axis=1) / (
        np.linalg.norm(means[scored, 1], axis=1) * np.linalg.norm(means[scored, 2], axis=1))
    similarities = np.clip(similarities, -1., 1.)
    summary = {"pairs": len(records), "both": int(both.sum()), "positive_only": int((mask[:, 1] & ~mask[:, 2]).sum()),
               "negative_only": int((~mask[:, 1] & mask[:, 2]).sum()),
               "caption_rows": int(records.n_all.sum()),
               "within_pair_cosine_quantiles": {str(q): float(np.quantile(similarities, q))
                                                for q in (0, .1, .5, .9, 1)} if len(similarities) else {}}
    variations = []
    for index, kind in enumerate(KINDS):
        subset = records.loc[mask[:, index]].copy()
        selected = means[mask[:, index], index]
        subset["row"] = np.arange(len(subset))
        norms = np.linalg.norm(selected, axis=1)
        summary[kind] = {"pairs": len(selected),
                         "mean_component_variance": float(selected.var(axis=0).mean()) if len(selected) else None,
                         "norm_min": float(norms.min()) if len(selected) else None,
                         "norm_max": float(norms.max()) if len(selected) else None,
                         "degenerate_raw_means": int((norms <= 1e-8).sum())}
        for field, label in (("language", "between_paintings"), ("painting", "between_languages")):
            for key, rows in subset.groupby(field):
                cosine = average_pairwise_cosine(selected[rows.row])
                if cosine is not None:
                    variations.append({"kind": kind, "comparison": label, "group": key,
                                       "pairs": len(rows), "mean_pairwise_cosine": cosine})
    records["positive_negative_cosine"] = np.nan
    records.loc[scored, "positive_negative_cosine"] = similarities
    return summary, records, pd.DataFrame(variations), means, mask
