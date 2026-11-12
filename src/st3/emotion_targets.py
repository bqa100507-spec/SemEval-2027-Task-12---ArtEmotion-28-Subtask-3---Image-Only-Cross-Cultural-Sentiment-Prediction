"""Eight original emotion means and shared, existing pooled-caption PCA geometry."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import data
from .caption_data import allowed_captions
from .polarity_targets import average_pairwise_cosine

EMOTIONS = ("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness")
POLARITY = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def emotion_means(captions, pairs, vectors, *, destination=None):
    selected = allowed_captions(captions, pairs)
    if not set(selected.emotion).issubset(EMOTIONS):
        raise ValueError("Unexpected binary emotion")
    records = pairs[data.KEYS].sort_values(data.KEYS).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(records[data.KEYS])
    width = len(next(iter(vectors.values())))
    shape = (len(records), 8, width)
    means = (np.zeros(shape, dtype=np.float32) if destination is None else
             np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=shape))
    means[:] = 0
    counts = np.zeros((len(records), 8), dtype=np.int32)
    dispersion = []
    for key, group in selected.groupby(data.KEYS + ["emotion"], sort=True):
        row, emotion = int(index.get_loc(key[:2])), EMOTIONS.index(key[2])
        matrix = np.stack([vectors[key] for key in group.caption_id])
        means[row, emotion] = matrix.mean(axis=0)
        counts[row, emotion] = len(matrix)
        if len(matrix) >= 2:
            dispersion.append({"painting":key[0], "language":key[1], "emotion":key[2],
                               "captions":len(matrix), "caption_mean_pairwise_cosine":average_pairwise_cosine(matrix)})
    expected = records.merge(pairs[data.KEYS + ["k", "n"]], on=data.KEYS, validate="one_to_one")
    if ((counts[:, :4].sum(axis=1) > expected.k).any() or
            (counts[:, 4:].sum(axis=1) > expected.n-expected.k).any()):
        raise ValueError("Emotion caption counts exceed original votes")
    if isinstance(means, np.memmap):
        means.flush()
    return records, means, counts, pd.DataFrame(dispersion)


def normalize(values, available):
    norms = np.linalg.norm(values, axis=-1)
    valid = available & np.isfinite(values).all(axis=-1) & (norms > 1e-8)
    unit = np.zeros_like(values, dtype=np.float32)
    unit[valid] = values[valid] / norms[valid, None]
    return unit, valid


def cosine_matrix(values, valid):
    size = values.shape[1]
    sums, counts = np.zeros((size, size)), np.zeros((size, size), dtype=np.int64)
    # Normalize one channel at a time to avoid a second full raw-mean allocation.
    for left in range(size):
        a, av = normalize(values[:, left], valid[:, left])
        for right in range(left, size):
            b, bv = normalize(values[:, right], valid[:, right])
            keep = av & bv
            counts[left, right] = counts[right, left] = keep.sum()
            if keep.any():
                sums[left, right] = sums[right, left] = np.clip((a[keep]*b[keep]).sum(axis=1), -1., 1.).sum()
    matrix = np.full((size, size), np.nan)
    np.divide(sums, counts, out=matrix, where=counts > 0)
    return matrix, counts


def geometry_statistics(records, values, valid, polarity_values, polarity_valid, threshold):
    matrix, pair_counts = cosine_matrix(values, valid)
    variation, similarities = [], np.full(valid.shape, np.nan, dtype=np.float32)
    for channel, emotion in enumerate(EMOTIONS):
        unit, keep = normalize(values[:, channel], valid[:, channel])
        punit, pkeep = normalize(polarity_values[:, POLARITY[channel]], polarity_valid[:, POLARITY[channel]])
        both = keep & pkeep
        similarities[both, channel] = np.clip((unit[both]*punit[both]).sum(axis=1), -1., 1.)
        subset = records.loc[keep].copy().assign(row=np.flatnonzero(keep))
        for field, label in (("painting", "between_languages"), ("language", "between_paintings")):
            for group, rows in subset.groupby(field):
                cosine = average_pairwise_cosine(unit[rows.row])
                if cosine is not None:
                    variation.append({"emotion":emotion, "comparison":label, "group":group,
                                      "pairs":len(rows), "mean_pairwise_cosine":cosine})
    within = across = within_count = across_count = 0.
    for left in range(8):
        for right in range(left+1, 8):
            n = pair_counts[left, right]
            if n:
                if POLARITY[left] == POLARITY[right]:
                    within += matrix[left, right]*n; within_count += n
                else:
                    across += matrix[left, right]*n; across_count += n
    finite = np.isfinite(similarities)
    summary = {"within_polarity_cosine":within/within_count if within_count else None,
               "across_polarity_cosine":across/across_count if across_count else None,
               "coobserved_within_channels":int(within_count), "coobserved_across_channels":int(across_count),
               "emotion_polarity_median_cosine":float(np.nanmedian(similarities)) if finite.any() else None,
               "near_polarity_fraction":float((similarities[finite] >= threshold).mean()) if finite.any() else None,
               "emotion_polarity_comparisons":int(finite.sum())}
    summary["per_emotion_polarity_similarity"] = {
        emotion:{"comparisons":int(finite[:,channel].sum()),
                 "median":float(np.nanmedian(similarities[:,channel])) if finite[:,channel].any() else None,
                 "near_fraction":float((similarities[finite[:,channel],channel]>=threshold).mean()) if finite[:,channel].any() else None}
        for channel,emotion in enumerate(EMOTIONS)}
    return summary, matrix, pair_counts, pd.DataFrame(variation), similarities


def project_targets(records, means, counts, polarity_records, polarity_raw, partition, pca_path, polarity_bundle):
    """No PCA fitting: transform with exactly the original components and mean."""
    ordered = partition[data.KEYS].sort_values(data.KEYS).reset_index(drop=True)
    ix = pd.MultiIndex.from_frame(records[data.KEYS]).get_indexer(pd.MultiIndex.from_frame(ordered[data.KEYS]))
    pi = pd.MultiIndex.from_frame(polarity_records[data.KEYS]).get_indexer(pd.MultiIndex.from_frame(ordered[data.KEYS]))
    if (ix < 0).any() or (pi < 0).any():
        raise ValueError("Target cache misses training pairs")
    with np.load(pca_path, allow_pickle=False) as pca:
        components, mean = pca["components"], pca["mean"]
    projected = np.zeros((len(ordered), 8, len(components)), dtype=np.float32)
    polarity = np.zeros((len(ordered), 2, len(components)), dtype=np.float32)
    with threadpool_limits(limits=4):
        for channel in range(8):
            projected[:, channel] = (means[ix, channel]-mean) @ components.T
        for channel in range(2):
            polarity[:, channel] = (polarity_raw[pi, channel+1]-mean) @ components.T
    evalues, evalid = normalize(projected, counts[ix] > 0)
    pvalues, pvalid = normalize(polarity, np.isfinite(polarity_raw[pi, 1:]).all(axis=-1) &
                                (np.linalg.norm(polarity_raw[pi, 1:], axis=-1) > 1e-8))
    reference_index = pd.MultiIndex.from_frame(polarity_bundle["records"][data.KEYS]).get_indexer(pd.MultiIndex.from_frame(ordered[data.KEYS]))
    if (reference_index < 0).any():
        raise ValueError("Parent PCA target rows missing")
    np.testing.assert_allclose(pvalues, polarity_bundle["targets"][reference_index, 1:], atol=3e-5, rtol=0.)
    np.testing.assert_array_equal(pvalid, polarity_bundle["valid"][reference_index, 1:])
    # Intersection is fixed, never condition-specific: even a numerical
    # degeneracy removes that channel from both conditions and denominators.
    valid = evalid & pvalid[:, POLARITY]
    avalues = pvalues[:, POLARITY].copy()
    evalues[~valid] = 0; avalues[~valid] = 0
    return {"records":ordered, "A8":avalues, "E8":evalues, "valid":valid,
            "counts":counts[ix], "polarity":pvalues, "polarity_valid":pvalid,
            "emotion_projected_degenerate":int(((counts[ix]>0) & ~evalid).sum()),
            "common_mask_removed":int(((counts[ix]>0) & ~valid).sum())}


def condition_targets(bundle, condition):
    if condition not in ("A8", "E8"):
        raise ValueError("Unknown eight-query condition")
    keys = list(bundle["records"][data.KEYS].itertuples(index=False, name=None))
    present = bundle["valid"].any(axis=1)
    counts = bundle["records"].loc[present].groupby("painting").size()
    targets = {key:(value, mask) for key, value, mask in zip(keys, bundle[condition], bundle["valid"])}
    weights = {key:1./counts[key[0]] if keep else 0. for key, keep in zip(keys, present)}
    return targets, weights
