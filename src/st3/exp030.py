"""Data and count-weighted objectives for EXP-030."""
import csv

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .caption_data import require_isolation
from .data import AMBIGUOUS, KEYS, require_same_pairs
from .vlm_semantics import fingerprint

EMOTIONS = ("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness")


def count_votes(path, pairs):
    """Parse vote labels only for in-scope keys from the released training split."""
    counts = {key: np.zeros(8, dtype=np.int64) for key in pairs[KEYS].itertuples(index=False, name=None)}
    images = dict(zip(pairs.painting, pairs.image))
    emotion_index = {name: i for i, name in enumerate(EMOTIONS)}
    canonical = {"english": "English", "arabic": "Arabic", "chinese": "Chinese"}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"].strip() != "train":
                continue
            name, language = row["painting"].strip(), row["language"].strip()
            key = (name, canonical.get(language, language))
            if key not in counts:
                continue
            if row["image_name"].strip() != images[name]:
                raise ValueError("Vote image identity differs from frozen P2")
            emotion = row["emotion"].strip().lower()
            if emotion not in AMBIGUOUS:
                if emotion not in emotion_index:
                    raise ValueError("Unknown in-scope emotion")
                counts[key][emotion_index[emotion]] += 1
    matrix = np.stack(list(counts.values()))
    if not np.array_equal(matrix.sum(1), pairs.n) or not np.array_equal(matrix[:, :4].sum(1), pairs.k):
        raise ValueError("Eight-way vote reconstruction disagrees with pinned P2 counts")
    return pairs.assign(**dict(zip(EMOTIONS, matrix.T)))


def fold3(rows, manifest):
    train, val = rows[rows.fold.ne(3)], rows[rows.fold.eq(3)]
    require_isolation(train, val)
    for frame, name, size in ((train, "refit", 1068), (val, "held", 267)):
        frozen = manifest["populations"][name]
        paintings = sorted(frame.painting.unique())
        if (len(paintings) != size or paintings != frozen["paintings"] or len(frame) != frozen["pairs"]
                or fingerprint(frame[["painting", "language", "image_sha256"]].values.tolist()) != frozen["key_sha256"]):
            raise ValueError(f"P2 {name} fold identity differs")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def matched_reference(path, val):
    frame = pd.read_csv(path, float_precision="round_trip", usecols=[*KEYS, "fold", "p_positive"])
    pred = frame.loc[frame.fold.eq(3), [*KEYS, "p_positive"]]
    require_same_pairs(val, pred)
    return pred


def loss_sums(logits, counts):
    """Return weighted eight-way CE and binary BCE numerators in nats."""
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = -(counts * logp).sum()
    bce = -(counts[:, :4].sum(1) * logp[:, :4].logsumexp(1)
            + counts[:, 4:].sum(1) * logp[:, 4:].logsumexp(1)).sum()
    return ce, bce


def positive_probability(logits):
    return logits.float().softmax(dim=-1)[:, :4].sum(1).clamp(0., 1.)


def best_epoch(history):
    """Highest recorded Contrast and earliest epoch for an exact tie."""
    valid = [r for r in history if r["contrast"] is not None and np.isfinite(r["contrast"])]
    if not valid:
        raise ValueError("No eligible validation Contrast for checkpoint selection")
    return max(valid, key=lambda r: (r["contrast"], -r["epoch"]))["epoch"]
