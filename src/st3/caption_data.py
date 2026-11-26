"""Official training captions and the locked EXP-017 painting protocol."""

import csv
import hashlib
import math

import numpy as np
import pandas as pd

from . import data
from .baseline import file_sha256


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def training_captions(root, *, allowed_paintings=None):
    """Never retain a caption from an official validation row."""
    annotations = data.read_annotations(root, training_only=True)
    if allowed_paintings is not None:
        allowed_paintings = set(allowed_paintings)
        annotations = annotations[annotations.painting.isin(allowed_paintings)]
    rows = []
    with (root / data.ANNOTATIONS).open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if row["split"].strip() == "train" and (allowed_paintings is None or row["painting"].strip() in allowed_paintings):
                rows.append({"annotation_id": index, **{key: row[key] for key in data.ANNOTATION_COLUMNS},
                             "caption": row["caption"]})
    result = pd.DataFrame(rows)
    for key in data.ANNOTATION_COLUMNS:
        result[key] = annotations[key].to_numpy()
    if len(result) != len(annotations):
        raise ValueError("Caption rows differ from the original training annotations")
    result["valid_caption"] = result["caption"].str.strip().ne("") & ~result["caption"].str.contains("\x00", regex=False)
    result["binary"] = result["emotion"].isin(data.POSITIVE | data.NEGATIVE)
    result["caption_id"] = result["caption"].map(text_digest)
    return result


def require_isolation(*frames):
    for index, left in enumerate(frames):
        for right in frames[index + 1:]:
            for column in ("painting", "image", "image_sha256"):
                if column in left and column in right and set(left[column]) & set(right[column]):
                    raise ValueError(f"Partitions overlap on {column}")


def ranked_groups(frame, seed):
    groups = frame.groupby("image_sha256")["painting"].agg(lambda names: min(names))
    return sorted(groups.index, key=lambda digest: text_digest(f"{seed}\0{groups[digest]}\0{digest}"))


def lock_protocol(train, root, annotations, config):
    images = annotations[["painting", "image_name", "split"]].drop_duplicates().rename(columns={"image_name": "image"})
    images["image_sha256"] = images["image"].map(lambda name: file_sha256(data.image_path(root, name)))
    require_isolation(images[images.split.eq("train")], images[images.split.eq("val")])
    paintings = train[["painting", "image"]].drop_duplicates().merge(
        images[["painting", "image_sha256"]], on="painting", validate="one_to_one")
    ranked = ranked_groups(paintings, config["holdout_seed"])
    holdout = set(ranked[:math.ceil(len(ranked) * config["holdout_fraction"])])
    paintings["role"] = np.where(paintings.image_sha256.isin(holdout), "holdout", "development")
    remaining = paintings[paintings.role.eq("development")]
    assignments = {key: index % config["cv_folds"] + 1 for index, key in enumerate(ranked_groups(remaining, config["cv_seed"]))}
    paintings["fold"] = paintings.image_sha256.map(assignments).fillna(0).astype(int)
    return paintings.sort_values("painting").reset_index(drop=True)


def inner_partition(frame, config, fold):
    ranked = ranked_groups(frame, config["early_stop_seed"] + fold)
    stopping = set(ranked[:math.ceil(len(ranked) * config["early_stop_fraction"])])
    fit, held = frame[~frame.image_sha256.isin(stopping)], frame[frame.image_sha256.isin(stopping)]
    require_isolation(fit, held)
    if fit.empty or held.empty:
        raise ValueError("Empty internal training or stopping partition")
    return fit, held


def allowed_captions(captions, pairs):
    if not captions.split.eq("train").all():
        raise ValueError("Text targets require official training captions")
    keys = pairs[data.KEYS].drop_duplicates()
    return captions[captions.valid_caption & captions.binary].merge(keys, on=data.KEYS, validate="many_to_one")


def audit_captions(captions, pairs):
    per_language = []
    eligible = pd.MultiIndex.from_frame(pairs[data.KEYS])
    for language, rows in captions.groupby("language", sort=True):
        valid = rows[rows.valid_caption]
        mapped = rows.valid_caption & rows.binary
        available = pd.MultiIndex.from_frame(rows.loc[mapped, data.KEYS].drop_duplicates())
        language_pairs = pairs[pairs.language.eq(language)]
        counts = rows.emotion.value_counts().to_dict()
        per_language.append({"language": language, "annotations": len(rows), "valid_captions": len(valid),
                             "empty_or_invalid": len(rows) - len(valid), "unique_captions": valid.caption_id.nunique(),
                             "duplicate_text_rows": len(valid) - valid.caption_id.nunique(),
                             "positive": int(rows.emotion.isin(data.POSITIVE).sum()),
                             "negative": int(rows.emotion.isin(data.NEGATIVE).sum()),
                             "other": int((~rows.binary).sum()), "eligible_pairs": len(language_pairs),
                             "eligible_pairs_with_text": len(available.intersection(eligible)),
                             "mean_characters": float(valid.caption.str.len().mean()),
                             **{"emotion_" + name: count for name, count in counts.items()}})
    lengths = captions.loc[captions.valid_caption, "caption"].str.len()
    summary = {"training_annotations": len(captions), "valid_captions": int(captions.valid_caption.sum()),
               "empty_or_invalid": int((~captions.valid_caption).sum()),
               "unique_valid_captions": int(captions.loc[captions.valid_caption, "caption_id"].nunique()),
               "binary_annotations": int(captions.binary.sum()),
               "positive": int(captions.emotion.isin(data.POSITIVE).sum()),
               "negative": int(captions.emotion.isin(data.NEGATIVE).sum()), "other": int((~captions.binary).sum()),
               "eligible_pairs": len(pairs), "emotion_counts": captions.emotion.value_counts().to_dict(),
               "character_quantiles": {str(q): float(lengths.quantile(q)) for q in (0, .5, .9, .95, .99, 1)},
               "duplicate_annotation_rows": int(captions.duplicated([*data.KEYS, "emotion", "caption"]).sum())}
    return summary, pd.DataFrame(per_language)
