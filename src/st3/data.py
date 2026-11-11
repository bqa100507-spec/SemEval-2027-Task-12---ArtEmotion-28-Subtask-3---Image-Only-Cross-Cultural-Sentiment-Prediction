"""Public ST3 data and train/validation integrity checks."""

import csv
from pathlib import PurePosixPath

import pandas as pd
from PIL import Image

KEYS = ["painting", "language"]
INPUT_COLUMNS = [*KEYS, "image"]
ANNOTATION_COLUMNS = [*KEYS, "image_name", "split", "emotion"]
ANNOTATIONS = "annotations/artelingo28_train_val.csv"
DEV_INPUT = "phase_inputs/dev/st3_test.csv"
VERIFIER = "scripts/verify_dataset.py"
POSITIVE = {"amusement", "awe", "contentment", "excitement"}
NEGATIVE = {"anger", "disgust", "fear", "sadness"}
AMBIGUOUS = {"other", "something else"}
MIN_ANNOTATORS = 5


def read_exact_csv(handle, columns):
    reader = csv.reader(handle, strict=True)
    if next(reader, None) != columns:
        raise ValueError(f"Expected CSV columns {columns}, in that order")
    rows = []
    for row in reader:
        if len(row) != len(columns) or any(not cell.strip() for cell in row):
            raise ValueError(f"Invalid CSV record ending on line {reader.line_num}")
        rows.append(row)
    if not rows:
        raise ValueError("CSV contains no records")
    return pd.DataFrame(rows, columns=columns)


def unique_pairs(frame):
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate painting-language pairs")
    return set(frame[KEYS].itertuples(index=False, name=None))


def require_same_pairs(expected, actual):
    expected, actual = unique_pairs(expected), unique_pairs(actual)
    if expected != actual:
        raise ValueError(f"Pair mismatch: {len(expected - actual)} missing, {len(actual - expected)} extra")


def check_split_integrity(frame):
    if set(frame["split"]) != {"train", "val"}:
        raise ValueError("Expected exactly train and val splits")
    if frame.groupby("painting")["split"].nunique().gt(1).any():
        raise ValueError("Painting appears in both train and val")
    mapping = frame[["painting", "image_name"]].drop_duplicates()
    if mapping["painting"].duplicated().any() or mapping["image_name"].duplicated().any():
        raise ValueError("Painting identifiers and image paths must map one-to-one")


def read_annotations(root):
    frame = pd.read_csv(root / ANNOTATIONS, usecols=ANNOTATION_COLUMNS, dtype=str, keep_default_na=False)
    for name in ANNOTATION_COLUMNS:
        frame[name] = frame[name].str.strip()
        if frame[name].eq("").any():
            raise ValueError(f"Blank annotation field: {name}")
    frame["language"] = frame["language"].replace({"arabic": "Arabic", "chinese": "Chinese", "english": "English"})
    frame["emotion"] = frame["emotion"].str.lower()
    unknown = set(frame["emotion"]) - POSITIVE - NEGATIVE - AMBIGUOUS
    if unknown:
        raise ValueError(f"Unknown emotions: {sorted(unknown)}")
    check_split_integrity(frame)
    return frame


def read_dev_input(root):
    with (root / DEV_INPUT).open(encoding="utf-8", newline="") as handle:
        dev = read_exact_csv(handle, INPUT_COLUMNS)
    unique_pairs(dev)
    dev["image"] = dev["image"].str.strip()
    return dev


def check_dev_isolation(annotations, dev):
    train = annotations[annotations["split"] == "train"]
    if set(dev["painting"]) & set(train["painting"]) or set(dev["image"]) & set(train["image_name"]):
        raise ValueError("Development input overlaps training paintings or images")
    val = annotations[annotations["split"] == "val"].set_index("painting")["image_name"]
    val = val[~val.index.duplicated()]
    if not dev["painting"].map(val).eq(dev["image"]).all():
        raise ValueError("Development image paths do not match validation painting IDs")


def aggregate(frame):
    """One row per eligible pair. Each released binary-mappable row is one vote."""
    votes = frame.loc[frame["emotion"].isin(POSITIVE | NEGATIVE), [*KEYS, "emotion"]].copy()
    votes["positive"] = votes["emotion"].isin(POSITIVE).astype(int)
    counts = votes.groupby(KEYS)["positive"].agg(k="sum", n="size").reset_index()
    return counts.loc[counts["n"] >= MIN_ANNOTATORS].assign(p=lambda pairs: pairs["k"] / pairs["n"])


def validation_gold(annotations, dev):
    gold = aggregate(annotations[annotations["split"] == "val"])
    require_same_pairs(dev, gold)
    return gold


def image_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Invalid image path: {name}")
    return root / "images" / relative


def audit(root):
    annotations = read_annotations(root)
    dev = read_dev_input(root)
    check_dev_isolation(annotations, dev)
    validation_gold(annotations, dev)
    names = set(annotations["image_name"])
    on_disk = {path.relative_to(root / "images").as_posix() for path in (root / "images").rglob("*")
               if path.is_file() and not path.name.startswith(".")}
    if missing := names - on_disk:
        raise ValueError(f"{len(missing)} referenced images missing, including {min(missing)}")
    for name in sorted(names):
        with Image.open(image_path(root, name)) as image:
            image.load()

    report = {"images_verified": len(names), "unreferenced_images": len(on_disk - names),
              "dev_pairs": len(dev), "dev_paintings": dev["painting"].nunique(),
              "dev_languages": dev["language"].nunique(), "splits": {}, "languages": {}}
    for split, frame in [*annotations.groupby("split"), ("total", annotations)]:
        pairs = aggregate(frame)
        positive = int(frame["emotion"].isin(POSITIVE).sum())
        negative = int(frame["emotion"].isin(NEGATIVE).sum())
        report["splits"][split] = {
            "annotations": len(frame), "paintings": frame["painting"].nunique(),
            "languages": frame["language"].nunique(), "positive": positive, "negative": negative,
            "ambiguous": len(frame) - positive - negative, "invalid_labels": 0,
            "positive_vote_rate": positive / (positive + negative), "eligible_pairs": len(pairs),
            "eligible_languages": pairs["language"].nunique(),
            "eligible_positive_vote_rate": float(pairs["k"].sum() / pairs["n"].sum()),
            "eligible_pair_share_quantiles": {str(q): float(pairs["p"].quantile(q)) for q in (0, .25, .5, .75, 1)},
        }
        report["languages"][split] = {
            language: {"annotations": int(count), "eligible_pairs": int((pairs["language"] == language).sum())}
            for language, count in sorted(frame["language"].value_counts().items())
        }
    return report
