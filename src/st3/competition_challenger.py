"""Fit and predict the separate ST3 ridge challenger on pinned frozen features."""

import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.special import expit

from . import data, submission
from .baseline import file_sha256
from .cli import ROOT, git_head, json_output
from .competition_ridge import Head, fit_head, make_panel, painting_splits
from .competition_stack import ENCODERS, aligned_values, blend, fit_stack
from .contrast_study import new_output
from .e50_heads import inference_config, load_legacy, read_json
from .e50_inference import read_input, run_feature_worker, safe_image_path
from .e50l_heads import clip_large_config
from .incremental import valid_vectors

CONFIG = ROOT / "configs" / "competition_track1.json"
BUNDLE_VERSION = 1


def frame_sha(frame):
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def check_hash(path, expected):
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Missing or incorrectly hashed file: {path}")


def dev_isolation(training, inputs):
    if set(training.painting) & set(inputs.painting) or set(training.image) & set(inputs.image):
        raise ValueError("Official training and Development input overlap")


def fit_heads(gold, features, config):
    panel = make_panel(gold)
    heads, receipts = {}, {}
    for encoder in ENCODERS:
        x = np.ascontiguousarray(features[encoder].loc[list(panel.paintings)].to_numpy(dtype=np.float64))
        heads[encoder], receipts[encoder] = fit_head(panel, x, config["ridge"])
    return heads, receipts


def component_predictions(heads, inputs, features):
    paintings, languages = sorted(inputs.painting.unique()), sorted(inputs.language.unique())
    row = inputs.painting.map({name: index for index, name in enumerate(paintings)}).to_numpy()
    col = inputs.language.map({name: index for index, name in enumerate(languages)}).to_numpy()
    probabilities = {}
    for encoder, head in heads.items():
        x = np.ascontiguousarray(features[encoder].loc[paintings].to_numpy(dtype=np.float64))
        probabilities[encoder] = expit(head.logits(x, languages)[row, col]).round(6)
    return inputs[data.KEYS].assign(**probabilities)


def crossfit(gold, features, config, seed):
    paintings = sorted(gold.painting.unique())
    assignments = dict(zip(paintings, painting_splits(paintings, seed, config["inner_folds"])))
    assignments = gold.painting.map(assignments)
    receipts, predictions = [], []
    for fold in range(config["inner_folds"]):
        fit, held = gold.loc[assignments.ne(fold)], gold.loc[assignments.eq(fold)]
        if fit.empty or held.empty or set(fit.painting) & set(held.painting):
            raise ValueError("Invalid inner painting split")
        heads, fits = fit_heads(fit, features, config)
        predictions.append(component_predictions(heads, held[data.KEYS], features))
        receipts.append({"fold": fold, "fit_paintings": fit.painting.nunique(),
                         "held_paintings": held.painting.nunique(), "fit_gold_sha256": frame_sha(fit),
                         "held_keys_sha256": frame_sha(held[data.KEYS]), "heads": fits})
    output = pd.concat(predictions, ignore_index=True).sort_values(data.KEYS, ignore_index=True)
    data.require_same_pairs(output, gold)
    meta_gold = gold.sort_values(data.KEYS, ignore_index=True)
    learned = fit_stack(meta_gold, output, config)
    return learned | {"seed": seed, "inner_oof_sha256": frame_sha(output), "splits": receipts}


def production_encoders():
    return inference_config()["encoders"] | {"clip_large": clip_large_config()}


def write_bundle(path, heads, *, mode, weights, provenance):
    fields = ("intercept", "language_intercepts", "shared", "deviations", "languages")
    with new_output(path) as stage:
        np.savez_compressed(stage / "heads.npz", **{f"{name}:{field}": np.asarray(getattr(head, field))
                                                    for name, head in heads.items() for field in fields})
        json_output({"version": BUNDLE_VERSION, "mode": mode, "weights": list(weights),
                     "encoder_order": ENCODERS, "encoders": production_encoders(),
                     "official_languages": inference_config()["official_languages"],
                     "prediction_decimals": 6, "provenance": provenance,
                     "heads_sha256": file_sha256(stage / "heads.npz")}, stage / "manifest.json")
        load_bundle(stage)


def load_bundle(path):
    manifest = read_json(path / "manifest.json")
    weights = np.asarray(manifest["weights"], dtype=float)
    if (manifest["version"] != BUNDLE_VERSION or manifest["encoders"] != production_encoders()
            or tuple(manifest["encoder_order"]) != ENCODERS or manifest["prediction_decimals"] != 6
            or manifest["official_languages"] != inference_config()["official_languages"]
            or weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any()
            or abs(weights.sum() - 1) > 1e-10 or manifest["mode"] not in {"fixed", "learned"}
            or (manifest["mode"] == "fixed" and weights.tolist() != [.25, .25, .5])):
        raise ValueError("Ridge bundle has an incompatible encoder or fusion configuration")
    check_hash(path / "heads.npz", manifest["heads_sha256"])
    heads = {}
    with np.load(path / "heads.npz", allow_pickle=False) as state:
        for encoder in ENCODERS:
            def field(name):
                return state[f"{encoder}:{name}"]

            width = manifest["encoders"][encoder]["width"]
            languages = tuple(field("languages").tolist())
            intercept, language, shared, deviations = (field(k) for k in
                                                       ("intercept", "language_intercepts", "shared", "deviations"))
            if (not languages or languages != tuple(sorted(set(languages)))
                    or not set(languages) <= set(manifest["official_languages"])
                    or intercept.shape != () or language.shape != (len(languages),) or shared.shape != (width,)
                    or deviations.shape != (len(languages), width)
                    or not all(np.isfinite(a).all() for a in (intercept, language, shared, deviations))):
                raise ValueError(f"Invalid ridge head for {encoder}")
            heads[encoder] = Head(float(intercept), language, shared, deviations, languages)
    if len({head.languages for head in heads.values()}) != 1:
        raise ValueError("Ridge head vocabularies differ")
    return heads, manifest


def predict_with_features(heads, manifest, inputs, features):
    components = component_predictions(heads, inputs, features)
    q = blend(aligned_values(inputs, components, ENCODERS), np.asarray(manifest["weights"]),
              fixed=manifest["mode"] == "fixed")
    return inputs[data.KEYS].assign(p_positive=q)


def official_pairs(root, config, protocol):
    spec = config["official_training"]
    path = root / data.ANNOTATIONS
    if git_head(root) != spec["revision"]:
        raise ValueError("Full refit requires the pinned organizer revision")
    check_hash(path, spec["annotations_sha256"])
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"].strip() == "train":
                rows.append({name: row[name].strip() for name in data.ANNOTATION_COLUMNS})
    train = pd.DataFrame(rows)
    train["language"] = train.language.replace({"arabic": "Arabic", "chinese": "Chinese", "english": "English"})
    train["emotion"] = train.emotion.str.lower()
    if set(train.emotion) - data.POSITIVE - data.NEGATIVE - data.AMBIGUOUS:
        raise ValueError("Unrecognized training emotion")
    pairs = data.training_pairs(train).sort_values(data.KEYS, ignore_index=True)
    dev_isolation(pairs, data.read_dev_input(root))
    keys = pairs[["painting", "image"]].drop_duplicates().sort_values("painting").reset_index(drop=True)
    if (len(pairs) != spec["pairs"] or len(keys) != spec["paintings"]
            or not keys.equals(protocol[["painting", "image"]].sort_values("painting").reset_index(drop=True))):
        raise ValueError("Full official training population differs from the pinned painting inventory")
    return pairs


def official_features(args, train, protocol, config):
    expected = config["production_cache_manifests"]
    e50, large = args.e50_bundle, args.clip_large_bundle
    for directory, role in ((e50, "e50"), (large, "clip_large")):
        check_hash(directory / "manifest.json", expected[role])
    e50_manifest, large_manifest = (read_json(p / "manifest.json") for p in (e50, large))
    for directory, manifest in ((e50, e50_manifest), (large, large_manifest)):
        check_hash(directory / "legacy-index.json", manifest["files_sha256"]["legacy-index.json"])
    indexes = {name: read_json((large if name == "clip_large" else e50) / "legacy-index.json") for name in ENCODERS}
    indexes["clip_large"] = indexes["clip_large"]["train"]
    manifest = e50_manifest["legacy"] | {"clip_large": large_manifest["legacy"]["train"]}
    paintings = train[["painting", "image"]].drop_duplicates().set_index("painting").sort_index()
    hashes = paintings.image.map(lambda name: file_sha256(safe_image_path(args.data_dir, name)))
    if hashes.duplicated().any() or not hashes.eq(protocol.set_index("painting").image_sha256).all():
        raise ValueError("Official training image bytes differ from the pinned original protocol")
    if set(hashes) & {file_sha256(safe_image_path(args.data_dir, name))
                      for name in data.read_dev_input(args.data_dir).image.unique()}:
        raise ValueError("Development inputs duplicate original training image contents")
    features, sources = {}, {}
    for name, width in config["feature_widths"].items():
        cached, sources[name] = load_legacy(name, hashes, indexes[name], manifest[name], args.cache_dir, width)
        if set(cached) != set(hashes):
            raise ValueError(f"Required official training features are missing for {name}")
        features[name] = pd.DataFrame(np.stack([cached[h] for h in hashes]), index=paintings.index)
    return features, sources


def metadata_identity(config):
    names = [CONFIG, ROOT / "scripts" / "run_competition_track1.py"]
    names += [Path(__file__).with_name(f"{name}.py") for name in
              ("competition_ridge", "competition_stack", "competition_challenger", "competition_track1",
               "data", "metrics", "submission", "e50_inference", "e50_encoders", "incremental")]
    names += [ROOT / "configs" / name for name in ("exp010.json", "e50_inference.json")]
    import scipy

    return {"source_sha256": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in names},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "pandas": pd.__version__}, "config": config}


def predict_images(args):
    start = time.perf_counter()
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    heads, manifest = load_bundle(args.bundle)
    input_hash = file_sha256(args.input)
    inputs = read_input(args.input, manifest["official_languages"])
    unique = inputs[["painting", "image"]].drop_duplicates().set_index("painting")
    paths = {name: safe_image_path(args.data_dir, name) for name in unique.image.unique()}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    images = {hashes[name]: path for name, path in paths.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features, receipts = {}, {}
    with TemporaryDirectory(prefix=".ridge-work-", dir=args.output.parent) as scratch:
        for name in ENCODERS:
            profile = manifest["encoders"][name]
            vectors, receipts[name] = run_feature_worker(name, images, args, profile, Path(scratch))
            x = np.stack([vectors[hashes[image]] for image in unique.image])
            valid_vectors(x, len(unique), profile["width"])
            features[name] = pd.DataFrame(x, index=unique.index)
    if file_sha256(args.input) != input_hash or any(file_sha256(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Inference inputs changed while predictions were being generated")
    before = time.perf_counter()
    outputs = predict_with_features(heads, manifest, inputs, features)
    head_seconds = time.perf_counter() - before
    submission.validate_predictions(outputs, inputs)
    if not outputs[data.KEYS].equals(inputs[data.KEYS]):
        raise ValueError("Challenger changed submission row order")
    with new_output(args.output) as stage:
        csv_path, zip_path = stage / submission.FILENAME, stage / "submission.zip"
        outputs.to_csv(csv_path, index=False, float_format="%.6f", lineterminator="\n")
        submission.pack(csv_path, zip_path, inputs)
        if not submission.validate_archive(zip_path, inputs).equals(submission.read_predictions(csv_path, inputs)):
            raise ValueError("Candidate ZIP differs from the checked prediction CSV")
        json_output({"bundle_sha256": file_sha256(args.bundle / "manifest.json"), "input_sha256": input_hash,
                     "image_sha256": hashes, "rows": len(inputs), "features": receipts,
                     "head_seconds": head_seconds, "wall_seconds": time.perf_counter() - start,
                     "output_sha256": {path.name: file_sha256(path) for path in (csv_path, zip_path)}}, stage / "run.json")
    print(f"Validated {len(inputs)} {manifest['mode']} challenger predictions in {args.output / 'submission.zip'}", flush=True)
