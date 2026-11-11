"""General image-and-language prediction from the frozen EXP-007 heads."""

import importlib.metadata
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PureWindowsPath

import numpy as np

from . import data, submission
from .ablation import predict_lr
from .baseline import file_sha256
from .cli import ROOT, git_head, json_output
from .contrast_study import new_output
from .e50_heads import ENCODERS, MANIFEST, inference_config, load_bundle, read_json
from .fusion import blend
from .incremental import valid_vectors


def read_input(path, languages):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        inputs = data.read_exact_csv(handle, data.INPUT_COLUMNS)
    data.unique_pairs(inputs)
    unsupported = sorted(set(inputs["language"]) - set(languages))
    if unsupported:
        raise ValueError(f"Unsupported target language(s): {', '.join(unsupported)}")
    if inputs[["painting", "image"]].drop_duplicates()["painting"].duplicated().any():
        raise ValueError("Each painting must map to exactly one image path")
    return inputs


def safe_image_path(root, name):
    parts = name.split("/")
    if any(not part or part in (".", "..") or part.strip() != part or part.endswith(".")
           or PureWindowsPath(part).is_reserved() or re.search(r'[<>:"\\|?*\x00-\x1f]', part)
           for part in parts):
        raise ValueError(f"Unsafe image path: {name}")
    image_root = (root / "images").resolve(strict=True)
    try:
        path = data.image_path(root, name).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"Missing image: {name}") from error
    if not path.is_relative_to(image_root) or not path.is_file():
        raise ValueError(f"Image must be a regular file inside the images directory: {name}")
    return path


def load_legacy(name, wanted, index, legacy, cache_dir, width):
    matches = {digest: index[digest] for digest in wanted if digest in index}
    if not matches:
        return {}, {"matched": 0, "status": "no_matching_content"}
    path = (cache_dir / legacy["file"]).resolve()
    if not path.is_relative_to(cache_dir.resolve()):
        raise ValueError("Invalid legacy cache path")
    if not path.is_file():
        return {}, {"matched": len(matches), "status": "archive_unavailable", "archive": str(path)}
    if file_sha256(path) != legacy["sha256"]:
        raise ValueError(f"Historical {name} feature checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        names, features = archive["names"].tolist(), archive["features"]
    valid_vectors(features, len(names), width)
    if len(set(names)) != len(names) or not set(matches.values()).issubset(names):
        raise ValueError(f"Historical {name} cache has invalid image coverage")
    by_name = dict(zip(names, features))
    return {digest: by_name[image] for digest, image in matches.items()}, {
        "matched": len(matches), "status": "verified", "archive": str(path), "sha256": legacy["sha256"],
        "source": legacy["source"],
    }


def run_feature_worker(name, images, args, config, scratch):
    directory = scratch / name
    directory.mkdir()
    request = directory / "request.json"
    batch_size = getattr(args, f"{name}_batch_size") or config["batch_size"]
    json_output({"images": {digest: str(path) for digest, path in images.items()},
                 "device": args.device, "batch_size": batch_size, "cache_dir": str(args.cache_dir)}, request)
    python = args.siglip2_python if name == "siglip2" else Path(sys.executable)
    if not python.is_file():
        raise ValueError(f"{name} interpreter missing: {python}. Run scripts/setup_siglip2.py from .venv.")
    command = [str(python), "-u", str(ROOT / "scripts" / "run_st3.py"), "_e50_worker", name,
               str(request), str(directory)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{name} feature worker exited with status {error.returncode}. "
                           "Completed batches are cached. Retry the same command.") from error
    with np.load(directory / "vectors.npz", allow_pickle=False) as archive:
        vectors = archive["features"]
        if archive["digests"].tolist() != list(images):
            raise ValueError(f"{name} worker returned different image coverage")
    valid_vectors(vectors, len(images), config["width"])
    metadata = read_json(directory / "features.json")
    if set(metadata["images"]) != set(images):
        raise ValueError(f"{name} worker provenance does not cover the requested images")
    return dict(zip(images, vectors)), metadata


def run_prediction(args):
    start = time.perf_counter()
    if args.output_dir.exists():
        raise ValueError(f"Output already exists: {args.output_dir}. Choose a new output directory.")
    config = inference_config()
    for size in (args.clip_batch_size, args.siglip2_batch_size):
        if size is not None and size < 1:
            raise ValueError("Batch size must be positive")
    input_sha256 = file_sha256(args.input)
    inputs = read_input(args.input, config["official_languages"])
    paths = {name: safe_image_path(args.data_dir, name) for name in inputs["image"].unique()}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    images = {hashes[name]: path for name, path in paths.items()}
    active = ENCODERS if args.mode == "e50" else ("clip",)
    models, manifest, index = load_bundle(args.model_dir, config, active)
    languages = manifest["languages"]
    untrained = inputs.loc[~inputs["language"].isin(languages), "language"].value_counts().to_dict()
    if untrained:
        print(f"Using shared image terms for official languages absent from training: {untrained}", flush=True)

    predictions, features = {}, {}
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".e50-work-", dir=args.output_dir.parent) as temporary:
        for name in active:
            encoder = config["encoders"][name]
            cached, historical = load_legacy(name, images, index, manifest["legacy"][name], args.cache_dir,
                                              encoder["width"])
            if historical["status"] == "archive_unavailable":
                print(f"{name}: historical archive unavailable, resolving its matches through the incremental cache",
                      flush=True)
            missing = {digest: path for digest, path in images.items() if digest not in cached}
            incremental = None
            if missing:
                print(f"{name}: {len(cached)} historical hits, {len(missing)} to resolve", flush=True)
                new, incremental = run_feature_worker(name, missing, args, encoder, Path(temporary))
                cached.update(new)
            features[name] = {"historical": historical, "historical_hits": len(images) - len(missing),
                              "encoded": incremental["encoded"] if incremental else 0,
                              "incremental_hits": incremental["incremental_hits"] if incremental else 0,
                              "worker": incremental}
            predictions[name] = predict_lr(models[name], inputs, {key: cached[d] for key, d in hashes.items()}, languages)

    if (file_sha256(args.input) != input_sha256
            or any(file_sha256(path) != hashes[name] for name, path in paths.items())):
        raise ValueError("Input CSV or image bytes changed during prediction. Retry with stable input files.")
    pred = predictions["clip"]
    if args.mode == "e50":
        pred = blend(pred, predictions["siglip2"], config["blend_clip_weight"])
    submission.validate_predictions(pred, inputs)
    if not pred[data.KEYS].equals(inputs[data.KEYS]):
        raise ValueError("Prediction pair ordering differs from the input")
    with new_output(args.output_dir) as stage:
        csv_path = stage / submission.FILENAME
        zip_path = stage / "submission.zip"
        pred.to_csv(csv_path, index=False, lineterminator="\n")
        submission.pack(csv_path, zip_path, inputs)
        archived = submission.validate_archive(zip_path, inputs)
        if not archived.equals(submission.read_predictions(csv_path, inputs)):
            raise ValueError("Archived predictions differ from the output CSV")
        json_output({
            "mode": args.mode, "system": manifest["system"], "project_revision": git_head(ROOT),
            "input": str(args.input), "input_sha256": input_sha256, "data_dir": str(args.data_dir),
            "model_dir": str(args.model_dir), "model_manifest_sha256": file_sha256(args.model_dir / MANIFEST),
            "model_files_sha256": manifest["files_sha256"], "training": {
                key: manifest[key] for key in ("upstream_revision", "annotations_sha256", "training_pairs",
                                               "training_paintings", "lr", "languages")},
            "blend_clip_weight": config["blend_clip_weight"] if args.mode == "e50" else 1.0,
            "probability_decimals": 6, "requested_device": args.device,
            "encoders": {name: config["encoders"][name] for name in active},
            "pairs": len(inputs), "paintings": inputs["painting"].nunique(), "unique_image_paths": len(paths),
            "unique_image_contents": len(images), "image_sha256": hashes, "untrained_language_rows": untrained,
            "features": features, "packages": {name: importlib.metadata.version(name)
                                               for name in ("scikit-learn", "numpy", "scipy", "pandas")},
            "output_sha256": {path.name: file_sha256(path) for path in (csv_path, zip_path)},
            "wall_seconds": time.perf_counter() - start,
        }, stage / "run.json")
    print(f"Validated {len(inputs)} {args.mode} predictions in {args.output_dir / 'submission.zip'}", flush=True)


def validate_input_archive(args):
    inputs = read_input(args.input, inference_config()["official_languages"])
    submission.validate_archive(args.archive, inputs)
    print(f"Validated {args.archive} against all {len(inputs)} input pairs")
