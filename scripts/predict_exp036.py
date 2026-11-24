#!/usr/bin/env python3
"""Predict a fresh ST3 input with the frozen 75/25 EXP-036 competition ensemble."""

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3 import data, submission
from st3.competition_challenger import load_bundle, predict_with_features
from st3.competition_neighborhood import CONFIG, ResidualBank, predict_correction
from st3.contrast_study import new_output
from st3.e50_inference import read_input, run_feature_worker, safe_image_path
from st3.exp036_io import check, read, sha, write

SOURCE = "abc23cf9b9a3a138c51e85185b8e2003058436f4"
ALPHA = 0.25
PINNED = {
    "cscva/bundle.json": "fa84d1f1e4d64ea11eb6947438acf6d701bd71f351e9a4323b5f1f5c9883e043",
    "cscva/weights.pt": "302ae761c3d045fa4094362ce0f9495aed0874b117a1abf1438bdf877058f5d0",
    "cscva/preprocessor_config.json": "fb2817d3523ca3b666c859f15320c7138416bc38ffc515e2963f78c868c51c90",
    "ridge/fixed/manifest.json": "1d211fc70c27d5ababaf6b7f05afa75c498854c4201b6039e333891d5016185a",
    "ridge/fixed/heads.npz": "9105a0623867916c046dfd3ec4e1fa9fb6854c2d69068570c3b169bf47c89e76",
    "ridge/corrected/manifest.json": "004a2c1627625aeb7fe0b7f2411d6156ee9bc0dc39f5edb2aaf5d0b9ca1b345f",
    "ridge/corrected/bank.npz": "e2cd9dd4ca3acc67320e8165279b1c2d8b36010c7c64dd12759e786163c4f08e",
    "requirements-exp035-kaggle.lock.txt": "b562cb36ac14b55c422b43d63d26248885e40a25ce8ce7f1be33cbdbbd0d1812",
    "requirements-exp036-kaggle.txt": "6ed052cfad495f6f08397832cd41052d3a20b65028f2d08e2e89245e41f8e5af",
}


def verify_bundle(root):
    manifest = read(root / "manifest.json")
    if (manifest["version"], manifest["training_source"], manifest["cs_cva_weight"]) != (1, SOURCE, ALPHA):
        raise ValueError("Unexpected EXP-036 deployment recipe")
    for path, digest in PINNED.items():
        if manifest["files"].get(path) != digest:
            raise ValueError(f"Deployment manifest differs from the verified asset: {path}")
    check(root, manifest["files"])
    model = read(root / "cscva/bundle.json")
    if model["epoch"] != 5 or model["source"] != SOURCE or model["config"] != read(ROOT / "configs/exp036.json"):
        raise ValueError("The CS-CVA bundle does not match the selected Development checkpoint")
    if sorted(model["languages"].values()) != list(range(len(model["languages"]))):
        raise ValueError("Invalid CS-CVA language mapping")
    return manifest


def load_bank(folder, policy, heads_manifest):
    meta = read(folder / "manifest.json")
    if meta["recipe"] != policy or meta["ridge_manifest_sha256"] != sha(heads_manifest):
        raise ValueError("Residual bank no longer matches the sealed Ridge heads and recipe")
    check(folder, {"bank.npz": meta["bank_sha256"]})
    with np.load(folder / "bank.npz", allow_pickle=False) as state:
        names, languages, hashes = (state[n].tolist() for n in ("paintings", "languages", "image_sha256"))
        vectors, numerator, votes = (state[n] for n in ("visual", "numerator", "votes"))
    if (names != sorted(set(names)) or languages != sorted(set(languages)) or len(set(hashes)) != len(names)
            or len(names) != meta["bank_paintings"] or vectors.shape != (len(names), 1536)
            or numerator.shape != (len(names), len(languages)) or votes.shape != numerator.shape
            or not all(np.isfinite(v).all() for v in (vectors, numerator, votes)) or (votes < 0).any()):
        raise ValueError("The sealed corrected bank has inconsistent images, dimensions or votes")
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=3e-6, rtol=0)
    return ResidualBank(pd.DataFrame(vectors, index=names),
                        pd.DataFrame(numerator, index=names, columns=languages),
                        pd.DataFrame(votes, index=names, columns=languages)), set(hashes)


def ridge_with_features(heads, meta, bank, policy, inputs, features):
    keys = inputs[data.KEYS]
    encoders = policy["similarity_features"]
    visual = pd.concat([features[name] for name in encoders], axis=1, ignore_index=True) / np.sqrt(len(encoders))
    with threadpool_limits(limits=2):
        base = predict_with_features(heads, meta, inputs, features)
        result = predict_correction(bank, keys, visual, base, policy)
    return keys.assign(p_positive=result.corrected)


def aligned(frame, inputs):
    values = submission.validate_predictions(frame, inputs)
    return inputs[data.KEYS].merge(frame.assign(p_positive=values), on=data.KEYS, validate="one_to_one")


def write_submission(stage, inputs, preds, name, archive_name):
    csv = stage / name / "predictions.csv"
    csv.parent.mkdir()
    preds.to_csv(csv, index=False, lineterminator="\n", float_format="%.6f")
    archive = stage / archive_name
    submission.pack(csv, archive, inputs)
    pd.testing.assert_frame_equal(submission.validate_archive(archive, inputs), preds, check_exact=True)
    return {archive.name: sha(archive), csv.relative_to(stage).as_posix(): sha(csv)}


def write_submissions(stage, inputs, standalone, corrected):
    single, ridge = (aligned(frame, inputs) for frame in (standalone, corrected))
    single["p_positive"] = single.p_positive.round(6)
    mix = single.assign(p_positive=((1 - ALPHA) * ridge.p_positive.round(6) + ALPHA * single.p_positive).round(6))
    return (write_submission(stage, inputs, single, "standalone", "exp036_standalone_submission.zip")
            | write_submission(stage, inputs, mix, "ensemble", "exp036_ensemble_submission.zip"))


def run(args):
    if args.output.exists():
        raise ValueError("Choose a fresh output directory")
    if not args.ridge_only and (args.cscva_python is None or not args.cscva_python.is_file()):
        raise ValueError("Supply the pinned CS-CVA CUDA Python interpreter with --cscva-python")
    verify_bundle(args.bundle)
    policy = read(CONFIG)
    ridge = args.bundle / "ridge/fixed"
    heads, meta = load_bundle(ridge)
    bank, seen = load_bank(args.bundle / "ridge/corrected", policy, ridge / "manifest.json")
    input_digest = sha(args.input)
    inputs = read_input(args.input, meta["official_languages"])
    names = inputs[["painting", "image"]].drop_duplicates().set_index("painting")
    paths = {img: safe_image_path(args.data_dir, img) for img in names.image.unique()}
    hashes = {img: sha(path) for img, path in paths.items()}
    if seen.intersection(hashes.values()) or set(names.index) & set(bank.visual.index):
        raise ValueError("Query painting overlaps the fitted residual bank")
    if len(set(hashes.values())) != len(names):
        raise ValueError("Each query painting must have distinct image bytes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".exp036-features-", dir=args.output.parent) as name:
        scratch = Path(name)
        features, receipts = {}, {}
        for encoder, spec in meta["encoders"].items():
            vectors, receipt = run_feature_worker(encoder, {hashes[i]: paths[i] for i in paths}, args, spec, scratch)
            identity = receipt["identity"]
            if any(identity[key] != spec[key] for key in ("model", "revision", "assets_sha256", "width", "feature")):
                raise ValueError(f"{encoder} extraction fingerprint differs")
            features[encoder] = pd.DataFrame(np.stack([vectors[hashes[i]] for i in names.image]), index=names.index)
            receipts[encoder] = receipt
        corrected = ridge_with_features(heads, meta, bank, policy, inputs, features)
        if not args.ridge_only:
            cscva_out = scratch / "predictions.csv"
            command = [str(args.cscva_python), str(ROOT / "scripts/run_exp036.py"), "infer", "--bundle",
                       str(args.bundle / "cscva"), "--pairs", str(args.input), "--images", str(args.data_dir),
                       "--destination", str(cscva_out), "--microbatch", "4"]
            if args.model is not None:
                command.extend(("--model", str(args.model)))
            subprocess.run(command, check=True)
            single = submission.read_predictions(cscva_out, inputs)
        if input_digest != sha(args.input) or any(sha(paths[i]) != hashes[i] for i in paths):
            raise ValueError("Evaluation inputs changed during inference")
        with new_output(args.output) as stage:
            if args.ridge_only:
                results = write_submission(stage, inputs, corrected, "corrected_ridge",
                                           "corrected_ridge_submission.zip")
            else:
                results = write_submissions(stage, inputs, single, corrected)
            write(stage / "run.json", {"bundle_sha256": sha(args.bundle / "manifest.json"),
                  "runner_sha256": sha(Path(__file__)), "input_sha256": input_digest, "image_sha256": hashes,
                  "rows": len(inputs), "cs_cva_weight": 0. if args.ridge_only else ALPHA,
                  "features": receipts, "output_sha256": results})
    print(f"Verified submissions for all {len(inputs)} pairs in {args.output}.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("input", "data-dir", "bundle", "output", "cache-dir", "feature-cache-dir",
                  "siglip2-python"):
        p.add_argument(f"--{field}", type=Path, required=True)
    p.add_argument("--cscva-python", type=Path)
    p.add_argument("--ridge-only", action="store_true")
    p.add_argument("--model", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                   help="Ridge feature device; CS-CVA uses the pinned CUDA runtime")
    for encoder in ("clip", "siglip2", "clip-large"):
        p.add_argument(f"--{encoder}-batch-size", type=int)
    run(p.parse_args())
