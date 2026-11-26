#!/usr/bin/env python3
"""Build the small private EXP-038 overlay that sits beside the existing EXP-037 Kaggle input dataset.

The overlay holds the worker source at one commit, cached CLIP-B / SigLIP2-Base / CLIP-L features for all
1,656 Train and Development images, the pinned P2 fold assignment, the R0 reference artifacts, the saved
EXP-036 CS-CVA and ensemble Development predictions, and EXP-037's C6 Development predictions. It holds no
Evaluation data and no images.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3 import data, submission
from st3.competition_challenger import load_bundle, official_features, official_pairs, predict_with_features
from st3.competition_track1 import load_p2, load_protocol
from st3.contrast_study import new_output
from st3.exp036_io import read, sha, write
from st3.exp037_io import safe_members, unpack_inputs, verify_inputs
from st3.exp038_io import (ARCHIVE_PREFIX, OVERLAY_MANIFEST, archive_like, check_config_pins, encode_members,
                           restore_overlay, unpack_overlay)

SOURCES = ["src/st3", "configs", "scripts", "requirements-exp035-kaggle.lock.txt",
           "requirements-exp036-kaggle.txt", "requirements-exp038-kaggle.txt"]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def require(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Checksum differs: {path}")
    return path.read_bytes()


def load_builder(project):
    path = project / "runs/codabench_dev_release/build_corrected.py"
    spec = importlib.util.spec_from_file_location("build_corrected", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def features_table(project, refs, base_manifest, train_names, dev):
    track1 = read(ROOT / "configs/competition_track1.json")
    upstream = project / "data/upstream"
    protocol = load_protocol(project / "runs/exp023_budget_delivery/release/restore_check", track1)
    train = official_pairs(upstream, track1, protocol)
    args = Namespace(data_dir=upstream, cache_dir=project / "cache", e50_bundle=project / "runs/e50_inference_models",
                     clip_large_bundle=project / "runs/e50l_inference_models")
    training, receipts = official_features(args, train, protocol, track1)
    builder = load_builder(project)
    heads, manifest = load_bundle(project / "runs/competition_track1/january_bundle")
    inputs = dev[data.INPUT_COLUMNS]
    fixed_run = read(project / "runs/codabench_dev_release/fixed_ridge/run.json")
    development, _ = builder.cached_dev_features(inputs, manifest, fixed_run)
    replay = predict_with_features(heads, manifest, inputs, development)
    saved = submission.read_predictions(project / "runs/codabench_dev_release/fixed_ridge/predictions.csv", inputs)
    if not np.array_equal(replay.p_positive.to_numpy(), saved.p_positive.to_numpy()):
        raise ValueError("Development base features do not reproduce Fixed Ridge")
    gold, p2_features, _ = load_p2(project / "runs/exp023_budget_delivery/release/restore_check", track1)
    for name in p2_features:
        if not np.array_equal(p2_features[name].to_numpy(), training[name].loc[p2_features[name].index].to_numpy()):
            raise ValueError(f"P2 and official training features differ for {name}")
    rows = pd.concat([train[["painting", "image"]].drop_duplicates(), dev[["painting", "image"]].drop_duplicates()])
    if set(rows.painting) != set(train_names) | set(dev.painting):
        raise ValueError("Feature painting coverage differs from the base population")
    rows["image_sha256"] = rows.image.map(lambda name: Path(base_manifest["images"][name]).stem)
    order = rows.sort_values("image_sha256", ignore_index=True)
    arrays = {"image_sha256": order.image_sha256.to_numpy(dtype=str)}
    for name in ("clip", "siglip2", "clip_large"):
        frame = pd.concat([training[name], development[name]])
        arrays[name] = frame.loc[order.painting].to_numpy(dtype=np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue(), {"training": receipts, "development_rows": len(dev.painting.unique()),
                               "rows": len(order)}


def package(args):
    project = args.project.resolve()
    config = read(ROOT / "configs/exp038.json")
    check_config_pins(config)  # Refuse to build with a truncated or malformed hash pin.
    refs = config["references"]
    source = subprocess.check_output(["git", "rev-parse", f"{args.source}^{{commit}}"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "archive", "--format=zip", source, *SOURCES], cwd=ROOT)
    with ZipFile(io.BytesIO(tree)) as runtime:
        if json.loads(runtime.read("configs/exp038.json")) != config:
            raise ValueError("Committed EXP-038 config differs from the working copy")
    base_zip = args.base_zip.resolve()
    if sha(base_zip) != refs["base_inputs_zip_sha256"]:
        raise ValueError("EXP-037 base input ZIP checksum differs")
    with tempfile.TemporaryDirectory() as scratch, new_output(args.output) as stage:
        base = unpack_inputs(base_zip, Path(scratch), refs["base_manifest_sha256"], refs["base_inputs_zip_sha256"])
        base_manifest = verify_inputs(base, refs["base_manifest_sha256"])
        annotations = pd.read_csv(base / "training/annotations.csv", dtype=str, keep_default_na=False)
        train = data.training_pairs(annotations[data.ANNOTATION_COLUMNS])
        dev = data.read_dev_input(base)
        features, feature_receipt = features_table(project, refs, base_manifest, sorted(train.painting.unique()), dev)
        exp037 = project / "runs/exp037/closure/exp037_results.zip"
        require(exp037, refs["exp037_results_zip_sha256"])
        with ZipFile(exp037) as z:
            c6 = z.read("C/selected/standalone/predictions.csv")
            exported = json.loads(z.read("export_manifest.json"))["files"]
        if digest(c6) != refs["exp037_c6_standalone_csv_sha256"] or \
                exported["C/selected/standalone/predictions.csv"] != digest(c6):
            raise ValueError("EXP-037 C6 Development predictions differ")
        track = project / "runs/competition_track1"
        neighborhood = project / "runs/competition_neighborhood_residual/results"
        release = project / "runs/codabench_dev_release"
        p2 = project / "runs/exp023_budget_delivery/release/restore_check/training"
        members = {
            "features/base_features.npz": features,
            "p2/protocol.csv": require(p2 / "protocol.csv", refs["p2_protocol_sha256"]),
            "p2/pairs.csv": require(p2 / "pairs.csv", refs["p2_pairs_sha256"]),
            "reference/track1_oof.csv": require(track / "results/oof.csv", refs["track1_oof_sha256"]),
            "reference/track1_metrics.json": require(track / "results/metrics.json", refs["track1_metrics_sha256"]),
            "reference/corrected_oof.csv": require(neighborhood / "oof.csv", refs["corrected_oof_sha256"]),
            "reference/corrected_metrics.json": require(neighborhood / "metrics.json", refs["corrected_metrics_sha256"]),
            "reference/fixed_bundle/heads.npz": require(track / "january_bundle/heads.npz", refs["fixed_heads_sha256"]),
            "reference/fixed_bundle/manifest.json": require(track / "january_bundle/manifest.json",
                                                            refs["fixed_manifest_sha256"]),
            "reference/corrected_bundle/bank.npz": require(release / "corrected_bundle/bank.npz",
                                                           refs["corrected_bank_sha256"]),
            "reference/corrected_bundle/manifest.json": (release / "corrected_bundle/manifest.json").read_bytes(),
            "reference/fixed_ridge_submission.zip": require(release / "fixed_ridge/submission.zip",
                                                            refs["fixed_dev_zip_sha256"]),
            "reference/corrected_ridge_submission.zip": require(release / "corrected_ridge/submission.zip",
                                                                refs["corrected_dev_zip_sha256"]),
            "reference/exp036_standalone_submission.zip": require(
                project / "runs/exp036/closure/exp036_standalone_submission.zip", refs["exp036_standalone_zip_sha256"]),
            "reference/exp036_ensemble_submission.zip": require(
                project / "runs/exp036/closure/exp036_ensemble_submission.zip", refs["exp036_ensemble_zip_sha256"]),
            "reference/exp037_best_submission.zip": require(
                project / "runs/exp037/closure/exp037_best_submission.zip", refs["exp037_original_blend_zip_sha256"]),
            "reference/exp037_c6_standalone/predictions.csv": c6,
        }
        if json.loads(members["reference/corrected_bundle/manifest.json"])["bank_sha256"] != refs["corrected_bank_sha256"]:
            raise ValueError("Corrected bank manifest differs")
        with ZipFile(io.BytesIO(tree)) as runtime:
            for name in runtime.namelist():
                if not name.endswith("/"):
                    members[f"runtime/{name}"] = runtime.read(name)
        # Kaggle expands nested archives in uploaded datasets; store each one as base64 text instead.
        stored, encoded = encode_members(members)
        files = {name: digest(content) for name, content in sorted(stored.items())}
        manifest = {"experiment": "EXP-038", "source": source, "encoded": encoded,
                    "packaging": "patch-2: archive-like members are stored as .b64 text",
                    "base": {"zip_sha256": refs["base_inputs_zip_sha256"], "manifest_sha256": refs["base_manifest_sha256"]},
                    "files": files, "references": refs,
                    "population": base_manifest["population"] | {"p2_pairs": 13600, "p2_paintings": 1335},
                    "features": feature_receipt["rows"], "models": {arm: [spec.get("model"), spec.get("revision")]
                                                                     for arm, spec in config["arms"].items()},
                    "evaluation_data": "none"}
        content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii")
        archive = stage / "exp038_inputs.zip"
        safe_members([*(ARCHIVE_PREFIX + name for name in stored), ARCHIVE_PREFIX + OVERLAY_MANIFEST])
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
            for name, payload in sorted(stored.items()):
                z.writestr(ARCHIVE_PREFIX + name, payload)
            z.writestr(ARCHIVE_PREFIX + OVERLAY_MANIFEST, content)
        with ZipFile(archive) as z:  # Recheck the whole written overlay, member by member.
            bad = [name for name in z.namelist() if archive_like(name, z.read(name))]
            if bad:
                raise ValueError(f"Overlay ZIP contains archive-like members: {bad[:5]}")
        archive_sha, manifest_sha = sha(archive), digest(content)
        with tempfile.TemporaryDirectory() as check:
            mounted = unpack_overlay(archive, Path(check) / "mounted", manifest_sha, archive_sha)
            restored = restore_overlay(mounted, manifest_sha, refs["base_manifest_sha256"], Path(check) / "restored")
            receipt = read(restored / "restore_receipt.json")["restored"]
        (stage / OVERLAY_MANIFEST).write_bytes(content)
        write(stage / "exp038_handoff.json", {"archive_sha256": archive_sha, "manifest_sha256": manifest_sha,
                                              "archive_bytes": archive.stat().st_size, "source": source,
                                              "files": len(files), "feature_rows": feature_receipt["rows"],
                                              "restored_archives": receipt,
                                              "base_zip_sha256": refs["base_inputs_zip_sha256"],
                                              "base_manifest_sha256": refs["base_manifest_sha256"]})
    print(json.dumps(read(args.output / "exp038_handoff.json"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--project", type=Path, required=True, help="Project root holding the ignored runs/ artifacts")
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args())
