#!/usr/bin/env python3
"""Fetch, bound frozen extraction, and run CPU P2 scoring for DINOv2-L."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.baseline import file_sha256
from st3.competition_challenger import check_hash
from st3.competition_dinov2_large import CONFIG, TRACK1, extract, run
from st3.competition_semantic_evaluation import write_atomic
from st3.competition_semantic_generation import training_inventory
from st3.e50_heads import read_json
from st3.e50_inference import safe_image_path
from st3.frozen_features import load_frozen_cache, verify_asset_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "extract", "cv"))
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/upstream")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/competition_neighborhood_residual/results")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/dinov2-large-47b73ee")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/competition_dinov2_large")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config, track = read_json(CONFIG), read_json(TRACK1)
    if args.stage == "fetch":
        from huggingface_hub import snapshot_download

        snapshot_download(config["model"], revision=config["revision"], token=False, local_dir=args.checkpoint,
                          allow_patterns=list(config["assets_sha256"]), max_workers=2)
        verify_asset_hashes({name: file_sha256(args.checkpoint / name) for name in config["assets_sha256"]}, config)
        print("Verified all public DINOv2-L assets")
        return

    if args.stage == "extract" and args.device == "cuda" and not args.worker:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                       timeout=config["max_gpu_seconds"], check=True)
        return

    scope = training_inventory(args.inputs, track)
    receipt_path = args.run_dir / "features.json"
    if args.stage == "extract":
        for row in scope.itertuples(index=False):
            check_hash(safe_image_path(args.data_dir, row.image), row.image_sha256)
        _, receipt = extract(args.data_dir, scope, args.run_dir / "cache", args.checkpoint, config, args.device)
        receipt["inventory_sha256"] = track["p2_input_sha256"]["training/protocol.csv"]
        write_atomic(receipt_path, receipt)
    else:
        receipt = read_json(receipt_path)
        if receipt["inventory_sha256"] != track["p2_input_sha256"]["training/protocol.csv"]:
            raise ValueError("P2 image inventory changed")
        identity = receipt["identity"]
        for key in ("model", "revision", "width", "feature", "batch_size", "assets_sha256"):
            if identity[key] != config[key]:
                raise ValueError(f"Cached DINOv2-L changed {key}")
        vectors, loaded = load_frozen_cache(Path(receipt["path"]), sorted(scope.image), identity)
        verify_asset_hashes(loaded["assets_sha256"], config)
        if receipt["features_sha256"] != loaded["features_sha256"]:
            raise ValueError("Feature receipt differs from the actual cache")
        matrix = pd.DataFrame([vectors[name] for name in scope.image], index=scope.painting)
        run(args.inputs, matrix, receipt, args.reference, args.run_dir / "results")


if __name__ == "__main__":
    main()
