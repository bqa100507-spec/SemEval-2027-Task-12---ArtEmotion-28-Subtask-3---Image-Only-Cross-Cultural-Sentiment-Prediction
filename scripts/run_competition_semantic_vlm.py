#!/usr/bin/env python3
"""Bounded preflight, explicitly authorized extraction, and semantic ST3 CV."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.competition_semantic_generation import CONFIG, TRACK1, extract, preflight, verified_assets
from st3.e50_heads import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "preflight", "extract", "cv"))
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/competition_semantic_vlm")
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/upstream")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/qwen3-vl-4b-ebb281e")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "runs/competition_neighborhood_residual/results")
    parser.add_argument("--authorize-full-extraction", action="store_true")
    args = parser.parse_args()
    config, track1 = read_json(CONFIG), read_json(TRACK1)
    if args.stage == "fetch":
        from huggingface_hub import snapshot_download

        names = list(config["git_blobs"]) + list(config["weight_sha256"])
        snapshot_download(config["model"], revision=config["revision"], allow_patterns=names,
                          local_dir=args.checkpoint, token=False, max_workers=2)
        verified_assets(args.checkpoint, config)
        print(f"Verified all Qwen checkpoint assets in {args.checkpoint}")
    elif args.stage == "preflight":
        result = preflight(args, config, track1)
        if result["status"] != "TECHNICAL_PASS":
            print(f"Preflight failed. See {args.run_dir / 'preflight.json'}", file=sys.stderr)
            raise SystemExit(2)
    elif args.stage == "extract":
        extract(args, config, track1)
    else:
        from st3.competition_semantic_vlm import run_cv

        run_cv(args, config, track1)


if __name__ == "__main__":
    main()
