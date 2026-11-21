#!/usr/bin/env python3
"""CPU-only entry points for the separate ST3 competition challenger."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.competition_challenger import CONFIG, predict_images
from st3.competition_track1 import load_p2, refit, run_cv, verify_archive
from st3.e50_heads import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inputs = ROOT / "runs" / "exp023_budget_delivery" / "release" / "restore_check"
    archive = ROOT / "runs" / "exp032" / "p2_verified"
    run_dir = ROOT / "runs" / "competition_track1"
    for name in ("verify-history", "cv", "refit"):
        command = commands.add_parser(name)
        command.add_argument("--inputs", type=Path, default=inputs)
        if name in {"verify-history", "cv"}:
            command.add_argument("--archive", type=Path, default=archive)
        if name in {"cv", "refit"}:
            command.add_argument("--run-dir", type=Path, default=run_dir)
        if name == "refit":
            command.add_argument("--data-dir", type=Path, default=ROOT / "data" / "upstream")
            command.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
            command.add_argument("--e50-bundle", type=Path, default=ROOT / "runs" / "e50_inference_models")
            command.add_argument("--clip-large-bundle", type=Path, default=ROOT / "runs" / "e50l_inference_models")
            command.add_argument("--output", type=Path, default=run_dir / "january_bundle")

    prediction = commands.add_parser("predict")
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--data-dir", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--bundle", type=Path, default=run_dir / "january_bundle")
    prediction.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    prediction.add_argument("--feature-cache-dir", type=Path, default=run_dir / "prediction_features")
    prediction.add_argument("--siglip2-python", type=Path, default=ROOT / ".venv-siglip2" /
                            ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
    for name in ("clip", "siglip2", "clip-large"):
        prediction.add_argument(f"--{name}-batch-size", type=int)
    prediction.set_defaults(device="cpu")

    args = parser.parse_args()
    config = read_json(CONFIG)
    if args.command == "verify-history":
        gold, features, reference = load_p2(args.inputs, config)
        print(verify_archive(args.archive, gold, features, reference, config))
    elif args.command == "cv":
        run_cv(args, config)
    elif args.command == "refit":
        refit(args, config)
    else:
        predict_images(args)


if __name__ == "__main__":
    main()
