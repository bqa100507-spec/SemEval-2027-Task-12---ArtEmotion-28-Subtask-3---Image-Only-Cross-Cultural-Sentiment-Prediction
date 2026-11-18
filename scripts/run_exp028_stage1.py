"""CPU-only audit and fixed EXP-028 exploratory fold-3 screen."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.exp028_cache import audit_cache, expected_identity, partition, read_protocol, verify_pixels, visual_blocks
from st3.vlm_semantics import atomic_json, file_hash, fingerprint

BASE = "eeac6f89365a0a7b9c8e9a9588cb4998732ea992"
PROTOCOL_FILES = [
    "docs/exp028_design_spec.json", "reports/exp028_design_proposal.md",
    "docs/exp028_feature_inventory.json", "configs/exp026.json",
    "src/st3/vlm_supervised.py", "src/st3/vlm_supervised_training.py",
    "src/st3/baseline.py", "src/st3/caption_data.py", "src/st3/fusion.py", "src/st3/metrics.py",
]


def check_pins():
    for name in PROTOCOL_FILES:
        original = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=ROOT)
        if name == "src/st3/baseline.py":
            original = original.replace(b"with threadpool_limits(limits=4):",
                b'with threadpool_limits(limits=int(os.environ.get("ST3_EXP028_CPU_THREADS", "4"))):')
        if original != (ROOT / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"Frozen Stage 1 source or protocol changed: {name}")


def manifest(rows, spec):
    inner, stopping, fitting, held = partition(rows, spec)
    populations = {}
    for name, frame in zip(("inner", "stop", "refit", "held"), (inner, stopping, fitting, held)):
        populations[name] = {
            "paintings": sorted(frame.painting.unique()), "pairs": len(frame),
            "key_sha256": fingerprint(frame[["painting", "language", "image_sha256"]].values.tolist()),
        }
    return {"fold": 3, "populations": populations, "spec_sha256": file_hash(ROOT / "docs/exp028_design_spec.json"),
            "spec_logical_sha256": fingerprint(spec), "historical_reference_sha256":
            json.loads((ROOT / "configs/exp026.json").read_text())["input_sha256"]["history/exp022_R0.csv"]}


def execute(args):
    import numpy as np
    import pandas as pd
    import sklearn
    import scipy

    started = time.perf_counter()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    run = args.run_dir
    try:
        check_pins()
        rows = read_protocol(ROOT, args.inputs)
        spec = json.loads((ROOT / "docs/exp028_design_spec.json").read_text())
        frozen = manifest(rows, spec)
        committed = ROOT / "configs/exp028_fold3_manifest.json"
        if json.loads(committed.read_text()) != {k: v for k, v in frozen.items() if k != "spec_sha256"}:
            raise ValueError("Frozen fold-3 identities differ")
        atomic_json(run / "manifest.json", frozen)
        cache, qwen, missing = audit_cache(rows, args.qwen_cache, expected_identity(ROOT),
                                          keep_vectors=args.stage == "screen")
        if args.workspace is not None:
            cache["pixel_files_verified"] = int(verify_pixels(args.inputs, rows))
        atomic_json(run / "cache_audit.json", cache)
        atomic_json(run / "missing_pairs.json", [list(key) for key in missing])
        if args.workspace is not None:
            vision = visual_blocks(ROOT, args.workspace, rows)
        if cache["invalid_entries"]:
            raise ValueError("Qwen cache verification failed; inspect cache_audit.json")
        if getattr(args, "require_full_vision_cache", False) and cache["missing_vision_images"]:
            raise ValueError("Qwen vision coverage differs from the full Colab audit requirement")
        if not cache["complete"]:
            status = "CPU_IMPLEMENTED / BLOCKED_MISSING_QWEN_FEATURES"
        elif args.stage == "audit":
            status = "COMPLETE_CACHE_VERIFIED"
        else:
            from st3.exp028_stage1 import run_screen
            if args.workspace is None:
                raise ValueError("Screen requires --workspace with the three pinned visual caches")
            labeled = read_protocol(ROOT, args.inputs, labels=True)
            reference = pd.read_csv(args.inputs / "history/exp022_R0.csv", float_precision="round_trip")
            overlap = labeled[["painting", "language", "fold"]].merge(
                reference[["painting", "language", "fold"]], on=["painting", "language"],
                suffixes=("_actual", "_reference"), validate="one_to_one")
            if len(overlap) != len(labeled) or len(reference) != len(labeled) or not overlap.fold_actual.equals(overlap.fold_reference):
                raise ValueError("Historical reference keys/folds differ")
            preds, evaluation, paintings, fits = run_screen(labeled, vision, qwen, reference, spec)
            for name, frame in preds.items():
                frame.to_csv(run / (name + "_predictions.csv"), index=False, float_format="%.6f", lineterminator="\n")
                paintings[name].to_csv(run / (name + "_paintings.csv"), lineterminator="\n")
            atomic_json(run / "evaluation.json", evaluation)
            atomic_json(run / "fits.json", fits)
            status = "FOLD3_SCREEN_COMPLETE / EXPLORATORY"
        report = {"status": status, "fold": 3, "cache_audit_sha256": file_hash(run / "cache_audit.json"),
                  "elapsed_seconds": time.perf_counter() - started, "device": "CPU", "production": "E50+L"}
        code = 0 if args.stage == "audit" or cache["complete"] else 2
    except Exception as error:
        report = {"status": "BLOCKED_VERIFICATION", "error": f"{type(error).__name__}: {error}",
                  "elapsed_seconds": time.perf_counter() - started, "device": "CPU"}
        code = 2
    report["source_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    report["source_sha256"] = {name: file_hash(ROOT / name) for name in [*PROTOCOL_FILES, "src/st3/exp028_cache.py",
                               "src/st3/exp028_stage1.py", "scripts/run_exp028_stage1.py", "configs/exp028_fold3_manifest.json"]}
    report["runtime"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    report["files_sha256"] = {p.name: file_hash(p) for p in sorted(run.iterdir()) if p.is_file()}
    atomic_json(run / "status.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["audit", "screen"], default="audit")
    for name in ("inputs", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, action="append", required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--require-full-vision-cache", action="store_true",
                        help="Require complete verified vision payloads as well as decoder features")
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
