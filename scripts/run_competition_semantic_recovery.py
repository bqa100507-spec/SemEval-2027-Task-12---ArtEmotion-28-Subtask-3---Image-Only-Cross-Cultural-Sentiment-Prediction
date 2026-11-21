#!/usr/bin/env python3
"""Run the bounded Qwen inference recovery benchmark on the pinned training sample."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.baseline import file_sha256
from st3.cli import json_output
from st3.competition_semantic_generation import CONFIG, TRACK1, training_inventory
from st3.competition_semantic_recovery import (
    RECOVERY, batch_profile, load_model, read_sample, recover_benchmark, verified_assets,
)
from st3.e50_heads import read_json


def check_wide_batch(args, base, recovery, sample):
    """One fixed batch-width check using the exact same six paintings and recipe."""
    dest = args.run_dir / "batch_six.json"
    core = ROOT / "src/st3/competition_semantic_recovery.py"
    if dest.exists():
        raise FileExistsError(dest)
    prior = read_json(args.run_dir / "benchmark.json")
    if (len(sample) != 6 or prior["status"] != "MEASURED"
            or prior["source_sha256"] != file_sha256(core) or prior["recovery_sha256"] != file_sha256(RECOVERY)
            or prior["base_config_sha256"] != file_sha256(CONFIG) or not prior["repeat_exact"]):
        raise ValueError("A matching completed and repeatable primary benchmark is required")
    if any("description" not in row for batch in prior["compact"] for row in batch["records"]):
        raise ValueError("The compact format must pass on all six paintings before the batch-width check")
    began = time.perf_counter()
    receipt = {"status": "RUNNING", "benchmark_sha256": file_sha256(args.run_dir / "benchmark.json"),
               "runner_sha256": file_sha256(Path(__file__)), "batch_size": len(sample), "batches": []}
    try:
        verified_assets(args.checkpoint, base)
        torch, model, processor, receipt["hardware"] = load_model(args.checkpoint, base)
        model.set_attn_implementation(recovery["attention"])
        for repeat in range(2):
            batch = batch_profile(torch, model, processor, args.data_dir, sample, base, recovery)
            receipt["batches"].append(batch)
            json_output(receipt, dest)
            print(f"Width six run {repeat + 1}: {batch['wall_seconds']:.2f} s", flush=True)
        first, second = receipt["batches"]
        receipt["repeat_exact"] = [r["raw"] for r in first["records"]] == [r["raw"] for r in second["records"]]
        receipt["status"] = "MEASURED"
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - began
        json_output(receipt, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/competition_semantic_vlm/recovery")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/qwen3-vl-4b-ebb281e")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/upstream")
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--historical", type=Path, default=ROOT / "runs/competition_semantic_vlm/preflight_attempt1/preflight.json")
    parser.add_argument("--check-batch-six", action="store_true")
    args = parser.parse_args()
    args.base_config = CONFIG
    base, recovery = read_json(CONFIG), read_json(RECOVERY)
    scope = training_inventory(args.inputs, read_json(TRACK1))
    sample = read_sample(args.historical, scope, recovery)
    if args.check_batch_six:
        check_wide_batch(args, base, recovery, sample)
    else:
        recover_benchmark(args, base, recovery, sample)


if __name__ == "__main__":
    main()
