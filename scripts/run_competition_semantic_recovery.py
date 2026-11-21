#!/usr/bin/env python3
"""Run the bounded Qwen inference recovery benchmark on the pinned training sample."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.competition_semantic_generation import CONFIG, TRACK1, training_inventory
from st3.competition_semantic_recovery import RECOVERY, read_sample, recover_benchmark
from st3.e50_heads import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/competition_semantic_vlm/recovery")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/qwen3-vl-4b-ebb281e")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/upstream")
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--historical", type=Path, default=ROOT / "runs/competition_semantic_vlm/preflight_attempt1/preflight.json")
    args = parser.parse_args()
    args.base_config = CONFIG
    base, recovery = read_json(CONFIG), read_json(RECOVERY)
    scope = training_inventory(args.inputs, read_json(TRACK1))
    sample = read_sample(args.historical, scope, recovery)
    recover_benchmark(args, base, recovery, sample)


if __name__ == "__main__":
    main()
