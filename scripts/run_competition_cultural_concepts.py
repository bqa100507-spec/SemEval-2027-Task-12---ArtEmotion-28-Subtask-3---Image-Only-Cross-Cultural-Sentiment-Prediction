#!/usr/bin/env python3
"""Fit the fixed observable-concept comparison on the five original P2 folds."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.competition_cultural_concepts import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--semantic-run", type=Path, default=ROOT / "runs/competition_semantic_vlm/evaluation")
    parser.add_argument("--cached-sample", type=Path, default=ROOT / "runs/competition_semantic_vlm/recovery/batch_six.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/competition_neighborhood_residual/results")
    parser.add_argument("--destination", type=Path, default=ROOT / "runs/competition_cultural_concepts/results")
    args = parser.parse_args()
    args.semantic_args = SimpleNamespace(inputs=args.inputs, cached=args.cached_sample, run_dir=args.semantic_run)
    run(args)


if __name__ == "__main__":
    main()
