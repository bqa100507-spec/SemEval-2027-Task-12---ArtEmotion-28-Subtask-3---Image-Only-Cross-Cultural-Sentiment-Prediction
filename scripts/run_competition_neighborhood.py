#!/usr/bin/env python3
"""Run the single, CPU-only P2 image-neighborhood correction comparison."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.competition_neighborhood import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs" / "exp023_budget_delivery" / "release" / "restore_check")
    parser.add_argument("--track1-dir", type=Path, default=ROOT / "runs" / "competition_track1")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "competition_neighborhood_residual")
    run(parser.parse_args())
