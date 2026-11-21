#!/usr/bin/env python3
"""Run the frozen local semantic evaluation with one cumulative GPU budget."""

import argparse
import sys
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.competition_challenger import check_hash
from st3.competition_semantic_evaluation import (
    frozen_work, prepare, sealed_recovery, supervised_worker, worker, write_atomic,
)
from st3.competition_semantic_vlm import run_cv
from st3.e50_heads import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/competition_semantic_vlm/evaluation")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/qwen3-vl-4b-ebb281e")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/upstream")
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "runs/competition_neighborhood_residual/results")
    parser.add_argument("--cached", type=Path, default=ROOT / "runs/competition_semantic_vlm/recovery/batch_six.json")
    parser.add_argument("--authorize-3p5h-rtx4060", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_deadline is not None:
        ledger = read_json(args.run_dir / "gpu_budget.json")
        if (not args.authorize_3p5h_rtx4060 or args.prepare_only or ledger["active_since"] is None
                or args.worker_deadline != ledger["active_since"] + (ledger["limit_seconds"] - ledger["charged_seconds"])):
            raise ValueError("GPU worker must run under the authorized budget supervisor")
        worker(args, args.worker_deadline)
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.run_dir / "run.lock"), timeout=0):
        status = args.run_dir / "status.json"
        stage = "prepare"
        try:
            plan = frozen_work(args)
            _, base, _, track1, _, _, batches, identity = plan
            count = prepare(args, plan)
            for name, digest in track1["p2_input_sha256"].items():
                check_hash(args.inputs / name, digest)
            check_hash(args.reference_dir / "oof.csv", base["incumbent_oof_sha256"])
            check_hash(args.reference_dir / "metrics.json", base["incumbent_metrics_sha256"])
            if args.prepare_only:
                return
            if count < len(batches):
                if not args.authorize_3p5h_rtx4060:
                    raise PermissionError("GPU extraction requires explicit owner authorization")
                stage = "extraction"
                write_atomic(status, {"state": "RUNNING", "stage": stage, "identity": identity})
                supervised_worker(args, [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:]], plan)
            stage = "cv"
            descriptions, digest = sealed_recovery(args)
            write_atomic(status, {"state": "RUNNING", "stage": stage, "extraction_sha256": digest})
            if not (args.run_dir / "cv").exists():
                run_cv(args, base, track1, descriptions_loader=lambda *_: (descriptions, digest))
            cv = args.run_dir / "cv"
            seal = read_json(cv / "seal.json")
            if set(seal) != {"oof.csv", "metrics.json"}:
                raise ValueError("CV output seal is incomplete")
            for file, expected in seal.items():
                check_hash(cv / file, expected)
            metrics = read_json(cv / "metrics.json")
            if metrics["extraction_sha256"] != digest or metrics["config_sha256"] != identity["base"]:
                raise ValueError("Completed CV does not match the frozen training cache")
            write_atomic(status, {"state": "COMPLETE", "stage": stage, "decision": metrics["decision"],
                                  "extraction_sha256": digest, "metrics_sha256": seal["metrics.json"],
                                  "development": "DISABLED_PENDING_CONFIRMED_ELIGIBILITY"})
        except BaseException as exc:
            write_atomic(status, {"state": "STOPPED", "stage": stage, "error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    main()
