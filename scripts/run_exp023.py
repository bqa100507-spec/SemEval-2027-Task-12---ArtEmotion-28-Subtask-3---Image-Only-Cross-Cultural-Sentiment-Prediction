"""Single EXP-023 child entry point; cloud cleanup belongs to the supervising parent."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from st3.experiment_workflow import StopRun, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--started-at", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAILED_TRAINING", "experiment": "EXP-023", "production": "E50+L", "development_gold_read": False, "automatic_codabench_upload": False}
    try:
        from st3.exp023_runner import Experiment, ROOT
        config = json.loads((ROOT / "configs/exp023.json").read_text())
        auth = json.loads(args.authorization.read_text())
        engine = Experiment(args.inputs, args.run_dir, config, auth, device=args.device, started=args.started_at)
        stages = engine.run()
        report.update(status="SUCCESS_SUBMISSION_READY", stages=stages, model="EC2_R0_1335", production_promotion=False)
    except StopRun as exc:
        report.update(status=exc.status, error=str(exc))
    except Exception as exc:
        report.update(status="FAILED_TRAINING", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        report["observed_wall_seconds_since_Run_all"] = time.time() - args.started_at
        write_json(args.run_dir / "child_report.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["status"] in ("SUCCESS_SUBMISSION_READY", "CV_NO_GO", "CONFIRMATION_NO_GO") else 2


if __name__ == "__main__":
    raise SystemExit(main())
