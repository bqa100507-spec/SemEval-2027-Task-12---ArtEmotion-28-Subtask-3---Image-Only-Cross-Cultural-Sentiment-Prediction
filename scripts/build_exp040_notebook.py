#!/usr/bin/env python3
"""Write notebooks/EXP040_Kaggle.ipynb pinned to one packaged overlay and the existing EXP-037 base dataset.

New launcher template (checklist item 9): ``SMOKE_ONLY = True`` runs Setup, Validate (inputs, restore, pins, key
audit, the step-0 equality on one fold; replays skipped), one fold x 30 optimizer steps plus one inner pass, and
writes ``exp040_smoke.json`` with throughput, VRAM, devices, finiteness, determinism and a full-run forecast. It
produces no submission and no SUCCESS marker. The locate code and single-relay ``stream`` are shared with EXP-039.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from build_exp039_notebook import DEFS  # noqa: E402
from build_exp038_notebook import cell  # noqa: E402

INTRO = """# EXP-040 · E3-LoRA (Qwen3-VL-Embedding-2B adapted from a ridge-initialized interaction head)

Attach **two private datasets**: the existing EXP-037 input (`exp037_inputs.zip`) and the EXP-040 overlay
(`exp040_inputs.zip`). Select **GPU T4 ×2** (E3L trains on GPU 0 only) and **Internet on**.

1. **Smoke first:** keep `SMOKE_ONLY = True`, set `AUTHORIZE_RUN = True`, **Run All**. It validates the inputs, the
   key audit and the step-0 equality, trains one fold for 30 optimizer steps, runs one inner pass, and writes
   `exp040_smoke.json` (throughput, peak VRAM, devices, finite losses, two-pass determinism, full-run forecast).
   No submission is produced.
2. **Full run:** set `SMOKE_ONLY = False` and **Run All** once. Stages: Setup → Validate → Benchmark → P2 LoRA
   folds (sealed per fold, resumable) → P2 evaluation → Full refit and Development → Submission validation →
   Export → Release. If the forecast exceeds the session, the run stops before training and exports a NO_GO report.
   C2 is recomputed from the sealed EXP-038 features and is the gate reference. Development is reporting only.

Keep everything private. Download from **Output**: `exp040_smoke.json`, or `exp040_results.zip`,
`exp040_best_submission.zip` and their `.sha256` files (or `exp040_diagnostic.zip`; `exp040_progress.zip` to
resume). Every ZIP is `UNCONFIRMED_QWEN_ST3`. Never upload from here.
"""

CONFIG = """AUTHORIZE_RUN = False
SMOKE_ONLY = True   # Run the smoke first; set False for the full run
RESUME_INPUT = ""   # Optional absolute path to a previous exp040_progress.zip / exp040_results.zip or its extracted root
E3_SNAPSHOT = ""    # Optional local snapshot of Qwen/Qwen3-VL-Embedding-2B at the pinned revision
SESSION_HOURS = 10.5
"""

LOCATE = """WORKER_SOURCE = "{source}"
BASE_MANIFEST_SHA256 = "{base_manifest}"
BASE_ARCHIVE_SHA256 = "{base_archive}"
OVERLAY_MANIFEST_SHA256 = "{overlay_manifest}"
OVERLAY_ARCHIVE_SHA256 = "{overlay_archive}"

STARTED = time.time()
MOUNT, OUTPUT = Path("/kaggle/input"), Path("/kaggle/working")
if not AUTHORIZE_RUN:
    raise RuntimeError("Attach both private inputs, enable GPU and Internet, set AUTHORIZE_RUN = True, then Run All.")
if not MOUNT.is_dir() or not OUTPUT.is_dir():
    raise RuntimeError("Open this notebook in Kaggle with both private inputs attached.")
probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True) if Path("/dev/nvidia0").exists() else None
if probe is None or probe.returncode != 0 or "GPU 0" not in probe.stdout:
    raise RuntimeError("Select a CUDA GPU in Kaggle Settings, then Run All again.")
print(probe.stdout.strip())
base_root, base_manifest = locate(MOUNT, Path("/kaggle/temp/exp040_base"), "exp037_manifest.json",
                                  BASE_MANIFEST_SHA256, "exp037_inputs.zip", BASE_ARCHIVE_SHA256, "exp037_inputs")
overlay_root, overlay_manifest = locate(MOUNT, Path("/kaggle/temp/exp040_overlay_zip"), "exp040_manifest.json",
                                        OVERLAY_MANIFEST_SHA256, "exp040_inputs.zip", OVERLAY_ARCHIVE_SHA256,
                                        "exp040_inputs")
if overlay_manifest["source"] != WORKER_SOURCE or overlay_manifest["base"]["manifest_sha256"] != BASE_MANIFEST_SHA256:
    raise ValueError("The overlay was prepared for a different worker or base dataset")
print("Verified", len(base_manifest["files"]), "base files and", len(overlay_manifest["files"]), "overlay files;",
      len(overlay_manifest.get("encoded", {{}})), "archives are stored as .b64 and restored by the launcher")
"""

LAUNCH = """launcher = overlay_root / "runtime/scripts/exp040_kaggle.py"
command = [sys.executable, "-u", str(launcher), "--base", str(base_root), "--overlay", str(overlay_root),
           "--base-sha", BASE_MANIFEST_SHA256, "--overlay-sha", OVERLAY_MANIFEST_SHA256,
           "--started", str(STARTED), "--hours", str(SESSION_HOURS)]
for option, value in (("resume", RESUME_INPUT), ("e3-snapshot", E3_SNAPSHOT)):
    if value:
        path = Path(value)
        if not path.is_absolute() or not path.exists():
            raise FileNotFoundError(f"Set {option} to its actual absolute Kaggle path: {path}")
        command.extend((f"--{option}", str(path)))
if SMOKE_ONLY:
    command.append("--smoke")
stream(command, env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"})
"""

SUMMARY = """run_dir = OUTPUT / "exp040_run"
for name in ("status.json", "budget_plan.json", "decision.json", "selection.json"):
    path = run_dir / name
    if path.is_file():
        print(name, json.dumps(json.loads(path.read_text()), indent=1)[:4000])
smoke = OUTPUT / "exp040_smoke.json"
if smoke.is_file():
    result = json.loads(smoke.read_text())
    print("SMOKE_ONLY receipt:", json.dumps({k: result[k] for k in ("validate", "step0", "forecast")}, indent=1)[:6000])
for name in ("exp040_SUCCESS.json", "exp040_results.zip", "exp040_best_submission.zip", "exp040_progress.zip",
             "exp040_diagnostic.zip"):
    path = OUTPUT / name
    if path.is_file():
        receipt = OUTPUT / f"{name}.sha256"
        print(path, receipt.read_text().strip() if receipt.is_file() else "")
"""


def build(handoff):
    pins = {"source": handoff["source"], "base_manifest": handoff["base_manifest_sha256"],
            "base_archive": handoff["base_zip_sha256"], "overlay_manifest": handoff["manifest_sha256"],
            "overlay_archive": handoff["archive_sha256"]}
    cells = [cell("markdown", INTRO), cell("code", CONFIG), cell("code", DEFS), cell("code", LOCATE.format(**pins)),
             cell("code", LAUNCH), cell("code", SUMMARY)]
    return {"cells": cells, "metadata": {"kaggle": {"accelerator": "gpu", "isInternetEnabled": True,
                                                    "language": "python"},
                                         "kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python", "version": "3.12.10"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "notebooks/EXP040_Kaggle.ipynb")
    args = parser.parse_args()
    notebook = build(json.loads(args.handoff.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
