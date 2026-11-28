#!/usr/bin/env python3
"""Write notebooks/EXP039_Kaggle.ipynb pinned to one packaged overlay and the existing EXP-037 base dataset.

The locate code is shared with EXP-038 (three layouts: expanded, intact ZIP, recursive expansion). The launch
cell streams the launcher through a pipe, so every line reaches the Kaggle log once (EXP-038 printed each
line twice because the child also wrote straight to the kernel's stdout).
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from build_exp038_notebook import DEFS as LOCATE_DEFS  # noqa: E402
from build_exp038_notebook import cell  # noqa: E402

INTRO = """# EXP-039 · Scale the VLM-embedding axis (size, instructions, resolution)

Attach **two private datasets**: the existing EXP-037 input (`exp037_inputs.zip`) and the EXP-039 overlay
(`exp039_inputs.zip`). Select **GPU T4 ×2** (E5 splits Qwen3-VL-Embedding-8B in FP16 across both GPUs; with one
GPU it runs NF4 and is labelled `E5_nf4`) and **Internet on**. Set `AUTHORIZE_RUN = True` and select **Run All** once.

Stages: Setup → Validate → Benchmark → Feature extraction (E5, E6, E7) → P2 evaluation → Full refit and
Development → Submission validation → Export → Release. Validate checks inputs, the overlay restore, pin formats,
both Qwen snapshots, the sealed EXP-038 features and the R0 and C2 replays, and reports every failure together
before stopping. E5, E6 and E7 are optional; when over budget, E5 is skipped first, then E7. C2 is recomputed
from the sealed EXP-038 features and is the gate reference. Development is reporting only.

Keep everything private. Save a version or download from **Output**: `exp039_results.zip`,
`exp039_best_submission.zip` and their `.sha256` files (or `exp039_diagnostic.zip`; `exp039_progress.zip` to resume).
Qwen-dependent ZIPs are `UNCONFIRMED_QWEN_ST3`. Never upload from here.
"""

CONFIG = """AUTHORIZE_RUN = False
RESUME_INPUT = ""   # Optional absolute path to a previous exp039_progress.zip / exp039_results.zip or its extracted root
E3_SNAPSHOT = ""    # Optional local snapshot of Qwen/Qwen3-VL-Embedding-2B at the pinned revision
E5_SNAPSHOT = ""    # Optional local snapshot of Qwen/Qwen3-VL-Embedding-8B at the pinned revision
SESSION_HOURS = 10.5
"""

STREAM = '''
def stream(command, write=None, env=None):
    # Read the child through a pipe and print each line once; the child never writes to the kernel's stdout.
    write = write or (lambda line: print(line, end="", flush=True))
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", env=env) as child:
        for line in child.stdout:
            write(line)
        code = child.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
'''

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
base_root, base_manifest = locate(MOUNT, Path("/kaggle/temp/exp039_base"), "exp037_manifest.json",
                                  BASE_MANIFEST_SHA256, "exp037_inputs.zip", BASE_ARCHIVE_SHA256, "exp037_inputs")
overlay_root, overlay_manifest = locate(MOUNT, Path("/kaggle/temp/exp039_overlay_zip"), "exp039_manifest.json",
                                        OVERLAY_MANIFEST_SHA256, "exp039_inputs.zip", OVERLAY_ARCHIVE_SHA256,
                                        "exp039_inputs")
if overlay_manifest["source"] != WORKER_SOURCE or overlay_manifest["base"]["manifest_sha256"] != BASE_MANIFEST_SHA256:
    raise ValueError("The overlay was prepared for a different worker or base dataset")
print("Verified", len(base_manifest["files"]), "base files and", len(overlay_manifest["files"]), "overlay files;",
      len(overlay_manifest.get("encoded", {{}})), "archives are stored as .b64 and restored by the launcher")
"""

LAUNCH = """launcher = overlay_root / "runtime/scripts/exp039_kaggle.py"
command = [sys.executable, "-u", str(launcher), "--base", str(base_root), "--overlay", str(overlay_root),
           "--base-sha", BASE_MANIFEST_SHA256, "--overlay-sha", OVERLAY_MANIFEST_SHA256,
           "--started", str(STARTED), "--hours", str(SESSION_HOURS)]
for option, value in (("resume", RESUME_INPUT), ("e3-snapshot", E3_SNAPSHOT), ("e5-snapshot", E5_SNAPSHOT)):
    if value:
        path = Path(value)
        if not path.is_absolute() or not path.exists():
            raise FileNotFoundError(f"Set {option} to its actual absolute Kaggle path: {path}")
        command.extend((f"--{option}", str(path)))
stream(command, env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"})
"""

SUMMARY = """run_dir = OUTPUT / "exp039_run"
for name in ("status.json", "budget_plan.json", "decision.json", "selection.json"):
    path = run_dir / name
    if path.is_file():
        print(name, json.dumps(json.loads(path.read_text()), indent=1)[:4000])
for name in ("exp039_SUCCESS.json", "exp039_results.zip", "exp039_best_submission.zip", "exp039_progress.zip",
             "exp039_diagnostic.zip"):
    path = OUTPUT / name
    if path.is_file():
        receipt = OUTPUT / f"{name}.sha256"
        print(path, receipt.read_text().strip() if receipt.is_file() else "")
"""

DEFS = LOCATE_DEFS + STREAM


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
    parser.add_argument("--output", type=Path, default=ROOT.parent / "notebooks/EXP039_Kaggle.ipynb")
    args = parser.parse_args()
    notebook = build(json.loads(args.handoff.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
