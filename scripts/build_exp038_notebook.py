#!/usr/bin/env python3
"""Write notebooks/EXP038_Kaggle.ipynb pinned to one packaged overlay and the existing EXP-037 base dataset."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# EXP-038 · Cultural-deviation ensemble (frozen encoders + interaction heads)

Attach **two private datasets**: the existing EXP-037 input (`exp037_inputs.zip`) and the EXP-038 overlay
(`exp038_inputs.zip`). Switch on **GPU** (T4 ×2 is fine; only the first GPU is used) and **Internet**, set
`AUTHORIZE_RUN = True` in the next cell, and select **Run All** once.

Stages: Setup → Validate → Benchmark → Feature extraction → CS-CVA seeds → P2 evaluation → Full refit and
Development scoring → Submission validation → Export → Release. Validate replays the R0 references on CPU and
hashes every model file before any GPU work; a failed replay stops the run. The benchmark forecast may skip, in
order, CS-CVA seed 3, CS-CVA seed 2, then E4 (C3), and records why. Development is reporting only.
E3 and E4 (Qwen3-VL-Embedding) are optional: if E3 fails or is skipped, E4 is skipped as a dependency, C2/C3 are listed as unavailable, and the run still completes with the VLM-free C1 and its ZIPs.

Keep everything private. When the run ends, save a version or download from **Output**: `exp038_results.zip`,
`exp038_best_submission.zip` and their `.sha256` files (or `exp038_diagnostic.zip` after a failure, and
`exp038_progress.zip` to resume). C2, C3 and every B2 ZIP are `UNCONFIRMED_QWEN_ST3`. Never upload from here.
"""

CONFIG = """AUTHORIZE_RUN = False
RESUME_INPUT = ""   # Optional absolute path to a previous exp038_progress.zip / exp038_results.zip or its extracted root
E1_SNAPSHOT = ""    # Optional local snapshot of google/siglip2-so400m-patch14-384 at the pinned revision
E2_SNAPSHOT = ""    # Optional local snapshot of facebook/metaclip-2-worldwide-huge-quickgelu at the pinned revision
E3_SNAPSHOT = ""    # Optional local snapshot of Qwen/Qwen3-VL-Embedding-2B at the pinned revision
SESSION_HOURS = 10.5
"""

LOCATE = """import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from zipfile import ZipFile

WORKER_SOURCE = "{source}"
BASE_MANIFEST_SHA256 = "{base_manifest}"
BASE_ARCHIVE_SHA256 = "{base_archive}"
OVERLAY_MANIFEST_SHA256 = "{overlay_manifest}"
OVERLAY_ARCHIVE_SHA256 = "{overlay_archive}"

def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def verify_root(root, manifest_name, manifest_sha):
    manifest_path = root / manifest_name
    if sha256(manifest_path) != manifest_sha:
        raise ValueError(f"Manifest checksum differs at {{manifest_path}}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        target = root / relative
        if relative.is_absolute() or ".." in relative.parts or not target.is_file() or sha256(target) != expected:
            raise ValueError(f"Missing or corrupt file {{target}}. Reattach the verified private dataset.")
    return manifest

def locate(mount, work, manifest_name, manifest_sha, archive_name, archive_sha, prefix):
    found = [p.parent for p in sorted(mount.rglob(manifest_name)) if sha256(p) == manifest_sha]
    found = [p for p in found if (p / "runtime").is_dir()]
    if len(found) > 1:
        raise ValueError(f"Several extracted copies of {{archive_name}} match. Attach only one.")
    if found:
        return found[0], verify_root(found[0], manifest_name, manifest_sha)
    archives = [p for p in sorted(mount.rglob(archive_name)) if sha256(p) == archive_sha]
    if len(archives) != 1:
        raise FileNotFoundError(f"Attach exactly one private dataset with the pinned {{archive_name}}.")
    target = work / manifest_sha[:16]
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archives[0]) as archive:
        archive.extractall(target)
    root = target / prefix
    return root, verify_root(root, manifest_name, manifest_sha)

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
base_root, base_manifest = locate(MOUNT, Path("/kaggle/temp/exp038_base"), "exp037_manifest.json",
                                  BASE_MANIFEST_SHA256, "exp037_inputs.zip", BASE_ARCHIVE_SHA256, "exp037_inputs")
overlay_root, overlay_manifest = locate(MOUNT, Path("/kaggle/temp/exp038_overlay"), "exp038_manifest.json",
                                        OVERLAY_MANIFEST_SHA256, "exp038_inputs.zip", OVERLAY_ARCHIVE_SHA256,
                                        "exp038_inputs")
if overlay_manifest["source"] != WORKER_SOURCE or overlay_manifest["base"]["manifest_sha256"] != BASE_MANIFEST_SHA256:
    raise ValueError("The overlay was prepared for a different worker or base dataset")
print("Verified", len(base_manifest["files"]), "base files and", len(overlay_manifest["files"]), "overlay files")
print("Populations:", overlay_manifest["population"])
"""

LAUNCH = """launcher = overlay_root / "runtime/scripts/exp038_kaggle.py"
command = [sys.executable, "-u", str(launcher), "--base", str(base_root), "--overlay", str(overlay_root),
           "--base-sha", BASE_MANIFEST_SHA256, "--overlay-sha", OVERLAY_MANIFEST_SHA256,
           "--started", str(STARTED), "--hours", str(SESSION_HOURS)]
for option, value in (("resume", RESUME_INPUT), ("e1-snapshot", E1_SNAPSHOT), ("e2-snapshot", E2_SNAPSHOT),
                      ("e3-snapshot", E3_SNAPSHOT)):
    if value:
        path = Path(value)
        if not path.is_absolute() or not path.exists():
            raise FileNotFoundError(f"Set {option} to its actual absolute Kaggle path: {path}")
        command.extend((f"--{option}", str(path)))
subprocess.run(command, check=True, env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"})
"""

SUMMARY = """run_dir = OUTPUT / "exp038_run"
for name in ("status.json", "budget_plan.json", "decision.json", "selection.json"):
    path = run_dir / name
    if path.is_file():
        print(name, json.dumps(json.loads(path.read_text()), indent=1)[:4000])
for name in ("exp038_SUCCESS.json", "exp038_results.zip", "exp038_best_submission.zip", "exp038_progress.zip",
             "exp038_diagnostic.zip"):
    path = OUTPUT / name
    if path.is_file():
        receipt = OUTPUT / f"{name}.sha256"
        print(path, receipt.read_text().strip() if receipt.is_file() else "")
"""

OUTRO = """Save a version or download the files above from **Output** before closing the session. A progress or
results ZIP plus the same two inputs, the same GPU type and `RESUME_INPUT` resume the run; sealed stages and
feature caches are reused, and corrupt or incompatible caches stop the run instead of being recomputed.
"""


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    body = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        body |= {"execution_count": None, "outputs": []}
    return body


def build(handoff, base_archive, base_manifest):
    pins = {"source": handoff["source"], "base_manifest": base_manifest, "base_archive": base_archive,
            "overlay_manifest": handoff["manifest_sha256"], "overlay_archive": handoff["archive_sha256"]}
    cells = [cell("markdown", INTRO), cell("code", CONFIG), cell("code", LOCATE.format(**pins)),
             cell("code", LAUNCH), cell("code", SUMMARY), cell("markdown", OUTRO)]
    return {"cells": cells, "metadata": {"kaggle": {"accelerator": "gpu", "isInternetEnabled": True,
                                                    "language": "python"},
                                         "kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python", "version": "3.12.10"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/EXP038_Kaggle.ipynb")
    args = parser.parse_args()
    handoff = json.loads(args.handoff.read_text(encoding="utf-8"))
    notebook = build(handoff, handoff["base_zip_sha256"], handoff["base_manifest_sha256"])
    args.output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
