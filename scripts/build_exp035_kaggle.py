"""Build and statically verify the private EXP-035 handoff. Never runs the worker."""

import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/exp035_kaggle/handoff"
P2 = ROOT / "runs/exp023_budget_delivery/release/restore_check"
RIDGE = ROOT / "runs/competition_neighborhood_residual/results"
PREPARED = ROOT / "runs/exp035_fold3_pilot/prepared"
BASELINE_SOURCE = "bffef12895c8f7abdc5c343d7979f371a8f8e287"
PREPARED_SHA = "baa3c118856a4867af80f734a093024baccb8ff110913e5b2e6c02e9e0753e07"
INPUT_MOUNT = "/kaggle/input/exp035-fold3-inputs/exp035_inputs"
MODEL_MOUNT = "/kaggle/input/exp035-qwen"
SOURCES = (
    "src/st3", "scripts/run_exp035.py", "scripts/run_competition_direct_qwen_pilot.py",
    "scripts/run_competition_direct_qwen.py",
    "scripts/exp035_kaggle.py", "requirements-exp035-kaggle.lock.txt", "configs/st3.json",
    "configs/exp035.json", "configs/competition_track1.json", "configs/competition_semantic_vlm.json",
    "configs/competition_neighborhood_residual.json",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def notebook(manifest_sha, archive_sha):
    def cell(name, kind, source):
        metadata = {"id": name, "cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        return metadata | ({"execution_count": None, "outputs": []} if kind == "code" else {})

    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": [
        cell("start", "markdown", """# EXP-035 · Matched fold-3 pilot

Import this notebook into a **private** Kaggle notebook. Attach your private EXP-035 input dataset
and the pinned Qwen files as a separate private dataset or model. Select **GPU L4** if offered,
or **GPU T4 ×2**. The worker uses only GPU 0. Enable Internet to install the pinned Python environment.
Set `AUTHORIZE_RUN = True` below and click **Run All**. The preflight starts the existing EXP-035
worker, which continues to four epochs only if its measured forecast fits the five-hour ceiling.

The handoff requires no Kaggle or Hugging Face credentials inside the notebook.
"""),
        cell("settings", "code", f"""# Change the authorization before running on Kaggle.
AUTHORIZE_RUN = False
INPUT_ROOT = {INPUT_MOUNT!r}  # Optional override for your actual mount.
MODEL_ROOT = {MODEL_MOUNT!r}
RESUME_ARCHIVE = ""  # Optional: mounted exp035_results.zip or its extracted directory.
EXPECTED_MANIFEST = {manifest_sha!r}
EXPECTED_ARCHIVE = {archive_sha!r}
"""),
        cell("verify-guide", "markdown", """## 1. Find and verify the inputs

Extracted Kaggle datasets are discovered recursively. The original ZIP works too.
All bundled files are checked before the launcher is executed. The worker verifies the checkpoint
shards and processor against their original revision before loading Qwen.
"""),
        cell("verify", "code", f"""import hashlib
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

if AUTHORIZE_RUN is not True:
    raise RuntimeError("Attach both private inputs, enable Kaggle GPU and Internet, then set AUTHORIZE_RUN = True and Run All.")
started = time.time()
kaggle_input = Path("/kaggle/input")
working = Path("/kaggle/working")
if not kaggle_input.is_dir() or not working.is_dir():
    raise RuntimeError("Run this notebook inside Kaggle.")
if sys.version_info < (3, 11):
    raise RuntimeError("This notebook launcher needs a Kaggle Python 3.11 or newer host.")

def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def only(candidates, label):
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f"Expected one {{label}}, found {{len(candidates)}}. Attach the private input or set its exact mount in Settings.")
    return candidates[0]

handoff = Path(INPUT_ROOT)
if not (handoff / "exp035_handoff.json").is_file():
    unpacked = Path("/kaggle/temp/exp035_handoff/exp035_inputs")
    manifests = list(kaggle_input.rglob("exp035_handoff.json"))
    if manifests:
        handoff = only((p.parent for p in manifests), "EXP-035 handoff")
    else:
        archive = only(kaggle_input.rglob("exp035_fold3_inputs.zip"), "EXP-035 input ZIP")
        if sha256(archive) != EXPECTED_ARCHIVE:
            raise ValueError("Input ZIP SHA-256 mismatch. Attach the original EXP-035 archive.")
        with ZipFile(archive) as source:
            source.extractall(unpacked.parent)  # Only the exact, checksum-pinned ZIP can reach this line.
        handoff = unpacked

manifest_path = handoff / "exp035_handoff.json"
if sha256(manifest_path) != EXPECTED_MANIFEST:
    raise ValueError("Input manifest SHA-256 mismatch. Use the notebook and archive from the same handoff.")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for name, expected in manifest["files"].items():
    path = handoff / name
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Missing or mismatched handoff file {{name}}. Reattach the verified private archive.")

checkpoint = Path(MODEL_ROOT)
if not all((checkpoint / name).is_file() for name in manifest["model"]["files"]):
    checkpoint = only((path.parent for path in kaggle_input.rglob("model.safetensors.index.json")
                       if all((path.parent / name).is_file() for name in manifest["model"]["files"])),
                      "complete pinned Qwen model folder including README.md")
print("Verified inputs and source", manifest["source"]["kaggle_revision"], flush=True)
"""),
        cell("launch-guide", "markdown", """## 2. Install, preflight and run the matched experiment

The launcher checks CUDA and FP16, verifies the sealed Ridge offsets, loads the pinned NF4 model,
checks parity and LoRA gradients, and measures throughput. It stops before training if the forecast
exceeds the remaining budget. Scientific epochs retain their original optimizer, loss and
inner-only checkpoint selection. Logs and sealed epochs appear under `/kaggle/working/exp035_fold3`.
"""),
        cell("launch", "code", """import subprocess
import sys

launcher = handoff / "runtime/scripts/exp035_kaggle.py"
command = [sys.executable, "-u", str(launcher), "--input-root", str(handoff),
           "--checkpoint", str(checkpoint), "--handoff-sha", EXPECTED_MANIFEST,
           "--started", str(started)]
if RESUME_ARCHIVE:
    command += ["--resume", RESUME_ARCHIVE]
subprocess.run(command, check=True)
"""),
        cell("download-guide", "markdown", """## 3. Save results outside this session

Download **exp035_results.zip** and its **.sha256** from the Kaggle Output panel, or save a private
notebook version with outputs. The archive carries results or the exact stop reason and all
sealed epochs. To resume a controlled stop, attach the saved ZIP privately in a fresh Kaggle
session and set `RESUME_ARCHIVE` to the mounted ZIP or extracted directory. Use the same
accelerator model and handoff.
If Kaggle terminates a session abruptly, only artifacts exported or downloaded beforehand survive.
"""),
        cell("download", "code", """from IPython.display import FileLink, display

for name in ("exp035_results.zip", "exp035_results.zip.sha256"):
    path = working / name
    if path.is_file():
        display(FileLink(str(path)))
status = working / "exp035_fold3/status.json"
if status.is_file():
    print(status.read_text(encoding="utf-8"))
preflights = sorted((working / "exp035_fold3").glob("preflight_*/result.json"))
if preflights:
    probe = json.loads(preflights[-1].read_text(encoding="utf-8"))
    sample_size = len(probe["sample"])
    print("Kaggle preflight", probe["state"], probe["hardware"]["gpu"],
          "warm B / live C paintings per second",
          round(sample_size / probe["frozen_cached_step_seconds"], 2),
          round(sample_size / probe["adapter_step_seconds"], 2),
          "allocated / reserved GiB",
          round(probe["peak_allocated_bytes"] / 2**30, 2),
          round(probe["peak_reserved_bytes"] / 2**30, 2))
metrics = working / "exp035_fold3/results/metrics.json"
if metrics.is_file():
    report = json.loads(metrics.read_text(encoding="utf-8"))
    for scope in ("full", "test_like"):
        for arm in ("prior", "frozen", "merger"):
            row = report[scope][arm]
            print(scope, arm, row["cultural_contrast"], row["log_loss"])
"""),
    ]}


def build(revision):
    if git("rev-parse", revision).decode().strip() != revision:
        raise ValueError("Use a full Kaggle worker commit SHA")
    for relative in ("src/st3/exp035.py", "src/st3/exp035_protocol.py", "configs/exp035.json",
                     "configs/competition_semantic_vlm.json"):
        if git("show", f"{revision}:{relative}") != git("show", f"{BASELINE_SOURCE}:{relative}"):
            raise ValueError(f"Frozen experiment differs from the verified source: {relative}")
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "exp035_fold3_inputs.zip"
    scratch = archive.with_suffix(".tmp")
    with ZipFile(io.BytesIO(git("archive", "--format=zip", revision, *SOURCES))) as source, ZipFile(
            scratch, "w", compression=ZIP_DEFLATED, compresslevel=3, allowZip64=True) as bundle:
        files = {}

        def add(name, payload, expected=None):
            digest = sha(payload)
            if name in files or (expected is not None and digest != expected):
                raise ValueError(f"Duplicate or changed handoff input: {name}")
            files[name] = digest
            bundle.writestr(f"exp035_inputs/{name}", payload)

        for member in source.infolist():
            if not member.is_dir():
                add(f"runtime/{member.filename}", source.read(member))
        base = json.loads(source.read("configs/competition_semantic_vlm.json"))
        track = json.loads(source.read("configs/competition_track1.json"))
        for name, expected in track["p2_input_sha256"].items():
            add(f"p2/{name}", (P2 / name).read_bytes(), expected)
        for filename, key in (("oof.csv", "incumbent_oof_sha256"), ("metrics.json", "incumbent_metrics_sha256")):
            add(f"ridge/{filename}", (RIDGE / filename).read_bytes(), base[key])
        prepared = (PREPARED / "manifest.json").read_bytes()
        add("prepared/manifest.json", prepared, PREPARED_SHA)
        for name, expected in json.loads(prepared)["files"].items():
            add(f"prepared/{name}", (PREPARED / name).read_bytes(), expected)
        with (P2 / "training/protocol.csv").open(newline="", encoding="utf-8") as stream:
            paintings = [row for row in csv.DictReader(stream) if row["role"] == "development"]
        if len(paintings) != 1335 or len({row["image"] for row in paintings}) != 1335:
            raise ValueError("Expected 1,335 distinct P2 training images")
        for row in paintings:
            name = row["image"]
            add(f"artwork/images/{name}", (ROOT / "data/upstream/images" / name).read_bytes(), row["image_sha256"])
        model_files = {name: {"git_blob_sha1": value} for name, value in base["git_blobs"].items()}
        model_files.update({name: {"sha256": value} for name, value in base["weight_sha256"].items()})
        for name in model_files:
            cached = ROOT / "cache/qwen3-vl-4b-ebb281e" / name
            model_files[name]["local_file_bytes"] = cached.stat().st_size
        manifest = {"source": {"kaggle_revision": revision, "verified_local_revision": BASELINE_SOURCE},
                    "input_mount": INPUT_MOUNT, "model_mount": MODEL_MOUNT,
                    "model": {"id": base["model"], "revision": base["revision"], "files": model_files},
                    "preparation_manifest_sha256": PREPARED_SHA,
                    "population": {"fit": 854, "inner": 214, "outer": 267}, "files": files}
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        bundle.writestr("exp035_inputs/exp035_handoff.json", manifest_bytes)

    with ZipFile(scratch) as bundle:
        prefix = "exp035_inputs/"
        names = [prefix + name for name in (*files, "exp035_handoff.json")]
        if sorted(bundle.namelist()) != sorted(names):
            raise ValueError("Input archive coverage differs")
        for name, expected in files.items():
            with bundle.open(prefix + name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Archive readback failed: {name}")
        if bundle.read(prefix + "exp035_handoff.json") != manifest_bytes:
            raise ValueError("Manifest archive readback failed")
    scratch.replace(archive)
    archive_sha = file_sha(archive)
    (OUT / "exp035_handoff.json").write_bytes(manifest_bytes)
    (OUT / "exp035_fold3_inputs.zip.sha256").write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    path = ROOT / "notebooks/EXP035_Kaggle.ipynb"
    path.write_text(json.dumps(notebook(sha(manifest_bytes), archive_sha), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": archive_sha,
                      "manifest_sha256": sha(manifest_bytes), "files": len(files), "notebook": str(path)}, indent=2))


if __name__ == "__main__":
    build(sys.argv[1])
