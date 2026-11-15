"""Generate pinned, dual-platform one-click notebooks from an actual private delivery receipt."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


ORCHESTRATION = '''import hashlib, io, json, time, traceback, zipfile
from pathlib import Path
MOUNT_OK = PLATFORM != "colab"
RELEASE_ATTEMPTED = False

def request_release():
    global RELEASE_ATTEMPTED
    RELEASE_ATTEMPTED = True
    if PLATFORM == "colab" and AUTO_RELEASE_COLAB:
        from google.colab import runtime
        runtime.unassign()

def browser_request(path):
    if PLATFORM == "colab":
        from google.colab import files
        files.download(path)

try:
    if PLATFORM == "colab":
        from google.colab import drive
        drive.mount("/content/drive")
        MOUNT_OK = True
    archive = Path(INPUT_ZIP)
    if PLATFORM == "kaggle" and not archive.is_file():
        matches = list(Path("/kaggle/input").glob("**/exp023_handoff.zip"))
        if len(matches) != 1:
            raise FileNotFoundError("Attach one private exp023_handoff.zip dataset, or set INPUT_ZIP in Configuration")
        archive = matches[0]
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ZIP_SHA256:
        raise ValueError("ZIP SHA-256 mismatch: use the original delivered exp023_handoff.zip")
    with zipfile.ZipFile(archive) as z:
        raw = z.read("input_manifest.json")
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
            raise ValueError("Input manifest SHA-256 mismatch")
        manifest = json.loads(raw)
        source_bytes = z.read("source/source.zip")
    if manifest["source_revision"] != SOURCE_REVISION or hashlib.sha256(source_bytes).hexdigest() != manifest["files"]["source/source.zip"]:
        raise ValueError("Pinned source identity mismatch")
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as source:
        entry = source.read("scripts/exp023_notebook_entry.py")
    if hashlib.sha256(entry).hexdigest() != ENTRY_SHA256:
        raise ValueError("Notebook entry-point source checksum mismatch")
    namespace = {}
    exec(compile(entry, "pinned_exp023_notebook_entry.py", "exec"), namespace)
    settings = {key: value for key, value in globals().copy().items() if key.isupper() and isinstance(value, (str, bool, int, float))}
    settings["INPUT_ZIP"] = str(archive)
    run_cloud = namespace["run_cloud"]
    RESULT = run_cloud(settings, platform=PLATFORM, mount=lambda:None,
                       download=browser_request, release=request_release, started=RUN_STARTED)
except Exception as exc:
    # Bootstrap failure before trusted helpers exist: produce only a verified diagnostic export.
    RESULT = {"status":"BLOCKED_INPUT", "error":str(exc), "traceback":traceback.format_exc(),
              "browser_arrival":"UNVERIFIED", "physical_runtime_termination":"UNVERIFIED"}
    try:
        if not MOUNT_OK:
            raise RuntimeError("Drive mount unavailable; no persistent-copy claim is made")
        output = Path(PERSISTENT_ROOT) / "export"
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "bootstrap_failure.json"
        report_path.write_text(json.dumps(RESULT, indent=2))
        final = output / "bootstrap_failure_outputs.zip"
        with zipfile.ZipFile(final, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(report_path, report_path.name)
        payload = final.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if z.read(report_path.name) != report_path.read_bytes():
                raise OSError("Diagnostic export read-back failed")
        (output / "bootstrap_failure_outputs.sha256").write_text(checksum + "  " + final.name + "\\n")
        RESULT["persistent_readback"] = "PASS"
        if PLATFORM == "colab" and AUTO_BROWSER_DOWNLOAD:
            try:
                browser_request(str(final))
                RESULT["download_request"] = "API_RETURNED"
            except Exception as download_error:
                RESULT["download_request"] = "FAILED: " + str(download_error)
        (output / "final_report.json").write_text(json.dumps(RESULT, indent=2))
    except Exception as export_error:
        RESULT["export_error"] = str(export_error)
        RESULT["persistent_readback"] = "FAIL"
    print(json.dumps(RESULT, indent=2))
finally:
    if PLATFORM == "colab" and AUTO_RELEASE_COLAB and not RELEASE_ATTEMPTED:
        try:
            request_release()
        except Exception as release_error:
            print("Runtime release request failed:", release_error)
'''


def make_notebook(receipt, platform):
    revision = receipt["source_revision"]
    entry = subprocess.check_output(["git", "show", revision + ":scripts/exp023_notebook_entry.py"], cwd=ROOT)
    prefix = "/content" if platform == "colab" else "/kaggle/working"
    input_zip = "/content/drive/MyDrive/exp023_handoff.zip" if platform == "colab" else "/kaggle/input/exp023-inputs/exp023_handoff.zip"
    persistent = "/content/drive/MyDrive/exp023_runs/" + receipt["manifest_sha256"][:12] if platform == "colab" else "/kaggle/working/exp023_outputs"
    configuration = f'''# OWNER_CONFIGURATION — the only owner authorization/configuration section
import time
RUN_STARTED = time.time()
AUTHORIZE_EXP023 = False
AUTHORIZE_EXPERIMENT = AUTHORIZE_EXP023
AUTHORIZE_GUARDED_CONFIRMATION = False
CAPTION_ELIGIBILITY_STATUS = "UNRESOLVED"
RESOURCE_ELIGIBILITY_STATUS = CAPTION_ELIGIBILITY_STATUS
MAX_GPU_HOURS = 1.0
MAX_TOTAL_WALL_HOURS = 2.0
REQUIRE_SUBMISSION_ZIP = True
AUTO_BROWSER_DOWNLOAD = True
AUTO_RELEASE_COLAB = True
INPUT_ZIP = {input_zip!r}
PERSISTENT_ROOT = {persistent!r}
EVAL_INPUT_CSV = ""
EVAL_DATA_DIR = ""
# Immutable delivery identities; do not edit to bypass a mismatch.
PLATFORM = {platform!r}
LOCAL_ROOT = {str(prefix + '/exp023_' + revision[:12])!r}
SOURCE_REVISION = {revision!r}
ZIP_SHA256 = {receipt['zip_sha256']!r}
MANIFEST_SHA256 = {receipt['manifest_sha256']!r}
ENTRY_SHA256 = {hashlib.sha256(entry).hexdigest()!r}
'''
    text = """# EXP-023 — Painting-centered caption supervision

**Implemented, CPU-tested; GPU execution blocked while organizer eligibility is unresolved.**

Upload the single private `exp023_handoff.zip` to MyDrive (Colab) or attach it as a private Kaggle dataset. After a source-backed organizer-permission update and advance authorization, select a CUDA GPU, configure the single cell, and choose **Run all**. No manual Verify, Continue, Export or Cleanup is required. Current default settings stop safely and release Colab; they do not authorize any training.

The source archive is the exact immutable Git revision shown below. All inputs are restored locally; completed models are published directly to the persistent run directory. The isolated child uses Python **3.12.10**, the exact recorded dependency lock, FP32 heads, and no encoder training/extraction. Internet is required only to restore that pinned Python/dependency environment. A missing package/download blocks instead of changing versions.

Automatic order: validate → historical parity → representative benchmark/budget gate → 45 missing C0/C1/C2 fits → independent replay → CV decision → guarded final fits/confirmation only if preauthorized and CV_GO → reuse exact confirmed models → real input-only inference → ZIP validation → verified export → browser download requests → runtime release. NO_GO exports diagnostics and produces no promotion-ready submission. Production E50+L stays unchanged.

Limits: 1 GPU allocation-hour / 2 wall-hours, including setup since this Run all, plus the explicit persistence reserve. These are planning caps, **not measured EXP-023 runtimes**. Allocation before Run all is not observable here. A representative benchmark may stop for insufficient budget. Same-identity completed models resume; an interrupted current fit may restart. Source/runtime changes require a new run directory. Confirmation opens once after verified CV_GO, never during preparation or this notebook's default blocked run.

Downloads are requests only: browser arrival and physical runtime deallocation remain unverified. Drive retains verified archives even if the browser blocks a request. Logs and progress are visible below; durable outputs are under `PERSISTENT_ROOT`. No Codabench upload or automatic production promotion occurs.
"""
    if platform == "kaggle":
        text += "\n**Kaggle:** enable Internet and a GPU. No Colab APIs run here. `/kaggle/working` persists only for the live session unless captured by **Save Version / Save & Run All** and retained outputs. Unsaved-session recovery is not guaranteed. Use a private dataset; never publish the handoff.\n"
    def cell(kind, source, identity):
        return {"cell_type":kind,"id":identity,"metadata":{},"source":source.splitlines(keepends=True),
                **({"execution_count":None,"outputs":[]} if kind=="code" else {})}
    return {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"},"accelerator":"GPU"},
            "cells":[cell("markdown",text,"overview"),cell("code",configuration,"owner-configuration"),
                     cell("code",ORCHESTRATION,"one-run-entry"),cell("markdown","Inspect `export/final_report.json`, `outputs.zip` and the checksummed per-model artifacts. Colab may disconnect immediately after the final report/export. Historical EXP-022 is closed; this notebook does not resume it.","outputs")]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt",type=Path)
    args=parser.parse_args()
    receipt=json.loads(args.receipt.read_text())
    (ROOT/"notebooks").mkdir(exist_ok=True)
    for platform in ("colab","kaggle"):
        path=ROOT/"notebooks"/f"EXP023_{platform.title()}.ipynb"
        path.write_bytes((json.dumps(make_notebook(receipt,platform),indent=1)+"\n").encode())
        print(path)
