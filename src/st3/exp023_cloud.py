"""Stdlib-only supervisor: bootstrap, one child pipeline, verified export/download/finally release."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import threading
import traceback
import zipfile

from .experiment_workflow import (Budget, StopRun, finalize, restore_handoff, safe_member, sha256, write_json,
                                  execution_scope, eligibility_evidence, COMPETITION_READY)


def child(command, *, log, deadline, env=None):
    """Hard parent deadline includes imports/CPU fits; output survives controlled termination."""
    remaining = deadline - time.time()
    if remaining <= 0:
        raise StopRun("STOPPED_BUDGET", "No time remains before the persistence reserve")
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                   text=True, encoding="utf-8", errors="replace", start_new_session=os.name != "nt")
        def tee():
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
        reader = threading.Thread(target=tee, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True)
                process.wait()
            reader.join(timeout=5)
            raise StopRun("STOPPED_BUDGET", "Child deadline reached; completed models persist, interrupted fit may restart")
        except BaseException:
            # A user interrupt must stop the child before export/release as well.
            if process.poll() is None:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            reader.join(timeout=5)
            raise
        reader.join(timeout=5)
    if code:
        raise StopRun("FAILED_TRAINING", f"Stage exited {code}; inspect {log}")


def restore_source(inputs, local, manifest):
    source = Path(local) / "source"
    if source.exists():
        for name, digest in manifest["source_files_sha256"].items():
            if not (source / name).is_file() or sha256(source / name) != digest:
                raise StopRun("BLOCKED_INPUT", "Existing local source changed; use a new local workspace")
        return source
    source.mkdir(parents=True)
    with zipfile.ZipFile(Path(inputs) / "source/source.zip") as z:
        for name in z.namelist():
            if not name.endswith("/"):
                safe_member(name)
        z.extractall(source)
    for name, digest in manifest["source_files_sha256"].items():
        if sha256(source / name) != digest:
            raise StopRun("BLOCKED_INPUT", "Pinned restored source checksum differs")
    return source


def bootstrap(source, local, deadline):
    """Pinned CPython and full known L4 lock. Download failures never change versions."""
    local = Path(local)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(local / "python"), UV_CACHE_DIR=str(local / "uv_cache"))
    tools = local / "bootstrap_tools"
    if not (tools / "uv").exists():
        child([sys.executable, "-m", "pip", "install", "--target", str(tools), "uv==0.7.13"],
              log=local / "bootstrap.log", deadline=deadline)
    env["PYTHONPATH"] = str(tools)
    uv = [sys.executable, "-m", "uv"]
    child([*uv, "python", "install", "3.12.10"], log=local / "bootstrap.log", deadline=deadline, env=env)
    venv = local / "venv"
    if not venv.exists():
        child([*uv, "venv", "--python", "3.12.10", str(venv)], log=local / "bootstrap.log", deadline=deadline, env=env)
    python = venv / "bin/python"
    lock = source / "requirements-exp023.lock.txt"
    receipt = local / "installed_lock.sha256"
    if not receipt.exists() or receipt.read_text().strip() != sha256(lock):
        child([*uv, "pip", "sync", "--python", str(python), "--extra-index-url", "https://download.pytorch.org/whl/cu126",
               "--index-strategy", "unsafe-best-match", str(lock)],
              log=local / "bootstrap.log", deadline=deadline, env=env)
        receipt.write_text(sha256(lock))
    child([*uv, "pip", "check", "--python", str(python)], log=local / "bootstrap.log", deadline=deadline, env=env)
    return python


def run_cloud(settings, *, platform, mount=None, download=None, release=None, started=None):
    """One callable for Colab and Kaggle. Platform APIs are injected by the notebook."""
    started = time.time() if started is None else started
    local = Path(settings["LOCAL_ROOT"])
    persistent = Path(settings["PERSISTENT_ROOT"])
    local.mkdir(parents=True, exist_ok=True)
    report = {"experiment": "EXP-023", "status": "BLOCKED_INPUT", "production": "E50+L", "platform": platform,
              "development_gold_read": False, "confirmation_labels_opened": False, "promotion": False,
              "competition_eligibility": "NOT_ESTABLISHED", "development_submission_created": False}
    run_dir, destination = local / "bootstrap_diagnostics", persistent / "export"
    persistence_available = platform != "colab"
    try:
        if platform == "colab":
            mount()
            persistence_available = True
        persistent.mkdir(parents=True, exist_ok=True)
        run_dir = persistent / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        scope = execution_scope(settings)
        report["execution_scope"] = scope
        budget = Budget(settings["MAX_GPU_HOURS"], settings["MAX_TOTAL_WALL_HOURS"], cuda=True, started=started, reserve=180)
        # Check authorizations and source-backed eligibility before a model/GPU child exists.
        if not settings["AUTHORIZE_EXPERIMENT"]:
            raise StopRun("BLOCKED_AUTHORIZATION", "EXP-023 is not authorized. Eligibility and one advance run authorization are required")
        if settings["MAX_GPU_HOURS"] > 1.0 or settings["MAX_TOTAL_WALL_HOURS"] > 2.0:
            raise StopRun("STOPPED_BUDGET", "Frozen resource caps are 1 GPU allocation-hour / 2 wall-hours")
        inputs = local / "inputs"
        manifest = restore_handoff(settings["INPUT_ZIP"], inputs, settings["ZIP_SHA256"], settings["MANIFEST_SHA256"])
        if manifest["source_revision"] != settings["SOURCE_REVISION"]:
            raise StopRun("BLOCKED_INPUT", "Notebook source revision differs from handoff")
        source = restore_source(inputs, local, manifest)
        # Competition permission gates only competition scope. Research still requires
        # advance GPU authorization, pinned official artifacts, compliance and budget checks.
        if scope == COMPETITION_READY:
            evidence, _ = eligibility_evidence(settings, source / "configs/exp023_eligibility.json")
            if evidence["status"] != "VERIFIED_ORGANIZER_PERMISSION" or settings["CAPTION_ELIGIBILITY_STATUS"] != "VERIFIED":
                raise StopRun("BLOCKED_ELIGIBILITY", "Organizer permission remains unresolved; owner status cannot replace source evidence")
        if settings["REQUIRE_SUBMISSION_ZIP"] and not settings["AUTHORIZE_GUARDED_CONFIRMATION"]:
            raise StopRun("BLOCKED_AUTHORIZATION", "Include guarded confirmation in initial authorization for submission scope")
        budget.check()
        python = bootstrap(source, local, started + budget.seconds - budget.reserve)
        compliance_env = dict(os.environ, PYTHONPATH=str(source / "src"), CUDA_VISIBLE_DEVICES="-1")
        child([str(python), "-m", "pytest", "-q", str(source / "tests/test_experiment_workflow.py"),
               str(source / "tests/test_exp023.py"), str(source / "tests/test_exp023_notebooks.py"),
               str(source / "tests/test_exp023_scopes.py")],
              log=run_dir / f"workflow_compliance_{time.time_ns()}.log", deadline=started + budget.seconds - budget.reserve,
              env=compliance_env)
        write_json(run_dir / "workflow_compliance.json", {"status": "PASS", "CUDA_VISIBLE_DEVICES": "-1", "source_revision": settings["SOURCE_REVISION"]})
        authorization = run_dir / "authorization.json"
        write_json(authorization, {key: value for key, value in settings.items() if key.isupper()})
        write_json(run_dir / "source_receipt.json", {"revision": settings["SOURCE_REVISION"], "input_manifest_sha256": settings["MANIFEST_SHA256"],
                                                    "zip_sha256": settings["ZIP_SHA256"]})
        env = dict(os.environ, PYTHONPATH=str(source / "src"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        command = [str(python), "-u", str(source / "scripts/run_exp023.py"), "--inputs", str(inputs), "--run-dir", str(run_dir),
                   "--authorization", str(authorization), "--started-at", str(started), "--device", "cuda"]
        child_report = run_dir / "child_report.json"
        write_json(child_report, {"status": "RUNNING", "previous_report_revoked": True})
        try:
            child(command, log=run_dir / f"pipeline_{time.time_ns()}.log", deadline=started + budget.seconds - budget.reserve, env=env)
        except StopRun:
            if child_report.exists():
                saved = json.loads(child_report.read_text())
                if saved.get("status") != "RUNNING":
                    report.update(saved)
            raise
        saved = json.loads(child_report.read_text())
        if saved.get("status") == "RUNNING":
            raise StopRun("FAILED_VERIFICATION", "Child returned without an atomic final report")
        report.update(saved)
    except StopRun as exc:
        if report["status"] in ("BLOCKED_INPUT", "RUNNING") or exc.status == "STOPPED_BUDGET":
            report.update(status=exc.status, error=str(exc))
    except Exception as exc:
        report.update(status="BLOCKED_INPUT", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "supervisor_report.json", report)
            if (local / "bootstrap.log").exists():
                shutil.copyfile(local / "bootstrap.log", run_dir / "bootstrap.log")
        except Exception as exc:
            report["diagnostic_persistence_error"] = str(exc)
        # finalize has its own nested finally; export/report/download failure cannot suppress release.
        options = {}
        if not persistence_available:
            from .experiment_workflow import export_artifacts
            def unavailable(*args):
                raise OSError("Drive mount failed; persistent export is unavailable. Local diagnostics only.")
            options["exporter"] = unavailable
        report = finalize(run_dir, destination, report, platform=platform, download=download, release=release,
                          auto_download=settings["AUTO_BROWSER_DOWNLOAD"], auto_release=settings["AUTO_RELEASE_COLAB"], **options)
    print(json.dumps(report, indent=2))
    return report
