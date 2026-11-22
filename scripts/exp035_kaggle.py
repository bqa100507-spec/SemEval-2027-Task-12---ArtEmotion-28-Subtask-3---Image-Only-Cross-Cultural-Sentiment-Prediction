"""Kaggle setup, budget and export for the existing EXP-035 worker (stdlib only)."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

WORK = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp/exp035")
RUN = WORK / "exp035_fold3"
CAP = 18_000
EXPORT_RESERVE = 120


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name or "\\" in name or ":" in name:
        raise ValueError(f"Invalid artifact path: {name!r}")
    return path


def run_command(command, deadline, log, env):
    remaining = deadline - time.time()
    if remaining <= 2:
        raise TimeoutError("EXP-035 budget reached before the next command")
    print("Running", Path(command[0]).name, str(command[1]), flush=True)
    with log.open("ab") as stream, subprocess.Popen(
        command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True
    ) as child:
        try:
            code = child.wait(timeout=remaining - 2)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    if code:
        raise RuntimeError(f"{Path(command[0]).name} exited {code}. See {log.name}.")


def restore_archive(archive, expected_handoff):
    if any(RUN.iterdir()):
        raise ValueError("Resume input requires a fresh Kaggle working directory")
    with ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("result_manifest.json"))
        files = manifest["files"]
        if manifest["handoff_sha256"] != expected_handoff or sorted(bundle.namelist()) != sorted(
                ["result_manifest.json", *files]):
            raise ValueError("Resume archive belongs to a different handoff or has unexpected members")
        for name, sha in files.items():
            relative(name)
            with bundle.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                    raise ValueError(f"Corrupt resume file: {name}")
        for name in files:
            bundle.extract(name, RUN)


def export(handoff_sha):
    # Omit any incomplete atomic-write directories and the lock.
    files = {}
    for path in sorted(RUN.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(RUN)
        if any(part.startswith("tmp") for part in name.parts) or path.suffix in {".tmp", ".lock"}:
            continue
        files[name.as_posix()] = digest(path)
    archive = WORK / "exp035_results.zip"
    temporary = archive.with_suffix(".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr("result_manifest.json", json.dumps({"handoff_sha256": handoff_sha, "files": files}, indent=2))
        for name in files:
            bundle.write(RUN / name, name)
    with ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Result archive failed its ZIP readback")
        for name, expected in files.items():
            with bundle.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Result archive checksum mismatch: {name}")
    temporary.replace(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest(archive)}  {archive.name}\n", encoding="utf-8")
    print(f"Verified archive: {archive}", flush=True)


def execute(args, started):
    handoff = args.input_root
    manifest = read(handoff / "exp035_handoff.json")
    runtime = handoff / "runtime"
    ledger = RUN / "gpu_budget.json"
    used = 0.
    charge = False
    log = RUN / f"kaggle_{time.time_ns()}.log"
    try:
        if args.resume:
            restore_archive(args.resume, args.handoff_sha)
        if ledger.exists():
            previous = read(ledger)
            if previous["platform"] != "kaggle" or previous["handoff_sha256"] != args.handoff_sha:
                raise ValueError("Budget ledger belongs to a different run")
            if previous["active_since"] is not None:
                raise ValueError("The saved lease was still active. Only a closed, exported attempt can be resumed.")
            used = float(previous["charged_seconds"])
            if not 0 <= used <= CAP:
                raise ValueError("Invalid accumulated GPU time")
        elif any(RUN.glob("*/epoch_*/seal.json")):
            raise ValueError("Existing checkpoints are missing the cumulative budget ledger")
        if (RUN / "results/seal.json").exists():
            for name, sha in read(RUN / "results/seal.json").items():
                if digest(RUN / "results" / relative(name)) != sha:
                    raise ValueError(f"Completed results failed verification: {name}")
            print("Reusing completed EXP-035 results.", flush=True)
            return
        charge = True
        deadline = started + CAP - used - EXPORT_RESERVE
        if deadline - time.time() < 30:
            raise TimeoutError("Less than 30 seconds of remaining execution budget")
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("This handoff requires Kaggle Linux x86-64")
        for name in manifest["model"]["files"]:
            if not (args.checkpoint / name).is_file():
                raise FileNotFoundError(f"Pinned Qwen file missing: {args.checkpoint / name}")
        RUN.mkdir(parents=True, exist_ok=True)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        if not (RUN / "prepared").exists():
            shutil.copytree(handoff / "prepared", RUN / "prepared")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_IMPLICIT_TOKEN="1", UV_CACHE_DIR=str(SCRATCH / "cache"))
        run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                    deadline, log, env)
        bootstrap = SCRATCH / "bootstrap"
        if not (bootstrap / "bin/uv").exists():
            run_command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--target",
                         str(bootstrap), "--index-url", "https://pypi.org/simple", "uv==0.9.11"],
                        deadline, log, env)
        uv = str(bootstrap / "bin/uv")
        python = SCRATCH / "venv/bin/python"
        lock = runtime / "requirements-exp035-kaggle.lock.txt"
        run_command([uv, "venv", "--allow-existing", "--python", "3.12.10", str(python.parents[1])],
                    deadline, log, env)
        run_command([uv, "pip", "sync", "--python", str(python), "--default-index", "https://pypi.org/simple",
                     "--extra-index-url", "https://download.pytorch.org/whl/cu126", "--index-strategy",
                     "unsafe-best-match", str(lock)], deadline, log, env)
        run_command([uv, "pip", "check", "--python", str(python)], deadline, log, env)
        subprocess.run([str(python), "-c", "import sys; assert sys.version_info[:3] == (3, 12, 10)"],
                       env=env, check=True, timeout=min(10, max(0, deadline-time.time())))
        write(RUN / "provenance.json", {"source": manifest["source"],
              "source_manifest_sha256": args.handoff_sha, "model": manifest["model"],
              "runtime": {"python": "3.12.10", "lock_sha256": digest(lock),
                          "linux": platform.platform(terse=True)}})
        for name in ("configs/exp035.json", "requirements-exp035-kaggle.lock.txt"):
            shutil.copyfile(runtime / name, RUN / Path(name).name)
        write(ledger, {"platform": "kaggle", "handoff_sha256": args.handoff_sha,
                       "charged_seconds": used, "active_since": started, "deadline": deadline})
        write(RUN / "status.json", {"state": "GPU_WORKER_STARTED", "deadline": deadline})
        run_command([str(python), "-u", str(runtime / "scripts/run_exp035.py"),
                     "--inputs", str(handoff / "p2"), "--reference", str(handoff / "ridge"),
                     "--data-dir", str(handoff / "artwork"), "--checkpoint", str(args.checkpoint),
                     "--run-dir", str(RUN), "--run-bounded-gpu", "--kaggle", "--worker"], deadline, log, env)
        print(json.dumps(read(RUN / "status.json"), indent=2), flush=True)
    except Exception as error:
        with log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        write(RUN / "status.json", {"state": "KAGGLE_STOPPED", "reason": str(error), "log": log.name})
        print(f"Kaggle stopped: {error}", flush=True)
    finally:
        if charge:
            # Count notebook setup and a conservative allowance for publishing the result.
            used = min(CAP, used + max(0., time.time() - started) + EXPORT_RESERVE)
            write(ledger, {"platform": "kaggle", "handoff_sha256": args.handoff_sha,
                           "charged_seconds": used, "active_since": None})
        export(args.handoff_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--handoff-sha", required=True)
    parser.add_argument("--started", required=True, type=float)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not Path("/kaggle/input").is_dir():
        raise RuntimeError("This launcher must run inside Kaggle")
    if digest(args.input_root / "exp035_handoff.json") != args.handoff_sha:
        raise ValueError("Kaggle input manifest SHA-256 mismatch")
    RUN.mkdir(parents=True, exist_ok=True)
    # Keep the session lock outside the resumable run directory.
    with (WORK / "exp035_session.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, args.started)


if __name__ == "__main__":
    main()
