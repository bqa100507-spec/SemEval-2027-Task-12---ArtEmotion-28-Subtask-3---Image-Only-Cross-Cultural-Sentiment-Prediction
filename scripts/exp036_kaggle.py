"""One-session launcher. Scientific code stays in the pinned worker."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from st3.exp036_io import export, read, sha, write


def command(args, deadline, log, env):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Session budget elapsed")
    print("Running", " ".join(map(str, args[:3])), flush=True)
    with log.open("ab") as stream, subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
                                                     env=env, start_new_session=True) as child:
        try:
            result = child.wait(timeout=remaining)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    if result:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        raise RuntimeError(f"Exit {result}: " + "\n".join(tail))


def launch(args, output):
    scratch = Path("/kaggle/temp/exp036")
    scratch.mkdir(parents=True, exist_ok=True)
    runtime = args.inputs / "runtime"
    log = output / f"exp036_worker_{time.time_ns()}.log"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
               HF_HOME=str(scratch / "hf"), UV_CACHE_DIR=str(scratch / "uv"))
    deadline = args.started + args.hours * 3600
    try:
        command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], deadline-120, log, env)
        bootstrap = scratch / "bootstrap"
        uv = bootstrap / "bin/uv"
        if not uv.exists():
            command([sys.executable, "-m", "pip", "install", "--target", str(bootstrap),
                     "--index-url", "https://pypi.org/simple", "uv==0.9.11"], deadline-120, log, env)
        venv = scratch / "venv"
        python = venv / "bin/python"
        command([str(uv), "venv", "--allow-existing", "--python", "3.12.10", str(venv)], deadline-120, log, env)
        command([str(uv), "pip", "sync", "--python", str(python), "--default-index", "https://pypi.org/simple",
                 "--extra-index-url", "https://download.pytorch.org/whl/cu126", "--index-strategy", "unsafe-best-match",
                 str(runtime / "requirements-exp035-kaggle.lock.txt"),
                 str(runtime / "requirements-exp036-kaggle.txt")], deadline-120, log, env)
        command([str(uv), "pip", "check", "--python", str(python)], deadline-120, log, env)
        command([str(python), "-c", "import sys; assert sys.version_info[:3] == (3, 12, 10)"], deadline-120, log, env)
        worker = [str(python), "-u", str(runtime / "scripts/run_exp036.py"), "train", "--inputs", str(args.inputs),
                  "--output", str(output / "exp036_run"), "--log", str(log), "--handoff", args.handoff,
                  "--deadline", str(deadline-120)]
        for option in ("model", "resume"):
            value = getattr(args, option)
            if value is not None:
                worker.extend((f"--{option}", str(value)))
        command(worker, deadline, log, env)
    except Exception as exc:
        status = output / "exp036_run/status.json"
        state = read(status) if status.is_file() else {}
        if state.get("state") not in {"FAILED", "RESUMABLE_STOP"}:
            state = {"state": "LAUNCHER_STOP", "reason": str(exc)}
        print(state, flush=True)
        diagnostic = output / "exp036_diagnostic.json"
        write(diagnostic, state)
        report = output / "exp036_results.zip"
        if not report.exists():
            files = [diagnostic]
            if log.exists():
                files.append(log)
            export(output / "exp036_diagnostic.zip", output, files, args.handoff)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "model", "resume"):
        p.add_argument(f"--{name}", type=Path, required=name == "inputs")
    p.add_argument("--handoff", required=True)
    p.add_argument("--started", type=float, required=True)
    p.add_argument("--hours", type=float, default=10.5)
    args = p.parse_args()
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("This launcher runs in Kaggle")
    if sha(args.inputs / "manifest.json") != args.handoff or not 0.1 <= args.hours <= 11:
        raise ValueError("Handoff or session allowance is invalid")
    output = Path("/kaggle/working")
    with (output / "exp036.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch(args, output)
