"""Launch the verified EXP-038 worker in a pinned, session-scoped Kaggle environment.

Kaggle never imports or calls Colab APIs. Outputs survive only in a saved version or a download.
"""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from st3.exp036_io import export, read, write
from st3.exp038_io import verify_both


def command(args, deadline, log, env):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Kaggle session allowance elapsed during setup")
    print("Running", *args[:3], flush=True)
    with log.open("a", encoding="utf-8") as stream, subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        text=True, encoding="utf-8", errors="replace",
    ) as child:
        def relay():
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)

        reader = Thread(target=relay)
        reader.start()
        try:
            result = child.wait(timeout=remaining)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            reader.join()
    if result:
        raise RuntimeError(f"{Path(args[0]).name} exited with status {result}. See {log.name}.")


def environment(scratch):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    # The first GPU only; Haswell OpenBLAS kernels and two ridge threads match the machine that produced the
    # Track 1 and Corrected Ridge reference artifacts, so the CPU replay is bit-comparable.
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", OPENBLAS_CORETYPE="Haswell",
               HF_HOME=str(scratch / "hf"), UV_CACHE_DIR=str(scratch / "uv"), UV_MANAGED_PYTHON="1",
               TOKENIZERS_PARALLELISM="false")
    return env


def launch(args, output, log):
    scratch = Path("/kaggle/temp/exp038")
    scratch.mkdir(parents=True, exist_ok=True)
    runtime = args.overlay / "runtime"
    env = environment(scratch)
    deadline = args.started + args.hours * 3600
    setup_deadline = deadline - 1200
    command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], setup_deadline, log, env)
    bootstrap = scratch / "bootstrap"
    uv = bootstrap / "bin/uv"
    if not uv.exists():
        command([sys.executable, "-m", "pip", "install", "--target", str(bootstrap),
                 "--index-url", "https://pypi.org/simple", "uv==0.9.11"], setup_deadline, log, env)
    venv = scratch / "venv"
    python = venv / "bin/python"
    command([str(uv), "venv", "--allow-existing", "--python", "3.12.10", str(venv)], setup_deadline, log, env)
    command([str(uv), "pip", "sync", "--python", str(python), "--default-index", "https://pypi.org/simple",
             "--extra-index-url", "https://download.pytorch.org/whl/cu126", "--index-strategy", "unsafe-best-match",
             str(runtime / "requirements-exp035-kaggle.lock.txt"), str(runtime / "requirements-exp036-kaggle.txt"),
             str(runtime / "requirements-exp038-kaggle.txt")], setup_deadline, log, env)
    command([str(uv), "pip", "check", "--python", str(python)], setup_deadline, log, env)
    command([str(python), "-c", "import sys; assert sys.version_info[:3] == (3, 12, 10)"], setup_deadline, log, env)
    worker = [str(python), "-u", str(runtime / "scripts/run_exp038.py"), "run", "--base", str(args.base),
              "--overlay", str(args.overlay), "--output", str(output), "--base-sha", args.base_sha,
              "--overlay-sha", args.overlay_sha, "--started", str(args.started), "--hours", str(args.hours)]
    for option in ("resume", "e1_snapshot", "e2_snapshot", "e3_snapshot"):
        value = getattr(args, option)
        if value is not None:
            worker.extend((f"--{option.replace('_', '-')}", str(value)))
    command(worker, deadline, log, env)


def main(args, output):
    verify_both(args.base, args.overlay, args.base_sha, args.overlay_sha)
    log = output / f"exp038_launcher_{time.time_ns()}.log"
    try:
        launch(args, output, log)
    except Exception as error:
        status = output / "exp038_run/status.json"
        state = read(status) if status.is_file() else {}
        diagnostic = output / "exp038_launcher_diagnostic.json"
        write(diagnostic, {"worker_status": state, "launcher_error": str(error), "submission_claimed": False})
        files = [diagnostic, *([log] if log.exists() else [])]
        if not (output / "exp038_diagnostic.zip").is_file():
            export(output / "exp038_diagnostic.zip", output, files, args.overlay_sha)
        if state.get("state") == "RESUMABLE_STOP" and (output / "exp038_progress.zip").is_file():
            print("Session allowance reached. Save exp038_progress.zip and its checksum, then resume.", flush=True)
        else:
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "overlay"):
        p.add_argument(f"--{name}", type=Path, required=True)
    for name in ("resume", "e1-snapshot", "e2-snapshot", "e3-snapshot"):
        p.add_argument(f"--{name}", type=Path)
    for name in ("base-sha", "overlay-sha"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--started", type=float, required=True)
    p.add_argument("--hours", type=float, default=10.5)
    args = p.parse_args()
    if not Path("/kaggle/input").is_dir() or not .1 <= args.hours <= 11:
        p.error("Run inside Kaggle with a session allowance from 0.1 to 11 hours")
    output = Path("/kaggle/working")
    with (output / "exp038.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args, output)
