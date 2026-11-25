"""Launch the verified EXP-037 worker in a pinned, session-scoped Kaggle environment."""

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
from st3.exp037_io import verify_inputs


def command(args, deadline, log, env):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Kaggle session allowance elapsed")
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


def launch(args, output, log):
    scratch = Path("/kaggle/temp/exp037")
    scratch.mkdir(parents=True, exist_ok=True)
    runtime = args.inputs / "runtime"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
               HF_HOME=str(scratch / "hf"), UV_CACHE_DIR=str(scratch / "uv"),
               UV_MANAGED_PYTHON="1")
    deadline = args.started + args.hours * 3600
    training_deadline = deadline - 240
    command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], training_deadline, log, env)
    bootstrap = scratch / "bootstrap"
    uv = bootstrap / "bin/uv"
    if not uv.exists():
        command([sys.executable, "-m", "pip", "install", "--target", str(bootstrap),
                 "--index-url", "https://pypi.org/simple", "uv==0.9.11"], training_deadline, log, env)
    venv = scratch / "venv"
    python = venv / "bin/python"
    command([str(uv), "venv", "--allow-existing", "--python", "3.12.10", str(venv)], training_deadline, log, env)
    command([str(uv), "pip", "sync", "--python", str(python), "--default-index", "https://pypi.org/simple",
             "--extra-index-url", "https://download.pytorch.org/whl/cu126", "--index-strategy", "unsafe-best-match",
             str(runtime / "requirements-exp035-kaggle.lock.txt"), str(runtime / "requirements-exp036-kaggle.txt")],
            training_deadline, log, env)
    command([str(uv), "pip", "check", "--python", str(python)], training_deadline, log, env)
    command([str(python), "-c", "import sys; assert sys.version_info[:3] == (3, 12, 10)"], training_deadline, log, env)
    worker = [str(python), "-u", str(runtime / "scripts/run_exp037.py"), "train",
              "--inputs", str(args.inputs), "--output", str(output / "exp037_run"),
              "--log", str(log), "--handoff", args.handoff, "--deadline", str(training_deadline)]
    for option in ("siglip", "qwen", "resume"):
        value = getattr(args, option)
        if value is not None:
            worker.extend((f"--{option}", str(value)))
    command(worker, deadline, log, env)


def main(args, output):
    verify_inputs(args.inputs, args.handoff)
    log = output / f"exp037_worker_{time.time_ns()}.log"
    try:
        launch(args, output, log)
    except Exception as error:
        status = output / "exp037_run/status.json"
        state = read(status) if status.is_file() else {}
        diagnostic = output / "exp037_diagnostic.json"
        write(diagnostic, {"worker_status": state, "launcher_error": str(error)})
        export(output / "exp037_diagnostic.zip", output, [diagnostic, *([log] if log.exists() else [])], args.handoff)
        if state.get("state") == "RESUMABLE_STOP" and (output / "exp037_results.zip").is_file():
            print("Session allowance reached. Publish or download exp037_results.zip and its checksum for resume.", flush=True)
        else:
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "siglip", "qwen", "resume"):
        p.add_argument(f"--{name}", type=Path, required=name == "inputs")
    p.add_argument("--handoff", required=True)
    p.add_argument("--started", type=float, required=True)
    p.add_argument("--hours", type=float, default=10.5)
    args = p.parse_args()
    if not Path("/kaggle/input").is_dir() or not .1 <= args.hours <= 11:
        p.error("Run inside Kaggle with a session allowance from 0.1 to 11 hours")
    output = Path("/kaggle/working")
    with (output / "exp037.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args, output)
