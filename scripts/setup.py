#!/usr/bin/env python3
"""Bootstrap the pinned public ST3 checkout and an isolated Python 3.12 runtime."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "st3.json").read_text(encoding="utf-8"))
SPARSE_PATHS = ("/README.md", "/LICENSE", "/baselines/LICENSE", "/baselines/st3/",
                "/annotations/artelingo28_train_val.csv", "/images/",
                "/phase_inputs/dev/st3_test.csv", "/scripts/verify_dataset.py")
PIP_VERSION = "25.2"


def run(command, **kwargs):
    command = [str(part) for part in command]
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True, **kwargs)


def output(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def checkout_data(data_dir):
    revision = CONFIG["upstream_revision"]
    if not data_dir.exists():
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data_dir.parent) as scratch:
            stage = Path(scratch) / "upstream"
            run(["git", "clone", "--no-checkout", "--filter=blob:none", "--depth", "1", CONFIG["upstream_url"], stage])
            if output(["git", "rev-parse", "HEAD"], cwd=stage) != revision:
                run(["git", "fetch", "--depth", "1", "origin", revision], cwd=stage)
            run(["git", "sparse-checkout", "set", "--no-cone", *SPARSE_PATHS], cwd=stage)
            run(["git", "checkout", "--detach", revision], cwd=stage)
            stage.rename(data_dir)
    if not (data_dir / ".git").exists() or output(["git", "rev-parse", "HEAD"], cwd=data_dir) != revision:
        raise ValueError(f"{data_dir} must be a Git checkout of the pinned organizer revision {revision}")
    for name in ("annotations/artelingo28_train_val.csv", "phase_inputs/dev/st3_test.csv", "scripts/verify_dataset.py"):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(data_dir / name)
    if not (data_dir / "images").is_dir():
        raise FileNotFoundError(data_dir / "images")
    print(f"Verified upstream resources at {data_dir} ({revision})", flush=True)


def python312():
    if sys.version_info[:2] == (3, 12):
        return [sys.executable]
    if interpreter := shutil.which("python3.12"):
        return [interpreter]
    if os.name == "nt" and (launcher := shutil.which("py")):
        probe = subprocess.run([launcher, "-3.12", "--version"], capture_output=True)
        if probe.returncode != 0:
            run([launcher, "install", "3.12"])
        return [launcher, "-3.12"]
    raise ValueError("Python 3.12 is required. Run this script with python3.12.")


def installed_version(interpreter, package):
    code = ("from importlib.metadata import distributions; import sys; "
            "print(next((d.version for d in distributions(name=sys.argv[1])), ''))")
    return output([interpreter, "-c", code, package])


def cuda_driver_present():
    executable = shutil.which("nvidia-smi")
    return executable is not None and subprocess.run([executable, "-L"], capture_output=True).returncode == 0


def setup_environment(profile):
    venv_dir = ROOT / ".venv"
    interpreter = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.is_file():
        run([*python312(), "-m", "venv", venv_dir])
    if not output([interpreter, "--version"]).startswith("Python 3.12."):
        raise ValueError(f"{venv_dir} must use Python 3.12. Move the existing environment and rerun setup.")

    pip_tmp = ROOT / "cache" / "pip-tmp"
    pip_tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(pip_tmp), TMP=str(pip_tmp), TEMP=str(pip_tmp))
    pip = [interpreter, "-m", "pip", "install", "--disable-pip-version-check", "--timeout", "120"]
    if installed_version(interpreter, "pip") != PIP_VERSION:
        run([*pip, f"pip=={PIP_VERSION}"], env=env)
    pip.extend(["--resume-retries", "5", "--progress-bar", "off"])
    torch_version = CONFIG["torch_version"]
    installed = installed_version(interpreter, "torch")
    if profile == "auto":
        # Reuse either supported build on repeated setup. An explicit flag can switch it.
        selected = installed.split("+")[1] if installed in {f"{torch_version}+cpu", f"{torch_version}+cu126"} else None
        profile_to_install = selected or ("cu126" if cuda_driver_present() else "cpu")
    else:
        profile_to_install = profile

    def install_torch(build):
        requirement = f"torch=={torch_version}+{build}"
        if installed_version(interpreter, "torch") != requirement.removeprefix("torch=="):
            run([*pip, "--no-cache-dir", "--index-url", f"https://download.pytorch.org/whl/{build}", requirement], env=env)

    try:
        install_torch(profile_to_install)
    except subprocess.CalledProcessError:
        if profile != "auto" or profile_to_install != "cu126":
            raise
        print("CUDA wheel installation failed. Retrying setup with the pinned CPU wheel.", flush=True)
        install_torch("cpu")
    run([*pip, "-r", ROOT / "requirements.txt"], env=env)
    run([interpreter, "-m", "pip", "check"])
    run([interpreter, "-c", "import torch; print(f'PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}')"])
    return interpreter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "upstream")
    parser.add_argument("--torch", choices=("auto", "cpu", "cu126"), default="auto")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    checkout_data(data_dir)
    interpreter = setup_environment(args.torch)
    run([interpreter, ROOT / "scripts" / "run_st3.py", "--data-dir", data_dir, "audit", "--official",
         "--output", ROOT / "runs" / "data_audit.json"])
    print(f"Setup complete. Use {interpreter} to run scripts/run_st3.py.")


if __name__ == "__main__":
    main()
