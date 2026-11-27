#!/usr/bin/env python3
"""Linux CPU preflight of an exact Kaggle upload ZIP before handoff (workflow-standard pre-upload checklist).

Mounts the overlay ZIP twice, as an intact archive and as a simulated recursive Kaggle expansion, locates
each with the notebook's own ``locate`` cell, and runs ``cpu-validate`` from the runtime inside the ZIP.
Writes one receipt with the platform, the tolerance checks and both layouts. Run it on Linux, normally via
``scripts/linux_preflight.sh`` (pinned ``python:3.12.10`` image and the CPU lock).
"""

import argparse
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz", ".7z", ".bz2", ".xz", ".whl")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-zip", type=Path, required=True, help="The exact overlay ZIP that will be uploaded")
    parser.add_argument("--handoff", type=Path, required=True, help="exp038_handoff.json written by the packager")
    parser.add_argument("--base", type=Path, required=True, help="Base dataset: exp037_inputs.zip or its extracted root")
    parser.add_argument("--work", type=Path, required=True, help="Fresh scratch directory")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--allow-non-linux", action="store_true", help="Only for tests; the receipt says so")
    args = parser.parse_args(argv)
    for name in ("overlay_zip", "handoff", "base"):
        if not getattr(args, name).exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {getattr(args, name)}")
    handoff = json.loads(args.handoff.read_text(encoding="utf-8"))
    if sha(args.overlay_zip) != handoff["archive_sha256"]:
        parser.error("The overlay ZIP is not the archive recorded in the handoff")
    if args.work.exists() and any(args.work.iterdir()):
        parser.error("--work must be empty or absent")
    if platform.system() != "Linux" and not args.allow_non_linux:
        parser.error("Run the preflight on Linux (use scripts/linux_preflight.sh)")
    return args, handoff


def expand_recursively(target):
    """Replace every archive-like member by a directory of its contents until none is left."""
    while True:
        nested = [p for p in target.rglob("*") if p.is_file() and p.name.lower().endswith(ARCHIVE_SUFFIXES)
                  and zipfile.is_zipfile(p)]
        if not nested:
            return
        for path in nested:
            payload = path.read_bytes()
            path.unlink()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                archive.extractall(path)


def layouts(overlay_zip, work):
    intact = work / "intact" / "input" / "exp038-inputs"
    intact.mkdir(parents=True)
    shutil.copyfile(overlay_zip, intact / "exp038_inputs.zip")
    expanded = work / "recursive" / "input" / "exp038-inputs"
    with zipfile.ZipFile(overlay_zip) as archive:
        archive.extractall(expanded)
    expand_recursively(expanded)
    return {"intact": work / "intact" / "input", "recursive": work / "recursive" / "input"}


def base_root(base, work):
    if base.is_dir():
        return base
    target = work / "base"
    with zipfile.ZipFile(base) as archive:
        archive.extractall(target)
    return target / "exp037_inputs"


def notebook_locate():
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_exp038_notebook

    scope = {}
    exec(build_exp038_notebook.DEFS, scope)
    return scope["locate"]


def validate(root, base, handoff, output):
    command = [sys.executable, str(root / "runtime/scripts/run_exp038.py"), "cpu-validate", "--base", str(base),
               "--overlay", str(root), "--output", str(output), "--base-sha", handoff["base_manifest_sha256"],
               "--overlay-sha", handoff["manifest_sha256"]]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        return {"status": "FAILED", "error": (result.stdout + result.stderr)[-6000:]}
    receipt = json.loads((output / "cpu_validate.json").read_text(encoding="utf-8"))
    checks = receipt["r0_p2"]["checks"] + receipt["r0_development"]["checks"]
    return {"status": "PASS", "exact": receipt["exact"], "checks": len(checks),
            "exact_checks": sum(c["exact"] for c in checks),
            "max_ratio_to_threshold": max(c["max_abs_difference"] / c["threshold"] for c in checks),
            "platform": receipt["platform"], "pooled": receipt["r0_p2"]["pooled"]}


def main(argv=None, run_validate=validate):
    args, handoff = parse(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    base = base_root(args.base, args.work)
    locate = notebook_locate()
    out = {"overlay_zip_sha256": handoff["archive_sha256"], "overlay_manifest_sha256": handoff["manifest_sha256"],
           "source": handoff["source"], "os": platform.platform(), "linux": platform.system() == "Linux",
           "layouts": {}}
    for name, mount in layouts(args.overlay_zip, args.work).items():
        try:
            root, _ = locate(mount, args.work / f"{name}_zip", "exp038_manifest.json", handoff["manifest_sha256"],
                             "exp038_inputs.zip", handoff["archive_sha256"], "exp038_inputs")
        except Exception as error:
            out["layouts"][name] = {"status": "FAILED", "error": f"locate: {type(error).__name__}: {error}"}
            continue
        out["layouts"][name] = run_validate(root, base, handoff, args.work / f"{name}_validate")
    out["status"] = "PASS" if all(v["status"] == "PASS" for v in out["layouts"].values()) else "FAILED"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("status", "os")} | {n: v["status"] for n, v in out["layouts"].items()}))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
