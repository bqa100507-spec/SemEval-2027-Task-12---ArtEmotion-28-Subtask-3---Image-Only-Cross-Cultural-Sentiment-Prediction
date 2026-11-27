#!/usr/bin/env python3
"""Build the private EXP-039 overlay beside the existing EXP-037 base dataset (pre-upload checklist).

Contents: the EXP-038 overlay's references and P2/feature inputs (restored from its exact upload ZIP), the
sealed EXP-038 run-4 E1/E2/E3 caches, P2 OOF, Development predictions, CS-CVA-3 and incumbent ZIPs, and the
worker source at one commit. Every archive-like member is stored as .b64 text; the packager refuses any
archive extension or ZIP magic, checks every hash pin format, and verifies a full restore. No Evaluation data.
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.contrast_study import new_output
from st3.exp036_io import read, sha, write
from st3.exp037_io import safe_members
from st3.exp038_io import (archive_like, check_config_pins, encode_members, restore_overlay, unpack_overlay)

MANIFEST = "exp039_manifest.json"
PREFIX = "exp039_inputs/"
SOURCES = ["src/st3", "configs", "scripts", "requirements-exp035-kaggle.lock.txt", "requirements-exp036-kaggle.txt",
           "requirements-exp038-kaggle.txt", "requirements-exp038-cpu-preflight.txt"]
CARRIED = {"exp038/run.json": ("run.json", "run"),
           "exp038/p2_oof.csv": ("p2/oof.csv", "p2_oof"),
           "exp038/development_predictions.csv": ("development/predictions.csv", "development_predictions"),
           "exp038/CSCVA_3_submission.zip": ("development/submissions/exp038_CSCVA_3_submission.zip", "cscva3_zip"),
           "exp038/incumbent_submission.zip": ("development/submissions/exp038_B2_C2_cscva3_submission.zip",
                                               "incumbent_zip"),
           **{f"exp038/features/{arm}/{kind}": (f"features/{arm}/{kind}", f"feature_{arm}_{kind}")
              for arm in ("E1", "E2", "E3") for kind in ("features.npz", "seal.json")}}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def overlay_members(exp038_overlay, results_zip, runtime, references):
    """Logical members: EXP-038 overlay inputs (without its runtime and manifest), carried run-4 artifacts, runtime."""
    members = {}
    for path in sorted(Path(exp038_overlay).rglob("*")):
        rel = path.relative_to(exp038_overlay).as_posix()
        if path.is_file() and not rel.startswith("runtime/") and rel not in {"exp038_manifest.json",
                                                                              "restore_receipt.json"}:
            members[rel] = path.read_bytes()
    pins = references["exp038_carried_sha256"]
    with ZipFile(results_zip) as z:
        for target, (source, key) in CARRIED.items():
            content = z.read(source)
            if digest(content) != pins[key]:
                raise ValueError(f"EXP-038 artifact differs from its pin: {source}")
            members[target] = content
    for name, content in runtime.items():
        members[name] = content
    return members


def write_overlay(members, destination, manifest):
    stored, encoded = encode_members(members)
    files = {name: digest(content) for name, content in sorted(stored.items())}
    content = (json.dumps(manifest | {"files": files, "encoded": encoded}, indent=2, sort_keys=True) + "\n").encode("ascii")
    safe_members([*(PREFIX + name for name in stored), PREFIX + MANIFEST])
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for name, payload in sorted(stored.items()):
            z.writestr(PREFIX + name, payload)
        z.writestr(PREFIX + MANIFEST, content)
    with ZipFile(destination) as z:  # Recheck the whole written overlay, member by member.
        bad = [name for name in z.namelist() if archive_like(name, z.read(name))]
        if bad:
            raise ValueError(f"Overlay ZIP contains archive-like members: {bad[:5]}")
    return digest(content), encoded


def package(args):
    config = read(ROOT / "configs/exp039.json")
    check_config_pins(config)  # Refuse to build with a truncated or malformed hash pin.
    refs = config["references"]
    source = subprocess.check_output(["git", "rev-parse", f"{args.source}^{{commit}}"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "archive", "--format=zip", source, *SOURCES], cwd=ROOT)
    with ZipFile(io.BytesIO(tree)) as runtime:
        if json.loads(runtime.read("configs/exp039.json")) != config:
            raise ValueError("Committed EXP-039 config differs from the working copy")
        runtime_members = {f"runtime/{n}": runtime.read(n) for n in runtime.namelist() if not n.endswith("/")}
    if sha(args.exp038_results) != refs["exp038_results_zip_sha256"]:
        raise ValueError("EXP-038 run-4 results archive checksum differs")
    with ZipFile(args.exp038_results) as z:
        run38 = json.loads(z.read("run.json"))
    with tempfile.TemporaryDirectory() as scratch, new_output(args.output) as stage:
        mounted = unpack_overlay(args.exp038_overlay, Path(scratch) / "m38", run38["overlay_manifest_sha256"],
                                 sha(args.exp038_overlay))
        exp038 = restore_overlay(mounted, run38["overlay_manifest_sha256"], refs["base_manifest_sha256"],
                                 Path(scratch) / "r38")
        members = overlay_members(exp038, args.exp038_results, runtime_members, refs)
        manifest = {"experiment": "EXP-039", "source": source, "packaging": "patch-2 rules: archive-like members as .b64",
                    "base": {"zip_sha256": refs["base_inputs_zip_sha256"], "manifest_sha256": refs["base_manifest_sha256"]},
                    "references": refs, "exp038_overlay_zip_sha256": sha(args.exp038_overlay),
                    "population": read(exp038 / "exp038_manifest.json")["population"],
                    "models": {arm: [s.get("model"), s.get("revision"), s.get("license")]
                               for arm, s in config["arms"].items()}, "evaluation_data": "none"}
        archive = stage / "exp039_inputs.zip"
        manifest_sha, encoded = write_overlay(members, archive, manifest)
        with tempfile.TemporaryDirectory() as check:
            m39 = unpack_overlay(archive, Path(check) / "m", manifest_sha, sha(archive), manifest_name=MANIFEST,
                                 prefix=PREFIX)
            restored = restore_overlay(m39, manifest_sha, refs["base_manifest_sha256"], Path(check) / "r",
                                       manifest_name=MANIFEST)
            receipt = read(restored / "restore_receipt.json")["restored"]
        with ZipFile(archive) as z:
            (stage / MANIFEST).write_bytes(z.read(PREFIX + MANIFEST))
        write(stage / "exp039_handoff.json", {"archive_sha256": sha(archive), "manifest_sha256": manifest_sha,
                                              "archive_bytes": archive.stat().st_size, "source": source,
                                              "files": len(read_manifest(archive)["files"]),
                                              "restored_archives": receipt,
                                              "base_zip_sha256": refs["base_inputs_zip_sha256"],
                                              "base_manifest_sha256": refs["base_manifest_sha256"],
                                              "models": manifest["models"]})
    print(json.dumps({k: v for k, v in read(args.output / "exp039_handoff.json").items() if k != "restored_archives"},
                     indent=1))


def read_manifest(archive):
    with ZipFile(archive) as z:
        return json.loads(z.read(PREFIX + MANIFEST))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--exp038-results", type=Path, required=True)
    parser.add_argument("--exp038-overlay", type=Path, required=True, help="The exact EXP-038 run-4 overlay ZIP")
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args())
