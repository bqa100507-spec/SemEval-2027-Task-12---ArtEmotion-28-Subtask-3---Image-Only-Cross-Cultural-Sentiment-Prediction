#!/usr/bin/env python3
"""Build the private EXP-040 overlay beside the existing EXP-037 base dataset (pre-upload checklist).

Contents: every input member of the exact EXP-039 overlay (references, P2 folds, the sealed EXP-038 run-4
E1/E2/E3 caches, P2 OOF, Development predictions, CS-CVA-3 and incumbent ZIPs, the saved C6 predictions and the
E3 key-audit files fetched by code), without its runtime and the 8B audit files, plus the worker source at one
commit. Archive-like members are stored as .b64 text; the packager refuses archive extensions or ZIP magic,
checks every hash pin format, and verifies a full restore. No Evaluation data.
"""

import argparse
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from package_exp039 import digest  # noqa: E402
from st3.contrast_study import new_output  # noqa: E402
from st3.exp036_io import read, sha, write  # noqa: E402
from st3.exp037_io import safe_members  # noqa: E402
from st3.exp038_io import archive_like, check_config_pins, encode_members, restore_overlay, unpack_overlay  # noqa: E402

MANIFEST = "exp040_manifest.json"
PREFIX = "exp040_inputs/"
SOURCES = ["src/st3", "configs", "scripts", "requirements-exp035-kaggle.lock.txt", "requirements-exp036-kaggle.txt",
           "requirements-exp038-kaggle.txt", "requirements-exp038-cpu-preflight.txt"]
DROPPED = ("runtime/", "keys/E5_")
DROPPED_FILES = {"exp039_manifest.json", "restore_receipt.json"}


def carried_members(restored):
    members = {}
    for path in sorted(Path(restored).rglob("*")):
        rel = path.relative_to(restored).as_posix()
        if path.is_file() and not rel.startswith(DROPPED) and rel not in DROPPED_FILES:
            members[rel] = path.read_bytes()
    for required in ("keys/E3_config.json", "keys/E3_checkpoint_keys.json", "exp038/incumbent_submission.zip",
                     "exp038/features/E3/features.npz", "reference/exp037_c6_standalone/predictions.csv"):
        if required not in members:
            raise ValueError(f"EXP-039 overlay lacks {required}")
    return members


def write_overlay(members, destination, manifest):
    stored, encoded = encode_members(members)
    files = {name: digest(content) for name, content in sorted(stored.items())}
    content = (json.dumps(manifest | {"files": files, "encoded": encoded}, indent=2, sort_keys=True) + "\n").encode("ascii")
    safe_members([*(PREFIX + name for name in stored), PREFIX + MANIFEST])
    for name, payload in stored.items():
        if archive_like(name, payload):
            raise ValueError(f"Archive-like member survived encoding: {name}")
    from zipfile import ZIP_DEFLATED

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
    config = read(ROOT / "configs/exp040.json")
    check_config_pins(config)  # Refuse to build with a truncated or malformed hash pin.
    refs = config["references"]
    if sha(args.exp039_overlay) != refs["exp039_overlay_zip_sha256"]:
        raise ValueError("EXP-039 overlay ZIP differs from its pin")
    source = subprocess.check_output(["git", "rev-parse", f"{args.source}^{{commit}}"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "archive", "--format=zip", source, *SOURCES], cwd=ROOT)
    with ZipFile(io.BytesIO(tree)) as runtime:
        if json.loads(runtime.read("configs/exp040.json")) != config:
            raise ValueError("Committed EXP-040 config differs from the working copy")
        runtime_members = {f"runtime/{n}": runtime.read(n) for n in runtime.namelist() if not n.endswith("/")}
    with tempfile.TemporaryDirectory() as scratch, new_output(args.output) as stage:
        mounted = unpack_overlay(args.exp039_overlay, Path(scratch) / "m39", refs["exp039_overlay_manifest_sha256"],
                                 refs["exp039_overlay_zip_sha256"], manifest_name="exp039_manifest.json",
                                 prefix="exp039_inputs/")
        restored = restore_overlay(mounted, refs["exp039_overlay_manifest_sha256"], refs["base_manifest_sha256"],
                                   Path(scratch) / "r39", manifest_name="exp039_manifest.json")
        manifest39 = read(restored / "exp039_manifest.json")
        if {k: v for k, v in refs.items() if not k.startswith("exp039_overlay")} != manifest39["references"]:
            raise ValueError("EXP-040 reference pins differ from the EXP-039 overlay's")
        members = carried_members(restored) | runtime_members
        manifest = {"experiment": "EXP-040", "source": source, "packaging": "patch-2 rules: archive-like members as .b64",
                    "base": {"zip_sha256": refs["base_inputs_zip_sha256"], "manifest_sha256": refs["base_manifest_sha256"]},
                    "references": refs, "exp039_overlay_zip_sha256": sha(args.exp039_overlay),
                    "population": manifest39["population"],
                    "models": {arm: [s.get("model"), s.get("revision"), s.get("license")]
                               for arm, s in config["arms"].items()}, "evaluation_data": "none"}
        archive = stage / "exp040_inputs.zip"
        manifest_sha, encoded = write_overlay(members, archive, manifest)
        with tempfile.TemporaryDirectory() as check:
            m40 = unpack_overlay(archive, Path(check) / "m", manifest_sha, sha(archive), manifest_name=MANIFEST,
                                 prefix=PREFIX)
            restored40 = restore_overlay(m40, manifest_sha, refs["base_manifest_sha256"], Path(check) / "r",
                                         manifest_name=MANIFEST)
            receipt = read(restored40 / "restore_receipt.json")["restored"]
        with ZipFile(archive) as z:
            (stage / MANIFEST).write_bytes(z.read(PREFIX + MANIFEST))
            files = len(json.loads(z.read(PREFIX + MANIFEST))["files"])
        write(stage / "exp040_handoff.json", {"archive_sha256": sha(archive), "manifest_sha256": manifest_sha,
                                              "archive_bytes": archive.stat().st_size, "source": source,
                                              "files": files, "restored_archives": receipt,
                                              "base_zip_sha256": refs["base_inputs_zip_sha256"],
                                              "base_manifest_sha256": refs["base_manifest_sha256"],
                                              "models": manifest["models"]})
    print(json.dumps({k: v for k, v in read(args.output / "exp040_handoff.json").items() if k != "restored_archives"},
                     indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--exp039-overlay", type=Path, required=True, help="The exact EXP-039 overlay ZIP")
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args())
