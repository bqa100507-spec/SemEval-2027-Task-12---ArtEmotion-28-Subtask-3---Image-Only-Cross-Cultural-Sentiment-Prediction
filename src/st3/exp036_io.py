"""Checksum-pinned input and output bundles for the EXP-036 workflow."""

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZIP_STORED, ZipFile


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def relative(name):
    p = PurePosixPath(name)
    if not p.parts or p.as_posix() != name or p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError(f"Invalid artifact member {name}")
    return p


def check(root, files):
    for name, digest in files.items():
        p = root / relative(name)
        if not p.is_file() or sha(p) != digest:
            raise ValueError(f"Missing or corrupt artifact: {p}")


def export(destination, root, files, identity):
    """Publish only explicit files. A readback validates the payload before rename."""
    files = {path.relative_to(root).as_posix(): sha(path) for path in sorted(set(files))}
    scratch = destination.with_suffix(".tmp")
    with ZipFile(scratch, "w", compression=ZIP_STORED) as archive:
        archive.writestr("export_manifest.json", json.dumps({"identity": identity, "files": files}, indent=2))
        for member in files:
            archive.write(root / member, member)
    with ZipFile(scratch) as archive:
        if sorted(archive.namelist()) != sorted([*files, "export_manifest.json"]):
            raise ValueError("Archive coverage changed")
        for name, digest in files.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Failed export verification: {name}")
    scratch.replace(destination)
    receipt = destination.with_suffix(".zip.sha256")
    receipt.write_text(f"{sha(destination)}  {destination.name}\n", encoding="ascii")


def restore(source, root, identity):
    if any(root.iterdir()):
        raise ValueError("Restore an archive into a fresh run directory")
    if source.is_dir():
        manifest = read(source / "export_manifest.json")
        if manifest["identity"] != identity:
            raise ValueError("Resume input belongs to a different handoff")
        check(source, manifest["files"])
        for name in manifest["files"]:
            target = root / relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        return
    with ZipFile(source) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        files = manifest["files"]
        if manifest["identity"] != identity or sorted(archive.namelist()) != sorted([*files, "export_manifest.json"]):
            raise ValueError("Resume ZIP identity or content differs")
        for name, digest in files.items():
            relative(name)
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Corrupt resume file: {name}")
        for name in files:
            archive.extract(name, root)
