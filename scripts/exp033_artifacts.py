"""Export and verify Kaggle handoffs independently of the training process."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import zipfile

import json


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(archive, target=None):
    with zipfile.ZipFile(archive) as stream:
        files = json.loads(stream.read("artifact_manifest.json"))
        names = stream.namelist()
        if len(names) != len(set(names)) or set(names) != set(files) | {"artifact_manifest.json"}:
            raise ValueError("Invalid artifact archive inventory")
        for name, expected in files.items():
            path = Path(name)
            if (name.startswith("/") or "\\" in name or ":" in name or ".." in path.parts
                    or path.as_posix() != name):
                raise ValueError("Unsafe artifact member")
            with stream.open(name) as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                    raise ValueError(f"Artifact checksum mismatch: {name}")
        if target is not None:
            for name in files:
                stream.extract(name, target)


def export(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    members = {p.relative_to(source).as_posix(): p for p in source.rglob("*")
               if p.is_file() and not p.name.startswith(".partial-")}
    hashes = {name: digest(p) for name, p in sorted(members.items())}
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        stage = Path(staging) / "written.zip"
        with zipfile.ZipFile(stage, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
            for name, path in members.items():
                stream.write(path, name)
            stream.writestr("artifact_manifest.json", json.dumps(hashes, sort_keys=True))
        verify(stage)
        stage.rename(destination)
    destination.with_suffix(".sha256").write_text(digest(destination) + "\n", encoding="ascii")


def restore(archive, target, fingerprint):
    if digest(archive) != fingerprint:
        raise ValueError("Published archive SHA-256 mismatch")
    if target.exists():
        raise FileExistsError(f"Restore destination must be new: {target}")
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        folder = Path(staging) / "complete"
        folder.mkdir()
        verify(archive, folder)
        folder.rename(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "restore"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sha")
    args = parser.parse_args()
    if args.stage == "export":
        export(args.source, args.destination)
    else:
        restore(args.source, args.destination, args.sha)
