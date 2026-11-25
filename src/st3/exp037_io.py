"""Upload-safe EXP-037 input names and verified stage publication."""

import hashlib
import json
import re
from pathlib import PureWindowsPath
from zipfile import ZipFile

from .exp036_io import check, read, relative, sha


def safe_members(names):
    seen = set()
    directories = {}
    for name in names:
        path = relative(name)
        for part in path.parts:
            if (not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", part)
                    or part.endswith(".") or PureWindowsPath(part).is_reserved()):
                raise ValueError(f"Unsafe Kaggle member: {name}")
        for i in range(1, len(path.parts)):
            original = "/".join(path.parts[:i])
            key = original.casefold()
            if key in seen or directories.setdefault(key, original) != original:
                raise ValueError(f"Kaggle path collision: {name}")
        key = name.casefold()
        if key in seen or key in directories:
            raise ValueError(f"Kaggle path collision: {name}")
        seen.add(key)


def verify_inputs(root, expected):
    if sha(root / "exp037_manifest.json") != expected:
        raise ValueError("EXP-037 handoff manifest differs")
    manifest = read(root / "exp037_manifest.json")
    safe_members([*manifest["files"], "exp037_manifest.json"])
    check(root, manifest["files"])
    for logical, physical in manifest["images"].items():
        relative(logical)
        if physical != f"artwork/{manifest['files'][physical]}.jpg":
            raise ValueError(f"Artwork namespace differs for {logical}")
    return manifest


def unpack_inputs(archive, root, expected_manifest, expected_archive):
    if sha(archive) != expected_archive:
        raise ValueError("Private input ZIP checksum differs")
    with ZipFile(archive) as z:
        safe_members(z.namelist())
        content = z.read("exp037_inputs/exp037_manifest.json")
        if hashlib.sha256(content).hexdigest() != expected_manifest:
            raise ValueError("Input manifest checksum differs")
        manifest = json.loads(content)
        expected = {f"exp037_inputs/{name}" for name in [*manifest["files"], "exp037_manifest.json"]}
        if set(z.namelist()) != expected:
            raise ValueError("ZIP manifest coverage differs")
        z.extractall(root)
    return root / "exp037_inputs"
