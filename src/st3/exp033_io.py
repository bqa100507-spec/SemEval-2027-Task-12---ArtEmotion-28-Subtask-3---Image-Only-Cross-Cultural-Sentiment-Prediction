"""Checksum-verified inputs and atomic checkpoint boundaries for EXP-033."""

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import PurePosixPath

import pandas as pd
import torch

from .data import KEYS
from .submission import validate_predictions


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, allow_nan=False, indent=2) + "\n"


def atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    os.close(fd)
    try:
        write(name)
        os.replace(name, path)
    finally:
        if os.path.isfile(name):
            os.unlink(name)


def save_json(path, obj):
    def write(name):
        with open(name, "w", encoding="utf8", newline="\n") as stream:
            stream.write(canonical(obj))
    atomic(path, write)


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def safe_path(name):
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or p.as_posix() != name or ".." in p.parts or any(
        marker in name for marker in ("\\", ":", "\x00")
    ):
        raise ValueError(f"Unsafe artifact path {name!r}")
    return p


def restore_inputs(archive, destination, archive_sha, manifest_sha, *, verify_existing=True):
    if destination.exists() and verify_existing:
        return verify_inputs(destination, manifest_sha)
    if sha256(archive) != archive_sha:
        raise ValueError("Input ZIP SHA-256 mismatch")
    with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        manifest = source.read("manifest.json")
        if hashlib.sha256(manifest).hexdigest() != manifest_sha:
            raise ValueError("Input manifest SHA-256 mismatch")
        names = source.namelist()
        expected = json.loads(manifest)["files"]
        if len(names) != len(set(names)) or set(names) != set(expected) | {"manifest.json"}:
            raise ValueError("Unexpected ZIP contents")
        for name in names:
            safe_path(name)
        source.extractall(staging)
        from pathlib import Path
        staged = Path(staging)
        verify_inputs(staged, manifest_sha)
        staged.rename(destination)
    return verify_inputs(destination, manifest_sha)


def verify_inputs(root, pin):
    path = root / "manifest.json"
    if sha256(path) != pin:
        raise ValueError("Input manifest SHA-256 mismatch")
    manifest = read_json(path)
    actual = {x.relative_to(root).as_posix() for x in root.rglob("*") if x.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Extracted input inventory differs from its manifest")
    for name, expected in manifest["files"].items():
        local = root / safe_path(name)
        if local.is_symlink() or sha256(local) != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
    return manifest


def load_population(root, config):
    frame = pd.read_csv(root / "pairs.csv")
    fields = KEYS + ["fold", "image", "k", "n"]
    if list(frame) != fields or frame.isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("P2 pair schema or coverage invalid")
    if not frame["k"].between(0, frame["n"]).all() or not frame["n"].ge(5).all():
        raise ValueError("Invalid votes")
    if (frame[["k", "n"]].to_numpy() % 1 != 0).any():
        raise ValueError("Expected integer vote counts")
    images = frame[["painting", "image", "fold"]].drop_duplicates().sort_values("painting").reset_index(drop=True)
    if (images["painting"].duplicated().any() or images["image"].duplicated().any()
            or sorted(frame["fold"].unique()) != list(range(1, 6))
            or frame.groupby("fold").size().tolist() != config["fold_pairs"]
            or not images.groupby("fold").size().eq(config["fold_paintings"]).all()):
        raise ValueError("P2 fold or painting isolation mismatch")
    for path in images["image"]:
        if not safe_path(path).parts[0] == "images" or not (root / path).is_file():
            raise ValueError("Missing or unexpected painting image")
    references = {}
    for name, expected in config["reference_sha256"].items():
        path = root / ("oof_E50_L.csv" if name == "E50+L" else "oof_no_graph.csv")
        if sha256(path) != expected:
            raise ValueError(f"Historical OOF SHA-256 mismatch: {name}")
        references[name] = pd.read_csv(path)
        validate_predictions(references[name], frame)
    side = pd.read_csv(root / "ridge_side_heads.csv")
    for column in ("clip", "siglip", "large"):
        validate_predictions(side[KEYS + [column]].rename(columns={column: "p_positive"}), frame)
    ridge = fuse(side, side["siglip"]).rename(columns={"p_positive": "replay"})
    matched = references["ridge"].merge(ridge, on=KEYS, validate="one_to_one")
    if not matched["p_positive"].eq(matched["replay"]).all():
        raise ValueError("Reconstructed side-head blend differs from archived ridge OOF")
    return frame, images, references, side


def fuse(side, probabilities):
    intermediate = (.5 * side["clip"].to_numpy() + .5 * probabilities).round(6)
    return side[KEYS].assign(p_positive=(.5 * intermediate + .5 * side["large"].to_numpy()).round(6))


def publish_checkpoint(directory, identity, state):
    # Callers give each completed epoch its own directory. A seal is written last.
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "seal.json").exists():
        raise FileExistsError(f"Checkpoint already sealed: {directory}")
    path = directory / "state.pt"
    def write(name):
        with open(name, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
    atomic(path, write)
    save_json(directory / "seal.json", {"identity": identity, "sha256": sha256(path)})


def read_checkpoint(directory, identity):
    seal = read_json(directory / "seal.json")
    path = directory / "state.pt"
    if seal["identity"] != identity or seal["sha256"] != sha256(path):
        raise ValueError(f"Checkpoint identity or hash mismatch: {directory}")
    return torch.load(path, map_location="cpu", weights_only=True)
