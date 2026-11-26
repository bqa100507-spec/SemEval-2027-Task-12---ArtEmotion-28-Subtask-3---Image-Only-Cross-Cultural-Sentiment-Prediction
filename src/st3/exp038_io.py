"""EXP-038 input verification, sealed caches, budget planning and verified export."""

import hashlib
import json
import shutil
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import data, submission
from .exp036_io import check, export, read, relative, sha, write
from .exp037_io import safe_members, verify_inputs

OVERLAY_MANIFEST = "exp038_manifest.json"
BASE_MANIFEST = "exp037_manifest.json"
ARCHIVE_PREFIX = "exp038_inputs/"


class BlockedCache(RuntimeError):
    """A completed cache exists but is corrupt, stale or incomplete. It is never silently recomputed."""


class BudgetStop(TimeoutError):
    """A declared budget cap was reached for one optional piece."""


# Inputs.

def verify_overlay(root, expected_manifest, base_manifest_sha):
    if sha(root / OVERLAY_MANIFEST) != expected_manifest:
        raise ValueError("EXP-038 overlay manifest checksum differs")
    manifest = read(root / OVERLAY_MANIFEST)
    safe_members([*manifest["files"], OVERLAY_MANIFEST])
    check(root, manifest["files"])
    if manifest["base"]["manifest_sha256"] != base_manifest_sha:
        raise ValueError("The overlay was prepared for a different EXP-037 base dataset")
    return manifest


def unpack_overlay(archive, destination, expected_manifest, expected_archive):
    if sha(archive) != expected_archive:
        raise ValueError("Overlay ZIP checksum differs")
    with ZipFile(archive) as z:
        safe_members(z.namelist())
        content = z.read(ARCHIVE_PREFIX + OVERLAY_MANIFEST)
        if hashlib.sha256(content).hexdigest() != expected_manifest:
            raise ValueError("Overlay manifest checksum differs")
        manifest = json.loads(content)
        expected = {ARCHIVE_PREFIX + name for name in [*manifest["files"], OVERLAY_MANIFEST]}
        if set(z.namelist()) != expected:
            raise ValueError("Overlay ZIP coverage differs from its manifest")
        z.extractall(destination)
    return destination / ARCHIVE_PREFIX.rstrip("/")


def verify_both(base, overlay, base_manifest_sha, overlay_manifest_sha):
    base_manifest = verify_inputs(base, base_manifest_sha)
    overlay_manifest = verify_overlay(overlay, overlay_manifest_sha, base_manifest_sha)
    return base_manifest, overlay_manifest


def artwork(base, manifest, logical):
    try:
        physical = manifest["images"][logical]
    except KeyError:
        raise ValueError(f"Missing artwork mapping for {logical}") from None
    path = base / relative(physical)
    if not path.is_file():
        raise ValueError(f"Missing artwork file for {logical}")
    return path, Path(physical).stem


def training_population(base, expected_pair_sha):
    """Official Train pairs exactly as Track 1 and Corrected Ridge fitted them."""
    from .competition_challenger import frame_sha

    annotations = pd.read_csv(base / "training/annotations.csv", dtype=str, keep_default_na=False)
    if set(annotations.split) != {"train"}:
        raise ValueError("Training annotations contain a non-training split")
    pairs = data.training_pairs(annotations[data.ANNOTATION_COLUMNS]).sort_values(data.KEYS, ignore_index=True)
    if frame_sha(pairs[data.KEYS + ["image", "k", "n"]]) != expected_pair_sha:
        raise ValueError("Official training pairs differ from the Corrected Ridge training population")
    development = data.read_dev_input(base)
    if set(pairs.painting) & set(development.painting) or set(pairs.image) & set(development.image):
        raise ValueError("Training and Development overlap")
    return pairs, development


def p2_population(overlay, train, references):
    protocol_path, pairs_path = overlay / "p2/protocol.csv", overlay / "p2/pairs.csv"
    if sha(protocol_path) != references["p2_protocol_sha256"] or sha(pairs_path) != references["p2_pairs_sha256"]:
        raise ValueError("The pinned P2 fold assignment differs")
    protocol = pd.read_csv(protocol_path)
    gold = pd.read_csv(pairs_path).sort_values(data.KEYS, ignore_index=True)
    data.unique_pairs(gold)
    folds = protocol.loc[protocol.role.eq("development")].set_index("painting").fold
    if (set(gold.painting) != set(folds.index) or not gold.painting.map(folds).eq(gold.fold).all()
            or sorted(gold.fold.unique()) != [1, 2, 3, 4, 5] or not gold.split.eq("train").all()):
        raise ValueError("P2 pairs differ from the pinned painting-to-fold assignment")
    official = gold[data.KEYS].merge(train[data.KEYS + ["k", "n"]], on=data.KEYS, how="left", validate="one_to_one")
    if official.k.isna().any() or not np.array_equal(official.k.to_numpy(), gold.k.to_numpy()) \
            or not np.array_equal(official.n.to_numpy(), gold.n.to_numpy()):
        raise ValueError("P2 votes differ from the official training annotations")
    holdout = set(protocol.loc[protocol.role.eq("holdout"), "painting"])
    if holdout & set(gold.painting):
        raise ValueError("A sealed P2 holdout painting is in the P2 population")
    return gold[data.KEYS + ["fold", "k", "n"]], protocol


def base_features(overlay, paintings, widths):
    """Frozen CLIP-B / SigLIP2-Base / CLIP-L features for every Train and Development painting."""
    with np.load(overlay / "features/base_features.npz", allow_pickle=False) as cache:
        digests = cache["image_sha256"].tolist()
        index = {digest: i for i, digest in enumerate(digests)}
        if len(index) != len(digests):
            raise ValueError("Duplicate base feature image")
        missing = set(paintings.image_sha256) - set(index)
        if missing:
            raise ValueError(f"Base features missing for {len(missing)} images")
        rows = paintings.image_sha256.map(index).to_numpy()
        features = {}
        for name, width in widths.items():
            values = cache[name][rows]
            if values.shape != (len(paintings), width) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"Invalid base features for {name}")
            features[name] = pd.DataFrame(values, index=paintings.painting.to_numpy())
    return features


def painting_images(base, manifest, train, development):
    frames = [train[["painting", "image"]], development[["painting", "image"]]]
    paintings = pd.concat(frames).drop_duplicates().sort_values("painting", ignore_index=True)
    if paintings.painting.duplicated().any():
        raise ValueError("A painting maps to several images")
    paintings["image_sha256"] = [artwork(base, manifest, name)[1] for name in paintings.image]
    return paintings


# Sealed caches.

def seal(folder, identity, extra=None):
    files = {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name != "seal.json" and not p.name.endswith(".tmp")}
    write(folder / "seal.json", {"identity": identity, "files": files} | (extra or {}))


def verify_seal(folder, identity, required=()):
    path = folder / "seal.json"
    if not path.is_file():
        raise BlockedCache(f"Incomplete stage without a seal: {folder}")
    try:
        meta = read(path)
    except (OSError, ValueError) as error:
        raise BlockedCache(f"Unreadable seal at {folder}: {error}") from None
    if meta.get("identity") != identity:
        raise BlockedCache(f"Stale cache at {folder}: source, runtime or input identity changed. "
                           "Use a fresh run directory or the original runtime.")
    try:
        check(folder, meta["files"])
    except ValueError as error:
        raise BlockedCache(str(error)) from None
    if not set(required) <= set(meta["files"]):
        raise BlockedCache(f"Sealed stage at {folder} lacks {sorted(set(required) - set(meta['files']))}")
    present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} - {"seal.json"}
    if present - set(meta["files"]):
        raise BlockedCache(f"Unsealed extra files at {folder}: {sorted(present - set(meta['files']))[:3]}")
    return meta


def completed(folder, identity, required=()):
    """True for a valid sealed stage, False when absent, and BlockedCache otherwise."""
    if not folder.exists():
        return False
    if not (folder / "seal.json").is_file():
        raise BlockedCache(f"Incomplete stage without a seal: {folder}")
    verify_seal(folder, identity, required)
    return True


def save_features(folder, keys, values, identity, extra=None):
    """Write and seal one feature cache. ``keys`` are strings; ``values`` finite float32 rows."""
    values = np.asarray(values, dtype=np.float32)
    if len(keys) != len(values) or len(set(keys)) != len(keys) or not np.isfinite(values).all():
        raise ValueError("Feature cache rows must be finite and uniquely keyed")
    folder.mkdir(parents=True, exist_ok=True)
    scratch = folder / "features.tmp.npz"
    np.savez(scratch, keys=np.asarray(keys, dtype=str), values=values)
    scratch.replace(folder / "features.npz")
    seal(folder, identity, {"rows": len(keys), "width": int(values.shape[1]),
                            "keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest()} | (extra or {}))


def load_features(folder, identity, expected_keys):
    meta = verify_seal(folder, identity, ["features.npz"])
    with np.load(folder / "features.npz", allow_pickle=False) as cache:
        keys, values = cache["keys"].tolist(), cache["values"]
    if keys != list(expected_keys) or values.shape[0] != len(keys) or not np.isfinite(values).all():
        raise BlockedCache(f"Feature cache coverage differs at {folder}")
    if meta["keys_sha256"] != hashlib.sha256("\n".join(keys).encode()).hexdigest():
        raise BlockedCache(f"Feature cache key receipt differs at {folder}")
    return values


# Budget.

def plan_budget(forecast, remaining, budget):
    """Skip whole optional pieces in the declared order until the margin-scaled forecast fits."""
    margin = budget["forecast_margin"]
    skipped = {}
    if forecast.get("E4") is not None and forecast["E4"] > budget["e4_cap_seconds"]:
        skipped["E4"] = f"E4 forecast {forecast['E4']:.0f}s exceeds its {budget['e4_cap_seconds']}s cap"

    def total():
        return margin * sum(v for k, v in forecast.items() if k not in skipped)

    for piece in budget["skip_order"]:
        if total() <= remaining:
            break
        if piece in forecast and piece not in skipped:
            skipped[piece] = f"forecast {total():.0f}s exceeds remaining {remaining:.0f}s"
    return {"forecast_seconds": forecast, "margin": margin, "remaining_seconds": remaining,
            "planned_seconds": total(), "skipped": skipped,
            "kept": [k for k in forecast if k not in skipped],
            "mandatory_exceeds_budget": total() > remaining}


# Submissions and export.

def write_submission(frame, dev, folder, name):
    """Six-decimal CSV, top-level predictions.csv ZIP, and an exact round-trip check."""
    folder.mkdir(parents=True, exist_ok=True)
    directory = folder / name
    directory.mkdir(exist_ok=True)
    csv = directory / submission.FILENAME
    frame = frame[submission.COLUMNS]
    if not frame[data.KEYS].reset_index(drop=True).equals(dev[data.KEYS].reset_index(drop=True)):
        raise ValueError(f"{name} changed the official key order")
    q = frame.p_positive.to_numpy(dtype=float)
    if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any() or not np.array_equal(q, np.round(q, 6)):
        raise ValueError(f"{name} probabilities are not finite six-decimal values in [0, 1]")
    frame.to_csv(csv, index=False, lineterminator="\n", float_format="%.6f")
    archive = folder / f"{name}.zip"
    submission.pack(csv, archive, dev)
    restored = submission.validate_archive(archive, dev)
    if not np.array_equal(restored.p_positive.to_numpy(), submission.read_predictions(csv, dev).p_positive.to_numpy()):
        raise ValueError(f"{name} ZIP differs from its CSV")
    return archive


def zip_member(path):
    with ZipFile(path) as z:
        if z.namelist() != [submission.FILENAME]:
            raise ValueError(f"{path.name} must contain only top-level predictions.csv")
        return z.read(submission.FILENAME)


def readback(path, identity):
    """Second, independent read of a published archive and its checksum receipt."""
    receipt = path.with_suffix(".zip.sha256")
    digest = sha(path)
    if not receipt.is_file() or receipt.read_text(encoding="ascii").split()[0] != digest:
        raise ValueError(f"Checksum receipt differs for {path.name}")
    with ZipFile(path) as z:
        manifest = json.loads(z.read("export_manifest.json"))
        if manifest["identity"] != identity or sorted(z.namelist()) != sorted([*manifest["files"], "export_manifest.json"]):
            raise ValueError(f"Export coverage differs for {path.name}")
        for name, expected in manifest["files"].items():
            with z.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Export member corrupt after publication: {name}")
    return digest


def publish(destination, root, files, identity):
    export(destination, root, files, identity)
    return readback(destination, identity)


def copy_verified(source, destination):
    scratch = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, scratch)
    if sha(scratch) != sha(source):
        raise ValueError(f"Copy verification failed for {destination.name}")
    scratch.replace(destination)
    destination.with_name(destination.name + ".sha256").write_text(f"{sha(destination)}  {destination.name}\n",
                                                                    encoding="ascii")
    return sha(destination)


def exportable(root):
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not any(part.startswith(("tmp", ".")) for part in rel.parts) \
                and not p.name.endswith(".tmp") and ".tmp." not in p.name:
            yield p


class Clock:
    def __init__(self, started, hours, reserve):
        self.started = started
        self.deadline = started + hours * 3600
        self.reserve = reserve

    def remaining(self, now=None):
        return self.deadline - self.reserve - (time.time() if now is None else now)

    def work_deadline(self):
        return self.deadline - self.reserve
