"""Read-only P2 feature verification for the EXP-028 CPU screen."""
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .caption_data import inner_partition, require_isolation
from .vlm_semantics import file_hash, fingerprint

KEYS = ["painting", "language"]
IMAGE = ["painting", "image", "image_sha256", "fold"]
SOURCE_SHA = "551e2dd8471cb276bd8a6603c5c28b0a6ec6a47c971c9a52e423d3f0d8faaf24"
QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def read_protocol(root, inputs, *, labels=False):
    config = json.loads((root / "configs/exp026.json").read_text())
    names = ["training/pairs.csv", "training/protocol.csv"]
    if labels:
        names.append("history/exp022_R0.csv")
    for name in names:
        if file_hash(inputs / name) != config["input_sha256"][name]:
            raise ValueError(f"Pinned input SHA-256 differs for {name}")
    pairs = pd.read_csv(inputs / names[0], usecols=[*IMAGE, "language", "split", *(["k", "n"] if labels else [])])
    protocol = pd.read_csv(inputs / names[1], usecols=[*IMAGE, "role"])
    scope = protocol.loc[protocol.role.eq("development"), IMAGE].sort_values("painting").reset_index(drop=True)
    mapping = pairs[IMAGE].drop_duplicates().sort_values("painting").reset_index(drop=True)
    if (len(pairs) != 13600 or len(scope) != 1335 or pairs.language.nunique() != 25
            or not mapping.equals(scope) or pairs.duplicated(KEYS).any()
            or scope[["painting", "image", "image_sha256"]].nunique().ne(1335).any()
            or not pairs.split.eq("train").all() or set(pairs.fold) != {1, 2, 3, 4, 5}):
        raise ValueError("Pinned P2 image/language/fold coverage differs")
    require_isolation(scope, protocol[protocol.role.eq("holdout")])
    if labels and ((pairs.n < 5) | (pairs.k < 0) | (pairs.k > pairs.n) | (pairs.k % 1 != 0) | (pairs.n % 1 != 0)).any():
        raise ValueError("Invalid observed vote counts")
    return pairs.sort_values(KEYS).reset_index(drop=True)


def partition(rows, spec):
    fitting = rows[rows.fold.ne(3)]
    held = rows[rows.fold.eq(3)]
    inner, stopping = inner_partition(fitting, {
        "early_stop_seed": spec["inner_split"]["early_stop_seed"],
        "early_stop_fraction": spec["inner_split"]["fraction"],
    }, 3)
    require_isolation(inner, stopping, held)
    return inner, stopping, fitting, held


def verify_pixels(inputs, rows):
    folder = (inputs / "artwork/images").resolve()
    for image, digest in rows[["image", "image_sha256"]].drop_duplicates().itertuples(index=False, name=None):
        path = (folder / image).resolve()
        if not path.is_relative_to(folder) or file_hash(path) != digest:
            raise ValueError(f"Original training artwork sha256 differs: {image}")
    return rows.image.nunique()


def expected_identity(root):
    config = json.loads((root / "configs/exp026.json").read_text())
    source = (root / "src/st3/vlm_supervised.py").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA or config["vlm"]["revision"] != QWEN_REVISION:
        raise ValueError("Pinned extractor source/model differs")
    return {key: config[key] for key in ("vlm", "runtime", "neutral_prompt", "language_prompt", "hidden_size")} | {
        "cache_format": "qwen-state-arrays-v1", "extraction_source_sha256": SOURCE_SHA,
    }


def state_key(language):
    return "language_" + fingerprint(language)[:20]


def verified_arrays(path, identity, kind):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate arrays in NPZ")
    with np.load(path, allow_pickle=False) as arrays:
        values = {k: arrays[k] for k in arrays.files}
    metadata = json.loads(str(values.pop("__metadata")))
    receipts = {k: {"sha256": hashlib.sha256(v.tobytes()).hexdigest(), "shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in values.items()}
    if metadata != {"identity": fingerprint(identity), "filename": path.name, "arrays": receipts}:
        raise ValueError("Array identity or checksum differs")
    if not values or any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Empty or nonfinite arrays")
    if kind == "vision":
        if set(values) != {"tokens", "grid"}:
            raise ValueError("Invalid visual cache keys")
        grid, tokens = values["grid"], values["tokens"]
        if (grid.shape != (3,) or grid.dtype != np.int64 or grid[0] != 1 or (grid <= 0).any()
                or (grid[1:] % 2).any() or tokens.dtype != np.float16
                or tokens.shape != (int(np.prod(grid)) // 4, 2048) or not 128 <= len(tokens) <= 256):
            raise ValueError("Invalid vision dimensions, grid or dtype")
    else:
        for key, value in values.items():
            if ((key != "neutral" and not re.fullmatch(r"language_[0-9a-f]{20}", key))
                    or value.shape != (2048,) or value.dtype != np.float32 or not np.any(value)):
                raise ValueError("Invalid hidden state key, shape, dtype or zero vector")
    return values


def audit_cache(rows, directories, identity, *, keep_vectors=False):
    expected = {sha: {state_key(l): (p, l) for p, l in group[KEYS].itertuples(index=False, name=None)}
                for sha, group in rows.groupby("image_sha256", sort=True)}
    known_languages = {state_key(l) for l in rows.language.unique()}
    visions, states, files, errors = set(), {}, [], []
    available_bytes = 0
    extras = Counter()
    for index, directory in enumerate(directories):
        try:
            if json.loads((directory / "identity.json").read_text()) != identity:
                raise ValueError("Unexpected pinned cache identity")
        except (OSError, ValueError) as error:
            errors.append({"directory_index": index, "name": "identity.json", "error": str(error)})
            continue
        for path in sorted(directory.glob("*.npz")):
            # A terminated atomic write can leave an unpublished staging file.
            if path.name.endswith(".pending.npz"):
                extras["unpublished_files"] += 1
                continue
            match = re.fullmatch(r"([0-9a-f]{64})_(states|vision)\.npz", path.name)
            try:
                if match is None:
                    raise ValueError("Unrecognized cache filename")
                digest, kind = match.groups()
                arrays = verified_arrays(path, identity, kind)
                if kind == "states" and set(arrays) - known_languages - {"neutral"}:
                    raise ValueError("Unknown language prompt fingerprint")
                if digest not in expected:
                    extras["out_of_scope_files_verified"] += 1
                elif kind == "vision":
                    if digest in visions:
                        raise ValueError("Duplicate vision cache key")
                    visions.add(digest)
                else:
                    keys = expected[digest]
                    found = {keys[k]: v for k, v in arrays.items() if k in keys}
                    if states.keys() & found.keys():
                        raise ValueError("Duplicate or conflicting painting-language state")
                    states.update({k: v if keep_vectors else None for k, v in found.items()})
                    extras["other_valid_prompt_arrays"] += len(arrays) - len(found)
                files.append({"directory_index": index, "name": path.name, "sha256": file_hash(path)})
                available_bytes += path.stat().st_size
            except (ValueError, KeyError, OSError, TypeError, OverflowError, EOFError, zipfile.BadZipFile) as error:
                errors.append({"directory_index": index, "name": path.name, "error": str(error)})
    by_painting = dict(zip(rows.painting, rows.image_sha256))
    missing_pairs = sorted(set(rows[KEYS].itertuples(index=False, name=None)) - states.keys())
    missing_visions = sorted(set(expected) - visions)
    required_visions = {by_painting[p] for p, _ in missing_pairs} - visions
    remaining = Counter(p for p, _ in missing_pairs)
    receipt = {
        "identity_sha256": fingerprint(identity),
        "verified_vision_images": len(visions), "verified_language_states": len(states),
        "missing_vision_images": len(missing_visions), "missing_language_states": len(missing_pairs),
        "missing_vision_keys_sha256": fingerprint(missing_visions), "missing_language_keys_sha256": fingerprint(missing_pairs),
        "invalid_entries": len(errors), "errors": errors, "extra_arrays": dict(extras),
        "payload_sha256": files, "bytes_verified": available_bytes,
        "remaining_forwards_relative_to_supplied_cache": {
            "vision": len(required_visions), "decoder_sequences": len(missing_pairs),
            "decoder_batches_max8_per_image": sum((n + 7) // 8 for n in remaining.values()),
        },
        "complete": not errors and not missing_pairs,
    }
    return receipt, states, missing_pairs


def visual_blocks(root, workspace, rows):
    inventory = json.loads((root / "docs/exp028_feature_inventory.json").read_text())
    required = sorted(set(rows.image))
    blocks = {}
    for entry in inventory["visual_caches"]:
        path = workspace / entry["path"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"Pinned visual cache differs for {entry['model']}")
        with np.load(path, allow_pickle=False) as saved:
            names = saved["names"].tolist()
            values = saved["features"]
        if (len(names) != len(set(names)) or set(required) - set(names)
                or values.shape != (len(names), entry["width"]) or values.dtype != np.float32
                or not np.isfinite(values).all()):
            raise ValueError(f"Invalid feature keys/shape for {entry['model']}")
        lookup = dict(zip(names, values))
        selected = np.stack([lookup[n] for n in required])
        if not np.allclose(np.linalg.norm(selected, axis=1), 1, atol=3e-6, rtol=0):
            raise ValueError(f"Visual normalization differs for {entry['model']}")
        blocks[entry["model"]] = dict(zip(required, selected))
    if set(blocks) != {"CLIP-B", "CLIP-L", "SigLIP2-Base"}:
        raise ValueError("Incomplete visual cache inventory")
    return blocks
