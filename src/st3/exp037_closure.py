"""Exact EXP-037 archive checks and label-free probability recipes."""

import hashlib
import io
import json
import math
from zipfile import ZipFile

import pandas as pd

from . import data, submission
from .exp036 import rounded
from .exp036_io import relative


def digest(content):
    return hashlib.sha256(content).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def json_member(archive, name):
    return json.loads(archive.read(name))


def verify_export(archive, identity, *, name="export_manifest.json", prefix=""):
    manifest = json_member(archive, prefix+name)
    if name == "export_manifest.json":
        require(manifest.get("identity") == identity, "Unexpected archive identity")
    files = manifest["files"]
    require(sorted(archive.namelist()) == sorted([prefix+name, *(prefix+member for member in files)]),
            "Archive membership differs from manifest")
    for member, expected in files.items():
        relative(member)
        with archive.open(prefix+member) as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        require(actual == expected, f"Archive digest mismatch: {member}")
    return manifest


def checked_metrics(actual, saved):
    if isinstance(actual, dict):
        require(actual.keys() == saved.keys(), "Metric keys differ")
        for key in actual:
            checked_metrics(actual[key], saved[key])
    elif isinstance(actual, float):
        require(math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-14),
                f"Metric differs: {actual} != {saved}")
    else:
        require(actual == saved, f"Value differs: {actual} != {saved}")


def prediction_bytes(blob, dev):
    prediction = pd.read_csv(io.BytesIO(blob), dtype={name: str for name in data.KEYS}, float_precision="round_trip")
    submission.validate_predictions(prediction, dev)
    require(prediction[data.KEYS].reset_index(drop=True).equals(dev[data.KEYS].reset_index(drop=True)),
            "Archived predictions differ from official key order")
    require(prediction.p_positive.equals(prediction.p_positive.round(6)), "Prediction precision differs")
    return prediction


def zip_prediction_bytes(blob, dev):
    with ZipFile(io.BytesIO(blob)) as archive:
        require(archive.namelist() == [submission.FILENAME], "Unexpected submission archive layout")
        return prediction_bytes(archive.read(submission.FILENAME), dev)


def mixed(dev, parts):
    """Mix one level, with rounded inputs and one six-decimal rounding at this level."""
    require(bool(parts) and all(math.isfinite(w) and 0 <= w <= 1 for w, _ in parts), "Invalid fusion weight")
    require(math.isclose(sum(w for w, _ in parts), 1., rel_tol=0., abs_tol=1e-12), "Fusion weights must sum to 1")
    return dev[data.KEYS].assign(p_positive=sum(w*rounded(frame, dev).p_positive for w, frame in parts).round(6))


def recipe_prediction(dev, frames, recipe):
    if isinstance(recipe, str):
        return rounded(frames[recipe], dev)
    return mixed(dev, [(weight, recipe_prediction(dev, frames, part)) for weight, part in recipe])


def recipe_leaves(recipe):
    if isinstance(recipe, str):
        return {recipe}
    return set().union(*(recipe_leaves(part) for _, part in recipe))
