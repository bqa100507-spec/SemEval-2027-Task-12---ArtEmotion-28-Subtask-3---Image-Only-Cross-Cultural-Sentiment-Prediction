"""Strict CSV and ZIP validation for development predictions."""

import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import KEYS, read_exact_csv, require_same_pairs

FILENAME = "predictions.csv"
COLUMNS = [*KEYS, "p_positive"]


def validate_predictions(frame, dev):
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"Expected prediction columns {COLUMNS}, in that order")
    require_same_pairs(dev, frame)
    q = pd.to_numeric(frame["p_positive"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return q


def _read_predictions(handle, dev):
    frame = read_exact_csv(handle, COLUMNS)
    return frame.assign(p_positive=validate_predictions(frame, dev))


def read_predictions(path, dev):
    if path.name != FILENAME:
        raise ValueError(f"Expected CSV filename {FILENAME}")
    with path.open(encoding="utf-8", newline="") as handle:
        return _read_predictions(handle, dev)


def validate_archive(path, dev):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != [FILENAME]:
            raise ValueError(f"ZIP must contain only top-level {FILENAME}")
        with archive.open(FILENAME) as member, io.TextIOWrapper(member, encoding="utf-8", newline="") as handle:
            return _read_predictions(handle, dev)


def pack(path, destination, dev):
    frame = read_predictions(path, dev)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        staged = Path(scratch) / "submission.zip"
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(FILENAME, frame.to_csv(index=False, lineterminator="\n"))
        validate_archive(staged, dev)
        staged.replace(destination)
