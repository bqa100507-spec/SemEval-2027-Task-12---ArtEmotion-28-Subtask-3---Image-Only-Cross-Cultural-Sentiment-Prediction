#!/usr/bin/env python3
"""Recover the sealed EXP-036 inference bundle without the training dataset."""

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

from predict_exp036 import ALPHA, PINNED, SOURCE, new_output, sha, verify_bundle, write

RESULTS_SHA = "9c89585579a2af68e8589027d9f1269c1192bd4bf9373c99984f3e61fe554aa1"
HANDOFF_SHA = "ff34edd58a442082b6848f77dabcc4e2804582d15350d73e033de2caf7c2b4d2"


def recover(results, fixed, corrected, output):
    if sha(results) != RESULTS_SHA:
        raise ValueError("Results archive differs from the verified Kaggle export")
    with ZipFile(results) as archive, new_output(output) as folder:
        if json.loads(archive.read("status.json"))["state"] != "COMPLETE":
            raise ValueError("Incomplete EXP-036 export")
        for path, fingerprint in PINNED.items():
            destination = folder / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.startswith("ridge/fixed/"):
                shutil.copyfile(fixed / destination.name, destination)
            elif path.startswith("ridge/corrected/"):
                shutil.copyfile(corrected / destination.name, destination)
            else:
                member = path.replace("cscva/", "inference/", 1)
                destination.write_bytes(archive.read(member))
            if sha(destination) != fingerprint:
                raise ValueError(f"Deployment asset checksum differs: {path}")
        write(folder / "manifest.json", {
            "version": 1, "training_source": SOURCE, "training_handoff_sha256": HANDOFF_SHA,
            "results_sha256": RESULTS_SHA, "selected_epoch": 5, "cs_cva_weight": ALPHA,
            "unseen_language": "CS-CVA shared image logit plus mean fitted language intercept",
            "files": PINNED,
        })
        verify_bundle(folder)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("results", "fixed", "corrected", "output"):
        p.add_argument(f"--{field}", required=True, type=Path)
    args = p.parse_args()
    recover(args.results, args.fixed, args.corrected, args.output)
