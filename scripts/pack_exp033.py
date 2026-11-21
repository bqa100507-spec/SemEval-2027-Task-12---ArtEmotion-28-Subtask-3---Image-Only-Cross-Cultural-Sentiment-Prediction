"""Build the P2-only private handoff from pinned EXP-023 and EXP-032 artifacts."""
import argparse
import io
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from st3.exp033_io import KEYS, load_population, read_json, safe_path, save_json, sha256

PINS = {
    "training/pairs.csv": "50798ddac6d9bda709c53b0ab7fc5ff2bfc6fbc33a4e581c1ab5c45f94d57534",
    "visual/features.npz": "17374581aa7b65dd9a24542b18e0a8449388b6b04262fc18d4da235d9ce68896",
}
RIDGE_ARCHIVE = "d5c5cad843d05ec4fde5b7adaf976107f8944e7c8da515fcd113baa543e4291f"


def build(parent, previous, destination):
    for name, expected in PINS.items():
        if sha256(parent / name) != expected:
            raise ValueError(f"Pinned source differs: {name}")
    if sha256(previous) != RIDGE_ARCHIVE or destination.exists():
        raise ValueError("Ridge source hash differs or output already exists")
    source = pd.read_csv(parent / "training/pairs.csv", usecols=KEYS + [
        "fold", "role", "split", "image", "image_sha256", "k", "n",
    ])
    if not source["role"].eq("development").all() or not source["split"].eq("train").all():
        raise ValueError("Handoff is not the isolated P2 training pool")
    mapping = source[["painting", "image", "image_sha256"]].drop_duplicates().sort_values("painting")
    files = {}
    config = read_json(ROOT / "configs/exp033.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging, zipfile.ZipFile(previous) as archive:
        folder = Path(staging)
        (folder / "images").mkdir()
        for i, (_, image, expected) in enumerate(mapping.itertuples(index=False, name=None)):
            source_image = parent / "artwork/images" / safe_path(image)
            if sha256(source_image) != expected:
                raise ValueError(f"Source image differs: {image}")
            name = f"images/{i:04d}{source_image.suffix.lower()}"
            shutil.copyfile(source_image, folder / name)
            files[image] = name
        parts = []
        with np.load(parent / "visual/features.npz", allow_pickle=False) as visual:
            indices = {name: i for i, name in enumerate(visual["names"])}
            blocks = {name: visual[name].astype(np.float64) for name in ("clip", "siglip", "large")}
            for fold, rows in source.groupby("fold"):
                values = rows[KEYS].copy()
                raw = archive.read(f"fold_{fold}/coefficients.npz")
                with np.load(io.BytesIO(raw), allow_pickle=False) as head:
                    for name, embeddings in blocks.items():
                        p = f"no_graph:{name}:"
                        ls = {lang: j for j, lang in enumerate(head[p + "languages"])}
                        w, d = head[p + "shared"], head[p + "deviations"]
                        b0, b = head[p + "intercept"], head[p + "language_intercepts"]
                        logits = []
                        for image, language in rows[["image", "language"]].itertuples(index=False, name=None):
                            v = embeddings[indices[image]]
                            logit = b0 + w @ v
                            if language in ls:
                                j = ls[language]
                                logit += b[j] + d[j] @ v
                            logits.append(logit)
                        values[name] = expit(np.asarray(logits)).round(6)
                parts.append(values)
        pd.concat(parts).to_csv(folder / "ridge_side_heads.csv", index=False, float_format="%.6f")
        source["image"] = source["image"].map(files)
        source[KEYS + ["fold", "image", "k", "n"]].to_csv(folder / "pairs.csv", index=False)
        for name in ("oof_E50_L.csv", "oof_no_graph.csv"):
            (folder / name).write_bytes(archive.read(name))
        load_population(folder, config)
        names = {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()}
        manifest = folder / "manifest.json"
        save_json(manifest, {"experiment": "EXP-033", "scope": "P2_ONLY", "files": names,
                             "source_sha256": PINS | {"ridge_archive": RIDGE_ARCHIVE}})
        manifest_hash = sha256(manifest)
        tmp = destination.with_suffix(".partial")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as output:
            for name in list(names) + ["manifest.json"]:
                output.write(folder / name, name)
        tmp.rename(destination)
    print(json.dumps({"zip": sha256(destination), "manifest": manifest_hash}))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "ridge-archive", "output"):
        args.add_argument("--" + key, type=Path, required=True)
    options = args.parse_args()
    build(options.parent, options.ridge_archive, options.output)
