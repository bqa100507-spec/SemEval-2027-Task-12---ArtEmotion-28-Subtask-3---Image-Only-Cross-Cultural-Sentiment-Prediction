"""Build the private Kaggle input from the sealed EXP-036 handoff and closure."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from st3 import data, submission
from st3.exp036_io import sha, write
from st3.exp036_train import panels
from st3.exp037 import counts_for, diagnostics
from st3.exp037_io import safe_members, unpack_inputs, verify_inputs
from st3.contrast_study import new_output

PARENT = "e12bb7d11e0b0bddca4e0aa4ed1017cf0feaf353"
PRIOR_ZIP = "3e59a91ea6e4c9cdda0ecc3164a2a557d6425cb5ddc4e0e868bac1d6128ecddc"
PRIOR_MANIFEST = "ff34edd58a442082b6848f77dabcc4e2804582d15350d73e033de2caf7c2b4d2"
PRIOR_RESULTS = "9c89585579a2af68e8589027d9f1269c1192bd4bf9373c99984f3e61fe554aa1"
TARGETS = "1df461afd51011e37a6b9af1e34e20c9d2983b4e5606fe2130bc2739b4b41cb8"
INCUMBENT_SCORES = {"cultural_contrast": .4703387872416294, "log_loss": .7297085084022875}
SOURCES = ["src/st3", "configs/exp037.json", "configs/exp036.json", "configs/competition_semantic_vlm.json",
           "configs/e50_inference.json", "configs/st3.json", "scripts/run_exp036.py", "scripts/run_exp037.py",
           "scripts/exp037_kaggle.py", "requirements-exp035-kaggle.lock.txt", "requirements-exp036-kaggle.txt"]
DATA = ["training/annotations.csv", "phase_inputs/dev/st3_test.csv", "scoring/dev_gold.csv"]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_checked(z, name, expected):
    content = z.read(name)
    if digest(content) != expected:
        raise ValueError(f"EXP-036 member differs: {name}")
    return content


def package(args):
    source = subprocess.check_output(["git", "rev-parse", f"{args.source}^{{commit}}"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", PARENT, source], cwd=ROOT, check=True)
    tree = subprocess.check_output(["git", "archive", "--format=zip", source, *SOURCES], cwd=ROOT)
    for path, expected in ((args.prior_inputs, PRIOR_ZIP), (args.prior_results, PRIOR_RESULTS)):
        if sha(path) != expected:
            raise ValueError(f"EXP-036 artifact checksum differs: {path}")
    with new_output(args.output) as scratch, ZipFile(io.BytesIO(tree)) as runtime, ZipFile(args.prior_inputs) as prior, \
            ZipFile(args.prior_results) as results:
        prefix = "exp036_inputs/"
        pre = json.loads(read_checked(prior, prefix+"manifest.json", PRIOR_MANIFEST))
        visual = json.loads(runtime.read("configs/exp036.json"))
        config = json.loads(runtime.read("configs/exp037.json"))
        qwen = json.loads(runtime.read("configs/competition_semantic_vlm.json"))
        values = read_checked(results, "targets.npz", TARGETS)
        if sha(args.incumbent) != config["incumbent_zip_sha256"]:
            raise ValueError("EXP-036 incumbent archive differs")
        prior_export = json.loads(results.read("export_manifest.json"))
        if prior_export["identity"] != PRIOR_MANIFEST or prior_export["files"]["targets.npz"] != TARGETS:
            raise ValueError("Caption target artifact does not bind to the exact training handoff")
        # Parse the eligible populations using the same training loader as the worker.
        for name in DATA:
            path = scratch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(read_checked(prior, prefix+name, pre["files"][name]))
        pairs, dev, _ = panels(scratch, visual)
        votes = counts_for(pd.read_csv(scratch / DATA[0], dtype=str, keep_default_na=False), pairs)
        with np.load(io.BytesIO(values), allow_pickle=False) as targets:
            if (targets["values"].shape != (len(pairs), visual["hidden_size"])
                    or targets["mask"].shape != (len(pairs),) or targets["mask"].sum() != 16001
                    or not np.isfinite(targets["values"]).all()):
                raise ValueError("The EXP-036 caption target matrix differs")
        with ZipFile(args.incumbent) as inz:
            incumbent = submission.validate_archive(args.incumbent, dev)
            incumbent_bytes = inz.read("predictions.csv")
        gold = pd.read_csv(scratch / DATA[2])
        if list(gold) != [*data.KEYS, "k", "n"]:
            raise ValueError("Unexpected columns in scoring-only Development counts")
        metrics = diagnostics(gold, incumbent, config["test_like_excluded"])
        for metric, expected in INCUMBENT_SCORES.items():
            if abs(metrics["full"][metric]-expected) > 1e-12:
                raise ValueError(f"EXP-036 incumbent {metric} differs")
        logical = sorted(set(pairs.image) | set(dev.image))
        images = {name: f"artwork/{pre['files']['images/'+name]}.jpg" for name in logical}
        archive = scratch / "exp037_inputs.zip"
        files = {}
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=1) as z:
            def put(name, content):
                if name in files:
                    if files[name] != digest(content):
                        raise ValueError(f"Artifact path collision {name}")
                    return
                files[name] = digest(content)
                z.writestr("exp037_inputs/"+name, content)

            for name in runtime.namelist():
                if not name.endswith("/"):
                    put("runtime/"+name, runtime.read(name))
            for name in DATA:
                put(name, (scratch / name).read_bytes())
            for name in logical:
                member = prefix+"images/"+name
                put(images[name], read_checked(prior, member, pre["files"]["images/"+name]))
            put("training/targets.npz", values)
            put("incumbent/predictions.csv", incumbent_bytes)
            manifest = {"source": source, "parent": PARENT, "upstream": pre["upstream"], "files": files,
                        "images": images, "population": pre["population"],
                        "incumbent_scores": INCUMBENT_SCORES, "incumbent_zip_sha256": sha(args.incumbent),
                        "caption_targets_sha256": TARGETS, "exp036_handoff_manifest_sha256": PRIOR_MANIFEST,
                        "models": {"siglip": [visual["model"], visual["revision"]],
                                   "qwen": [qwen["model"], qwen["revision"]]}}
            safe_members([*z.namelist(), "exp037_inputs/exp037_manifest.json"])
            content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii")
            z.writestr("exp037_inputs/exp037_manifest.json", content)
        archive_sha, manifest_sha = sha(archive), digest(content)
        with tempfile.TemporaryDirectory() as directory:
            verify_inputs(unpack_inputs(archive, Path(directory), manifest_sha, archive_sha), manifest_sha)
        (scratch / "exp037_manifest.json").write_bytes(content)
        write(scratch / "exp037_handoff.json", {"archive_sha256": archive_sha, "manifest_sha256": manifest_sha,
              "source": source, "files_verified": len(files), "unique_artwork_files": len(set(images.values())),
              "logical_artworks": len(images), "languages": int(pairs.language.nunique()),
              "votes": int(votes.sum()), "incumbent": metrics})
        # Retain only the final deliverables.
        for name in DATA:
            path = scratch / name
            path.unlink()
        for name in ("phase_inputs/dev", "phase_inputs", "training", "scoring"):
            (scratch / name).rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    for field in ("prior_inputs", "prior_results", "incumbent", "output"):
        parser.add_argument(f"--{field.replace('_', '-')}", type=Path, required=True)
    package(parser.parse_args())
