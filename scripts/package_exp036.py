"""Build the private EXP-036 handoff and its manifest-pinned Kaggle launcher."""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3 import data, submission
from st3.caption_data import require_isolation, training_captions
from st3.exp036_io import sha, write
from st3.metrics import score

SOURCES = ("src/st3", "scripts/run_exp036.py", "scripts/exp036_kaggle.py", "configs/exp036.json",
           "requirements-exp035-kaggle.lock.txt", "requirements-exp036-kaggle.txt")


def notebook(manifest_sha, archive_sha):
    def cell(name, kind, source):
        doc = {"id": name, "cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        return doc | ({"execution_count": None, "outputs": []} if kind == "code" else {})

    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {
        "display_name": "Python 3", "name": "python3", "language": "python"}}, "cells": [
        cell("guide", "markdown", """# EXP-036 · Caption-supervised visual adaptation

Attach the private `exp036_inputs.zip` dataset. Enable a GPU (T4 ×2 uses GPU 0) and Internet
for dependencies and the pinned public SigLIP2 download. Set `AUTHORIZE_RUN = True` then
**Run All**. The run uses official training captions only as auxiliary targets, then selects
the final checkpoint and blend on Development.

Optional mounts are the exact pinned SigLIP2 files and a prior results or progress archive.
Keep the notebook and inputs private. Results are for manual review and upload.
"""),
        cell("settings", "code", f"""AUTHORIZE_RUN = False
MODEL_ROOT = ""  # Optional exact local Kaggle model mount.
RESUME = ""      # Optional mounted progress/results ZIP or extracted directory.
SESSION_HOURS = 10.5
MANIFEST_SHA = {manifest_sha!r}
ARCHIVE_SHA = {archive_sha!r}
"""),
        cell("launch", "code", """import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

if AUTHORIZE_RUN is not True:
    raise RuntimeError("Attach the private EXP-036 input dataset and enable GPU/Internet before setting AUTHORIZE_RUN = True.")
started = time.time()
mount = Path("/kaggle/input")
out = Path("/kaggle/working")
if not mount.is_dir() or not out.is_dir():
    raise RuntimeError("Run this notebook in Kaggle.")

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

try:
    matches = [p for p in mount.rglob("manifest.json") if p.parent.name == "exp036_inputs"
               and sha(p) == MANIFEST_SHA]
    if len(matches) == 1:
        inputs = matches[0].parent
    elif not matches:
        zips = list(mount.rglob("exp036_inputs.zip"))
        if len(zips) != 1 or sha(zips[0]) != ARCHIVE_SHA:
            raise ValueError("Attach the exact exp036_inputs.zip dataset or its extracted directory.")
        scratch = Path("/kaggle/temp/exp036_input")
        with ZipFile(zips[0]) as zipped:
            zipped.extractall(scratch)  # This path is reached only for the exact pinned ZIP.
        inputs = scratch / "exp036_inputs"
    else:
        raise ValueError("Attach exactly one copy of the EXP-036 handoff.")
    if sha(inputs / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Input manifest SHA-256 mismatch.")
    m = json.loads((inputs / "manifest.json").read_text())
    for name, digest in m["files"].items():
        if not (inputs / name).is_file() or sha(inputs / name) != digest:
            raise ValueError(f"Missing or corrupt pinned input {name}")
    launcher = inputs / "runtime/scripts/exp036_kaggle.py"
    command = [sys.executable, "-u", str(launcher), "--inputs", str(inputs), "--handoff", MANIFEST_SHA,
               "--started", str(started), "--hours", str(SESSION_HOURS)]
    if MODEL_ROOT:
        command += ["--model", MODEL_ROOT]
    if RESUME:
        command += ["--resume", RESUME]
    subprocess.run(command, check=True)
except Exception as error:
    diagnostic = out / "exp036_bootstrap_error.json"
    diagnostic.write_text(json.dumps({"state": "BOOTSTRAP_FAILED", "reason": str(error)}, indent=2))
    print(diagnostic.read_text(), flush=True)
"""),
        cell("outputs", "markdown", """## Outputs

The Output panel holds the sealed results and separate submissions. Save a **private** version
with outputs or download the files before retiring the session. A results archive with
`RESUMABLE_STOP` or a periodic progress archive can be mounted in a fresh session with the
same GPU model. Fill in `RESUME`; training continues at the sealed optimizer boundary.
An abruptly terminated session can lose work that Kaggle has not saved or you have not downloaded.
"""),
        cell("report", "code", """from IPython.display import FileLink, display
from pathlib import Path

working = Path("/kaggle/working")
run = working / "exp036_run"
for path in (run / "status.json", run / "preflight.json", working / "exp036_diagnostic.json"):
    if path.is_file():
        print(path.read_text())
for path in (working / "exp036_results.zip", working / "exp036_results.zip.sha256",
             working / "exp036_progress.zip", working / "exp036_diagnostic.zip",
             run / "exp036_standalone_submission.zip", run / "exp036_ensemble_submission.zip"):
    if path.is_file():
        display(FileLink(str(path)))
if (run / "metrics.json").is_file():
    report = json.loads((run / "metrics.json").read_text())
    print("Selected epoch", report["selected_epoch"], "; selected ensemble alpha", report["ensemble"]["alpha"])
    for name, row in (("Corrected Ridge", report["reference"]), ("CS-CVA", report["standalone"]),
                      ("Ensemble", report["ensemble"]["metrics"])):
        print(name, row["cultural_contrast"], row["log_loss"])
""")]}


def package(args):
    source = subprocess.check_output(["git", "rev-parse", args.source], cwd=ROOT, text=True).strip()
    payload = subprocess.check_output(["git","archive","--format=zip",source,*SOURCES], cwd=ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "exp036_inputs.zip"
    scratch = archive.with_suffix(".tmp")
    from hashlib import sha256

    with ZipFile(io.BytesIO(payload)) as code, ZipFile(scratch,"w",compression=ZIP_DEFLATED,compresslevel=3) as bundle:
        cfg = json.loads(code.read("configs/exp036.json"))
        origin = subprocess.check_output(["git","-C",str(args.upstream),"rev-parse","HEAD"], text=True).strip()
        if origin != cfg["upstream_revision"] or any(sha(args.upstream / name) != cfg[key] for name, key in (
                (data.ANNOTATIONS,"annotations_sha256"),(data.DEV_INPUT,"dev_input_sha256"))):
            raise ValueError("Organizer input revision or bytes differ")
        if sha(args.ridge / "corrected_ridge/submission.zip") != cfg["ridge_zip_sha256"]:
            raise ValueError("Corrected Ridge is not the sealed Development submission")
        ann = data.read_annotations(args.upstream)
        train = training_captions(args.upstream)
        dev = data.read_dev_input(args.upstream)
        data.check_dev_isolation(ann, dev)
        pairs = data.training_pairs(train)
        gold = data.validation_gold(ann, dev)[[*data.KEYS,"k","n"]]
        if (len(pairs), pairs.painting.nunique(), len(dev)) != (cfg["train_pairs"],cfg["train_paintings"],cfg["dev_pairs"]):
            raise ValueError("Official population changed")
        q = submission.validate_archive(args.ridge / "corrected_ridge/submission.zip", dev)
        incumbent = score(gold, q)
        check = submission.validate_archive(args.ridge / "corrected_replay/submission.zip", dev)
        if not q.equals(check):
            raise ValueError("Corrected Ridge replay differs")
        all_images = []
        for values in (pairs,dev):
            unique = values[["painting","image"]].drop_duplicates().copy()
            unique["image_sha256"] = [sha(data.image_path(args.upstream, i)) for i in unique.image]
            all_images.append(unique)
        require_isolation(*all_images)

        files = {}
        def add(name, content):
            if name in files:
                raise ValueError(f"Duplicate archive member {name}")
            files[name] = sha256(content).hexdigest()
            bundle.writestr("exp036_inputs/"+name,content)

        def csv(name, frame):
            add(name,frame.to_csv(index=False,lineterminator="\n").encode("utf-8"))

        for name in code.namelist():
            if not name.endswith("/"):
                add(f"runtime/{name}",code.read(name))
        csv("training/annotations.csv",train[[*data.ANNOTATION_COLUMNS,"caption"]])
        add(data.DEV_INPUT,(args.upstream/data.DEV_INPUT).read_bytes())
        csv("scoring/dev_gold.csv",gold)
        with ZipFile(args.ridge/"corrected_ridge/submission.zip") as sealed:
            add("ridge/predictions.csv", sealed.read("predictions.csv"))
        for folder, names in (("corrected_bundle",("manifest.json","bank.npz")),("fixed_bundle",("manifest.json","heads.npz"))):
            src = args.ridge / folder if folder == "corrected_bundle" else args.fixed
            for name in names:
                add(f"ridge/{folder}/{name}", (src / name).read_bytes())
        corrected = json.loads((args.ridge / "corrected_bundle/manifest.json").read_bytes())
        fixed = json.loads((args.fixed / "manifest.json").read_bytes())
        if (files["ridge/corrected_bundle/bank.npz"] != corrected["bank_sha256"]
                or files["ridge/fixed_bundle/manifest.json"] != corrected["ridge_manifest_sha256"]
                or files["ridge/fixed_bundle/heads.npz"] != fixed["heads_sha256"]):
            raise ValueError("Corrected Ridge bundle seal disagrees with the supplied resources")
        for subset in all_images:
            for _, row in subset.iterrows():
                blob = data.image_path(args.upstream,row.image).read_bytes()
                if sha256(blob).hexdigest() != row.image_sha256:
                    raise ValueError("Official image changed during packaging")
                add("images/"+row.image,blob)
        receipt = {"source": source, "upstream": origin, "files": files, "ridge_zip_sha256": cfg["ridge_zip_sha256"],
                   "model": {"name": cfg["model"],"revision":cfg["revision"]},
                   "population":{"train_paintings":pairs.painting.nunique(),"train_pairs":len(pairs),
                                 "development_paintings":dev.painting.nunique(),"development_pairs":len(dev)},
                   "incumbent_development":{"cultural_contrast":incumbent["cultural_contrast"],
                                            "log_loss":incumbent["log_loss"]}}
        manifest = (json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
        bundle.writestr("exp036_inputs/manifest.json",manifest)
    with ZipFile(scratch) as z:
        assert sorted(z.namelist()) == sorted("exp036_inputs/"+n for n in [*files,"manifest.json"])
        assert all(sha256(z.read("exp036_inputs/"+p)).hexdigest()==h for p,h in files.items())
        assert z.read("exp036_inputs/manifest.json") == manifest
    scratch.replace(archive)
    manifest_sha, zip_sha = sha256(manifest).hexdigest(), sha(archive)
    (args.output/"manifest.json").write_bytes(manifest)
    write(args.output/"receipt.json",{"source":source,"archive_sha256":zip_sha,"manifest_sha256":manifest_sha,
          "files":len(files),"population":receipt["population"],"incumbent_development":receipt["incumbent_development"]})
    args.notebook.write_text(json.dumps(notebook(manifest_sha,zip_sha),indent=2)+"\n",encoding="utf-8")
    print((args.output/"receipt.json").read_text(),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    for option in ("upstream","ridge","fixed","output","notebook"):
        p.add_argument(f"--{option}",type=Path,required=True)
    package(p.parse_args())
