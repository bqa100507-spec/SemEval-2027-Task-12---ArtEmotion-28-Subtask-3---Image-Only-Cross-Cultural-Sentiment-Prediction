#!/usr/bin/env python3
"""EXP-040 January path: prepare the sealed bundle and infer the selected candidate from pixels and language.

``prepare`` copies the hash-checked Development bundle (Track 1 heads, banks, train features including the
adapted E3L embeddings, and ``e3l/`` with the LoRA + head weights) out of ``exp040_results.zip``.
``infer --selected-only`` extracts E1, E2 and E3 with the pinned encoders (plus the base Fixed Ridge features),
runs the E3L forward (saved LoRA and head; at deployment epoch 0 E3L is the frozen E3), applies the candidate
mix and its neighbor bank, composes B2 = round6(0.80 * round6(0.75 * Fx + 0.25 * CS-CVA-3) + 0.20 * C6) with
externally supplied CS-CVA and C6 ZIPs for the same keys, and writes ZIPs plus a receipt with measured E3L
throughput. For NO_GO the selected recipe is the EXP-038 incumbent: use scripts/deploy_exp038.py instead.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(_path))

import deploy_exp038 as d38
import deploy_exp039 as d39
from st3 import data, submission
from st3 import exp038 as science
from st3 import exp040 as plan
from st3.competition_stack import ENCODERS
from st3.contrast_study import new_output
from st3.e50_inference import read_input, safe_image_path
from st3.exp036_io import check, read, sha, write
from st3.exp038_io import write_submission


def prepare(args):
    with ZipFile(args.results) as z, new_output(args.output) as stage:
        manifest = json.loads(z.read("export_manifest.json"))
        if args.identity and manifest["identity"] != args.identity:
            raise ValueError("Results archive belongs to a different EXP-040 handoff")
        files = manifest["files"]
        wanted = [n for n in files if n.startswith("development/bundle/")] + ["decision.json", "selection.json",
                                                                              "run.json"]
        for name in wanted:
            content = z.read(name)
            if hashlib.sha256(content).hexdigest() != files[name]:
                raise ValueError(f"Corrupt results member {name}")
            target = stage / name.replace("development/bundle/", "bundle/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        write(stage / "deployment_index.json", {
            "results_identity": manifest["identity"], "results_sha256": sha(args.results),
            "files": {p.relative_to(stage).as_posix(): sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}})


def load_bundle(bundle):
    manifest = read(bundle / "manifest.json")
    check(bundle, manifest["files"])
    record = read(bundle / "e3l/e3l.json")
    if record["epoch"] > 0:
        check(bundle / "e3l", {"weights.pt": record["weights_sha256"], "init_head.npz": record["init_head_sha256"]})
    return manifest, record


def e3l_forward(bundle, record, inputs, paths, snapshot):
    """Adapted embeddings and E3L probabilities for every requested pair (GPU); None at deployment epoch 0."""
    import torch

    from st3 import exp040_gpu as gpu
    from st3.exp038_gpu import fetch_snapshot

    if record["epoch"] == 0:
        return None, None, None
    directory, _ = fetch_snapshot(record["spec"], snapshot)
    gpu.deterministic(0)
    head_fields = dict(np.load(bundle / "e3l/init_head.npz", allow_pickle=False))
    e3l, _ = gpu.build(record["spec"], record["lora"], head_fields, record["languages"], record["lambda"], directory,
                       torch.device("cuda:0"), 0)
    e3l.load_state(torch.load(bundle / "e3l/weights.pt", weights_only=True, map_location="cpu"))
    pairs = inputs[data.KEYS].assign(k=0., n=0.)
    panel = gpu.Paintings(pairs, paths, record["languages"])
    embeddings, probabilities, rate = gpu.predict(e3l, panel)
    return embeddings, probabilities, {"paintings_per_second": rate, "paintings": len(panel),
                                       "gpu": torch.cuda.get_device_name(0)}


def predict(bundle, manifest, inputs, test, e3l_probabilities, keep):
    candidates = {k: tuple(v) for k, v in manifest["candidates"].items() if set(v) <= set(keep)}
    heads = {n: h for n, h in d39.load_heads(bundle).items() if n in ENCODERS or n in keep}
    with np.load(bundle / "train_features.npz", allow_pickle=False) as saved:
        paintings = saved["paintings"].tolist()
        train = {n: pd.DataFrame(saved[n], index=paintings) for n in saved.files if n != "paintings" and n in
                 {*ENCODERS, *keep}}
    if set(paintings) & set(inputs.painting):
        raise ValueError("A requested painting ID is also a training bank painting")
    features = {n: pd.concat([train[n], test[n].astype(train[n].dtypes.iloc[0])]) for n in train}
    visual = science.spaces(features, candidates)
    banks = d38.bank_set(d38.load_banks(bundle, ["R0", *candidates], train), visual)
    arms = [a for a in (*plan.BASE_ARMS, plan.ARM) if a in keep]
    frame, _, _ = plan.predict_with_banks(heads, banks, visual, inputs[data.KEYS], features, e3l_probabilities,
                                          manifest["neighborhood"], candidates, arms)
    return frame, manifest["config"], candidates


def compose(args):
    index = read(args.bundle / "deployment_index.json")
    check(args.bundle, index["files"])
    bundle = args.bundle / "bundle"
    manifest, record = load_bundle(bundle)
    selection = read(args.bundle / "selection.json")
    selected = selection.get("selected_candidate")
    if not selected:
        raise ValueError("EXP-040 was NO_GO; its best ZIP is the EXP-038 incumbent (use deploy_exp038.py)")
    keep = tuple(manifest["candidates"][selected]) if args.selected_only else tuple(manifest["arms"])
    inputs = read_input(args.input, read(ROOT / "configs/e50_inference.json")["official_languages"])
    if args.features:
        with np.load(args.features, allow_pickle=False) as saved:
            names = saved["paintings"].tolist()
            test = {n: pd.DataFrame(saved[n], index=names) for n in saved.files if n not in {"paintings"}}
        probabilities = pd.read_csv(args.e3l_probabilities) if args.e3l_probabilities else None
        receipt_features, throughput = {"precomputed": sha(args.features)}, None
    else:
        config = manifest["config"]
        test, _, hashes = d38.extract_features(inputs, args, [a for a in keep if a in plan.BASE_ARMS], config)
        probabilities, throughput = None, None
        if plan.ARM in keep:
            unique = inputs[["painting", "image"]].drop_duplicates().set_index("painting")
            paths = {p: str(safe_image_path(args.images, image)) for p, image in unique.image.items()}
            tick = time.time()
            embeddings, probabilities, throughput = e3l_forward(bundle, record, inputs, paths, args.e3_snapshot)
            test[plan.ARM] = test["E3"] if embeddings is None else embeddings
            if throughput:
                throughput["wall_seconds"] = time.time() - tick
        receipt_features = {"image_sha256": hashes}
    frame, config, used = predict(bundle, manifest, inputs, test, probabilities, keep)
    outputs = {name: science.system_frame(frame, name) for name in ["R0_fixed", "R0_corrected", *keep, *used]}
    frames = [submission.validate_archive(path, inputs) for path in args.cscva]
    if len(frames) not in (1, 3):
        raise ValueError("Supply CS-CVA-3 as one ZIP, or the three seed ZIPs (original, 203836, 203936)")
    cscva3 = frames[0] if len(frames) == 1 else science.mean_of(frames, inputs)
    c6 = submission.validate_archive(args.c6, inputs)
    outputs |= science.blends({name: outputs[name] for name in used}, {"cscva3": cscva3}, c6, inputs, config)
    with new_output(args.output) as stage:
        archives = {}
        for name, prediction in outputs.items():
            archive = write_submission(science.canonical(prediction, inputs), inputs, stage / "submissions",
                                       f"exp040_{name}_submission")
            archives[name] = {"sha256": sha(archive), "eligibility": science.eligibility(name, config)}
        best = stage / "submissions" / f"exp040_{selection['submission']}_submission.zip"
        (stage / "exp040_best_submission.zip").write_bytes(best.read_bytes())
        write(stage / "inference_receipt.json", {
            "bundle_index_sha256": sha(args.bundle / "deployment_index.json"), "input_sha256": sha(args.input),
            "pairs": len(inputs), "features": receipt_features, "arms_extracted": list(keep),
            "e3l": {"deployment_epoch": record["epoch"], "throughput": throughput},
            "external_components": {"cscva": [sha(p) for p in args.cscva], "c6": sha(args.c6)},
            "selection": selection, "archives": archives, "inputs_used": "artwork pixels and target language only"})
    print(f"Validated {len(inputs)} pairs; outputs in {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--identity")
    i = sub.add_parser("infer")
    for name in ("bundle", "input", "images", "output", "c6"):
        i.add_argument(f"--{name}", type=Path, required=True)
    i.add_argument("--cscva", type=Path, action="append", required=True)
    i.add_argument("--features", type=Path, help="Precomputed features (tests); E3L probabilities via the next flag")
    i.add_argument("--e3l-probabilities", type=Path)
    i.add_argument("--selected-only", action="store_true")
    i.add_argument("--device", default="cuda")
    i.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    i.add_argument("--siglip2-python", type=Path, default=ROOT / ".venv-siglip2" /
                   ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
    for arm in ("e1", "e2", "e3"):
        i.add_argument(f"--{arm}-snapshot", type=Path)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else compose)(args)
