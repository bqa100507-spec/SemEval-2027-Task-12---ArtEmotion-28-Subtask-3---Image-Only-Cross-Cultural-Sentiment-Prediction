#!/usr/bin/env python3
"""Prepare the sealed EXP-038 deployment bundle and run image-only inference on official input pairs.

``prepare`` copies the refitted heads, neighbor banks, training similarity features and the new CS-CVA seed
bundles out of a verified ``exp038_results.zip``. ``infer`` takes the official input-only CSV and an
``images/`` root, extracts E1-E4 (E4 only for requested pairs) and the three Fixed Ridge features, applies the
heads, mixes and banks, composes B1/B2 with externally supplied EXP-036 CS-CVA and EXP-037 C6 ZIPs for the same
keys, validates keys and writes ZIPs plus a checksum receipt. Inputs are pixels and target language only.
"""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3 import data, submission
from st3 import exp038 as science
from st3.competition_neighborhood import ResidualBank
from st3.competition_ridge import Head
from st3.competition_stack import ENCODERS
from st3.contrast_study import new_output
from st3.e50_inference import read_input, safe_image_path
from st3.exp036_io import check, read, sha, write
from st3.exp038_io import write_submission

BUNDLE_FILES = ("heads.npz", "banks.npz", "train_features.npz", "manifest.json")


def prepare(args):
    with ZipFile(args.results) as z, new_output(args.output) as stage:
        manifest = json.loads(z.read("export_manifest.json"))
        if args.identity and manifest["identity"] != args.identity:
            raise ValueError("Results archive belongs to a different EXP-038 handoff")
        files = manifest["files"]
        wanted = [f"development/bundle/{name}" for name in BUNDLE_FILES]
        wanted += [name for name in files if name.startswith("cscva/seed_") and "/inference/" in name]
        wanted += ["decision.json", "selection.json", "run.json"]
        for name in wanted:
            content = z.read(name)
            import hashlib

            if hashlib.sha256(content).hexdigest() != files[name]:
                raise ValueError(f"Corrupt results member {name}")
            target = stage / name.replace("development/bundle/", "bundle/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        write(stage / "deployment_index.json", {
            "results_identity": manifest["identity"], "results_sha256": sha(args.results),
            "files": {p.relative_to(stage).as_posix(): sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}})


def load_heads(bundle):
    heads = {}
    with np.load(bundle / "heads.npz", allow_pickle=False) as state:
        names = sorted({key.split(":")[0] for key in state.files})
        for name in names:
            head = Head(float(state[f"{name}:intercept"]), state[f"{name}:language_intercepts"],
                        state[f"{name}:shared"], state[f"{name}:deviations"],
                        tuple(state[f"{name}:languages"].tolist()))
            heads[name] = science.PairHead(head, state[f"{name}:gamma"]) if f"{name}:gamma" in state.files else head
    ordered = {name: heads[name] for name in ENCODERS}
    return ordered | {name: heads[name] for name in science.ARMS if name in heads}


def load_banks(bundle, systems, train_features):
    banks = {}
    with np.load(bundle / "banks.npz", allow_pickle=False) as state:
        for system in systems:
            paintings = state[f"{system}:paintings"].tolist()
            languages = state[f"{system}:languages"].tolist()
            numerator = pd.DataFrame(state[f"{system}:numerator"], index=paintings, columns=languages)
            votes = pd.DataFrame(state[f"{system}:votes"], index=paintings, columns=languages)
            banks[system] = (numerator, votes)
    return banks


def bank_set(banks, visual):
    return {system: ResidualBank(visual[system].loc[numerator.index], numerator, votes)
            for system, (numerator, votes) in banks.items()}


def extract_features(inputs, args, arms, config):
    """Real extraction: Track 1 workers for the base three, pinned EXP-038 encoders for E1-E4."""
    from types import SimpleNamespace

    from st3 import exp038_gpu as gpu
    from st3.competition_challenger import production_encoders
    from st3.e50_inference import run_feature_worker
    from st3.incremental import valid_vectors

    unique = inputs[["painting", "image"]].drop_duplicates().set_index("painting")
    paths = {name: safe_image_path(args.images, name) for name in unique.image.unique()}
    hashes = {name: sha(path) for name, path in paths.items()}
    features = {}
    worker_args = SimpleNamespace(device=args.device, cache_dir=args.cache_dir, feature_cache_dir=None,
                                  siglip2_python=args.siglip2_python, clip_batch_size=None, siglip2_batch_size=None,
                                  clip_large_batch_size=None)
    with TemporaryDirectory(prefix=".exp038-base-", dir=args.output.parent) as scratch:
        for name in ENCODERS:
            profile = production_encoders()[name]
            vectors, _ = run_feature_worker(name, {hashes[i]: paths[i] for i in paths}, worker_args, profile,
                                            Path(scratch))
            x = np.stack([vectors[hashes[image]] for image in unique.image])
            valid_vectors(x, len(unique), profile["width"])
            features[name] = pd.DataFrame(x, index=unique.index)
    import torch

    device = torch.device("cuda:0")
    gpu.deterministic_cuda()
    images = [(hashes[image], str(paths[image])) for image in unique.image]
    for arm in [a for a in ("E1", "E2", "E3") if any(science.image_encoder(x) == a for x in arms)]:
        spec = config["arms"][arm]
        snapshot, _ = gpu.fetch_snapshot(spec, getattr(args, f"{arm.lower()}_snapshot"))
        embed = gpu.encoder(arm, spec, snapshot, device)
        features[arm] = pd.DataFrame(gpu.embed_items(arm, spec, embed, images, spec["batch"]), index=unique.index)
        del embed
        torch.cuda.empty_cache()
    delta = None
    if "E4" in arms:
        spec = config["arms"]["E3"] | config["arms"]["E4"]
        snapshot, _ = gpu.fetch_snapshot(config["arms"]["E3"], args.e3_snapshot)
        embed = gpu.encoder("E4", spec, snapshot, device)
        items = [(f"{p}|{l}", str(paths[unique.image[p]]), l) for p, l in inputs[data.KEYS].itertuples(index=False)]
        values = gpu.embed_items("E4", spec, embed, items, spec["batch"])
        delta = science.PairDelta.build(inputs[data.KEYS].reset_index(drop=True), values, features["E3"])
    return features, delta, hashes


def recipe_arms(manifest, selection):
    """Arms the selected recipe needs; none for a NO_GO (the best ZIP is then the archived EXP-037 blend)."""
    selected = selection.get("selected_candidate")
    return tuple(manifest["candidates"][selected]) if selected else ()


def predict(bundle, inputs, test_features, delta, keep_arms=None):
    manifest = read(bundle / "manifest.json")
    check(bundle, manifest["files"])
    config = manifest["config"]
    arms = tuple(manifest["arms"])
    candidates = {k: tuple(v) for k, v in manifest["candidates"].items()}
    heads = load_heads(bundle)
    if keep_arms is not None:
        arms = tuple(a for a in arms if a in keep_arms)
        candidates = {k: v for k, v in candidates.items() if set(v) <= set(arms)}
        heads = {name: head for name, head in heads.items() if name in ENCODERS or name in arms}
    with np.load(bundle / "train_features.npz", allow_pickle=False) as saved:
        train_paintings = saved["paintings"].tolist()
        needed = {*ENCODERS, *[science.image_encoder(a) for a in arms]}
        train = {name: pd.DataFrame(saved[name], index=train_paintings) for name in saved.files
                 if name != "paintings" and name in needed}
    if set(train_paintings) & set(inputs.painting):
        raise ValueError("A requested painting ID is also a training bank painting")
    features = {name: pd.concat([train[name], test_features[name].astype(train[name].dtypes.iloc[0])])
                for name in train}
    visual = science.spaces(features, candidates)
    banks = bank_set(load_banks(bundle, ["R0", *candidates], train), visual)
    frame, _ = science.development_predictions(heads, banks, visual, inputs[data.KEYS], features, delta, candidates,
                                               manifest["neighborhood"])
    return frame, config, arms, candidates


def compose(args):
    index = read(args.bundle / "deployment_index.json")
    check(args.bundle, index["files"])
    bundle = args.bundle / "bundle"
    manifest = read(bundle / "manifest.json")
    official = read(ROOT / "configs/e50_inference.json")["official_languages"]
    inputs = read_input(args.input, official)
    selection = read(args.bundle / "selection.json")
    keep = recipe_arms(manifest, selection) if args.selected_only else None
    wanted = tuple(manifest["arms"]) if keep is None else keep
    if args.features:
        with np.load(args.features, allow_pickle=False) as saved:
            paintings = saved["paintings"].tolist()
            test = {name: pd.DataFrame(saved[name], index=paintings) for name in saved.files
                    if name not in {"paintings", "pair_keys", "E4"}}
            delta = None
            if "E4" in saved.files and "E4" in wanted:
                keys = pd.DataFrame([k.split("|", 1) for k in saved["pair_keys"].tolist()], columns=data.KEYS)
                delta = science.PairDelta.build(keys, saved["E4"], test["E3"])
        receipt_features = {"precomputed": sha(args.features)}
    else:
        test, delta, hashes = extract_features(inputs, args, wanted, manifest["config"])
        receipt_features = {"image_sha256": hashes}
    frame, config, arms, candidates = predict(bundle, inputs, test, delta, keep)
    dev = inputs
    outputs = {name: science.system_frame(frame, name) for name in ["R0_fixed", "R0_corrected", *arms, *candidates]}
    cscva = {}
    if args.cscva:
        frames = [submission.validate_archive(path, dev) for path in args.cscva]
        cscva["original"] = frames[0]
        if len(frames) == 3:
            cscva["cscva3"] = science.mean_of(frames, dev)
        elif len(frames) != 1:
            raise ValueError("Supply one CS-CVA ZIP (original) or three (original, seed 2, seed 3)")
    if cscva and args.c6:
        c6 = submission.validate_archive(args.c6, dev)
        bases = {"R0": outputs["R0_corrected"]} | {name: outputs[name] for name in candidates}
        outputs |= science.blends(bases, cscva, c6, dev, config)
    with new_output(args.output) as stage:
        archives = {}
        for name, prediction in outputs.items():
            archive = write_submission(science.canonical(prediction, dev), dev, stage / "submissions",
                                       f"exp038_{name}_submission")
            archives[name] = {"sha256": sha(archive), "eligibility": science.eligibility(name, config)}
        chosen = selection["submission"]
        if chosen in outputs:
            best = stage / "submissions" / f"exp038_{chosen}_submission.zip"
            (stage / "exp038_best_submission.zip").write_bytes(best.read_bytes())
        write(stage / "inference_receipt.json", {
            "bundle_index_sha256": sha(args.bundle / "deployment_index.json"), "input_sha256": sha(args.input),
            "pairs": len(inputs), "paintings": int(inputs.painting.nunique()), "features": receipt_features,
            "external_components": {"cscva": [sha(p) for p in args.cscva or []],
                                    "c6": sha(args.c6) if args.c6 else None},
            "selection": selection, "archives": archives,
            "best_available": chosen in outputs, "arms_extracted": list(wanted),
            "inputs_used": "artwork pixels and target language only"})
    print(f"Validated {len(inputs)} pairs; outputs in {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--identity", help="Expected overlay manifest SHA-256 recorded in the results archive")
    i = sub.add_parser("infer")
    for name in ("bundle", "input", "images", "output"):
        i.add_argument(f"--{name}", type=Path, required=True)
    i.add_argument("--cscva", type=Path, action="append", help="EXP-036 CS-CVA ZIP(s) for the same keys")
    i.add_argument("--c6", type=Path, help="EXP-037 C6 standalone ZIP for the same keys")
    i.add_argument("--features", type=Path, help="Precomputed feature NPZ (CPU replay and tests)")
    i.add_argument("--selected-only", action="store_true",
                   help="Extract and score only the arms of the selected recipe (C2: E1, E2, E3; no E4)")
    i.add_argument("--device", default="cuda")
    i.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    i.add_argument("--siglip2-python", type=Path, default=ROOT / ".venv-siglip2" /
                   ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
    for arm in ("e1", "e2", "e3"):
        i.add_argument(f"--{arm}-snapshot", type=Path)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else compose)(args)
