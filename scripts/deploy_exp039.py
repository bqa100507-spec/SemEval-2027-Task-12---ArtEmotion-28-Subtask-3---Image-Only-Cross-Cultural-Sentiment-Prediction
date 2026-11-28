#!/usr/bin/env python3
"""EXP-039 January path: prepare the sealed bundle and infer the selected candidate from pixels and language.

``prepare`` is EXP-038's (same results-archive layout). ``infer --selected-only`` extracts only the arms of the
selected candidate (E1, E2, E3 plus the selected E5/E6/E7), applies the refitted heads, the candidate mix
(including D4's VLM-third rule) and its neighbor bank, composes B2 = round6(0.80 * round6(0.75 * Dx + 0.25 *
CS-CVA-3) + 0.20 * C6) with externally supplied CS-CVA and C6 ZIPs for the same keys, and writes ZIPs plus a
receipt. For NO_GO the selected recipe is the EXP-038 incumbent: use scripts/deploy_exp038.py instead.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(_path))

import deploy_exp038 as d38
from st3 import data, submission
from st3 import exp038 as science
from st3 import exp039 as plan39
from st3.competition_ridge import Head
from st3.competition_stack import ENCODERS
from st3.contrast_study import new_output
from st3.e50_inference import read_input, safe_image_path
from st3.exp036_io import check, read, sha, write
from st3.exp038_io import write_submission

prepare = d38.prepare


def load_heads(bundle):
    heads = {}
    with np.load(bundle / "heads.npz", allow_pickle=False) as state:
        for name in sorted({key.split(":")[0] for key in state.files}):
            heads[name] = Head(float(state[f"{name}:intercept"]), state[f"{name}:language_intercepts"],
                               state[f"{name}:shared"], state[f"{name}:deviations"],
                               tuple(state[f"{name}:languages"].tolist()))
    return {name: heads[name] for name in ENCODERS} | {n: h for n, h in heads.items() if n not in ENCODERS}


def parse_candidates(manifest):
    out = {}
    for name, spec in manifest["candidates"].items():
        out[name] = (science.VlmThird(tuple(spec["clip_arms"]), tuple(spec["vlm_arms"]))
                     if isinstance(spec, dict) else tuple(spec))
    return out


def predict(bundle, inputs, test, keep):
    manifest = read(bundle / "manifest.json")
    check(bundle, manifest["files"])
    candidates = {k: v for k, v in parse_candidates(manifest).items() if set(v) <= set(keep)}
    heads = {n: h for n, h in load_heads(bundle).items() if n in ENCODERS or n in keep}
    with np.load(bundle / "train_features.npz", allow_pickle=False) as saved:
        paintings = saved["paintings"].tolist()
        train = {n: pd.DataFrame(saved[n], index=paintings) for n in saved.files if n != "paintings" and n in
                 {*ENCODERS, *keep}}
    if set(paintings) & set(inputs.painting):
        raise ValueError("A requested painting ID is also a training bank painting")
    features = {n: pd.concat([train[n], test[n].astype(train[n].dtypes.iloc[0])]) for n in train}
    visual = science.spaces(features, candidates)
    banks = d38.bank_set(d38.load_banks(bundle, ["R0", *candidates], train), visual)
    frame, _ = science.development_predictions(heads, banks, visual, inputs[data.KEYS], features, None, candidates,
                                               manifest["neighborhood"])
    return frame, manifest["config"], candidates


def extract(inputs, args, arms, config):
    """Fixed Ridge base features through the Track 1 workers; E1-E3 and new arms with the pinned encoders."""
    import torch

    from st3 import exp038_gpu as gpu

    base, _, hashes = d38.extract_features(inputs, args, ("E1", "E2", "E3"), read(ROOT / "configs/exp038.json"))
    unique = inputs[["painting", "image"]].drop_duplicates().set_index("painting")
    images = [(hashes[image], str(safe_image_path(args.images, image))) for image in unique.image]
    for arm in [a for a in arms if a not in plan39.SEALED]:
        spec = plan39.arm_spec(config, arm)
        snapshot, _ = gpu.fetch_snapshot(config["arms"]["E5" if arm.startswith("E5") else "E3"],
                                         args.e5_snapshot if arm.startswith("E5") else args.e3_snapshot)
        embed = gpu.encoder(arm, spec, snapshot, torch.device("cuda:0"))
        if plan39.config_arm(arm) == "E6":
            raw = gpu.embed_items(arm, spec, embed, plan39.pool_items(images, len(spec["instructions"])), spec["batch"])
            values = plan39.pooled_instructions(base["E3"].to_numpy(), raw, len(spec["instructions"]))
        else:
            values = gpu.embed_items(arm, spec, embed, images, spec["batch"])
        base[arm] = pd.DataFrame(values, index=unique.index)
        del embed
        torch.cuda.empty_cache()
    return base, hashes


def compose(args):
    index = read(args.bundle / "deployment_index.json")
    check(args.bundle, index["files"])
    bundle = args.bundle / "bundle"
    manifest = read(bundle / "manifest.json")
    selection = read(args.bundle / "selection.json")
    selected = selection.get("selected_candidate")
    if not selected:
        raise ValueError("EXP-039 was NO_GO; its best ZIP is the EXP-038 incumbent (use deploy_exp038.py)")
    candidates = parse_candidates(manifest)
    keep = tuple(candidates[selected]) if args.selected_only else tuple(manifest["arms"])
    inputs = read_input(args.input, read(ROOT / "configs/e50_inference.json")["official_languages"])
    if args.features:
        with np.load(args.features, allow_pickle=False) as saved:
            names = saved["paintings"].tolist()
            test = {n: pd.DataFrame(saved[n], index=names) for n in saved.files if n != "paintings"}
        receipt_features = {"precomputed": sha(args.features)}
    else:
        test, hashes = extract(inputs, args, keep, manifest["config"])
        receipt_features = {"image_sha256": hashes}
    frame, config, used = predict(bundle, inputs, test, keep)
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
                                       f"exp039_{name}_submission")
            archives[name] = {"sha256": sha(archive), "eligibility": science.eligibility(name, config)}
        best = stage / "submissions" / f"exp039_{selection['submission']}_submission.zip"
        (stage / "exp039_best_submission.zip").write_bytes(best.read_bytes())
        write(stage / "inference_receipt.json", {
            "bundle_index_sha256": sha(args.bundle / "deployment_index.json"), "input_sha256": sha(args.input),
            "pairs": len(inputs), "features": receipt_features, "arms_extracted": list(keep),
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
    i.add_argument("--features", type=Path)
    i.add_argument("--selected-only", action="store_true")
    i.add_argument("--device", default="cuda")
    i.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    i.add_argument("--siglip2-python", type=Path, default=ROOT / ".venv-siglip2" /
                   ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
    for arm in ("e1", "e2", "e3", "e5"):
        i.add_argument(f"--{arm}-snapshot", type=Path)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else compose)(args)
