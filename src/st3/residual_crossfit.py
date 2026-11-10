"""Verify frozen outer predictions and cross-fit base heads inside each outer training set."""

import importlib.metadata
from itertools import combinations

import pandas as pd

from . import data
from .ablation import load_study_embeddings, painting_folds
from .baseline import file_sha256
from .cli import ROOT, git_head, json_output
from .clip_large_study import E50_LARGE, LARGE
from .contrast_study import new_output
from .coverage_study import E50, fold_reference, saved_prediction
from .e50_heads import read_json
from .e50l_heads import clip_large_config, compose_e50l, load_experiment_cache
from .fusion import blend, verify_reference
from .incremental import digest_bytes
from .low_rank_study import FOLD_FILE, verify_folds
from .residual import COMPONENTS
from .siglip2 import load_siglip2_features
from .visual_study import REFERENCE, SIGLIP, fit_head


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def disjoint(*partitions):
    for left, right in combinations(partitions, 2):
        for key in ("painting", "image"):
            if not set(left[key]).isdisjoint(right[key]):
                raise ValueError(f"Cross-fitting leakage: overlapping {key} groups")


def signals(clip, siglip, large):
    e50 = blend(clip, siglip, .5)
    q0 = compose_e50l(e50, large)
    result = clip[data.KEYS].copy()
    for name, frame in zip((*COMPONENTS, "e50", "q0"), (clip, siglip, large, e50, q0)):
        data.require_same_pairs(result, frame)
        result = result.merge(frame.rename(columns={"p_positive": name}), on=data.KEYS, validate="one_to_one")
    return result


def residual_inputs(args, dev, config_name="exp012"):
    config_path = ROOT / "configs" / f"{config_name}.json"
    config = read_json(config_path)
    baseline = read_json(ROOT / "configs" / "st3.json")
    source = args.exp010_dir / "cv"
    prior = read_json(source / "run.json")
    decision = read_json(source / "selection.json")
    if (git_head(args.data_dir) != baseline["upstream_revision"] or decision["model"] != E50_LARGE
            or any(file_sha256(source / name) != decision["sha256"][name] for name in ("run.json", "summary.csv"))
            or file_sha256(source / "predictions.csv") != config["exp010_predictions_sha256"]):
        raise ValueError("Frozen EXP-010 outer reference or organizer revision differs")
    train = data.training_pairs(data.read_annotations(args.data_dir, training_only=True)).assign(split="train")
    disjoint(train, dev)
    clip, old = load_study_embeddings(args.data_dir, train, dev, args.cache_dir, args.embedding_run,
                                     baseline["upstream_revision"], baseline)
    siglip, siglip_meta = load_siglip2_features(args, dev, train, read_json(ROOT / "configs" / "exp006.json"),
                                               extraction_dir=args.exp006_dir / "extraction")
    large, large_meta = load_experiment_cache(args, train["image"], prior["clip_large_embedding"], clip_large_config())
    if (any(old[key] != prior[key] for key in ("annotations_sha256", "dev_input_sha256"))
            or old["embedding"]["sha256"] != prior["embedding"]["sha256"]
            or siglip_meta["features_sha256"] != prior["siglip2_embedding"]["features_sha256"]):
        raise ValueError("Base features or original training data differ from EXP-010")
    paths = {"exp002_folds": args.exp002_dir / FOLD_FILE,
             "exp006_predictions": args.exp006_dir / "cv" / "predictions.csv",
             "exp007_predictions": args.exp007_dir / "cv" / "predictions.csv"}
    hashes = {key + "_sha256": file_sha256(path) for key, path in paths.items()}
    if (hashes["exp002_folds_sha256"] != prior["exp002_folds_sha256"]
            or hashes["exp006_predictions_sha256"] != prior["exp006_predictions_sha256"]
            or hashes["exp007_predictions_sha256"] != prior["exp007_sha256"]["predictions.csv"]):
        raise ValueError("Historical fold inputs have changed")
    saved, exp006, exp007 = (read_csv(path) for path in paths.values())
    folds = verify_folds(train, saved, prior["exp002_config"])
    outer = replay_outer(train, folds, saved, exp006, exp007, source / "predictions.csv",
                         prior["exp002_config"]["fit_seed"])
    names = set(train["image"])
    features = {name: {path: bank[path] for path in names}
                for name, bank in zip(COMPONENTS, (clip, siglip, large))}
    provenance = {**hashes, **{key: old[key] for key in ("annotations_sha256", "dev_input_sha256")},
                  "upstream_revision": baseline["upstream_revision"], "exp002_config": prior["exp002_config"],
                  "exp010_run_sha256": file_sha256(source / "run.json"), "config_sha256": file_sha256(config_path),
                  "features_sha256": [old["embedding"]["sha256"], siglip_meta["features_sha256"], large_meta["sha256"]],
                  "reference_oof_sha256": digest_bytes(outer_reference_bytes(outer, prior["exp002_config"]["fit_seed"])),
                  "packages": {name: importlib.metadata.version(name)
                               for name in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")},
                  "config": config}
    return train, features, folds, outer, provenance


def outer_reference_bytes(frame, seed):
    reference = frame[data.KEYS].assign(p_positive=frame["q0"], model=E50_LARGE, fold=frame["outer_fold"], seed=seed)
    return reference.to_csv(index=False, lineterminator="\n").encode("utf-8")


def replay_outer(train, folds, saved, exp006, exp007, path, seed):
    exp010 = read_csv(path)
    raw = path.read_bytes().splitlines(keepends=True)
    expected = raw[0] + b"".join(line for line in raw[1:] if line.rsplit(b",", 3)[1] == E50_LARGE.encode())
    result = []
    for fold, (fit_idx, held_idx) in enumerate(folds, 1):
        fit, held = train.iloc[fit_idx], train.iloc[held_idx]
        disjoint(fit, held)
        e50, siglip = fold_reference(saved, exp006, exp007, held, fold, .5)
        clip = saved_prediction(exp007, REFERENCE, fold)
        rows = signals(clip, siglip, saved_prediction(exp010, LARGE, fold))
        verify_reference(rows[[*data.KEYS, "e50"]].rename(columns={"e50": "p_positive"}), e50)
        result.append(rows.merge(held[[*data.KEYS, "k", "n", "split"]], on=data.KEYS, validate="one_to_one")
                      .assign(outer_fold=fold))
    outer = pd.concat(result, ignore_index=True)
    data.require_same_pairs(train, outer)
    if outer_reference_bytes(outer, seed) != expected:
        raise ValueError("Replayed E50+L is not byte-identical to EXP-010 outer predictions")
    return outer


def inner_partitions(fit, forbidden, config, outer_fold):
    disjoint(fit, forbidden)
    seed = config["inner_seed"] + outer_fold
    for fit_idx, held_idx in painting_folds(fit, config["inner_folds"], seed):
        inner_fit, inner_held = fit.iloc[fit_idx], fit.iloc[held_idx]
        disjoint(inner_fit, inner_held, forbidden)
        if len(inner_fit) + len(inner_held) != len(fit):
            raise ValueError("Inner fold does not cover the outer training partition")
        yield inner_fit, inner_held


def verify_inner(rows, partitions, forbidden, outer_fold):
    keys = [*data.KEYS, "k", "n", "split", "inner_fold", "outer_fold"]
    expected = pd.concat([held.assign(inner_fold=fold, outer_fold=outer_fold)[keys]
                          for fold, (_, held) in enumerate(partitions, 1)], ignore_index=True)
    data.require_same_pairs(expected, rows)
    comparison = expected.merge(rows[keys], on=keys, validate="one_to_one")
    if len(comparison) != len(rows) or set(rows["painting"]) & set(forbidden["painting"]):
        raise ValueError("Inner OOF membership or training targets differ from the verified partitions")


def group_digest(frame, field):
    return digest_bytes("\n".join(sorted(set(frame[field]))).encode("utf-8"))


def inner_oof(directory, fit, forbidden, outer_fold, features, provenance):
    config = provenance["config"]
    partitions = list(inner_partitions(fit, forbidden, config, outer_fold))
    seed = provenance["exp002_config"]["fit_seed"]
    identity = {"version": 1, "outer_fold": outer_fold, "seed": config["inner_seed"] + outer_fold,
                "inputs": provenance, "outer_held_paintings_sha256": group_digest(forbidden, "painting")}
    output = directory / f"outer-{outer_fold}"
    csv = output / "predictions.csv"
    if output.exists():
        manifest = read_json(output / "run.json")
        if manifest["identity"] != identity or file_sha256(csv) != manifest["predictions_sha256"]:
            raise ValueError(f"Inner cache identity or checksum mismatch: {output}")
        rows = read_csv(csv)
        verify_inner(rows, partitions, forbidden, outer_fold)
        return rows, manifest

    records, audit = [], []
    for fold, (inner_fit, inner_held) in enumerate(partitions, 1):
        print(f"EXP-012 outer {outer_fold}, inner {fold}/{len(partitions)}", flush=True)
        predictions, fits = [], {}
        for name, bank in features.items():
            pred, fits[name] = fit_head(inner_fit, inner_held, bank, seed)
            if not fits[name]["converged"]:
                raise ValueError(f"{name} failed convergence in outer {outer_fold}, inner {fold}")
            predictions.append(pred)
        rows = signals(*predictions)
        records.append(rows.merge(inner_held[[*data.KEYS, "k", "n", "split"]], on=data.KEYS, validate="one_to_one")
                       .assign(outer_fold=outer_fold, inner_fold=fold))
        audit.append({"inner_fold": fold, "fit_pairs": len(inner_fit), "held_pairs": len(inner_held),
                      "fit_paintings": inner_fit["painting"].nunique(), "held_paintings": inner_held["painting"].nunique(),
                      "fit_paintings_sha256": group_digest(inner_fit, "painting"),
                      "held_paintings_sha256": group_digest(inner_held, "painting"),
                      "fit_images_sha256": group_digest(inner_fit, "image"),
                      "held_images_sha256": group_digest(inner_held, "image"), "fits": fits})
    rows = pd.concat(records, ignore_index=True)
    verify_inner(rows, partitions, forbidden, outer_fold)
    with new_output(output) as stage:
        rows.to_csv(stage / csv.name, index=False, lineterminator="\n")
        manifest = {"identity": identity, "predictions_sha256": file_sha256(stage / csv.name), "inner_folds": audit}
        json_output(manifest, stage / "run.json")
    return rows, manifest
