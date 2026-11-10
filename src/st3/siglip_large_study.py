"""Preflight and compare the four frozen EXP-015 systems."""

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, submission
from .baseline import file_sha256
from .cli import json_output
from .clip_large_study import E50_LARGE
from .contrast_study import new_output, write_development
from .e50_heads import read_json
from .fusion import blend
from .fusion_study import write_cv_analysis
from .low_rank_study import run_metadata, score_result
from .residual_crossfit import disjoint, outer_reference_bytes, residual_inputs
from .residual_study import development_components
from .siglip_large import extract_large, load_large_cache
from .visual_study import fit_head

MODELS = ("M0", "M1", "M2", "M3")


def run_large_preflight(args, dev):
    import torch

    start = time.perf_counter()
    train, _, _, _, provenance = residual_inputs(args, dev, "exp015")
    config = provenance["config"]
    encoder = config["encoder"]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with new_output(args.run_dir / "preflight") as stage:
        free = shutil.disk_usage(args.cache_dir).free
        cuda_free, cuda_total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        resource = {"free_disk_bytes": free, "free_cuda_bytes": cuda_free, "total_cuda_bytes": cuda_total}
        result = {**provenance, "resources": resource, "status": "RESOURCE_NO_GO"}
        if not torch.cuda.is_available():
            result["reason"] = "CUDA is unavailable"
        else:
            probe = {}
            names = sorted(set(train["image"]))[:config["probe_images"]]
            try:
                _, manifest = extract_large(args.data_dir, names, args.cache_dir, encoder,
                                            feature_dir=stage / "probe_cache", probe=probe)
            except torch.OutOfMemoryError as error:
                result["reason"] = str(error)
            else:
                if not probe.get("repeat_exact") or probe["output_dtype"] != "torch.float32":
                    raise ValueError("The probe did not verify deterministic FP32 feature extraction")
                result.update({"status": "FEASIBLE", "probe": probe, "embedding": {
                    **manifest, "path": str(Path(manifest["path"]).relative_to(stage))}})
        json_output({**run_metadata(result, start), "device": "CUDA"}, stage / "run.json")
    print(f"EXP-015 preflight: {result['status']}", flush=True)


def run_large_embed(args, dev):
    train, _, _, _, provenance = residual_inputs(args, dev, "exp015")
    preflight = read_json(args.run_dir / "preflight" / "run.json")
    if (preflight["status"] != "FEASIBLE" or not preflight["probe"]["repeat_exact"]
            or any(preflight[key] != value for key, value in provenance.items())):
        raise ValueError("EXP-015 requires a verified preflight on the same inputs")
    with new_output(args.run_dir / "extraction") as stage:
        _, manifest = extract_large(args.data_dir, train["image"], args.cache_dir, provenance["config"]["encoder"])
        json_output(manifest, stage / "features.json")


def signals(frame):
    return {key: frame[[*data.KEYS, field]].rename(columns={field: "p_positive"})
            for key, field in (("M0", "q0"), ("clip_b", "clip_b"), ("clip_l", "clip_l"))}


def compare_systems(frame, large):
    original = signals(frame)
    replacement = blend(blend(original["clip_b"], large, .5), original["clip_l"], .5)
    two_encoder = blend(original["clip_l"], large, .5)
    return {"M0": original["M0"], "M1": large, "M2": replacement, "M3": two_encoder}


def select_large(summary, config, folds):
    baseline = summary.loc["M0"]
    candidates = summary.loc[["M2", "M3"]]
    thresholds = {name: float(baseline["pooled_contrast"] + gain)
                  for name, gain in config["minimum_pooled_contrast_gain"].items()}
    loss = float(baseline["log_loss_mean"] + config["log_loss_guardrail_bits"])
    complete = candidates[["folds", "converged", "scored", "loss_scored"]].eq(folds).all(axis=1)
    eligible = candidates[complete & np.isfinite(candidates[["pooled_contrast", "log_loss_mean"]]).all(axis=1)
                          & candidates["pooled_contrast"].ge(pd.Series(thresholds))
                          & candidates["log_loss_mean"].le(loss)
                          & candidates["fold_wins"].ge(config["minimum_fold_wins"])]
    winner = eligible["pooled_contrast"].idxmax() if len(eligible) else "M0"
    return {"model": winner, "eligible_candidates": eligible.index.tolist(), "minimum_pooled_contrast": thresholds,
            "maximum_mean_log_loss": loss, "minimum_fold_wins": config["minimum_fold_wins"]}


def training_cache(args, train, config):
    source = read_json(args.run_dir / "extraction" / "features.json")
    return load_large_cache(args.data_dir, train["image"], args.cache_dir, config["encoder"], source)


def run_large_cv(args, dev):
    start = time.perf_counter()
    with new_output(args.run_dir / "cv") as stage:
        train, _, folds, outer, provenance = residual_inputs(args, dev, "exp015")
        config = provenance["config"]
        vectors, embedding = training_cache(args, train, config)
        records = {name: [] for name in ("predictions", "paintings", "languages")}
        rows, fits = [], []
        seed = provenance["exp002_config"]["fit_seed"]
        for fold, (fit_idx, held_idx) in enumerate(folds, 1):
            fit, held = train.iloc[fit_idx], train.iloc[held_idx]
            disjoint(fit, held)
            large, info = fit_head(fit, held, vectors, seed)
            fits.append({"fold": fold, **info, "fit_paintings": fit["painting"].nunique(),
                         "held_paintings": held["painting"].nunique()})
            frame = outer.loc[outer["outer_fold"].eq(fold)]
            for name, pred in compare_systems(frame, large).items():
                context = {"model": name, "fold": fold}
                contrast, loss = score_result(held, pred, context, records)
                fit_seconds = info["fit_seconds"] if name == "M1" else 0.
                predict_seconds = info["predict_seconds"] if name == "M1" else 0.
                rows.append({**context, "contrast": contrast, "log_loss": loss,
                             "converged": True if name == "M0" else info["converged"],
                             "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                             "seconds": fit_seconds + predict_seconds})
                print(f"Outer {fold} {name}: {contrast:.6f} / {loss:.6f}", flush=True)
        summary, _ = write_cv_analysis(stage, train, records, pd.DataFrame(rows), config, "M0", "M0")
        old = pd.read_csv(args.exp010_dir / "cv" / "summary.csv", index_col="model", float_precision="round_trip")
        columns = ["contrast_mean", "log_loss_mean", "pooled_contrast", "pooled_log_loss"]
        np.testing.assert_allclose(summary.loc["M0", columns].astype(float), old.loc[E50_LARGE, columns].astype(float),
                                   rtol=0., atol=1e-14)
        (stage / "reference_e50l.csv").write_bytes(outer_reference_bytes(outer, seed))
        json_output(fits, stage / "fits.json")
        json_output(run_metadata({**provenance, "training_embedding": embedding}, start), stage / "run.json")
        decision = select_large(summary, config, len(folds))
        decision["sha256"] = {name: file_sha256(stage / name) for name in ("run.json", "summary.csv", "fits.json")}
        json_output(decision, stage / "selection.json")
    print(f"Frozen EXP-015 decision: {decision['model']}", flush=True)


def run_large_dev(args, dev):
    start = time.perf_counter()
    cv = args.run_dir / "cv"
    decision = read_json(cv / "selection.json")
    for name in ("run.json", "summary.csv", "fits.json"):
        if decision["sha256"][name] != file_sha256(cv / name):
            raise ValueError(f"Frozen EXP-015 artifact changed: {name}")
    meta = read_json(cv / "run.json")
    config = meta["config"]
    summary = pd.read_csv(cv / "summary.csv", index_col="model", float_precision="round_trip")
    expected = select_large(summary, config, meta["exp002_config"]["cv_folds"])
    if any(decision[key] != value for key, value in expected.items()):
        raise ValueError("EXP-015 decision disagrees with the frozen gate")
    winner = decision["model"]
    if winner == "M0":
        print("EXP-015 NO-GO. Retained E50+L without development scoring or packaging.", flush=True)
        return
    with new_output(args.run_dir / "dev") as stage:
        train, _, _, _, provenance = residual_inputs(args, dev, "exp015")
        if any(meta[key] != value for key, value in provenance.items()):
            raise ValueError("Development inputs differ from the frozen EXP-015 run")
        reference, bundles = development_components(args, dev)
        train_features, embedding = training_cache(args, train, config)
        if embedding["features_sha256"] != meta["training_embedding"]["features_sha256"]:
            raise ValueError("Training features differ from CV")
        dev_features, dev_embedding = extract_large(args.data_dir, dev["image"], args.cache_dir, config["encoder"])
        large, info = fit_head(train, dev, {**train_features, **dev_features}, meta["exp002_config"]["fit_seed"])
        if not info["converged"]:
            raise ValueError("Full-training SigLIP 2 Large head did not converge")
        preds = compare_systems(reference, large)
        gold = data.validation_gold(data.read_annotations(args.data_dir), dev)
        write_development(stage, gold, {"M0": preds["M0"], winner: preds[winner]}, winner, config, "M0")
        archive = stage / "submission.zip"
        submission.pack(stage / "predictions" / winner / submission.FILENAME, archive, dev)
        submission.validate_archive(archive, dev)
        json_output(run_metadata({**provenance, "selection": decision, "fit": info, "dev_embedding": dev_embedding,
                                 "head_manifests_sha256": bundles, "submission_sha256": file_sha256(archive)},
                                 start), stage / "run.json")
    print(f"EXP-015 scored {winner} and validated {len(dev)} development predictions", flush=True)
