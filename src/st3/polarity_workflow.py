"""Pinned inputs, sealed artifacts and bounded execution for EXP-018."""

import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import data
from .baseline import file_sha256
from .caption_comparison import device017, image_banks
from .caption_data import require_isolation, training_captions
from .caption_study import audited_inputs
from .caption_text import load_teacher
from .cli import ROOT, git_head, json_output
from .contrast_study import new_output
from .polarity_targets import fit_shared_geometry, polarity_audit


class BudgetStop(RuntimeError):
    pass


class StageBlocked(RuntimeError):
    pass


class ExecutionBudget:
    def __init__(self, seconds):
        self.start = time.perf_counter()
        self.seconds = seconds

    def check(self, phase="stage", epoch=0):
        elapsed = time.perf_counter() - self.start
        if elapsed >= self.seconds:
            raise BudgetStop(f"Budget reached after {elapsed:.1f}s during {phase}; completed seeds remain resumable")
        if epoch == 1 or epoch % 10 == 0:
            print(f"{phase} epoch {epoch}, elapsed {elapsed:.1f}s / {self.seconds:.1f}s", flush=True)


def config018():
    return json.loads((ROOT / "configs/exp018.json").read_text())


def fingerprint(args):
    config = config018()
    if file_sha256(ROOT / "configs/exp017.json") != config["exp017_config_sha256"]:
        raise ValueError("The pinned EXP-017 configuration changed")
    for name, digest in config["exp017_sha256"].items():
        if file_sha256(args.exp017_dir / name) != digest:
            raise ValueError(f"Pinned EXP-017 artifact changed: {name}")
    files = sorted((ROOT / "src/st3").glob("*.py")) + [ROOT / "scripts/run_exp018.ps1", ROOT / "scripts/run_st3.py"]
    return {"config_sha256": file_sha256(ROOT / "configs/exp018.json"),
            "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in files},
            "exp017_sha256": config["exp017_sha256"], "device": str(device017(args.device)),
            "python": platform.python_version(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if str(device017(args.device)) == "cuda" else None,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "numpy", "pandas", "scipy", "scikit-learn", "transformers")}}


def seal(directory, identity, metadata):
    files = {path.relative_to(directory).as_posix(): file_sha256(path) for path in directory.rglob("*")
             if path.is_file() and path != directory / "run.json"}
    json_output({"identity": identity, "metadata": metadata, "files_sha256": files,
                 "project_revision": git_head(ROOT), "python": platform.python_version()}, directory / "run.json")


def read_sealed(directory, identity):
    if not (directory / "run.json").is_file():
        raise StageBlocked(f"Missing or unpublished stage/checkpoint: {directory}")
    result = json.loads((directory / "run.json").read_text())
    if result["identity"] != identity:
        raise ValueError(f"Incompatible source, inputs or runtime for {directory}")
    files = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
             if path.is_file() and path != directory / "run.json"}
    if files != set(result["files_sha256"]):
        raise ValueError(f"Missing, partial or extra checkpoint files in {directory}")
    for name, digest in result["files_sha256"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {directory / name}")
    return result["metadata"]


def old_inputs(args):
    previous = SimpleNamespace(run_dir=args.exp017_dir, data_dir=args.data_dir, cache_dir=args.cache_dir)
    _, train, audit = audited_inputs(previous)
    partition = train[train.role.eq("development")]
    locked = train[train.role.eq("holdout")]
    require_isolation(partition, locked)
    if partition.painting.nunique() != 1335 or locked.painting.nunique() != 236:
        raise ValueError("EXP-018 must reuse the exact EXP-017 painting protocol")
    return train, partition, audit


def context(args, *, teacher=False):
    config = config018()
    identity = fingerprint(args)
    read_sealed(args.run_dir / "preflight", identity)
    _, partition, audit = old_inputs(args)
    reference = pd.read_csv(args.exp017_dir / "cv/predictions.csv", float_precision="round_trip")
    reference = reference[reference.model.eq("M0")][[*data.KEYS, "p_positive", "fold"]]
    data.require_same_pairs(partition, reference)
    joined = partition[[*data.KEYS, "fold"]].merge(reference, on=data.KEYS, suffixes=("_expected", "_actual"), validate="one_to_one")
    if not joined.fold_expected.eq(joined.fold_actual).all():
        raise ValueError("E50+L OOF predictions do not match the locked fold identities")
    banks = image_banks(args, partition)
    vectors = captions = None
    if teacher:
        vectors, _ = load_teacher(args.exp017_dir / "text/clip", config["teacher"], audit, partition.painting.unique())
        captions = training_captions(args.data_dir, allowed_paintings=partition.painting.unique())
    return SimpleNamespace(config=config, identity=identity, train=partition, reference=reference,
                           banks=banks, vectors=vectors, captions=captions)


def run_preflight(args):
    start = time.perf_counter()
    config = config018()
    identity = fingerprint(args)
    output = args.run_dir / "preflight"
    if output.exists():
        read_sealed(output, identity)
        print("Reused verified EXP-018 preflight", flush=True)
        return
    if git_head(args.data_dir) != config["upstream_revision"]:
        raise ValueError("Organizer revision differs from the pinned checkout")
    _, partition, audit = old_inputs(args)
    load_teacher(args.exp017_dir / "text/clip", config["teacher"], audit, partition.painting.unique())
    image_banks(args, partition)
    original = json.loads((args.exp017_dir / "independent_verification.json").read_text())
    if original["status"] != "PASS" or original["checkpoints_replayed"] != 45:
        raise ValueError("The reused EXP-017 baseline lacks its successful replay audit")
    paths = [path for path in args.exp017_dir.rglob("*") if path.is_file()]
    paths += [path for name in ("e50_inference_models", "e50l_inference_models")
              for path in (ROOT / "runs" / name).iterdir() if path.is_file()]
    paths += list((args.cache_dir / "exp017_text").glob("caption-clip-*.sqlite3"))
    paths += [ROOT / "runs/exp010_clip_large/dev/submission.zip",
              ROOT / "runs/exp010_clip_large/dev/predictions/E50+L/predictions.csv"]
    history = {str(path.resolve()): file_sha256(path) for path in paths}
    old_run = json.loads((args.exp017_dir / "cv/run.json").read_text())
    with new_output(output) as stage:
        json_output(history, stage / "historical_sha256.json")
        metadata = {"status": "INPUTS_VERIFIED", "training_paintings": 1335, "locked_paintings": 236,
                    "development_training_pairs": len(partition), "history_files": len(history),
                    "disk_free_bytes": shutil.disk_usage(args.run_dir.parent).free,
                    "cuda_device": torch.cuda.get_device_name() if identity["device"] == "cuda" else None,
                    "cuda_free_bytes": torch.cuda.mem_get_info()[0] if identity["device"] == "cuda" else None,
                    "reference_cv_seconds": old_run["wall_seconds"],
                    "reference_verification_seconds": original["seconds"],
                    "initial_total_estimate_seconds": old_run["wall_seconds"] + original["seconds"] + 90.,
                    "seconds": time.perf_counter() - start}
        seal(stage, identity, metadata)
    print(f"EXP-018 preflight verified inputs in {metadata['seconds']:.2f}s; no encoder extraction", flush=True)


def run_audit(args):
    ctx = context(args, teacher=True)
    output = args.run_dir / "audit"
    if output.exists():
        read_sealed(output, ctx.identity)
        print("Reused verified polarity audit", flush=True)
        return
    start = time.perf_counter()
    summary, records, variation, means, available = polarity_audit(ctx.captions, ctx.train, ctx.vectors)
    data.require_same_pairs(ctx.train, records)
    with new_output(output) as stage:
        records.to_csv(stage / "pairs.csv", index=False, lineterminator="\n")
        variation.to_csv(stage / "variation.csv", index=False, lineterminator="\n")
        np.savez(stage / "raw_means.npz", means=means, available=available)
        json_output(summary, stage / "summary.json")
        seal(stage, ctx.identity, {"status": "AUDITED", "seconds": time.perf_counter() - start,
                                   "pairs": len(records), "holdout_captions_retained": 0})
    print(f"Polarity audit: {summary['both']} both, {summary['positive_only']} positive-only, "
          f"{summary['negative_only']} negative-only", flush=True)


def load_raw(args, identity):
    read_sealed(args.run_dir / "audit", identity)
    records = pd.read_csv(args.run_dir / "audit/pairs.csv")
    with np.load(args.run_dir / "audit/raw_means.npz", allow_pickle=False) as archive:
        means, available = archive["means"], archive["available"]
    if means.shape[:2] != (len(records), 3) or available.shape != (len(records), 3):
        raise ValueError("Raw polarity cache coverage differs")
    return records, means, available


def fit_partition_geometry(raw, partition, config):
    records, means, available = raw
    keys = pd.MultiIndex.from_frame(records[data.KEYS])
    ordered = partition.sort_values(data.KEYS)
    indices = keys.get_indexer(pd.MultiIndex.from_frame(ordered[data.KEYS]))
    if (indices < 0).any() or len(set(indices)) != len(indices):
        raise ValueError("Polarity cache does not cover each training pair exactly once")
    return fit_shared_geometry(records.iloc[indices].reset_index(drop=True), means[indices], available[indices], config)


def save_geometry(directory, bundle):
    bundle["records"].to_csv(directory / "pairs.csv", index=False, lineterminator="\n")
    pca = bundle["pca"]
    np.savez(directory / "targets.npz", targets=bundle["targets"], valid=bundle["valid"],
             components=pca.components_, mean=pca.mean_, variance=np.nan_to_num(pca.explained_variance_ratio_, nan=0.))
    json_output(bundle["metadata"], directory / "pca.json")


def load_geometry(directory, identity, paintings):
    read_sealed(directory, identity)
    metadata = json.loads((directory / "pca.json").read_text())
    if metadata["fit_paintings"] != sorted(set(paintings)) or metadata["fit_kind"] != "all" or metadata["whiten"]:
        raise ValueError("PCA fitted outside the applicable training partition")
    with np.load(directory / "targets.npz", allow_pickle=False) as archive:
        targets, valid = archive["targets"], archive["valid"]
    if (not np.isfinite(targets).all() or not np.allclose(np.linalg.norm(targets[valid], axis=-1), 1., atol=2e-6)
            or np.count_nonzero(targets[~valid])):
        raise ValueError("Invalid normalized target geometry or masks")
    return {"records": pd.read_csv(directory / "pairs.csv"), "targets": targets, "valid": valid, "metadata": metadata}


def geometry(args, ctx, raw, fit, inner, directory):
    bundles = {}
    for phase, partition in (("inner", inner), ("full", fit)):
        output = directory / phase
        if not output.exists():
            with new_output(output) as stage:
                bundle = fit_partition_geometry(raw, partition, ctx.config)
                save_geometry(stage, bundle)
                seal(stage, ctx.identity, {"phase": phase})
        bundles[phase] = load_geometry(output, ctx.identity, partition.painting)
    return bundles


def stage_budget(args, stage):
    pilot = read_sealed(args.run_dir / "pilot", fingerprint(args))
    if pilot["status"] != "READY_FOR_LOCAL_RUN":
        raise StageBlocked("A valid finite-gradient pilot is required before heavy execution")
    if args.max_seconds > 0:
        return ExecutionBudget(args.max_seconds)
    estimate = pilot["estimated_cv_seconds"] if stage == "CV" else pilot["estimated_verification_seconds"]
    if stage == "Verify" and args.deep_verify:
        estimate += pilot["two_pca_fits_seconds"] * config018()["cv_folds"]
    if stage in ("Confirm", "Dev"):
        estimate = max(180., pilot["estimated_cv_seconds"] / 3.)
    return ExecutionBudget(max(60., estimate * 1.5))
