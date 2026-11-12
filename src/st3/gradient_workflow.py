"""Read-only EXP-018 validation and separately sealed EXP-019 stage identities."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import data
from .baseline import file_sha256
from .caption_comparison import image_banks
from .caption_data import inner_partition, require_isolation
from .cli import ROOT, json_output
from .contrast_study import new_output
from .fusion import verify_reference
from .polarity_experiment import choose_candidate
from .polarity_workflow import (ExecutionBudget, StageBlocked, config018, fingerprint, load_geometry,
                                old_inputs, read_sealed, seal)


def config019():
    return json.loads((ROOT / "configs/exp019.json").read_text())


def parent_args(args):
    return SimpleNamespace(**{**vars(args), "run_dir": args.exp018_dir})


def validate_parent(args):
    """Check original source keys verbatim; new EXP-019 files may be additive."""
    cfg = config019()
    if file_sha256(ROOT / "configs/exp018.json") != cfg["parent_config_sha256"]:
        raise ValueError("Pinned EXP-018 config changed")
    root = args.exp018_dir
    if not (root / "decision/run.json").exists():
        raise StageBlocked("Completed EXP-018 CV/replay missing. On the EXP-018 branch run .\\scripts\\run_exp018.ps1 -Stage CV then -Stage Verify -DeepVerify")
    original = json.loads((root / "decision/run.json").read_text())["identity"]
    current = fingerprint(parent_args(args))
    original_sources = original["source_sha256"]
    if any(current["source_sha256"].get(key) != digest for key, digest in original_sources.items()):
        raise ValueError("Original EXP-018 implementation changed")
    current["source_sha256"] = original_sources
    if current != original:
        raise ValueError("EXP-018 source/config/cache/runtime identity differs")
    for name, digest in cfg["parent_manifests"].items():
        path = root / name
        if file_sha256(path) != digest:
            raise ValueError(f"Pinned EXP-018 manifest changed: {name}")
        read_sealed(path.parent, original)
    attempt_paths = sorted((root / "verification_attempts").glob("*.json"))
    if not attempt_paths:
        raise StageBlocked("EXP-018 has no successful replay attempt")
    latest = json.loads(attempt_paths[-1].read_text())
    cv_hash = file_sha256(root / "cv/run.json")
    if latest["status"] != "PASS" or latest["cv_manifest_sha256"] != cv_hash:
        raise StageBlocked("Latest EXP-018 replay incomplete/failed; rerun Verify on its original branch")
    decision = read_sealed(root / "decision", original)
    verification = read_sealed(root / "verification", original)
    summary = pd.read_csv(root / "cv/summary.csv", index_col="model", float_precision="round_trip")
    expected = choose_candidate(summary, config018(), True)
    if any(decision[key] != value for key, value in expected.items()) or verification["checkpoints_replayed"] != 45:
        raise ValueError("EXP-018 decision or replay coverage differs")
    if (decision["cv_manifest_sha256"] != cv_hash or decision["verification_manifest_sha256"] != file_sha256(root / "verification/run.json")):
        raise ValueError("EXP-018 decision evidence changed")
    if (root / "confirm").exists() or (root / "dev").exists():
        raise ValueError("Expected EXP-018 confirmation/public Development to remain sealed")
    _, train, _ = old_inputs(parent_args(args))
    cv = read_sealed(root / "cv", original)
    if cv["pilot_manifest_sha256"] != file_sha256(root / "pilot/run.json"):
        raise ValueError("EXP-018 pilot changed")
    for fold in range(1, 6):
        directory = root / "work" / f"fold_{fold}"
        if file_sha256(directory / "result/run.json") != cv["fold_manifests"][str(fold)]:
            raise ValueError("EXP-018 CV fold reference differs")
        result = read_sealed(directory / "result", original)
        fit, held = train[train.fold.ne(fold)], train[train.fold.eq(fold)]
        inner, stopping = inner_partition(fit, config018(), fold)
        require_isolation(inner, stopping, held)
        for phase, partition in (("inner", inner), ("full", fit)):
            load_geometry(directory / "geometry" / phase, original, partition.painting)
        for name, digest in result["seed_manifests"].items():
            path = directory / "seeds" / name
            if file_sha256(path / "run.json") != digest:
                raise ValueError("EXP-018 seed reference changed")
            metadata = read_sealed(path, original)
            for key, rows in (("fit", fit), ("inner", inner), ("stopping", stopping), ("held", held)):
                if metadata[key + "_paintings"] != sorted(rows.painting.unique()):
                    raise ValueError("EXP-018 seed painting/fold identities changed")
            if not metadata["finite"]:
                raise ValueError("EXP-018 nonfinite seed")
    history = json.loads((root / "preflight/historical_sha256.json").read_text())
    for name, digest in history.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"Historical artifact changed: {name}")
    # Reused predictions are compared to the exact original E50+L fold source.
    predictions = pd.read_csv(root / "cv/predictions.csv", float_precision="round_trip")
    reference = pd.read_csv(args.exp017_dir / "cv/predictions.csv", float_precision="round_trip")
    verify_reference(predictions[predictions.model.eq("M0")][[*data.KEYS, "p_positive"]],
                     reference[reference.model.eq("M0")][[*data.KEYS, "p_positive"]])
    data.require_same_pairs(train, predictions[predictions.model.eq("M0")])
    joined = train[data.KEYS + ["fold"]].merge(predictions[predictions.model.eq("M0")], on=data.KEYS, suffixes=("_expected", ""), validate="one_to_one")
    if not joined.fold.eq(joined.fold_expected).all():
        raise ValueError("Reused OOF painting-fold assignment changed")
    return original, train, predictions, decision, len(history)


def identity019(args, parent_identity):
    current = fingerprint(parent_args(args))
    files = [ROOT / "scripts/run_exp019.py", ROOT / "scripts/run_exp019.ps1"]
    return {"parent_identity": parent_identity, "runtime_and_source": current,
            "config_sha256": file_sha256(ROOT / "configs/exp019.json"),
            "runners": {p.name: file_sha256(p) for p in files}}


def context019(args, *, preflight=True):
    parent_identity, train, predictions, decision, history_count = validate_parent(args)
    identity = identity019(args, parent_identity)
    if preflight:
        read_sealed(args.run_dir / "preflight", identity)
    cfg = config018()
    return SimpleNamespace(identity=identity, parent_identity=parent_identity,
        config={**cfg, "text_weight": 1.}, protocol=config019(), train=train,
        predictions=predictions, parent_decision=decision, history_count=history_count,
        banks=image_banks(args, train))


def run_preflight(args):
    ctx = context019(args, preflight=False)
    output = args.run_dir / "preflight"
    if output.exists():
        read_sealed(output, ctx.identity)
        print("Reused verified EXP-019 preflight", flush=True)
        return
    files = [p for p in args.exp018_dir.rglob("*") if p.is_file()]
    history = {str(p.resolve()): file_sha256(p) for p in files}
    with new_output(output) as stage:
        json_output(history, stage / "parent_files_sha256.json")
        json_output({"parent_cv_decision": ctx.parent_decision, "reused_models": {"M0":"M0", "C0":"M1", "C1":"M3", "E0":"B1", "E1":"B3"},
                     "new_models": 15, "new_training_stages": 30, "confirmation_paintings_sealed": 236,
                     "parent_early_checkpoints": "UNAVAILABLE: EXP-018 saved only final refits; never reconstructed by retraining"}, stage / "provenance.json")
        seal(stage, ctx.identity, {"status":"INPUTS_VERIFIED", "parent_files":len(history), "historical_files":ctx.history_count})
    print(f"EXP-018 complete and replayed; {len(history)} files preserved; 15 C2 fits only", flush=True)


def preserve_parent(args, identity):
    read_sealed(args.run_dir / "preflight", identity)
    hashes = json.loads((args.run_dir / "preflight/parent_files_sha256.json").read_text())
    actual = {str(p.resolve()) for p in args.exp018_dir.rglob("*") if p.is_file()}
    if actual != set(hashes):
        raise ValueError("EXP-018 files added or removed")
    for name, digest in hashes.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"Preserved EXP-018 file changed: {name}")
    return len(hashes)


def budget019(args, ctx, stage):
    pilot = read_sealed(args.run_dir / "gradient_pilot", ctx.identity)
    if args.max_seconds:
        return ExecutionBudget(args.max_seconds)
    seconds = pilot["estimated_total_seconds"] if stage == "CV" else pilot["estimated_verification_seconds"]
    return ExecutionBudget(max(120., 1.5 * seconds))
