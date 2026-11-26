"""One-time locked confirmation and guarded original-training development fit."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import data, submission
from .baseline import file_sha256
from .caption_comparison import image_banks, refit_baseline
from .caption_data import allowed_captions, training_captions
from .caption_text import extract_teacher, load_teacher
from .cli import ROOT, json_output
from .contrast_study import new_output, write_development
from .fusion import verify_reference
from .metrics import score
from .polarity_experiment import fit_conditions, verified_decision
from .polarity_targets import polarity_means
from .polarity_workflow import (context, fingerprint, load_raw, old_inputs, read_sealed, seal, stage_budget)


def confirmation_passes(metrics, winner, config):
    gain = metrics[winner]["cultural_contrast"] - metrics["M0"]["cultural_contrast"]
    incremental = metrics[winner]["cultural_contrast"] - metrics["B1"]["cultural_contrast"]
    loss = metrics[winner]["log_loss"] - metrics["M0"]["log_loss"]
    passed = gain >= config["holdout_minimum_gain"] and incremental > 0 and loss <= config["holdout_loss_guardrail_bits"]
    return {"status": "PASS" if passed else "NO_GO", "contrast_gain": gain,
            "gain_over_B1": incremental, "log_loss_delta": loss}


def run_confirm(args):
    decision = verified_decision(args)
    if decision["status"] != "PASS":
        print("EXP-018 CV NO-GO; confirmation remains sealed", flush=True)
        return
    ctx = context(args)
    output = args.run_dir / "confirm"
    if output.exists():
        read_sealed(output, ctx.identity)
        print("Reused sealed one-time confirmation result", flush=True)
        return
    budget = stage_budget(args, "Confirm")
    all_train, fit, _ = old_inputs(args)
    held = all_train[all_train.role.eq("holdout")]
    ctx.banks = image_banks(args, all_train)
    winner = decision["model"]
    baseline, lr_fits = refit_baseline(fit, held, ctx.banks)
    directory = args.run_dir / "confirmation_work"
    predictions, fits = fit_conditions(args, ctx, load_raw(args, ctx.identity), fit, held, baseline, directory,
                                       6, budget, ("M1", "M" + winner[1:]))
    metrics = {name: score(held, predictions[name]) for name in ("M0", "B1", winner)}
    result = confirmation_passes(metrics, winner, ctx.config)
    with new_output(output) as stage:
        write_development(stage, held, {name: predictions[name] for name in ("M0", winner)}, winner, ctx.config, "M0")
        control = stage / "matched_control"
        control.mkdir()
        write_development(control, held, {name: predictions[name] for name in ("B1", winner)}, winner, ctx.config, "B1")
        json_output(metrics, stage / "confirmation_metrics.json")
        json_output({"lr_fits": lr_fits, "student_fits": fits}, stage / "fits.json")
        seal(stage, ctx.identity, {**result, "model": winner,
            "cv_decision_sha256": file_sha256(args.run_dir / "decision/run.json"),
            "seed_manifests": {f"{model}_{seed}": file_sha256(directory / "seeds" / f"{model}_{seed}" / "run.json")
                               for model in ("M1", "M" + winner[1:]) for seed in ctx.config["seeds"]}})
    print(f"One-time confirmation {result['status']} for {winner}", flush=True)


def confirmed_decision(args):
    cv = verified_decision(args)
    if cv["status"] != "PASS":
        return None
    ctx = context(args)
    result = read_sealed(args.run_dir / "confirm", ctx.identity)
    if result["model"] != cv["model"] or result["cv_decision_sha256"] != file_sha256(args.run_dir / "decision/run.json"):
        raise ValueError("Confirmation candidate differs from the frozen CV decision")
    metrics = json.loads((args.run_dir / "confirm/confirmation_metrics.json").read_text())
    expected = confirmation_passes(metrics, result["model"], ctx.config)
    if any(result[key] != value for key, value in expected.items()):
        raise ValueError("Confirmation result differs from its frozen guardrails")
    for name, digest in result["seed_manifests"].items():
        directory = args.run_dir / "confirmation_work/seeds" / name
        read_sealed(directory, ctx.identity)
        if file_sha256(directory / "run.json") != digest:
            raise ValueError("Confirmation checkpoint changed")
    return result if result["status"] == "PASS" else None


def full_training_text(args, ctx, train, budget):
    """Extend into an EXP-018 cache without writing the EXP-017 cache."""
    output = args.run_dir / "full_text"
    _, partition, old_audit = old_inputs(args)
    original, original_manifest = load_teacher(args.exp017_dir / "text/clip", ctx.config["teacher"], old_audit,
                                               partition.painting.unique())
    audit = {"config_sha256": ctx.identity["config_sha256"], "protocol_sha256": old_audit["protocol_sha256"]}
    if output.exists():
        return load_teacher(output, ctx.config["teacher"], audit, train.painting.unique())[0]
    captions = training_captions(args.data_dir, allowed_paintings=train.painting.unique())
    selected = allowed_captions(captions, train)
    needed = selected[~selected.caption_id.isin(original)]
    extension = {}
    extension_manifest = None
    if not needed.empty:
        extension_directory = args.run_dir / "text_extension"
        extension_pairs = train.merge(needed[data.KEYS].drop_duplicates(), on=data.KEYS, validate="one_to_one")
        if not extension_directory.exists():
            extraction_args = SimpleNamespace(**vars(args))
            extraction_args.teacher = "clip"
            extract_teacher(extraction_args, needed, extension_pairs, {"teachers": {"clip": ctx.config["teacher"]}},
                            audit, extension_directory, feature_directory=args.cache_dir / "exp018_text",
                            progress_hook=lambda count: budget.check("new official-training caption extraction"))
        extension, extension_manifest = load_teacher(extension_directory, ctx.config["teacher"], audit,
                                                      extension_pairs.painting.unique())
    keys = sorted(set(selected.caption_id))
    combined = {**original, **extension}
    if not set(keys).issubset(combined):
        raise ValueError("Extended CLIP text cache is incomplete")
    with new_output(output) as stage:
        np.savez(stage / "features.npz", keys=np.array(keys), features=np.stack([combined[key] for key in keys]))
        json_output({**audit, "identity": ctx.config["teacher"], "fit_paintings": sorted(train.painting.unique()),
                     "features_sha256": file_sha256(stage / "features.npz"),
                     "legacy_profile": original_manifest["identity"],
                     "extension_profile": extension_manifest["identity"] if extension_manifest else None,
                     "legacy_vectors_reused": len(set(keys) & set(original)), "new_vectors": len(extension)}, stage / "run.json")
    return load_teacher(output, ctx.config["teacher"], audit, train.painting.unique())[0]


def run_dev(args):
    confirmed = confirmed_decision(args)
    if confirmed is None:
        print("EXP-018 public development remains unscored because the promotion gates did not pass", flush=True)
        return
    ctx = context(args)
    output = args.run_dir / "dev"
    if output.exists():
        read_sealed(output, ctx.identity)
        print("Reused sealed one-time development result", flush=True)
        return
    budget = stage_budget(args, "Dev")
    train, _, _ = old_inputs(args)
    vectors = full_training_text(args, ctx, train, budget)
    captions = training_captions(args.data_dir, allowed_paintings=train.painting.unique())
    records, means, available = polarity_means(captions, train, vectors)
    raw = records, means, available
    dev = data.read_dev_input(args.data_dir)
    dev["image_sha256"] = dev.image.map(lambda name: file_sha256(data.image_path(args.data_dir, name)))
    ctx.banks = image_banks(args, pd.concat([train, dev], ignore_index=True))
    baseline, lr_fits = refit_baseline(train, dev, ctx.banks)
    reference_path = ROOT / "runs/exp010_clip_large/dev/predictions/E50+L/predictions.csv"
    verify_reference(baseline, pd.read_csv(reference_path, float_precision="round_trip"))
    winner = confirmed["model"]
    predictions, fits = fit_conditions(args, ctx, raw, train, dev, baseline, args.run_dir / "development_work",
                                       7, budget, ("M" + winner[1:],))
    gold = data.validation_gold(data.read_annotations(args.data_dir), dev)
    with new_output(output) as stage:
        write_development(stage, gold, {"M0": baseline, winner: predictions[winner]}, winner, ctx.config, "M0")
        submission.pack(stage / "predictions" / winner / submission.FILENAME, stage / "submission.zip", dev)
        submission.validate_archive(stage / "submission.zip", dev)
        json_output({"lr_fits": lr_fits, "student_fits": fits}, stage / "fits.json")
        seal(stage, ctx.identity, {"model": winner, "eligibility": ctx.config["eligibility"], "pairs": len(dev),
                                   "confirm_manifest_sha256": file_sha256(args.run_dir / "confirm/run.json")})
    print(f"Scored the fixed {winner} on {len(dev)} public development pairs and validated a separate ZIP", flush=True)
