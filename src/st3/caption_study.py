"""Audited, frozen-stage entry points for EXP-017."""

import json

import pandas as pd

from . import data
from .baseline import file_sha256
from .caption_data import audit_captions, lock_protocol, training_captions
from .cli import ROOT, json_output
from .contrast_study import new_output


def config017():
    return json.loads((ROOT / "configs/exp017.json").read_text(encoding="utf-8"))


def audited_inputs(args):
    config = config017()
    recorded = json.loads((args.run_dir / "audit/run.json").read_text())
    if (recorded["config_sha256"] != file_sha256(ROOT / "configs/exp017.json")
            or recorded["annotations_sha256"] != file_sha256(args.data_dir / data.ANNOTATIONS)
            or recorded["protocol_sha256"] != file_sha256(args.run_dir / "audit/protocol.csv")):
        raise ValueError("EXP-017 audit, locked split or original data changed")
    annotations = data.read_annotations(args.data_dir, training_only=True)
    train = data.training_pairs(annotations).assign(split="train")
    protocol = pd.read_csv(args.run_dir / "audit/protocol.csv")
    train = train.merge(protocol.drop(columns="image"), on="painting", validate="many_to_one")
    if len(train) != config["training_pairs"]:
        raise ValueError("Original eligible training coverage differs")
    return config, train, recorded


def run_caption_audit(args):
    config = config017()
    from .cli import git_head

    if git_head(args.data_dir) != config["upstream_revision"]:
        raise ValueError("Organizer checkout differs from the required revision")
    annotations = data.read_annotations(args.data_dir)
    captions = training_captions(args.data_dir)
    train = data.training_pairs(annotations).assign(split="train")
    if len(train) != config["training_pairs"]:
        raise ValueError("Expected 16,019 original eligible training pairs")
    protocol = lock_protocol(train, args.data_dir, annotations, config)
    summary, languages = audit_captions(captions, train)
    with new_output(args.run_dir / "audit") as stage:
        protocol.to_csv(stage / "protocol.csv", index=False, lineterminator="\n")
        languages.to_csv(stage / "languages.csv", index=False, lineterminator="\n")
        captions.groupby([*data.KEYS, "emotion"]).size().rename("annotations").to_csv(stage / "annotation_counts.csv")
        captions[["annotation_id", *data.KEYS, "emotion", "caption_id", "valid_caption", "binary"]].to_csv(
            stage / "annotation_mapping.csv", index=False, lineterminator="\n")
        summary.update({"status": "PASS", "config_sha256": file_sha256(ROOT / "configs/exp017.json"),
                        "annotations_sha256": file_sha256(args.data_dir / data.ANNOTATIONS),
                        "protocol_sha256": file_sha256(stage / "protocol.csv"),
                        "painting_counts": protocol.role.value_counts().to_dict(),
                        "image_content_groups": protocol.image_sha256.nunique(),
                        "holdout_paintings": protocol.loc[protocol.role.eq("holdout"), "painting"].tolist()})
        json_output(summary, stage / "run.json")
    print(f"EXP-017 audit PASS, {len(train)} pairs; locked groups {summary['painting_counts']}", flush=True)


def run_caption_extract(args):
    from .caption_text import extract_teacher

    config, train, audit = audited_inputs(args)
    partition = train if args.full_training else train[train.role.eq("development")]
    if args.full_training:
        decision = checked_decision(args, "holdout")
        if decision["status"] != "PASS":
            raise ValueError("Full-training text extraction requires locked confirmation")
    captions = training_captions(args.data_dir)
    stage_name = "full_text" if args.full_training else "text"
    output = args.run_dir / stage_name / args.teacher
    extract_teacher(args, captions, partition, config, audit, output)


def checked_decision(args, phase):
    directory = args.run_dir / phase
    decision = json.loads((directory / "selection.json").read_text())
    for name, digest in decision["sha256"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"Frozen EXP-017 decision changed: {name}")
    config, _, audit = audited_inputs(args)
    if decision["config_sha256"] != audit["config_sha256"]:
        raise ValueError("The frozen decision has a different EXP-017 configuration")
    if phase == "cv":
        from .caption_comparison import CONDITIONS, select_caption

        fits = json.loads((directory / "fits.json").read_text())
        expected = {(fold, model, seed) for fold in range(1, config["cv_folds"] + 1)
                    for model in CONDITIONS for seed in config["seeds"]}
        complete = {(row["fold"], row["model"], row["seed"]) for row in fits} == expected and len(fits) == len(expected)
        table = pd.read_csv(directory / "summary.csv", index_col="model", float_precision="round_trip")
        recomputed = select_caption(table, config, complete)
        if any(decision[key] != value for key, value in recomputed.items()):
            raise ValueError("EXP-017 selection disagrees with the frozen CV gate")
    else:
        prior = checked_decision(args, "cv")
        if prior["status"] != "PASS" or prior["model"] != decision["model"]:
            raise ValueError("Locked confirmation differs from the frozen CV winner")
        metrics = json.loads((directory / "metrics.json").read_text())
        winner = metrics[decision["model"]]
        gain = winner["cultural_contrast"] - metrics["M0"]["cultural_contrast"]
        incremental = winner["cultural_contrast"] - metrics["B1"]["cultural_contrast"]
        loss = winner["log_loss"] - metrics["M0"]["log_loss"]
        passed = gain >= config["holdout_minimum_gain"] and incremental > 0 and loss <= config["holdout_loss_guardrail_bits"]
        if decision["status"] != ("PASS" if passed else "NO_GO"):
            raise ValueError("EXP-017 confirmation disagrees with the frozen holdout gate")
    return decision


def run_caption(args):
    if args.command == "caption-audit":
        run_caption_audit(args)
    elif args.command == "caption-extract":
        run_caption_extract(args)
    elif args.command == "caption-token-audit":
        from .caption_text import audit_training_tokens

        config, _, _ = audited_inputs(args)
        output = args.run_dir / "audit" / f"{args.teacher}_all_training_tokens.csv"
        if output.exists():
            raise ValueError(f"Output already exists: {output}")
        audit_training_tokens(args, training_captions(args.data_dir), config["teachers"][args.teacher], output)
        print(f"Audited all original training tokens for {args.teacher}", flush=True)
    else:
        from .caption_comparison import run_caption_cv, run_caption_holdout, run_caption_dev

        runners = {"caption-cv": run_caption_cv, "caption-holdout": run_caption_holdout,
                   "caption-dev": run_caption_dev}
        runners[args.command](args)
