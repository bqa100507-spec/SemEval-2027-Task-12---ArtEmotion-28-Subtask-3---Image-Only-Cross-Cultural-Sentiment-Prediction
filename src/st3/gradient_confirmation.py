"""Guarded reuse of EXP-018 confirmation mechanics, writing only EXP-019 files."""

import json

import numpy as np
import pandas as pd

from . import data, submission
from .baseline import file_sha256
from .caption_comparison import image_banks, refit_baseline
from .caption_data import training_captions
from .caption_text import load_teacher
from .cli import ROOT, json_output
from .contrast_study import new_output, write_development
from .fusion import verify_reference
from .gradient_experiment import verified_decision
from .gradient_workflow import budget019, context019, parent_args
from .metrics import score
from .polarity_confirmation import confirmation_passes
from .polarity_experiment import fit_conditions
from .polarity_targets import polarity_means
from .polarity_workflow import StageBlocked, load_raw, old_inputs, read_sealed, seal


def run_confirm(args):
    decision=verified_decision(args)
    if decision["status"]!="PASS":
        raise StageBlocked("COMPETITION_NO_GO; confirmation remains sealed")
    ctx=context019(args)
    if (args.run_dir/"confirm").exists():
        read_sealed(args.run_dir/"confirm",ctx.identity);return
    train,fit,_=old_inputs(parent_args(args));held=train[train.role.eq("holdout")]
    ctx.banks=image_banks(args,train)
    baseline,lr_fits=refit_baseline(fit,held,ctx.banks)
    budget=budget019(args,ctx,"Confirm")
    predictions,fits=fit_conditions(args,ctx,load_raw(parent_args(args),ctx.parent_identity),fit,held,baseline,
        args.run_dir/"confirmation_work",6,budget,("M1","M3"))
    metrics={name:score(held,predictions[name]) for name in ("M0","B1","B3")}
    result=confirmation_passes(metrics,"B3",ctx.config)
    with new_output(args.run_dir/"confirm") as stage:
        write_development(stage,held,{name:predictions[name] for name in ("M0","B1","B3")},"B3",ctx.config,"M0")
        json_output(metrics,stage/"metrics.json");json_output({"lr":lr_fits,"students":fits},stage/"fits.json")
        seal(stage,ctx.identity,{**result,"model":"E2","cv_decision_sha256":file_sha256(args.run_dir/"decision/run.json"),
            "seed_manifests":{f"{name}_{seed}":file_sha256(args.run_dir/"confirmation_work/seeds"/f"{name}_{seed}"/"run.json") for name in ("M1","M3") for seed in ctx.config["seeds"]}})


def confirmed(args):
    decision=verified_decision(args)
    if decision["status"]!="PASS":
        raise StageBlocked("COMPETITION_NO_GO; Development remains sealed")
    ctx=context019(args);result=read_sealed(args.run_dir/"confirm",ctx.identity)
    if result["cv_decision_sha256"]!=file_sha256(args.run_dir/"decision/run.json"):
        raise ValueError("Confirmation decision changed")
    metrics=json.loads((args.run_dir/"confirm/metrics.json").read_text())
    if any(result[k]!=v for k,v in confirmation_passes(metrics,"B3",ctx.config).items()):
        raise ValueError("Confirmation gate differs")
    for name,digest in result["seed_manifests"].items():
        path=args.run_dir/"confirmation_work/seeds"/name
        read_sealed(path,ctx.identity)
        if file_sha256(path/"run.json")!=digest:raise ValueError("Confirmation checkpoint changed")
    if result["status"]!="PASS":raise StageBlocked("Confirmation failed; Development remains sealed")
    return ctx


def run_dev(args):
    ctx=confirmed(args)
    output=args.run_dir/"dev"
    if output.exists():read_sealed(output,ctx.identity);return
    train,_,audit=old_inputs(parent_args(args))
    # EXP-019 forbids new embedding extraction. A full-training cache is a real
    # conditional prerequisite, not permission to encode locked captions now.
    cache=args.full_text_dir
    if not (cache/"run.json").exists():
        raise StageBlocked("No verified full-original-training CLIP text cache. EXP-019 forbids new extraction; stop before Development scoring. Supply --full-text-dir only if a provenance-verified cache already exists.")
    vectors,_=load_teacher(cache,ctx.config["teacher"],audit,train.painting.unique())
    captions=training_captions(args.data_dir,allowed_paintings=train.painting.unique())
    raw=polarity_means(captions,train,vectors)
    dev=data.read_dev_input(args.data_dir)
    dev["image_sha256"]=dev.image.map(lambda name:file_sha256(data.image_path(args.data_dir,name)))
    ctx.banks=image_banks(args,pd.concat([train,dev],ignore_index=True))
    baseline,lr=refit_baseline(train,dev,ctx.banks)
    reference=ROOT/"runs/exp010_clip_large/dev/predictions/E50+L/predictions.csv"
    verify_reference(baseline,pd.read_csv(reference,float_precision="round_trip"))
    predictions,fits=fit_conditions(args,ctx,raw,train,dev,baseline,args.run_dir/"development_work",7,budget019(args,ctx,"Dev"),("M3",))
    gold=data.validation_gold(data.read_annotations(args.data_dir),dev)
    with new_output(output) as stage:
        write_development(stage,gold,{"M0":baseline,"E2":predictions["B3"]},"E2",ctx.config,"M0")
        submission.pack(stage/"predictions/E2/predictions.csv",stage/"submission.zip",dev)
        submission.validate_archive(stage/"submission.zip",dev)
        json_output({"lr":lr,"students":fits},stage/"fits.json")
        seal(stage,ctx.identity,{"model":"E2","pairs":len(dev),"confirm_manifest_sha256":file_sha256(args.run_dir/"confirm/run.json"),
                               "text_manifest_sha256":file_sha256(cache/"run.json")})
