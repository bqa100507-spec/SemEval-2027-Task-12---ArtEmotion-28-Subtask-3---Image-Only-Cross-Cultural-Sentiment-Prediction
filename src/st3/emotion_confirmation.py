"""Gate-only confirmation; locked captions/labels never enter any fitting set."""

import json

import numpy as np
import pandas as pd
import torch

from . import data,submission
from .baseline import file_sha256
from .caption_comparison import image_banks,refit_baseline
from .caption_data import inner_partition,require_isolation
from .caption_student import concatenate_features,fit_student,predict_student
from .cli import ROOT,json_output
from .contrast_study import new_output,write_development
from .emotion_experiment import ENSEMBLES,load_model
from .emotion_student import emotion_rows,seed_model
from .emotion_targets import condition_targets,project_targets
from .emotion_verification import verified_decision
from .emotion_workflow import context020,load_audit,old_args,stage_budget
from .fusion import blend
from .gradient_training import TrainingRecorder
from .metrics import score
from .polarity_confirmation import confirmation_passes
from .polarity_experiment import fit_conditions
from .polarity_workflow import StageBlocked,geometry,load_raw,old_inputs,read_sealed,seal


def run_confirm(args):
    decision=verified_decision(args)
    if decision["status"]!="PASS":raise StageBlocked("COMPETITION_NO_GO; confirmation remains sealed")
    ctx=context020(args)
    if (args.run_dir/"confirm").exists():read_sealed(args.run_dir/"confirm",ctx.identity);return
    train,fit,_=old_inputs(old_args(args));held=train[train.role.eq("holdout")]
    require_isolation(fit,held)
    budget=stage_budget(args,ctx,"Confirm");ctx.banks=image_banks(args,train)
    baseline,lr=refit_baseline(fit,held,ctx.banks)
    raw=load_raw(old_args(args),ctx.parent_identity);emotion_raw=load_audit(args,ctx)
    directory=args.run_dir/"confirmation_work";inner,stopping=inner_partition(fit,ctx.config,6)
    # This previously unused confirmation-training partition needs the original
    # pooled-caption PCA procedure. Never fit PCA on emotion-specific means.
    bundles=geometry(args,ctx,raw,fit,inner,directory/"geometry")
    control,controlfits=fit_conditions(args,ctx,raw,fit,held,baseline,directory,6,budget,("M1",))
    condition="A8" if decision["model"]=="EA8" else "E8"
    prepared={}
    for phase,partition in (("inner",inner),("full",fit)):
        bundle=project_targets(*emotion_raw,raw[0],raw[1],partition,directory/"geometry"/phase/"targets.npz",bundles[phase])
        prepared[phase]=condition_targets(bundle,condition)
    visual=concatenate_features(ctx.banks);device=torch.device(ctx.identity["runtime_and_source"]["device"])
    values=[];manifests={}
    for seed in ctx.config["seeds"]:
        budget.check(f"Confirmation {condition} {seed}")
        path=directory/f"{condition}_{seed}"
        if not path.exists():
            with new_output(path) as stage:
                recorder=TrainingRecorder(stage,budget,9999)
                model,curves,info=fit_student(fit,inner,stopping,visual,prepared,seed,ctx.config,device,
                    model_factory=seed_model,epoch_function=recorder.epoch_function,training_rows=emotion_rows,progress=recorder.progress)
                probabilities=predict_student(model,held[data.KEYS+["image"]],visual,baseline,device,ctx.config["batch_size"])
                torch.save({"languages":list(model.languages),"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}},stage/"student.pt")
                np.savez(stage/"predictions.npz",probabilities=probabilities)
                pd.DataFrame(curves).to_csv(stage/"curves.csv",index=False)
                seal(stage,ctx.identity,{**info,"condition":condition,"seed":seed,"fit_paintings":sorted(fit.painting.unique()),
                                        "held_paintings":sorted(held.painting.unique())})
        info=read_sealed(path,ctx.identity)
        if info["fit_paintings"]!=sorted(fit.painting.unique()) or info["held_paintings"]!=sorted(held.painting.unique()):
            raise ValueError("Confirmation checkpoint isolation differs")
        model=load_model(path/"student.pt",ctx.config,device)
        probabilities=predict_student(model,held[data.KEYS+["image"]],visual,baseline,device,ctx.config["batch_size"])
        with np.load(path/"predictions.npz",allow_pickle=False) as archive:np.testing.assert_array_equal(probabilities,archive["probabilities"])
        values.append(probabilities);manifests[path.name]=file_sha256(path/"run.json");del model
    student=held[data.KEYS].assign(p_positive=np.round(np.mean(values,axis=0),6))
    candidate=blend(baseline,student,.5)
    metrics={"M0":score(held,baseline),"B1":score(held,control["B1"]),decision["model"]:score(held,candidate)}
    result=confirmation_passes(metrics,decision["model"],ctx.config)
    with new_output(args.run_dir/"confirm") as stage:
        write_development(stage,held,{"R0":baseline,decision["model"]:candidate},decision["model"],ctx.config,"R0")
        json_output(metrics,stage/"metrics.json");json_output({"lr":lr,"control":controlfits},stage/"fits.json")
        seal(stage,ctx.identity,{**result,"model":decision["model"],"condition":condition,
            "cv_decision_sha256":file_sha256(args.run_dir/"decision/run.json"),"student_manifests":manifests,
            "control_manifests":{f"M1_{seed}":file_sha256(directory/"seeds"/f"M1_{seed}"/"run.json") for seed in ctx.config["seeds"]}})


def confirmed(args):
    decision=verified_decision(args)
    if decision["status"]!="PASS":raise StageBlocked("COMPETITION_NO_GO; public Development remains sealed")
    ctx=context020(args);result=read_sealed(args.run_dir/"confirm",ctx.identity)
    if result["cv_decision_sha256"]!=file_sha256(args.run_dir/"decision/run.json") or result["model"]!=decision["model"]:
        raise ValueError("Confirmation candidate/evidence changed")
    metrics=json.loads((args.run_dir/"confirm/metrics.json").read_text())
    if any(result[k]!=v for k,v in confirmation_passes(metrics,result["model"],ctx.config).items()):raise ValueError("Confirmation gate differs")
    for key,base in (("student_manifests",args.run_dir/"confirmation_work"),("control_manifests",args.run_dir/"confirmation_work/seeds")):
        for name,digest in result[key].items():
            read_sealed(base/name,ctx.identity)
            if file_sha256(base/name/"run.json")!=digest:raise ValueError("Confirmation model changed")
    if result["status"]!="PASS":raise StageBlocked("Confirmation failed; public Development remains sealed")
    return ctx,result


def run_dev(args):
    ctx,confirmation=confirmed(args)
    if (args.run_dir/"dev").exists():read_sealed(args.run_dir/"dev",ctx.identity);return
    budget=stage_budget(args,ctx,"Dev")
    inputs=data.read_dev_input(args.data_dir);banks=image_banks(args,inputs);visual=concatenate_features(banks)
    baseline=pd.read_csv(ROOT/"runs/exp010_clip_large/dev/predictions/E50+L/predictions.csv",float_precision="round_trip")
    data.require_same_pairs(inputs,baseline)
    device=torch.device(ctx.identity["runtime_and_source"]["device"]);values=[]
    # Reuse the confirmed 1,335-painting student. No locked caption/annotation is
    # ever admitted to fitting, even after confirmation. Release R0 stays frozen.
    for name in confirmation["student_manifests"]:
        budget.check(f"Development {name}")
        model=load_model(args.run_dir/"confirmation_work"/name/"student.pt",ctx.config,device)
        values.append(predict_student(model,inputs,visual,baseline,device,ctx.config["batch_size"]));del model
    student=inputs[data.KEYS].assign(p_positive=np.round(np.mean(values,axis=0),6));candidate=blend(baseline,student,.5)
    gold=data.validation_gold(data.read_annotations(args.data_dir),inputs)
    with new_output(args.run_dir/"dev") as stage:
        write_development(stage,gold,{"R0":baseline,confirmation["model"]:candidate},confirmation["model"],ctx.config,"R0")
        submission.pack(stage/"predictions"/confirmation["model"]/"predictions.csv",stage/"submission.zip",inputs)
        submission.validate_archive(stage/"submission.zip",inputs)
        seal(stage,ctx.identity,{"model":confirmation["model"],"pairs":len(inputs),"student_fit_paintings":1335,
            "confirm_manifest_sha256":file_sha256(args.run_dir/"confirm/run.json"),"no_locked_annotations_used_for_fitting":True})
