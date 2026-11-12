"""Independent thirty-checkpoint replay, shared geometry checks and frozen gate."""

import json
import time

import numpy as np
import pandas as pd
import torch

from . import data
from .baseline import file_sha256
from .caption_student import concatenate_features,predict_student
from .cli import json_output
from .contrast_study import new_output
from .emotion_experiment import (CONDITIONS,ENSEMBLES,REFERENCES,choose,collect,fit_seed,inputs_for_fold,
                                 load_model,memory_report,research_assessment)
from .emotion_student import emotion_rows
from .emotion_targets import condition_targets,project_targets
from .emotion_workflow import (context020,geometry_directory,load_audit,old_args,preserve,read_targets,stage_budget)
from .fusion import blend,verify_reference
from .metrics import score
from .polarity_workflow import StageBlocked,load_geometry,load_raw,read_sealed,seal


def compare_diagnostics(expected,actual):
    if isinstance(expected,dict):
        if expected.keys()!=actual.keys():raise ValueError("Diagnostic keys changed")
        for key in expected:compare_diagnostics(expected[key],actual[key])
    elif isinstance(expected,list):
        if len(expected)!=len(actual):raise ValueError("Diagnostic shape changed")
        for a,b in zip(expected,actual):compare_diagnostics(a,b)
    elif isinstance(expected,float):
        np.testing.assert_allclose(expected,actual,atol=1e-7,rtol=1e-6)
    elif expected!=actual:raise ValueError("Diagnostic replay identity changed")


def replay_targets(args,ctx,budget):
    raw=load_audit(args,ctx);pr,polarity,_=load_raw(old_args(args),ctx.parent_identity)
    from .caption_data import inner_partition
    completion=read_sealed(args.run_dir/"audit_completion",ctx.identity)
    for fold in range(1,ctx.config["cv_folds"]+1):
        full=ctx.train[ctx.train.fold.ne(fold)];inner,_=inner_partition(full,ctx.config,fold)
        for phase,partition in (("inner",inner),("full",full)):
            budget.check(f"Target replay {fold} {phase}")
            path=geometry_directory(args,fold,phase)
            if file_sha256(path/"run.json")!=completion["target_manifests"][f"{fold}/{phase}"]:
                raise ValueError("Audited target provenance changed")
            stored=read_targets(args,ctx,fold,phase,partition)
            parent=args.exp018_dir/"work"/f"fold_{fold}"/"geometry"/phase
            reference=load_geometry(parent,ctx.parent_identity,partition.painting)
            fresh=project_targets(*raw,pr,polarity,partition,parent/"targets.npz",reference)
            for key in ("A8","E8"):
                np.testing.assert_allclose(stored[key],fresh[key],atol=3e-5,rtol=0.)
            np.testing.assert_array_equal(stored["valid"],fresh["valid"])
            np.testing.assert_array_equal(stored["counts"],fresh["counts"])


def _verify(args,ctx):
    started=time.perf_counter();cv=read_sealed(args.run_dir/"cv",ctx.identity);budget=stage_budget(args,ctx,"Verify")
    replay_targets(args,ctx,budget)
    visual=concatenate_features(ctx.banks);device=torch.device(ctx.identity["runtime_and_source"]["device"])
    stored=pd.read_csv(args.run_dir/"cv/predictions.csv",float_precision="round_trip")
    summary=pd.read_csv(args.run_dir/"cv/summary.csv",index_col="model",float_precision="round_trip")
    for old,new in REFERENCES.items():
        verify_reference(stored[stored.model.eq(new)][data.KEYS+["p_positive"]],ctx.predictions[ctx.predictions.model.eq(old)][data.KEYS+["p_positive"]])
    seeds=[];curves=[];diagnostics=[];checked=0;keys=set()
    for fold in range(1,ctx.config["cv_folds"]+1):
        fit,held,inner,_,bundles=inputs_for_fold(args,ctx,fold,visual)
        baseline=held[data.KEYS].merge(ctx.predictions[ctx.predictions.model.eq("M0")][data.KEYS+["p_positive"]],on=data.KEYS)
        for condition in CONDITIONS:
            probabilities=[]
            for seed in ctx.config["seeds"]:
                budget.check(f"Checkpoint replay {fold} {condition} {seed}")
                path=args.run_dir/"work"/f"fold_{fold}"/f"{condition}_{seed}"
                if file_sha256(path/"run.json")!=cv["seed_manifests"][f"{fold}/{condition}_{seed}"]:
                    raise ValueError("CV checkpoint provenance changed")
                pred,info=fit_seed(args,ctx,fold,seed,condition,visual,budget)
                paired=args.run_dir/"work"/f"fold_{fold}"/f"{('E8' if condition=='A8' else 'A8')}_{seed}"
                matched=read_sealed(paired,ctx.identity)
                if info["initialization_sha256"]!=matched["initialization_sha256"]:
                    raise ValueError("A8/E8 matched initialization replay failed")
                model=load_model(path/"student.pt",ctx.config,device)
                if tuple(model.languages)!=tuple(sorted(fit.language.unique())):raise ValueError("Language vocabulary differs")
                values=predict_student(model,held[data.KEYS+["image"]],visual,baseline,device,ctx.config["batch_size"])
                np.testing.assert_array_equal(values,pred.p_positive.to_numpy());probabilities.append(values)
                seeds.append(held[data.KEYS].assign(p_positive=np.round(values,6),model=condition,seed=seed,fold=fold))
                curve=pd.read_csv(path/"curves.csv");selection=curve[curve.phase.eq("selection")]
                if int(selection.loc[selection.validation_bce.idxmin(),"epoch"])!=info["selected_epoch"] or len(curve[curve.phase.eq("refit")])!=info["selected_epoch"]:
                    raise ValueError("Epoch selection/refit replay differs")
                curves.append(curve.assign(fold=fold,seed=seed,model=condition))
                if fold in (1,3,5):
                    rows=emotion_rows(inner,visual,list(model.languages),device,*condition_targets(bundles["full"],condition))
                    fresh=collect(model,rows,ctx,fold,seed,condition,"selected_final_refit")
                    compare_diagnostics(json.loads((path/"gradients.json").read_text()),fresh);diagnostics.extend(fresh)
                checked+=1;keys.add((fold,condition,seed));del model
            pred=held[data.KEYS].assign(p_positive=np.round(np.mean(probabilities,axis=0),6))
            verify_reference(pred,stored[stored.model.eq(condition)&stored.fold.eq(fold)][data.KEYS+["p_positive"]])
            verify_reference(blend(baseline,pred,.5),stored[stored.model.eq(ENSEMBLES[condition])&stored.fold.eq(fold)][data.KEYS+["p_positive"]])
    if keys!={(f,c,s) for f in range(1,ctx.config["cv_folds"]+1) for c in CONDITIONS for s in ctx.config["seeds"]}:
        raise ValueError("Incomplete seed replay")
    folds={}
    for name,rows in stored.groupby("model"):
        metric=score(ctx.train,rows[data.KEYS+["p_positive"]])
        np.testing.assert_allclose([metric["cultural_contrast"],metric["log_loss"]],summary.loc[name,["pooled_contrast","pooled_log_loss"]].astype(float),atol=1e-14,rtol=0.)
        values=[score(ctx.train[ctx.train.fold.eq(f)],rows[rows.fold.eq(f)][data.KEYS+["p_positive"]]) for f in range(1,ctx.config["cv_folds"]+1)]
        folds[name]=values
        np.testing.assert_allclose([np.mean([v["cultural_contrast"] for v in values]),np.mean([v["log_loss"] for v in values])],summary.loc[name,["contrast_mean","log_loss_mean"]].astype(float),atol=1e-14,rtol=0.)
    for name,values in folds.items():
        if sum(x["cultural_contrast"]>y["cultural_contrast"] for x,y in zip(values,folds["R0"]))!=summary.loc[name,"fold_wins"]:
            raise ValueError("Fold-win replay differs")
    parentseeds=pd.read_csv(args.exp018_dir/"verification/seed_predictions.csv",float_precision="round_trip")
    baseline=ctx.predictions[ctx.predictions.model.eq("M0")][data.KEYS+["p_positive"]]
    metrics=[]
    for (name,seed),rows in pd.concat(seeds).groupby(["model","seed"]):
        pred=rows[data.KEYS+["p_positive"]]
        for label,values in ((name,pred),(ENSEMBLES[name],blend(baseline,pred,.5))):
            result=score(ctx.train,values);metrics.append({"model":label,"seed":seed,"contrast":result["cultural_contrast"],"log_loss":result["log_loss"]})
    for seed in ctx.config["seeds"]:
        for old,label in (("M1","RB1"),("M3","RB3")):
            pred=parentseeds[parentseeds.model.eq(old)&parentseeds.seed.eq(seed)][data.KEYS+["p_positive"]]
            comparisons=[(label,blend(baseline,pred,.5))]
            if old=="M3":comparisons.append(("R1",pred))
            for name,values in comparisons:
                result=score(ctx.train,values);metrics.append({"model":name,"seed":seed,"contrast":result["cultural_contrast"],"log_loss":result["log_loss"]})
    count=preserve(args,ctx)
    if not (args.run_dir/"verification").exists():
        with new_output(args.run_dir/"verification") as stage:
            pd.concat(curves).to_csv(stage/"training_curves.csv",index=False)
            frame=pd.DataFrame(metrics);frame.to_csv(stage/"seed_metrics.csv",index=False)
            frame.groupby("model")[["contrast","log_loss"]].agg(["mean","std","min","max"]).to_csv(stage/"seed_variance.csv")
            json_output(diagnostics,stage/"final_gradients.json")
            result=research_assessment(summary,pd.read_csv(args.run_dir/"cv/fold_metrics.csv"),json.loads((args.run_dir/"cv/matched_bootstrap.json").read_text()))
            json_output(result,stage/"research_assessment.json")
            seal(stage,ctx.identity,{"status":"PASS","checkpoints_replayed":checked,"target_partitions_replayed":2*ctx.config["cv_folds"],
                "protected_files_matched":count,"seconds":time.perf_counter()-started,"memory":memory_report(),
                "cv_manifest_sha256":file_sha256(args.run_dir/"cv/run.json")})
    else:read_sealed(args.run_dir/"verification",ctx.identity)
    result=choose(summary,ctx.config,True)
    if not (args.run_dir/"decision").exists():
        with new_output(args.run_dir/"decision") as stage:
            seal(stage,ctx.identity,{**result,"cv_manifest_sha256":file_sha256(args.run_dir/"cv/run.json"),
                                   "verification_manifest_sha256":file_sha256(args.run_dir/"verification/run.json")})
    return result


def run_verify(args):
    ctx=context020(args)
    directory=args.run_dir/"verification_attempts";directory.mkdir(parents=True,exist_ok=True)
    paths=sorted(directory.glob("*.json"));number=int(paths[-1].stem)+1 if paths else 1
    path=directory/f"{number:08}.json"
    with path.open("x") as handle:json.dump({"status":"STARTED"},handle)
    try:
        result=_verify(args,ctx)
        json_output({"status":"PASS","cv_manifest_sha256":file_sha256(args.run_dir/"cv/run.json")},path)
        verified_decision(args)
    except BaseException as error:
        json_output({"status":"FAILED","error":str(error)},path);raise
    print(f"EXP-020 replay-verified competition decision {result['status']}",flush=True)


def verified_decision(args):
    ctx=context020(args);result=read_sealed(args.run_dir/"decision",ctx.identity)
    attempts=sorted((args.run_dir/"verification_attempts").glob("*.json"))
    latest=json.loads(attempts[-1].read_text()) if attempts else {}
    cv=read_sealed(args.run_dir/"cv",ctx.identity);read_sealed(args.run_dir/"verification",ctx.identity)
    if latest.get("status")!="PASS" or latest.get("cv_manifest_sha256")!=file_sha256(args.run_dir/"cv/run.json"):
        raise StageBlocked("Latest replay missing/interrupted/failed; rerun Verify")
    for key,digest in cv["seed_manifests"].items():
        fold,name=key.split("/");directory=args.run_dir/"work"/f"fold_{fold}"/name
        read_sealed(directory,ctx.identity)
        if file_sha256(directory/"run.json")!=digest:raise ValueError("Verified checkpoint changed")
    completion=read_sealed(args.run_dir/"audit_completion",ctx.identity)
    for key,digest in completion["target_manifests"].items():
        fold,phase=key.split("/");directory=geometry_directory(args,int(fold),phase)
        read_sealed(directory,ctx.identity)
        if file_sha256(directory/"run.json")!=digest:raise ValueError("Verified target geometry changed")
    for phase in ("cv","verification"):
        if result[phase+"_manifest_sha256"]!=file_sha256(args.run_dir/phase/"run.json"):raise ValueError("Decision evidence changed")
    summary=pd.read_csv(args.run_dir/"cv/summary.csv",index_col="model",float_precision="round_trip")
    if any(result[k]!=v for k,v in choose(summary,ctx.config,True).items()):raise ValueError("Frozen selection differs")
    preserve(args,ctx)
    return result
