"""Independent screen/full replay and frozen allocation/promotion decisions."""

import json

import numpy as np
import pandas as pd

from . import data
from .baseline import file_sha256
from .caption_data import inner_partition
from .cli import json_output
from .contrast_study import new_output
from .emotion_verification import compare_diagnostics
from .fusion import blend,verify_reference
from .gradient_training import state_hash
from .metrics import painting_scores,score
from .polarity_workflow import ExecutionBudget,StageBlocked,read_sealed,seal
from .vote_experiment import (CONDITIONS,ENSEMBLES,auxiliary_probabilities,context,diagnostics,fit_seed,predict_binary)
from .vote_inputs import preserve
from .vote_model import VoteStudent,seed_model,vote_rows


def screening_choice(summary,folds,cfg,complete,major_failure=False):
    if not complete:return {"status":"INCOMPLETE"}
    control,treatment=summary.loc["D0"],summary.loc["D8"]
    gain=float(treatment.pooled_contrast-control.pooled_contrast)
    loss=float(treatment.pooled_log_loss-control.pooled_log_loss)
    wide=folds.pivot(index="fold",columns="model",values="contrast")
    wins=int((wide.D8>wide.D0).sum())
    if not np.isfinite([gain,loss]).all():raise ValueError("Nonfinite screening scores")
    status="PILOT_INCONCLUSIVE"
    if treatment.pooled_contrast<=control.pooled_contrast+cfg["screen_unpromising_gain"] or loss>cfg["screen_unpromising_loss_bits"] or major_failure:
        status="PILOT_UNPROMISING"
    elif treatment.pooled_contrast>=control.pooled_contrast+cfg["screen_promising_gain"] and wins>=1 and loss<=cfg["screen_promising_loss_bits"]:
        status="PILOT_PROMISING"
    return {"status":status,"D8_minus_D0_contrast":gain,"D8_minus_D0_loss_bits":loss,"fold_wins":wins,
            "major_reproducible_failure":bool(major_failure),"finite_nondegenerate_gradients":True,
            "scope":"Resource-allocation gate on repeatedly examined folds, not statistical proof or promotion"}


def competition_choice(summary,cfg,complete):
    if not complete:return {"status":"INCOMPLETE","model":None}
    base,control,candidate=summary.loc["R0"],summary.loc["ED0"],summary.loc["ED8"]
    if not np.isfinite(summary[["pooled_contrast","log_loss_mean"]].to_numpy()).all():raise ValueError("Nonfinite CV metrics")
    t=cfg["training"]
    checks={"gain_vs_R0":candidate.pooled_contrast>=base.pooled_contrast+t["minimum_pooled_contrast_gain"],
            "gain_vs_ED0":candidate.pooled_contrast>=control.pooled_contrast+t["minimum_incremental_gain"],
            "fold_wins":candidate.fold_wins>=t["minimum_fold_wins"],
            "loss_guardrail":candidate.log_loss_mean<=base.log_loss_mean+t["log_loss_guardrail_bits"],
            "five_folds":candidate.folds==t["cv_folds"]}
    passed=all(checks.values())
    return {"status":"PASS" if passed else "COMPETITION_NO_GO","model":"ED8" if passed else "R0",
            "checks":{k:bool(v) for k,v in checks.items()},"all_checkpoints_replayed":True,
            "eligibility":cfg["eligibility"],"confirmation_development":"Not implemented; locked resources remain inaccessible"}


def load_model(path,ctx):
    import torch
    payload=torch.load(path,weights_only=True,map_location="cpu")
    model=VoteStudent(payload["languages"],ctx.training).to(ctx.device);model.load_state_dict(payload["state_dict"],strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError("Nonfinite checkpoint")
    return model


def _verify(args,ctx,screen):
    phase="screen" if screen else "cv";suffix="screen_verification" if screen else "verification"
    study=read_sealed(args.run_dir/phase,ctx.identity);pilot=read_sealed(args.run_dir/"pilot",ctx.identity)
    if not pilot["finite_nondegenerate_gradients"]:raise ValueError("Pilot gradient checks failed")
    folds=ctx.config["screen_folds"] if screen else list(range(1,ctx.training["cv_folds"]+1))
    seeds=ctx.config["screen_seeds"] if screen else ctx.training["seeds"]
    expected={(fold,condition,seed) for fold in folds for condition in CONDITIONS for seed in seeds}
    if study["folds"]!=folds or study["seeds"]!=seeds or len(study["seed_manifests"])!=len(expected):raise ValueError("Incomplete screening/CV coverage")
    seconds=pilot["estimated_screen_verification_seconds"] if screen else pilot["estimated_cv_verification_seconds"]
    budget=ExecutionBudget(args.max_seconds or max(120.,1.5*seconds))
    stored=pd.read_csv(args.run_dir/phase/"predictions.csv",float_precision="round_trip")
    summary=pd.read_csv(args.run_dir/phase/"summary.csv",index_col="model",float_precision="round_trip")
    metrics=pd.read_csv(args.run_dir/phase/"fold_metrics.csv",float_precision="round_trip")
    seeds_predictions=[];constant=[];checks=set();curves=[]
    for fold in folds:
        fit=ctx.train[ctx.train.fold.ne(fold)];held=ctx.train[ctx.train.fold.eq(fold)]
        inner,_=inner_partition(fit,ctx.training,fold)
        baseline=held[data.KEYS].merge(ctx.reference[ctx.reference.fold.eq(fold)][data.KEYS+["p_positive"]],on=data.KEYS,validate="one_to_one")
        verify_reference(baseline,stored[stored.model.eq("R0")&stored.fold.eq(fold)][data.KEYS+["p_positive"]])
        for condition in CONDITIONS:
            values=[]
            for seed in seeds:
                budget.check(f"{phase} replay {fold} {condition} {seed}")
                directory=args.run_dir/"work"/f"fold_{fold}"/f"{condition}_{seed}"
                if file_sha256(directory/"run.json")!=study["seed_manifests"][f"{fold}/{condition}_{seed}"]:
                    raise ValueError("Checkpoint provenance changed")
                pred,auxiliary,info=fit_seed(args,ctx,fold,seed,condition,budget)
                paired=read_sealed(args.run_dir/"work"/f"fold_{fold}"/f"{('D8' if condition=='D0' else 'D0')}_{seed}",ctx.identity)
                if info["initialization_sha256"]!=paired["initialization_sha256"]:raise ValueError("Unmatched D0/D8 initialization")
                original,_,_=seed_model(sorted(inner.language.unique()),seed,ctx.training,ctx.device)
                if state_hash(original)!=info["initialization_sha256"]["selection"]:raise ValueError("Initialization replay differs")
                initial_rows=vote_rows(inner,ctx.visual,list(original.languages),ctx.device)
                initial=diagnostics(original,initial_rows,ctx,fold,condition,"initialization");del original
                model=load_model(directory/"student.pt",ctx)
                probabilities=predict_binary(model,held,ctx);diagnostic_aux=auxiliary_probabilities(model,held,ctx)
                np.testing.assert_array_equal(probabilities,pred.p_positive.to_numpy());np.testing.assert_array_equal(diagnostic_aux,auxiliary)
                final_rows=vote_rows(inner,ctx.visual,list(model.languages),ctx.device)
                final=diagnostics(model,final_rows,ctx,fold,condition,"final_refit")
                compare_diagnostics(json.loads((directory/"gradients.json").read_text()),[initial,final])
                curve=pd.read_csv(directory/"curves.csv");selection=curve[curve.phase.eq("selection")]
                if int(selection.loc[selection.validation_bce.idxmin(),"epoch"])!=info["selected_epoch"] or len(curve[curve.phase.eq("refit")])!=info["selected_epoch"]:
                    raise ValueError("Internal BCE stopping/refit replay differs")
                rounded=held[data.KEYS].assign(p_positive=np.round(probabilities,6))
                paintings=painting_scores(held[data.KEYS+["p"]].merge(rounded,on=data.KEYS,validate="one_to_one"))
                eligible=paintings[paintings.status.eq("scored")]
                constant.append({"fold":fold,"seed":seed,"model":condition,
                    "constant_fraction":float(eligible.constant_prediction.mean()) if len(eligible) else 0.})
                seeds_predictions.append(rounded.assign(fold=fold,seed=seed,model=condition));values.append(probabilities)
                curves.append(curve.assign(fold=fold,seed=seed,model=condition));checks.add((fold,condition,seed));del model
            pred=held[data.KEYS].assign(p_positive=np.round(np.mean(values,axis=0),6))
            verify_reference(pred,stored[stored.model.eq(condition)&stored.fold.eq(fold)][data.KEYS+["p_positive"]])
            verify_reference(blend(baseline,pred,.5),stored[stored.model.eq(ENSEMBLES[condition])&stored.fold.eq(fold)][data.KEYS+["p_positive"]])
    if checks!=expected:raise ValueError("Missing replayed seeds")
    subset=ctx.train[ctx.train.fold.isin(folds)];fold_metrics={}
    for name,rows in stored.groupby("model"):
        result=score(subset,rows[data.KEYS+["p_positive"]])
        np.testing.assert_allclose([result["cultural_contrast"],result["log_loss"]],summary.loc[name,["pooled_contrast","pooled_log_loss"]].astype(float),atol=1e-14,rtol=0.)
        values=[score(subset[subset.fold.eq(f)],rows[rows.fold.eq(f)][data.KEYS+["p_positive"]]) for f in folds]
        fold_metrics[name]=values
        np.testing.assert_allclose([np.mean([r["cultural_contrast"] for r in values]),np.mean([r["log_loss"] for r in values])],summary.loc[name,["contrast_mean","log_loss_mean"]].astype(float),atol=1e-14,rtol=0.)
        for fold,metric in zip(folds,values):
            original=metrics[metrics.model.eq(name)&metrics.fold.eq(fold)].iloc[0]
            np.testing.assert_allclose([metric["cultural_contrast"],metric["log_loss"]],[original.contrast,original.log_loss],atol=1e-14,rtol=0.)
    for name,values in fold_metrics.items():
        if sum(a["cultural_contrast"]>b["cultural_contrast"] for a,b in zip(values,fold_metrics["R0"]))!=summary.loc[name,"fold_wins"]:
            raise ValueError("Fold wins differ")
    seed_scores=[];base=ctx.reference[ctx.reference.fold.isin(folds)][data.KEYS+["p_positive"]]
    for (name,seed),rows in pd.concat(seeds_predictions).groupby(["model","seed"]):
        for model,pred in ((name,rows[data.KEYS+["p_positive"]]),(ENSEMBLES[name],blend(base,rows[data.KEYS+["p_positive"]],.5))):
            result=score(subset,pred);seed_scores.append({"model":model,"seed":seed,"contrast":result["cultural_contrast"],"log_loss":result["log_loss"]})
    constants=pd.DataFrame(constant).pivot(index=["fold","seed"],columns="model",values="constant_fraction")
    major=bool((constants.D8.ge(ctx.config["major_constant_prediction_fraction"]) & constants.D0.le(ctx.config["control_constant_prediction_maximum"])).all())
    count=preserve(args,ctx.audit_identity)
    if not (args.run_dir/suffix).exists():
        with new_output(args.run_dir/suffix) as stage:
            frame=pd.DataFrame(seed_scores);frame.to_csv(stage/"seed_metrics.csv",index=False)
            frame.groupby("model")[["contrast","log_loss"]].agg(["mean","std","min","max"]).to_csv(stage/"seed_variance.csv")
            constants.to_csv(stage/"constant_prediction_diagnostics.csv")
            pd.concat(curves).to_csv(stage/"training_curves.csv",index=False)
            seal(stage,ctx.identity,{"status":"PASS","checkpoints_replayed":len(checks),"protected_files_matched":count,
                 "major_reproducible_failure":major,"study_manifest_sha256":file_sha256(args.run_dir/phase/"run.json")})
    else:read_sealed(args.run_dir/suffix,ctx.identity)
    decision=screening_choice(summary,metrics,ctx.config,True,major) if screen else competition_choice(summary,ctx.config,True)
    output=args.run_dir/("screen_decision" if screen else "decision")
    if not output.exists():
        with new_output(output) as stage:
            seal(stage,ctx.identity,{**decision,"study_manifest_sha256":file_sha256(args.run_dir/phase/"run.json"),
                "verification_manifest_sha256":file_sha256(args.run_dir/suffix/"run.json")})
    return decision


def run_verify(args,*,screen):
    ctx=context(args);phase="screen" if screen else "cv"
    if not screen:
        if screen_decision(args)["status"]!="PILOT_PROMISING":raise StageBlocked("Screen does not permit full CV verification")
    attempts=args.run_dir/(phase+"_verification_attempts");attempts.mkdir(parents=True,exist_ok=True)
    previous=sorted(attempts.glob("*.json"));number=int(previous[-1].stem)+1 if previous else 1
    path=attempts/f"{number:08}.json"
    with path.open("x") as handle:json.dump({"status":"STARTED"},handle)
    try:
        decision=_verify(args,ctx,screen)
        json_output({"status":"PASS","study_manifest_sha256":file_sha256(args.run_dir/phase/"run.json")},path)
        verified_decision(args,screen=screen)
    except BaseException as error:
        json_output({"status":"FAILED","error":str(error)},path);raise
    print(f"{phase} independently verified: {decision['status']}",flush=True)


def verified_decision(args,*,screen):
    ctx=context(args);phase="screen" if screen else "cv";suffix="screen_verification" if screen else "verification"
    output="screen_decision" if screen else "decision"
    decision=read_sealed(args.run_dir/output,ctx.identity);study=read_sealed(args.run_dir/phase,ctx.identity)
    verification=read_sealed(args.run_dir/suffix,ctx.identity)
    attempts=sorted((args.run_dir/(phase+"_verification_attempts")).glob("*.json"))
    latest=json.loads(attempts[-1].read_text()) if attempts else {}
    digest=file_sha256(args.run_dir/phase/"run.json")
    if latest.get("status")!="PASS" or latest.get("study_manifest_sha256")!=digest:raise StageBlocked("Latest replay failed/interrupted; rerun verification")
    if decision["study_manifest_sha256"]!=digest or verification["study_manifest_sha256"]!=digest or decision["verification_manifest_sha256"]!=file_sha256(args.run_dir/suffix/"run.json"):
        raise ValueError("Decision evidence changed")
    for key,expected in study["seed_manifests"].items():
        fold,name=key.split("/");p=args.run_dir/"work"/f"fold_{fold}"/name
        read_sealed(p,ctx.identity)
        if file_sha256(p/"run.json")!=expected:raise ValueError("Verified checkpoint changed")
    summary=pd.read_csv(args.run_dir/phase/"summary.csv",index_col="model",float_precision="round_trip")
    expected=screening_choice(summary,pd.read_csv(args.run_dir/phase/"fold_metrics.csv"),ctx.config,True,verification["major_reproducible_failure"]) if screen else competition_choice(summary,ctx.config,True)
    if any(decision[k]!=v for k,v in expected.items()):raise ValueError("Frozen gate decision differs")
    if not screen:
        if screen_decision(args)["status"]!="PILOT_PROMISING" or study["screen_manifest_sha256"]!=file_sha256(args.run_dir/"screen/run.json"):
            raise StageBlocked("Full CV screening provenance changed")
    return decision


def screen_decision(args):
    return verified_decision(args,screen=True)
