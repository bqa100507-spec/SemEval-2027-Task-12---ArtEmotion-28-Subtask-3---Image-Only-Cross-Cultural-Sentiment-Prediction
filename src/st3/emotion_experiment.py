"""Matched eight-query fitting, historical references, paired analysis and gates."""

import json
import os
import time

import numpy as np
import pandas as pd
import torch

from . import data
from .ablation import paired_bootstrap
from .baseline import file_sha256
from .caption_comparison import paired_order
from .caption_data import inner_partition, require_isolation
from .caption_student import concatenate_features, fit_student, predict_student, validation_bce
from .cli import json_output
from .contrast_study import new_output
from .emotion_student import EmotionStudent, emotion_diagnostic, emotion_rows, seed_model
from .emotion_targets import condition_targets
from .emotion_workflow import context020, geometry_directory, preserve, read_targets, stage_budget
from .fusion import blend, verify_reference
from .fusion_study import write_cv_analysis
from .gradient_analysis import deterministic_batches
from .gradient_training import TrainingRecorder, state_hash
from .low_rank_study import score_result
from .metrics import score
from .polarity_workflow import ExecutionBudget, StageBlocked, read_sealed, seal

REFERENCES = {"M0":"R0", "M3":"R1", "B1":"RB1", "B3":"RB3"}
CONDITIONS = ("A8", "E8")
ENSEMBLES = {"A8":"EA8", "E8":"EE8"}
COMPARISONS = (("E8","A8"),("EE8","R0"),("A8","R1"),("E8","R1"),
               ("EE8","EA8"),("EE8","RB3"),("EE8","RB1"),("EA8","RB1"))


def memory_report():
    peak = None
    if os.name == "nt":
        import ctypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb",ctypes.c_ulong),("faults",ctypes.c_ulong)]+[(key,ctypes.c_size_t) for key in
                       ("peak","working","peak_paged","paged","peak_nonpaged","nonpaged","pagefile","peak_pagefile")]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(counters),counters.cb):
            peak = counters.peak
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    return {"peak_process_rss_bytes":peak,
            "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}


def inputs_for_fold(args,ctx,fold,visual):
    fit,held = ctx.train[ctx.train.fold.ne(fold)],ctx.train[ctx.train.fold.eq(fold)]
    inner,stopping = inner_partition(fit,ctx.config,fold)
    require_isolation(inner,stopping,held)
    bundles = {phase:read_targets(args,ctx,fold,phase,partition) for phase,partition in (("inner",inner),("full",fit))}
    return fit,held,inner,stopping,bundles


def collect(model,rows,ctx,fold,seed,condition,checkpoint):
    records=[]
    for number,batch in enumerate(deterministic_batches(len(rows["image"]),ctx.config["batch_size"],fold,202027)):
        result=emotion_diagnostic(model,rows,batch,ctx.config,ctx.protocol["collapsed_prediction_cosine"],
                                  require_nonzero=checkpoint!="selected_final_refit")
        records.append({"fold":fold,"seed":seed,"condition":condition,"checkpoint":checkpoint,"batch":number+1,
                        "scope":"inner-training rows only",**result})
    return records


def assert_matched(a,b,model_a,model_b):
    if state_hash(model_a)!=state_hash(model_b):
        raise ValueError("A8/E8 initialization differs")
    if a[0].keys()!=b[0].keys() or a[1]!=b[1]:
        raise ValueError("A8/E8 pair weights or keys differ")
    for key in a[0]:
        if not np.array_equal(a[0][key][1],b[0][key][1]):
            raise ValueError("A8/E8 availability masks differ")


def run_pilot(args):
    started=time.perf_counter();ctx=context020(args)
    read_sealed(args.run_dir/"audit_completion",ctx.identity)
    if (args.run_dir/"pilot").exists():
        read_sealed(args.run_dir/"pilot",ctx.identity);return
    budget=ExecutionBudget(args.max_seconds or 240.)
    visual=concatenate_features(ctx.banks);device=torch.device(ctx.identity["runtime_and_source"]["device"])
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
    fit,held,inner,stopping,bundles=inputs_for_fold(args,ctx,1,visual)
    prepared={condition:condition_targets(bundles["inner"],condition) for condition in CONDITIONS}
    languages=sorted(inner.language.unique());seed=ctx.config["seeds"][0]
    ma,_,_=seed_model(languages,seed,ctx.config,device);mb,_,_=seed_model(languages,seed,ctx.config,device)
    assert_matched(prepared["A8"],prepared["E8"],ma,mb)
    ma.eval();mb.eval()
    test=torch.as_tensor(np.stack([visual[k] for k in inner.image.iloc[:16]]),device=device)
    lang=torch.zeros(len(test),dtype=torch.long,device=device)
    with torch.no_grad():
        if not torch.equal(ma(test,lang)[0],mb(test,lang)[0]):raise ValueError("Initial sentiment predictions differ")
    parameters=sum(p.numel() for p in ma.parameters()); del ma,mb
    timings={};gradients=[]
    with new_output(args.run_dir/"pilot") as stage:
        for condition in CONDITIONS:
            rows=emotion_rows(inner,visual,languages,device,*prepared[condition])
            validation=emotion_rows(stopping,visual,languages,device)
            model,_,_=seed_model(languages,seed,ctx.config,device)
            gradients.extend(collect(model,rows,ctx,1,seed,condition,"initialization"));del model
            model,rng,optimizer=seed_model(languages,seed,ctx.config,device)
            recorder=TrainingRecorder(stage,budget,9999);epochs=[]
            for epoch in range(1,ctx.protocol["pilot_warmup_epochs"]+ctx.protocol["pilot_timed_epochs"]+1):
                recorder.progress("pilot",epoch);tick=time.perf_counter()
                bce,auxiliary=recorder.epoch_function(model,rows,rng,optimizer,ctx.config,.1)
                vbce=validation_bce(model,validation,ctx.config["batch_size"])
                if not np.isfinite([bce,auxiliary,vbce]).all():raise ValueError("Nonfinite pilot losses")
                epochs.append({"epoch":epoch,"train_bce":bce,"auxiliary":auxiliary,"validation_bce":vbce,"seconds":time.perf_counter()-tick})
            gradients.extend(collect(model,rows,ctx,1,seed,condition,"short_pilot_final"))
            timings[condition]={"epochs":epochs,"mean_epoch_seconds":float(np.mean([r["seconds"] for r in epochs[ctx.protocol["pilot_warmup_epochs"]:]])),"parameters":parameters}
            pd.DataFrame(recorder.records).to_csv(stage/f"{condition}_telemetry.csv",index=False)
            del model,optimizer,rows
        h18=[r for r in json.loads((args.exp018_dir/"cv/fits.json").read_text()) if r["model"]=="M3"]
        h19=json.loads((args.exp019_dir/"cv/fits.json").read_text())
        old18=sum(r["fit_seconds"] for r in h18);old19=sum(r["fit_seconds"] for r in h19)
        epochs18=sum(r["selection_epochs"]+r["selected_epoch"] for r in h18)
        epochs19=sum(r["selection_epochs"]+r["selected_epoch"] for r in h19)
        short_forecast=sum(r["mean_epoch_seconds"] for r in timings.values())*max(epochs18,epochs19)*1.15
        cv_seconds=max(2*old18,2*old19,short_forecast)*ctx.protocol["historical_margin"]
        audit=read_sealed(args.run_dir/"audit_completion",ctx.identity)
        preflight=read_sealed(args.run_dir/"preflight",ctx.identity)
        elapsed=time.perf_counter()-started
        total=cv_seconds+ctx.protocol["diagnostic_verification_allowance_seconds"]+audit["seconds"]+preflight["seconds"]+elapsed
        metadata={"status":"READY_FOR_LOCAL_RUN","seconds":elapsed,"new_cv_models":30,"training_stages":60,
            "historical_exp018_fit_seconds":old18,"historical_exp019_fit_seconds":old19,
            "historical_exp018_selection_epochs":sum(r["selection_epochs"] for r in h18),
            "historical_exp018_refit_epochs":sum(r["selected_epoch"] for r in h18),
            "historical_exp019_selection_epochs":sum(r["selection_epochs"] for r in h19),
            "historical_exp019_refit_epochs":sum(r["selected_epoch"] for r in h19),
            "short_epoch_forecast_seconds":short_forecast,"estimated_cv_seconds":cv_seconds,
            "estimated_verification_seconds":180.,"estimated_optional_confirmation_seconds":max(600.,cv_seconds/5.),
            "diagnostic_verification_allowance_seconds":ctx.protocol["diagnostic_verification_allowance_seconds"],
            "estimated_total_seconds":total,"memory":memory_report(),"matched_masks_weights_initialization":True}
        json_output(timings,stage/"timings.json");json_output(gradients,stage/"gradients.json")
        seal(stage,ctx.identity,metadata)
    print(f"Finite matched pilot; conservative complete workflow {total/60:.2f} minutes. CV remains separate.",flush=True)


def load_model(path,config,device):
    payload=torch.load(path,weights_only=True,map_location="cpu")
    model=EmotionStudent(payload["languages"],config).to(device);model.load_state_dict(payload["state_dict"],strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError("Nonfinite saved model")
    return model


def fit_seed(args,ctx,fold,seed,condition,visual,budget):
    fit,held,inner,stopping,bundles=inputs_for_fold(args,ctx,fold,visual)
    prepared={phase:condition_targets(bundle,condition) for phase,bundle in bundles.items()}
    baseline=held[data.KEYS].merge(ctx.predictions[ctx.predictions.model.eq("M0")][[*data.KEYS,"p_positive"]],on=data.KEYS,validate="one_to_one")
    directory=args.run_dir/"work"/f"fold_{fold}"/f"{condition}_{seed}"
    device=torch.device(ctx.identity["runtime_and_source"]["device"])
    if not directory.exists():
        with new_output(directory) as stage:
            initializations={}
            def factory(languages,seed,config,device):
                model,rng,optimizer=seed_model(languages,seed,config,device)
                initializations["selection" if not initializations else "refit"]=state_hash(model)
                return model,rng,optimizer
            recorder=TrainingRecorder(stage,budget,9999)
            model,curves,info=fit_student(fit,inner,stopping,visual,prepared,seed,ctx.config,device,
                model_factory=factory,epoch_function=recorder.epoch_function,training_rows=emotion_rows,progress=recorder.progress)
            values=predict_student(model,held[[*data.KEYS,"image"]],visual,baseline,device,ctx.config["batch_size"])
            torch.save({"languages":list(model.languages),"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}},stage/"student.pt")
            np.savez(stage/"predictions.npz",painting=held.painting.to_numpy(dtype=str),language=held.language.to_numpy(dtype=str),probabilities=values)
            pd.DataFrame(curves).to_csv(stage/"curves.csv",index=False);pd.DataFrame(recorder.records).to_csv(stage/"telemetry.csv",index=False)
            diagnostics=[]
            if fold in (1,3,5):
                rows=emotion_rows(inner,visual,list(model.languages),device,*prepared["full"])
                diagnostics=collect(model,rows,ctx,fold,seed,condition,"selected_final_refit")
            json_output(diagnostics,stage/"gradients.json")
            seal(stage,ctx.identity,{**info,"fold":fold,"seed":seed,"model":condition,"finite":True,"initialization_sha256":initializations,
                "fit_paintings":sorted(fit.painting.unique()),"inner_paintings":sorted(inner.painting.unique()),
                "stopping_paintings":sorted(stopping.painting.unique()),"held_paintings":sorted(held.painting.unique()),
                "target_manifest_sha256":{phase:file_sha256(geometry_directory(args,fold,phase)/"run.json") for phase in ("inner","full")}})
            del model
    info=read_sealed(directory,ctx.identity)
    for key,partition in (("fit",fit),("inner",inner),("stopping",stopping),("held",held)):
        if info[key+"_paintings"]!=sorted(partition.painting.unique()):raise ValueError("Resumed partition identities differ")
    if not info["finite"] or info["model"]!=condition or info["seed"]!=seed or info["fold"]!=fold:
        raise ValueError("Resumed seed identity differs")
    for phase,digest in info["target_manifest_sha256"].items():
        if digest!=file_sha256(geometry_directory(args,fold,phase)/"run.json"):raise ValueError("Resumed targets changed")
    with np.load(directory/"predictions.npz",allow_pickle=False) as archive:
        if archive["painting"].tolist()!=held.painting.tolist() or archive["language"].tolist()!=held.language.tolist():
            raise ValueError("Resumed prediction keys differ")
        values=archive["probabilities"].copy()
    if values.shape!=(len(held),) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError("Invalid probabilities")
    print(f"Completed/reused {condition} fold {fold}, seed {seed}, epoch {info['selected_epoch']}",flush=True)
    return held[data.KEYS].assign(p_positive=values),info


def paired_analysis(stage,ctx,records):
    paintings=pd.concat(records["paintings"]);wide=paintings[paintings.status.eq("scored")].pivot(index="painting",columns="model",values="contrast")
    changes={a+"_minus_"+b:wide[a]-wide[b] for a,b in COMPARISONS}
    frame=pd.DataFrame(changes).reset_index().assign(status="scored")
    draws,intervals=paired_bootstrap(frame,ctx.protocol["bootstrap_replicates"],ctx.protocol["bootstrap_seed"],columns=list(changes))
    frame.to_csv(stage/"matched_painting_changes.csv",index=False);draws.to_csv(stage/"matched_bootstrap_distribution.csv",index=False)
    json_output(intervals,stage/"matched_bootstrap.json")
    languages=pd.concat(records["languages"]).pivot(index=["fold","language"],columns="model",values="log_loss")
    pd.DataFrame({a+"_minus_"+b:languages[a]-languages[b] for a,b in COMPARISONS}).to_csv(stage/"matched_language_changes.csv")


def descriptive_subgroups(args,ctx,predictions,output):
    reliability=pd.read_csv(args.run_dir/"audit/pair_reliability.csv")
    rates=pd.read_csv(args.run_dir/"audit/tokenizer_languages.csv").set_index("language").truncation_rate
    joined=ctx.train.merge(reliability,on=data.KEYS,validate="one_to_one")
    masks={"one_emotion":joined.emotion_categories.eq(1),"multiple_emotions":joined.emotion_categories.ge(2),
           "two_caption_emotion":joined.has_two_caption_emotion,"all_emotions_single_caption":~joined.has_two_caption_emotion,
           "high_polarity_similarity":joined.mean_raw_polarity_cosine.ge(ctx.protocol["subgroup_polarity_similarity"]),
           "low_polarity_similarity":joined.mean_raw_polarity_cosine.lt(ctx.protocol["subgroup_polarity_similarity"]),
           "high_language_truncation":joined.language.map(rates).ge(ctx.protocol["high_truncation_rate"]),
           "low_language_truncation":joined.language.map(rates).lt(ctx.protocol["high_truncation_rate"])}
    results=[]
    for subgroup,keep in masks.items():
        gold=joined[keep]
        if gold.empty:continue
        for name,rows in predictions.groupby("model"):
            pred=gold[data.KEYS].merge(rows[data.KEYS+["p_positive"]],on=data.KEYS,validate="one_to_one")
            metric=score(gold,pred)
            results.append({"subgroup":subgroup,"model":name,"pairs":len(gold),"paintings":gold.painting.nunique(),
                            "contrast":metric["cultural_contrast"],"log_loss":metric["log_loss"],
                            "scored_paintings":metric["contrast_paintings"],"scored_languages":metric["log_loss_languages"]})
    pd.DataFrame(results).to_csv(output,index=False)


def run_cv(args):
    started=time.perf_counter();ctx=context020(args);read_sealed(args.run_dir/"pilot",ctx.identity)
    if (args.run_dir/"cv").exists():
        metadata=read_sealed(args.run_dir/"cv",ctx.identity)
        for key,digest in metadata["seed_manifests"].items():
            fold,name=key.split("/");directory=args.run_dir/"work"/f"fold_{fold}"/name
            read_sealed(directory,ctx.identity)
            if file_sha256(directory/"run.json")!=digest:raise ValueError("A completed seed changed")
        return
    budget=stage_budget(args,ctx,"CV");visual=concatenate_features(ctx.banks)
    records={key:[] for key in ("predictions","paintings","languages")};rows=[];fits=[]
    for fold in range(1,ctx.config["cv_folds"]+1):
        held=ctx.train[ctx.train.fold.eq(fold)]
        predictions={new:held[data.KEYS].merge(ctx.predictions[ctx.predictions.model.eq(old)][data.KEYS+["p_positive"]],on=data.KEYS,validate="one_to_one") for old,new in REFERENCES.items()}
        for condition in CONDITIONS:
            values=[]
            for seed in ctx.config["seeds"]:
                budget.check(f"fold {fold} {condition} {seed}")
                pred,info=fit_seed(args,ctx,fold,seed,condition,visual,budget);values.append(pred.p_positive.to_numpy());fits.append(info)
            predictions[condition]=held[data.KEYS].assign(p_positive=np.round(np.mean(values,axis=0),6))
            predictions[ENSEMBLES[condition]]=blend(predictions["R0"],predictions[condition],.5)
        for name,pred in predictions.items():
            contrast,loss=score_result(held,pred,{"fold":fold,"model":name},records)
            seconds=sum(r["fit_seconds"] for r in fits if r["fold"]==fold and r["model"]==name)
            rows.append({"fold":fold,"model":name,"contrast":contrast,"log_loss":loss,"converged":True,
                         "fit_seconds":seconds,"predict_seconds":0.,"seconds":seconds})
        print(f"EXP-020 fold {fold} complete; classification awaits replay",flush=True)
    budget.check("Scoring/bootstrap")
    with new_output(args.run_dir/"cv") as stage:
        _,predictions=write_cv_analysis(stage,ctx.train,records,pd.DataFrame(rows),ctx.config,"R0","R0")
        paired_analysis(stage,ctx,records);descriptive_subgroups(args,ctx,predictions,stage/"subgroups.csv")
        json_output({name:paired_order(ctx.train,frame[data.KEYS+["p_positive"]]) for name,frame in predictions.groupby("model")},stage/"pair_order.json")
        json_output(fits,stage/"fits.json")
        seal(stage,ctx.identity,{"status":"AWAITING_REPLAY_VERIFICATION","seconds":time.perf_counter()-started,
            "memory":memory_report(),"seed_manifests":{f"{r['fold']}/{r['model']}_{r['seed']}":file_sha256(args.run_dir/"work"/f"fold_{r['fold']}"/f"{r['model']}_{r['seed']}"/"run.json") for r in fits}})


def choose(summary,config,complete):
    if not complete:return {"status":"INCOMPLETE","model":None}
    baseline,control=summary.loc["R0"],summary.loc["RB1"]
    if not np.isfinite(summary[["pooled_contrast","log_loss_mean"]].to_numpy()).all():raise ValueError("Nonfinite CV metrics")
    candidates=summary.loc[["EA8","EE8"]]
    eligible=candidates[candidates.pooled_contrast.ge(baseline.pooled_contrast+config["minimum_pooled_contrast_gain"])
        & candidates.pooled_contrast.ge(control.pooled_contrast+config["minimum_incremental_gain"])
        & candidates.fold_wins.ge(config["minimum_fold_wins"])
        & candidates.log_loss_mean.le(baseline.log_loss_mean+config["log_loss_guardrail_bits"])
        & candidates.folds.eq(config["cv_folds"])]
    winner=eligible.pooled_contrast.idxmax() if len(eligible) else "R0"
    return {"status":"PASS" if winner!="R0" else "COMPETITION_NO_GO","model":winner,
            "qualified":eligible.index.tolist(),"all_seeds_finite_and_replayed":True,"eligibility":config["eligibility"]}


def research_assessment(summary,folds,intervals):
    gain=float(summary.loc["E8","pooled_contrast"]-summary.loc["A8","pooled_contrast"])
    table=folds.pivot(index="fold",columns="model",values="contrast")
    wins=int((table.E8>table.A8).sum());ci=intervals["E8_minus_A8"]["ci95"]
    status="RESEARCH_INCONCLUSIVE"
    if gain>=.001 and wins>=3 and ci[0]>0:status="RESEARCH_PROMISING"
    if gain<=-.001 and wins<=2 and ci[1]<0:status="RESEARCH_UNPROMISING"
    return {"status":status,"E8_minus_A8":gain,"fold_wins":wins,"ci95":ci,
            "E8_minus_R1":float(summary.loc["E8","pooled_contrast"]-summary.loc["R1","pooled_contrast"]),
            "scope":"Repeatedly examined frozen-feature CV; no trainable-encoder evidence",
            "criteria":"PROMISING: >=.001 matched gain, >=3 folds, positive CI; UNPROMISING: <=-.001, <=2 folds, negative CI; otherwise INCONCLUSIVE",
            "recommendation":"Consider a separately authorized, tightly controlled LoRA pilot only after inspecting target reliability, collapse and final gradient diagnostics; do not increase lambda based on EXP-019."}
