"""Resumable eight-fit screen and explicitly gated, checkpoint-reusing full CV."""

import importlib.metadata
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

from . import data
from .ablation import paired_bootstrap
from .baseline import file_sha256
from .caption_comparison import image_banks,paired_order
from .caption_data import inner_partition,require_isolation
from .caption_student import concatenate_features
from .cli import ROOT,json_output
from .contrast_study import new_output
from .emotion_experiment import memory_report
from .fusion import blend,verify_reference
from .fusion_study import write_cv_analysis
from .gradient_analysis import deterministic_batches
from .gradient_training import state_hash
from .low_rank_study import score_result
from .polarity_workflow import ExecutionBudget,StageBlocked,read_sealed,seal
from .vote_inputs import COUNT_COLUMNS,input_context,verify_parent
from .vote_model import (VoteStudent,batch_gradients,evaluate,prior_diagnostic,seed_model,train_epoch,vote_rows)

CONDITIONS=("D0","D8")
ENSEMBLES={"D0":"ED0","D8":"ED8"}


def context(args):
    inputs=input_context(args);verify_parent(args,inputs.config)
    read_sealed(args.run_dir/"audit",inputs.identity)
    stored=pd.read_csv(args.run_dir/"audit/targets.csv",float_precision="round_trip")
    pd.testing.assert_frame_equal(inputs.train,stored,check_dtype=False)
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else args.device
    if device=="auto":device="cpu"
    if device=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA unavailable; use -Device cpu")
    files=sorted((ROOT/"src/st3").glob("*.py"))+[ROOT/"scripts/run_exp021.py",ROOT/"scripts/run_exp021.ps1"]
    identity={"audit_identity":inputs.identity,"source_sha256":{str(p.relative_to(ROOT)):file_sha256(p) for p in files},
        "device":device,"python":platform.python_version(),"cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name() if device=="cuda" else None,
        "matmul_precision":torch.get_float32_matmul_precision(),"allow_tf32":torch.backends.cuda.matmul.allow_tf32,
        "versions":{name:importlib.metadata.version(name) for name in ("torch","numpy","pandas","scipy","scikit-learn")}}
    inputs.audit_identity=inputs.identity;inputs.identity=identity
    inputs.training={**inputs.config["training"],"bootstrap_replicates":inputs.config["bootstrap_replicates"],
                     "bootstrap_seed":inputs.config["bootstrap_seed"]}
    inputs.device=torch.device(device);inputs.visual=concatenate_features(image_banks(args,inputs.train))
    return inputs


def diagnostics(model,rows,ctx,fold,condition,checkpoint):
    batch=deterministic_batches(len(rows["image"]),ctx.training["batch_size"],fold,212027,count=1)[0]
    result=batch_gradients(model,rows,batch,0. if condition=="D0" else .1,ctx.training["gradient_clip"])
    return {"condition":condition,"checkpoint":checkpoint,**result}


def fit_model(fit,inner,stopping,ctx,fold,seed,condition,budget):
    require_isolation(inner,stopping)
    data.require_same_pairs(fit,pd.concat([inner,stopping]))
    start=time.perf_counter();cfg=ctx.training;weight=0. if condition=="D0" else .1
    languages=sorted(inner.language.unique())
    rows=vote_rows(inner,ctx.visual,languages,ctx.device);validation=vote_rows(stopping,ctx.visual,languages,ctx.device)
    model,rng,optimizer=seed_model(languages,seed,cfg,ctx.device);initial={"selection":state_hash(model)}
    gradients=[diagnostics(model,rows,ctx,fold,condition,"initialization")]
    if not gradients[0]["all_logits_valid_gradients"] or gradients[0]["groups"]["shared_representation"]["text_norm"]<=0:
        raise ValueError("Degenerate initial auxiliary gradients")
    # Diagnostics do not touch RNG, grads or optimizer state.
    best=float("inf");selected=0;curves=[]
    for epoch in range(1,cfg["max_epochs"]+1):
        budget.check(f"{condition} {seed} selection",epoch)
        metrics=train_epoch(model,rows,rng,optimizer,cfg,weight)
        train_eval,_=evaluate(model,rows,cfg["batch_size"]);val_eval,_=evaluate(model,validation,cfg["batch_size"])
        row={"phase":"selection","epoch":epoch,**metrics,"train_eval_ce":train_eval["ce"],
             "validation_bce":val_eval["bce"],"validation_ce":val_eval["ce"],
             "validation_binary_aux_mae":val_eval["binary_aux_mae"]}
        if not np.isfinite(list(metrics.values())+list(train_eval.values())+list(val_eval.values())).all():raise ValueError("Nonfinite selection loss")
        curves.append(row)
        if val_eval["bce"]<best:best=val_eval["bce"];selected=epoch
        if epoch-selected>=cfg["patience"]:break
    selection_epochs=epoch;selected_metrics=curves[selected-1].copy()
    del model,optimizer,rows,validation
    languages=sorted(fit.language.unique());rows=vote_rows(fit,ctx.visual,languages,ctx.device)
    model,rng,optimizer=seed_model(languages,seed,cfg,ctx.device);initial["refit"]=state_hash(model)
    for epoch in range(1,selected+1):
        budget.check(f"{condition} {seed} refit",epoch)
        metrics=train_epoch(model,rows,rng,optimizer,cfg,weight)
        if not np.isfinite(list(metrics.values())).all():raise ValueError("Nonfinite refit loss")
        curves.append({"phase":"refit","epoch":epoch,**metrics})
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError("Nonfinite model parameters")
    training_eval,probabilities=evaluate(model,rows,cfg["batch_size"])
    inner_rows=vote_rows(inner,ctx.visual,languages,ctx.device)
    gradients.append(diagnostics(model,inner_rows,ctx,fold,condition,"final_refit"))
    return model,curves,gradients,prior_diagnostic(fit,probabilities,fit),{
        "selected_epoch":selected,"selection_epochs":selection_epochs,"fit_seconds":time.perf_counter()-start,
        "best_validation_bce":best,"selected_training_ce":selected_metrics["train_eval_ce"],
        "selected_validation_ce":selected_metrics["validation_ce"],"full_refit_evaluation":training_eval,
        "initialization_sha256":initial,"parameters":sum(p.numel() for p in model.parameters())}


@torch.inference_mode()
def predict_binary(model,frame,ctx):
    inputs=frame[data.KEYS+["image"]]
    rows=vote_rows(inputs,ctx.visual,model.languages,ctx.device,training=False)
    model.eval();parts=[]
    for batch in torch.arange(len(inputs),device=ctx.device).split(ctx.training["batch_size"]):
        logits,_=model(rows["features"][rows["image"][batch]],rows["language"][batch],auxiliary=False)
        parts.append(logits.sigmoid().cpu().numpy())
    values=np.concatenate(parts).astype(np.float64)
    if not np.isfinite(values).all():raise ValueError("Nonfinite sentiment inference")
    return values


@torch.inference_mode()
def auxiliary_probabilities(model,frame,ctx):
    # Diagnostic forwards admit exactly the same image/language lookup inputs.
    inputs=frame[data.KEYS+["image"]]
    rows=vote_rows(inputs,ctx.visual,model.languages,ctx.device,training=False)
    model.eval();parts=[]
    for batch in torch.arange(len(inputs),device=ctx.device).split(ctx.training["batch_size"]):
        _,logits=model(rows["features"][rows["image"][batch]],rows["language"][batch],auxiliary=True)
        parts.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(parts)


def fit_seed(args,ctx,fold,seed,condition,budget):
    fit=ctx.train[ctx.train.fold.ne(fold)];held=ctx.train[ctx.train.fold.eq(fold)]
    inner,stopping=inner_partition(fit,ctx.training,fold);require_isolation(inner,stopping,held)
    baseline=held[data.KEYS].merge(ctx.reference[ctx.reference.fold.eq(fold)][data.KEYS+["p_positive"]],on=data.KEYS,validate="one_to_one")
    directory=args.run_dir/"work"/f"fold_{fold}"/f"{condition}_{seed}"
    if not directory.exists():
        with new_output(directory) as stage:
            model,curves,gradients,priors,info=fit_model(fit,inner,stopping,ctx,fold,seed,condition,budget)
            values=predict_binary(model,held,ctx)
            auxiliary=auxiliary_probabilities(model,held,ctx)
            torch.save({"languages":list(model.languages),"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}},stage/"student.pt")
            np.savez(stage/"predictions.npz",painting=held.painting.to_numpy(dtype=str),language=held.language.to_numpy(dtype=str),probabilities=values,auxiliary=auxiliary)
            pd.DataFrame(curves).to_csv(stage/"curves.csv",index=False)
            pd.DataFrame(priors).to_csv(stage/"training_prior_diagnostics.csv",index=False)
            json_output(gradients,stage/"gradients.json")
            seal(stage,ctx.identity,{**info,"fold":fold,"seed":seed,"condition":condition,"finite":True,
                "fit_paintings":sorted(fit.painting.unique()),"inner_paintings":sorted(inner.painting.unique()),
                "stopping_paintings":sorted(stopping.painting.unique()),"held_paintings":sorted(held.painting.unique()),
                "unseen_language_pairs":int((~held.language.isin(model.languages)).sum()),
                "audit_manifest_sha256":file_sha256(args.run_dir/"audit/run.json")})
            del model
    info=read_sealed(directory,ctx.identity)
    for key,frame in (("fit",fit),("inner",inner),("stopping",stopping),("held",held)):
        if info[key+"_paintings"]!=sorted(frame.painting.unique()):raise ValueError("Checkpoint painting isolation differs")
    if info["fold"]!=fold or info["seed"]!=seed or info["condition"]!=condition or not info["finite"]:
        raise ValueError("Checkpoint identity differs")
    if info["audit_manifest_sha256"]!=file_sha256(args.run_dir/"audit/run.json"):raise ValueError("Empirical targets changed")
    with np.load(directory/"predictions.npz",allow_pickle=False) as archive:
        if archive["painting"].tolist()!=held.painting.tolist() or archive["language"].tolist()!=held.language.tolist():raise ValueError("Prediction keys differ")
        values=archive["probabilities"].copy();auxiliary=archive["auxiliary"].copy()
    if values.shape!=(len(held),) or auxiliary.shape!=(len(held),8) or not np.isfinite(values).all() or not np.isfinite(auxiliary).all():
        raise ValueError("Invalid checkpoint predictions")
    if ((values<0)|(values>1)).any() or (auxiliary<0).any() or not np.allclose(auxiliary.sum(1),1.,atol=2e-6):raise ValueError("Invalid probability range")
    print(f"Completed/reused fold {fold} {condition} seed {seed}, epoch {info['selected_epoch']}",flush=True)
    return held[data.KEYS].assign(p_positive=values),auxiliary,info


def run_pilot(args):
    started=time.perf_counter();ctx=context(args)
    input_loading_seconds=time.perf_counter()-started
    if (args.run_dir/"pilot").exists():read_sealed(args.run_dir/"pilot",ctx.identity);return
    budget=ExecutionBudget(args.max_seconds or 240.)
    fit=ctx.train[ctx.train.fold.ne(1)];inner,stopping=inner_partition(fit,ctx.training,1)
    target_tick=time.perf_counter()
    languages=sorted(inner.language.unique());rows=vote_rows(inner,ctx.visual,languages,ctx.device)
    validation=vote_rows(stopping,ctx.visual,languages,ctx.device)
    target_preparation_seconds=time.perf_counter()-target_tick
    a,ra,oa=seed_model(languages,2027,ctx.training,ctx.device);b,rb,ob=seed_model(languages,2027,ctx.training,ctx.device)
    if state_hash(a)!=state_hash(b) or not torch.equal(ra.get_state(),rb.get_state()):raise ValueError("Unmatched initialization/RNG")
    a.eval();b.eval()
    with torch.no_grad():
        x=rows["features"][rows["image"][:32]];l=rows["language"][:32]
        if not torch.equal(a(x,l)[0],b(x,l)[0]):raise ValueError("Unmatched binary logits")
    del a,b,oa,ob
    if ctx.device.type=="cuda":torch.cuda.reset_peak_memory_stats()
    results={};gradients=[];prior_rows=[]
    with new_output(args.run_dir/"pilot") as stage:
        for condition in CONDITIONS:
            model,rng,optimizer=seed_model(languages,2027,ctx.training,ctx.device)
            gradients.append(diagnostics(model,rows,ctx,1,condition,"initialization"))
            epochs=[]
            for epoch in range(1,ctx.config["pilot_warmup_epochs"]+ctx.config["pilot_timed_epochs"]+1):
                budget.check(f"Pilot {condition}",epoch);tick=time.perf_counter()
                metrics=train_epoch(model,rows,rng,optimizer,ctx.training,0. if condition=="D0" else .1)
                val,_=evaluate(model,validation,ctx.training["batch_size"])
                epochs.append({"epoch":epoch,**metrics,"validation_bce":val["bce"],"validation_ce":val["ce"],"seconds":time.perf_counter()-tick})
            gradients.append(diagnostics(model,rows,ctx,1,condition,"short_pilot_final"))
            evaluation,probs=evaluate(model,rows,ctx.training["batch_size"])
            prior_rows.extend([{**r,"condition":condition} for r in prior_diagnostic(inner,probs,inner)])
            results[condition]={"epochs":epochs,"evaluation":evaluation,"mean_epoch_seconds":float(np.mean([r["seconds"] for r in epochs[ctx.config["pilot_warmup_epochs"]:]])),
                "parameters":sum(p.numel() for p in model.parameters())}
            del model,optimizer
        if not all(g["finite"] and g["all_logits_valid_gradients"] for g in gradients):raise ValueError("Invalid pilot gradients")
        histories=[]
        for parent,path,models in (("EXP018",args.exp018_dir/"cv/fits.json",{"M1","M3"}),
                                   ("EXP019",args.exp019_dir/"cv/fits.json",{"C2"}),
                                   ("EXP020",args.exp020_dir/"cv/fits.json",{"A8","E8"})):
            records=json.loads(path.read_text());records=[r for r in records if r.get("model") in models]
            histories.extend([{**r,"parent":parent} for r in records])
        upper_seconds=float(np.quantile([r["fit_seconds"] for r in histories],ctx.config["runtime_quantile"]))
        upper_epochs=float(np.quantile([r["selection_epochs"]+r["selected_epoch"] for r in histories],ctx.config["runtime_quantile"]))
        epoch_cost=max(r["mean_epoch_seconds"] for r in results.values())*1.15
        per_fit=max(upper_seconds,upper_epochs*epoch_cost)*ctx.config["runtime_margin"]
        screen_fits=len(ctx.config["screen_folds"])*len(ctx.config["screen_seeds"])*len(CONDITIONS)
        full_fits=ctx.training["cv_folds"]*len(ctx.training["seeds"])*len(CONDITIONS)
        screen=screen_fits*per_fit;full=full_fits*per_fit
        prep=read_sealed(args.run_dir/"audit",ctx.audit_identity)["seconds"]+read_sealed(args.run_dir/"preflight",ctx.audit_identity)["seconds"]
        metadata={"status":"READY_FOR_LOCAL_RUN","seconds":time.perf_counter()-started,
            "input_loading_and_identity_seconds":input_loading_seconds,"target_tensor_preparation_seconds":target_preparation_seconds,
            "screen_fits":screen_fits,"screen_training_stages":2*screen_fits,
            "full_fits":full_fits,"additional_fits_after_verified_screen":full_fits-screen_fits,"historical_fit_records":len(histories),
            "historical_p90_fit_seconds":upper_seconds,"historical_p90_total_epochs":upper_epochs,
            "estimated_screen_seconds":screen,"estimated_full_cv_seconds":full,"estimated_remaining_cv_seconds":(full_fits-screen_fits)*per_fit,
            "estimated_screen_verification_seconds":ctx.config["screen_verification_allowance_seconds"],
            "estimated_cv_verification_seconds":ctx.config["cv_verification_allowance_seconds"],
            "estimated_preparation_plus_screen_seconds":prep+(time.perf_counter()-started)+screen+ctx.config["screen_verification_allowance_seconds"],
            "finite_nondegenerate_gradients":True,"matched_initialization_logits_rng":True,"memory":memory_report()}
        json_output(results,stage/"timings.json");json_output(gradients,stage/"gradients.json")
        pd.DataFrame(prior_rows).to_csv(stage/"prior_diagnostics.csv",index=False)
        pd.DataFrame([{k:r[k] for k in ("parent","model","fold","seed","fit_seconds","selection_epochs","selected_epoch")} for r in histories]).to_csv(stage/"historical_fit_times.csv",index=False)
        seal(stage,ctx.identity,metadata)
    print(f"Pilot complete; estimated preparation + eight-fit screening + replay {metadata['estimated_preparation_plus_screen_seconds']/60:.2f} min. Full CV is NEVER automatic.",flush=True)


def run_study(args,*,screen):
    ctx=context(args);pilot=read_sealed(args.run_dir/"pilot",ctx.identity)
    if not screen:
        from .vote_verification import screen_decision
        decision=screen_decision(args)
        if decision["status"]!="PILOT_PROMISING":raise StageBlocked("Screen is not PILOT_PROMISING; full CV is prohibited")
    phase="screen" if screen else "cv";directory=args.run_dir/phase
    if directory.exists():
        m=read_sealed(directory,ctx.identity)
        for key,digest in m["seed_manifests"].items():
            fold,name=key.split("/");p=args.run_dir/"work"/f"fold_{fold}"/name
            read_sealed(p,ctx.identity)
            if file_sha256(p/"run.json")!=digest:raise ValueError("Published study checkpoint changed")
        return
    seconds=pilot["estimated_screen_seconds"] if screen else pilot["estimated_remaining_cv_seconds"]
    budget=ExecutionBudget(args.max_seconds or max(120.,1.5*seconds))
    folds=ctx.config["screen_folds"] if screen else list(range(1,ctx.training["cv_folds"]+1))
    seeds=ctx.config["screen_seeds"] if screen else ctx.training["seeds"]
    records={key:[] for key in ("predictions","paintings","languages")};rows=[];fits=[];consistency=[]
    for fold in folds:
        held=ctx.train[ctx.train.fold.eq(fold)]
        predictions={"R0":held[data.KEYS].merge(ctx.reference[ctx.reference.fold.eq(fold)][data.KEYS+["p_positive"]],on=data.KEYS,validate="one_to_one")}
        for condition in CONDITIONS:
            values=[]
            for seed in seeds:
                budget.check(f"{phase} fold {fold} {condition} {seed}")
                pred,aux,info=fit_seed(args,ctx,fold,seed,condition,budget);values.append(pred.p_positive.to_numpy());fits.append(info)
                consistency.append({"fold":fold,"model":condition,"seed":seed,"binary_aux_mae":float(np.mean(np.abs(pred.p_positive.to_numpy()-aux[:,:4].sum(1))))})
            predictions[condition]=held[data.KEYS].assign(p_positive=np.round(np.mean(values,axis=0),6))
            predictions[ENSEMBLES[condition]]=blend(predictions["R0"],predictions[condition],.5)
        for name,pred in predictions.items():
            contrast,loss=score_result(held,pred,{"fold":fold,"model":name},records)
            cost=sum(r["fit_seconds"] for r in fits if r["fold"]==fold and r["condition"]==name)
            rows.append({"fold":fold,"model":name,"contrast":contrast,"log_loss":loss,"converged":True,"fit_seconds":cost,"predict_seconds":0.,"seconds":cost})
    budget.check("Scoring and paired bootstrap")
    subset=ctx.train[ctx.train.fold.isin(folds)]
    with new_output(directory) as stage:
        _,predictions=write_cv_analysis(stage,subset,records,pd.DataFrame(rows),ctx.training,"R0","R0")
        wide=pd.concat(records["paintings"]).query("status == 'scored'").pivot(index="painting",columns="model",values="contrast")
        changes={a+"_minus_"+b:wide[a]-wide[b] for a,b in (("D8","D0"),("ED8","ED0"),("ED8","R0"))}
        frame=pd.DataFrame(changes).reset_index().assign(status="scored")
        draws,intervals=paired_bootstrap(frame,ctx.config["bootstrap_replicates"],ctx.config["bootstrap_seed"],columns=list(changes))
        frame.to_csv(stage/"matched_painting_changes.csv",index=False);draws.to_csv(stage/"matched_bootstrap.csv",index=False)
        json_output(intervals,stage/"matched_intervals.json")
        pd.DataFrame(consistency).to_csv(stage/"binary_aux_consistency.csv",index=False)
        json_output({name:paired_order(subset,pred[data.KEYS+["p_positive"]]) for name,pred in predictions.groupby("model")},stage/"pair_order.json")
        json_output(fits,stage/"fits.json")
        seal(stage,ctx.identity,{"status":"AWAITING_REPLAY_VERIFICATION","folds":folds,"seeds":seeds,
            "seed_manifests":{f"{r['fold']}/{r['condition']}_{r['seed']}":file_sha256(args.run_dir/"work"/f"fold_{r['fold']}"/f"{r['condition']}_{r['seed']}"/"run.json") for r in fits},
            "screen_manifest_sha256":file_sha256(args.run_dir/"screen/run.json") if not screen else None})
    print(f"{phase} complete; run {'ScreenVerify' if screen else 'Verify'} before any decision",flush=True)
