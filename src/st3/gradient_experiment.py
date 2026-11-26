"""Fixed lambda=1 C2 fits, historical controls, paired analysis and replay."""

import json
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
from .fusion import blend, verify_reference
from .fusion_study import write_cv_analysis
from .gradient_analysis import deterministic_batches, diagnostic
from .gradient_training import TrainingRecorder, matched_factory, state_hash
from .gradient_workflow import budget019, context019, preserve_parent
from .low_rank_study import score_result
from .metrics import score
from .polarity_student import PolarityStudent, polarity_rows, seed_polarity_model
from .polarity_targets import condition_targets
from .polarity_workflow import ExecutionBudget, StageBlocked, load_geometry, read_sealed, seal

REUSE = {"M0":"M0", "M1":"C0", "M3":"C1", "B1":"E0", "B3":"E1"}
COMPARISONS = (("C1","C0"),("C2","C0"),("C2","C1"),("E1","E0"),("E2","E0"),("E2","E1"),("E2","M0"))


def partition_inputs(args, ctx, fold, visual):
    fit, held = ctx.train[ctx.train.fold.ne(fold)], ctx.train[ctx.train.fold.eq(fold)]
    inner, stopping = inner_partition(fit, ctx.config, fold)
    require_isolation(inner, stopping, held)
    directory = args.exp018_dir / "work" / f"fold_{fold}" / "geometry"
    bundles = {phase: load_geometry(directory / phase, ctx.parent_identity, rows.painting)
               for phase, rows in (("inner",inner),("full",fit))}
    prepared = {phase: condition_targets(bundle, "M3") for phase, bundle in bundles.items()}
    # All checkpoint comparisons use inner-training images and the same full-refit
    # PCA targets; early selection diagnostics are explicitly a different scope.
    full_targets, full_weights = prepared["full"]
    diagnostic_rows = polarity_rows(inner, visual, sorted(fit.language.unique()),
                                    torch.device(ctx.identity["runtime_and_source"]["device"]), full_targets, full_weights)
    return fit, held, inner, stopping, prepared, diagnostic_rows


def load_model(path, config, device):
    payload = torch.load(path, weights_only=True, map_location="cpu")
    model = PolarityStudent(payload["languages"], config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Nonfinite checkpoint parameters")
    return model


def collect(model, rows, ctx, fold, seed, condition, checkpoint):
    weight = ctx.protocol["text_weights"][condition]
    output = []
    for index, batch in enumerate(deterministic_batches(len(rows["image"]), ctx.config["batch_size"], fold,
            ctx.protocol["diagnostic_seed"], ctx.protocol["diagnostic_batches"])):
        result = diagnostic(model, rows, batch, weight, ctx.config["gradient_clip"])
        output.append({"fold":fold,"seed":seed,"condition":condition,"checkpoint":checkpoint,
                       "batch":index+1,"training_scope":"inner training only; full outer-training PCA", **result})
    return output


def run_gradient_pilot(args):
    start = time.perf_counter()
    ctx = context019(args)
    output = args.run_dir / "gradient_pilot"
    if output.exists():
        read_sealed(output, ctx.identity)
        print("Reused sealed gradient pilot", flush=True)
        return
    budget = ExecutionBudget(args.max_seconds or 300.)
    visual = concatenate_features(ctx.banks)
    device = torch.device(ctx.identity["runtime_and_source"]["device"])
    diagnostics = []
    tick = time.perf_counter()
    for fold in ctx.protocol["diagnostic_folds"]:
        fit, _, _, _, _, rows = partition_inputs(args, ctx, fold, visual)
        for seed in ctx.config["seeds"]:
            for condition, old in (("C0","M1"),("C1","M3")):
                budget.check(f"GradientPilot fold {fold} {condition} {seed}")
                model, _, _ = seed_polarity_model(sorted(fit.language.unique()), seed, ctx.config, device)
                diagnostics.extend(collect(model, rows, ctx, fold, seed, condition, "initialization"))
                if condition == "C1":
                    diagnostics.extend(collect(model, rows, ctx, fold, seed, "C2", "initialization"))
                del model
                model = load_model(args.exp018_dir / "work" / f"fold_{fold}" / "seeds" / f"{old}_{seed}" / "student.pt", ctx.config, device)
                diagnostics.extend(collect(model, rows, ctx, fold, seed, condition, "selected_final_refit"))
                del model
        print(f"Completed training-only historical gradient diagnostics for fold {fold}", flush=True)
    diagnostic_seconds = time.perf_counter()-tick
    fold = 1
    fit, _, inner, stopping, prepared, _ = partition_inputs(args, ctx, fold, visual)
    languages = sorted(inner.language.unique())
    targets, weights = prepared["inner"]
    rows = polarity_rows(inner, visual, languages, device, targets, weights)
    validation = polarity_rows(stopping, visual, languages, device)
    epochs = []
    with new_output(output) as stage:
        recorder = TrainingRecorder(stage, budget, ctx.protocol["early_refit_epoch"])
        model, rng, optimizer = matched_factory(languages, ctx.config["seeds"][0], ctx.config, device)
        for epoch in range(1, ctx.protocol["pilot_epochs"]+1):
            recorder.progress("pilot_selection", epoch)
            tick = time.perf_counter()
            bce, text = recorder.epoch_function(model, rows, rng, optimizer, ctx.config, 1.)
            vbce = validation_bce(model, validation, ctx.config["batch_size"])
            seconds = time.perf_counter()-tick
            if not np.isfinite([bce,text,vbce]).all():
                raise ValueError("Nonfinite short C2 pilot")
            epochs.append({"epoch":epoch,"bce":bce,"text_loss":text,"validation_bce":vbce,"seconds":seconds})
        # The pilot is not one of the fifteen CV fits and is never resumed into C2.
        json_output(diagnostics, stage / "gradients.json")
        pd.DataFrame(epochs).to_csv(stage / "epochs.csv",index=False)
        pd.DataFrame(recorder.records).to_csv(stage / "telemetry.csv",index=False)
        historical = json.loads((args.exp018_dir / "cv/fits.json").read_text())
        fits = [r for r in historical if r["model"] == "M3"]
        selection_epochs = sum(r["selection_epochs"] for r in fits)
        refit_epochs = sum(r["selected_epoch"] for r in fits)
        historical_seconds = sum(r["fit_seconds"] for r in fits)
        measured_epoch = float(np.mean([r["seconds"] for r in epochs[1:]]))
        epoch_forecast = measured_epoch * (selection_epochs+refit_epochs) * 1.15
        estimate = max(historical_seconds, epoch_forecast) * ctx.protocol["historical_margin"]
        allowance = ctx.protocol["diagnostic_verification_allowance_seconds"]
        total = estimate + allowance + diagnostic_seconds
        metadata = {"status":"READY_FOR_LOCAL_RUN", "new_cv_fits":len(fits),"training_stages":2*len(fits),
            "historical_selection_epochs":selection_epochs,"historical_refit_epochs":refit_epochs,
            "historical_c1_fit_seconds":historical_seconds,"parent_last_cv_invocation_seconds":
                json.loads((args.exp018_dir / "cv/run.json").read_text())["metadata"]["seconds"],
            "mean_pilot_epoch_seconds":measured_epoch,"epoch_forecast_seconds":epoch_forecast,
            "estimated_cv_seconds":estimate,"estimated_verification_seconds":max(120.,diagnostic_seconds+60.),
            "diagnostic_verification_allowance_seconds":allowance,"diagnostic_seconds":diagnostic_seconds,
            "estimated_total_seconds":total,"actual_seconds":time.perf_counter()-start,
            "early_parent_checkpoints":"unavailable; no historical models retrained",
            "automatic_budget_seconds":ctx.protocol["codex_budget_seconds"]}
        seal(stage,ctx.identity,metadata)
    print(f"Pilot estimate {total/60:.2f} minutes, historical C1 fits {historical_seconds:.2f}s; full CV remains separate",flush=True)


def save_fit(args,ctx,fold,seed,visual,budget):
    fit, held, inner, stopping, prepared, diagnostic_rows = partition_inputs(args,ctx,fold,visual)
    output = args.run_dir / "work" / f"fold_{fold}" / f"C2_{seed}"
    device = torch.device(ctx.identity["runtime_and_source"]["device"])
    baseline = held[data.KEYS].merge(ctx.predictions[ctx.predictions.model.eq("M0")][[*data.KEYS,"p_positive"]],on=data.KEYS,validate="one_to_one")
    if not output.exists():
        with new_output(output) as stage:
            recorder = TrainingRecorder(stage,budget,ctx.protocol["early_refit_epoch"])
            model,curves,info = fit_student(fit,inner,stopping,visual,prepared,seed,ctx.config,device,
                model_factory=matched_factory,epoch_function=recorder.epoch_function,training_rows=polarity_rows,progress=recorder.progress)
            values = predict_student(model,held[[*data.KEYS,"image"]],visual,baseline,device,ctx.config["batch_size"])
            torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"languages":list(model.languages)},stage/"student.pt")
            np.savez(stage/"predictions.npz",painting=held.painting.to_numpy(dtype=str),language=held.language.to_numpy(dtype=str),probabilities=values)
            pd.DataFrame(curves).to_csv(stage/"curves.csv",index=False)
            pd.DataFrame(recorder.records).to_csv(stage/"telemetry.csv",index=False)
            gradients=[]
            if fold in ctx.protocol["diagnostic_folds"]:
                gradients.extend(collect(model,diagnostic_rows,ctx,fold,seed,"C2","selected_final_refit"))
                if (stage/"early.pt").exists():
                    early=load_model(stage/"early.pt",ctx.config,device)
                    gradients.extend(collect(early,diagnostic_rows,ctx,fold,seed,"C2","early_refit_epoch_5"))
                    del early
            json_output(gradients,stage/"gradients.json")
            seal(stage,ctx.identity,{**info,"fold":fold,"seed":seed,"model":"C2","finite":True,
                "initialization":"exact EXP-018 architecture/seed; fresh lambda=1 training",
                "fit_paintings":sorted(fit.painting.unique()),"inner_paintings":sorted(inner.painting.unique()),
                "stopping_paintings":sorted(stopping.painting.unique()),"held_paintings":sorted(held.painting.unique()),
                "parent_geometry_sha256":{phase:file_sha256(args.exp018_dir/"work"/f"fold_{fold}"/"geometry"/phase/"run.json") for phase in ("inner","full")}})
            del model
    info=read_sealed(output,ctx.identity)
    for key,partition in (("fit",fit),("inner",inner),("stopping",stopping),("held",held)):
        if info[key+"_paintings"]!=sorted(partition.painting.unique()):
            raise ValueError("C2 resumed fit painting partition differs")
    if not info["finite"] or info["seed"]!=seed or info["fold"]!=fold:
        raise ValueError("Invalid seed metadata")
    for phase,digest in info["parent_geometry_sha256"].items():
        if digest!=file_sha256(args.exp018_dir/"work"/f"fold_{fold}"/"geometry"/phase/"run.json"):
            raise ValueError("C2 reused PCA provenance differs")
    with np.load(output/"predictions.npz",allow_pickle=False) as archive:
        if archive["painting"].tolist()!=held.painting.tolist() or archive["language"].tolist()!=held.language.tolist():
            raise ValueError("C2 resumed prediction keys differ")
        values=archive["probabilities"].copy()
    if values.shape!=(len(held),) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError("Invalid C2 probabilities")
    print(f"Completed/reused C2 fold {fold}, seed {seed}, selected epoch {info['selected_epoch']}",flush=True)
    return held[data.KEYS].assign(p_positive=values),info


def compare_outputs(stage,ctx,records,rows):
    summary,predictions=write_cv_analysis(stage,ctx.train,records,pd.DataFrame(rows),ctx.config,"M0","M0")
    paintings=pd.concat(records["paintings"])
    wide=paintings[paintings.status.eq("scored")].pivot(index="painting",columns="model",values="contrast")
    changes={a+"_minus_"+b:wide[a]-wide[b] for a,b in COMPARISONS}
    frame=pd.DataFrame(changes).reset_index().assign(status="scored")
    draws,intervals=paired_bootstrap(frame,ctx.protocol["bootstrap_replicates"],ctx.protocol["bootstrap_seed"],columns=list(changes))
    frame.to_csv(stage/"matched_painting_changes.csv",index=False)
    draws.to_csv(stage/"matched_bootstrap_distribution.csv",index=False)
    json_output(intervals,stage/"matched_bootstrap.json")
    languages=pd.concat(records["languages"]).pivot(index=["fold","language"],columns="model",values="log_loss")
    pd.DataFrame({a+"_minus_"+b:languages[a]-languages[b] for a,b in COMPARISONS}).to_csv(stage/"matched_language_changes.csv")
    json_output({name:paired_order(ctx.train,frame[[*data.KEYS,"p_positive"]]) for name,frame in predictions.groupby("model")},stage/"pair_order.json")
    return summary


def run_cv(args):
    start=time.perf_counter();ctx=context019(args)
    read_sealed(args.run_dir/"gradient_pilot",ctx.identity)
    if (args.run_dir/"cv").exists():
        cv=read_sealed(args.run_dir/"cv",ctx.identity)
        for key,digest in cv["seed_manifests"].items():
            fold,name=key.split("/");path=args.run_dir/"work"/f"fold_{fold}"/name
            read_sealed(path,ctx.identity)
            if file_sha256(path/"run.json")!=digest:raise ValueError("Completed C2 seed changed")
        print("Reused complete C2 CV; Verify remains separate",flush=True);return
    budget=budget019(args,ctx,"CV"); visual=concatenate_features(ctx.banks)
    records={name:[] for name in ("predictions","paintings","languages")};rows=[]; fits=[]
    for fold in range(1,ctx.config["cv_folds"]+1):
        held=ctx.train[ctx.train.fold.eq(fold)]
        predictions={new:held[data.KEYS].merge(ctx.predictions[ctx.predictions.model.eq(old)][[*data.KEYS,"p_positive"]],on=data.KEYS,validate="one_to_one") for old,new in REUSE.items()}
        values=[]
        for seed in ctx.config["seeds"]:
            budget.check(f"C2 fold {fold} seed {seed}")
            pred,info=save_fit(args,ctx,fold,seed,visual,budget);values.append(pred.p_positive.to_numpy());fits.append(info)
        predictions["C2"]=held[data.KEYS].assign(p_positive=np.round(np.mean(values,axis=0),6))
        predictions["E2"]=blend(predictions["M0"],predictions["C2"],.5)
        for name,pred in predictions.items():
            contrast,loss=score_result(held,pred,{"model":name,"fold":fold},records)
            seconds=sum(f["fit_seconds"] for f in fits if f["fold"]==fold) if name=="C2" else 0.
            rows.append({"model":name,"fold":fold,"contrast":contrast,"log_loss":loss,"converged":True,"fit_seconds":seconds,"predict_seconds":0.,"seconds":seconds})
        print(f"Fold {fold} completed; no promotion decision before replay",flush=True)
    budget.check("Scoring and bootstrap")
    with new_output(args.run_dir/"cv") as stage:
        compare_outputs(stage,ctx,records,rows)
        json_output(fits,stage/"fits.json")
        seal(stage,ctx.identity,{"status":"AWAITING_REPLAY_VERIFICATION","seconds":time.perf_counter()-start,
            "seed_manifests":{f"{f['fold']}/C2_{f['seed']}":file_sha256(args.run_dir/"work"/f"fold_{f['fold']}"/f"C2_{f['seed']}"/"run.json") for f in fits}})
    print("C2 CV complete; run Verify before classification or locked confirmation",flush=True)


def choose(summary,config,complete):
    if not complete:
        return {"status":"INCOMPLETE","model":None}
    baseline,control,candidate=summary.loc["M0"],summary.loc["E0"],summary.loc["E2"]
    if not np.isfinite(summary[["pooled_contrast","log_loss_mean"]].to_numpy()).all():
        raise ValueError("Nonfinite CV scores")
    checks={"gain_over_M0":candidate.pooled_contrast>=baseline.pooled_contrast+config["minimum_pooled_contrast_gain"],
            "gain_over_E0":candidate.pooled_contrast>=control.pooled_contrast+config["minimum_incremental_gain"],
            "fold_wins":candidate.fold_wins>=config["minimum_fold_wins"],
            "loss_guardrail":candidate.log_loss_mean<=baseline.log_loss_mean+config["log_loss_guardrail_bits"],
            "all_folds":candidate.folds==config["cv_folds"]}
    passed=all(checks.values())
    return {"status":"PASS" if passed else "COMPETITION_NO_GO","model":"E2" if passed else "M0",
            "checks":{key:bool(value) for key,value in checks.items()},"all_seeds_finite_and_replayed":True,
            "eligibility":config["eligibility"]}


def summarize_gradients(records):
    rows=[]
    for record in records:
        for name,values in record["groups"].items():
            rows.append({**{k:record[k] for k in ("fold","seed","condition","checkpoint","batch")},"group":name,**values})
    return pd.DataFrame(rows)


def research_assessment(summary,gradients):
    # Research classification is independent of the competition gate. These
    # conservative criteria are fixed before C2 scores and allow uncertainty.
    final=gradients[(gradients.checkpoint=="selected_final_refit") & (gradients.group=="shared_representation")]
    c2=final[final.condition.eq("C2")];c1=final[final.condition.eq("C1")]
    if len(c2)==0 or len(c1)==0:
        return {"status":"RESEARCH_INCONCLUSIVE","reason":"Longitudinal C2 evidence incomplete"}
    higher=float(c2.scaled_ratio.median())>float(c1.scaled_ratio.median())
    conflict=float((c2.cosine < -.5).mean())
    heldout=float(summary.loc["C2","pooled_contrast"]-summary.loc["C1","pooled_contrast"])
    # A large stable gradient without held-out benefit remains inconclusive.
    status="RESEARCH_PROMISING" if higher and conflict<=.2 and heldout>=.002 else "RESEARCH_INCONCLUSIVE"
    if not higher and conflict>=.8 and heldout<=-.003:
        status="RESEARCH_UNPROMISING"
    return {"status":status,"c2_median_scaled_ratio":float(c2.scaled_ratio.median()),
            "c1_median_scaled_ratio":float(c1.scaled_ratio.median()),"strong_conflict_fraction":conflict,
            "C2_minus_C1_contrast":heldout,"scope":"Feature/head gradients only; encoder fine-tuning remains untested",
            "criteria":"PROMISING: stronger median signal, <=20% cosine<-.5, >=.002 C2-C1 Contrast; UNPROMISING: no stronger signal, >=80% conflict, <=-.003 Contrast; otherwise INCONCLUSIVE"}


def _verify(args,ctx):
    cv=read_sealed(args.run_dir/"cv",ctx.identity);budget=budget019(args,ctx,"Verify")
    visual=concatenate_features(ctx.banks);device=torch.device(ctx.identity["runtime_and_source"]["device"])
    stored=pd.read_csv(args.run_dir/"cv/predictions.csv",float_precision="round_trip")
    summary=pd.read_csv(args.run_dir/"cv/summary.csv",index_col="model",float_precision="round_trip")
    gradients=json.loads((args.run_dir/"gradient_pilot/gradients.json").read_text());seed_rows=[];curves=[]
    for old,new in REUSE.items():
        verify_reference(stored[stored.model.eq(new)][[*data.KEYS,"p_positive"]],ctx.predictions[ctx.predictions.model.eq(old)][[*data.KEYS,"p_positive"]])
    for fold in range(1,ctx.config["cv_folds"]+1):
        _,held,_,_,_,_=partition_inputs(args,ctx,fold,visual)
        baseline=held[data.KEYS].merge(ctx.predictions[ctx.predictions.model.eq("M0")][[*data.KEYS,"p_positive"]],on=data.KEYS)
        probabilities=[]
        for seed in ctx.config["seeds"]:
            budget.check(f"Replay C2 fold {fold} seed {seed}")
            path=args.run_dir/"work"/f"fold_{fold}"/f"C2_{seed}"
            if file_sha256(path/"run.json")!=cv["seed_manifests"][f"{fold}/C2_{seed}"]:
                raise ValueError("CV seed manifest changed")
            # Reuse the same metadata/coverage/partition checks without refitting.
            pred,info=save_fit(args,ctx,fold,seed,visual,budget)
            model=load_model(path/"student.pt",ctx.config,device)
            values=predict_student(model,held[[*data.KEYS,"image"]],visual,baseline,device,ctx.config["batch_size"])
            np.testing.assert_array_equal(values,pred.p_positive.to_numpy());probabilities.append(values)
            seed_rows.append(held[data.KEYS].assign(p_positive=np.round(values,6),model="C2",seed=seed,fold=fold))
            curve=pd.read_csv(path/"curves.csv");selection=curve[curve.phase.eq("selection")]
            if int(selection.loc[selection.validation_bce.idxmin(),"epoch"])!=info["selected_epoch"] or len(curve[curve.phase.eq("refit")])!=info["selected_epoch"]:
                raise ValueError("C2 early stopping/refit mismatch")
            curves.append(curve.assign(fold=fold,seed=seed,model="C2"))
            gradients.extend(json.loads((path/"gradients.json").read_text()));del model
        pred=held[data.KEYS].assign(p_positive=np.round(np.mean(probabilities,axis=0),6))
        verify_reference(pred,stored[stored.model.eq("C2") & stored.fold.eq(fold)][[*data.KEYS,"p_positive"]])
        verify_reference(blend(baseline,pred,.5),stored[stored.model.eq("E2") & stored.fold.eq(fold)][[*data.KEYS,"p_positive"]])
    for name,rows in stored.groupby("model"):
        metrics=score(ctx.train,rows[[*data.KEYS,"p_positive"]])
        np.testing.assert_allclose([metrics["cultural_contrast"],metrics["log_loss"]],summary.loc[name,["pooled_contrast","pooled_log_loss"]].astype(float),atol=1e-14,rtol=0.)
        fold_values=[score(ctx.train[ctx.train.fold.eq(f)],rows[rows.fold.eq(f)][[*data.KEYS,"p_positive"]]) for f in range(1,ctx.config["cv_folds"]+1)]
        np.testing.assert_allclose([np.mean([r["cultural_contrast"] for r in fold_values]),np.mean([r["log_loss"] for r in fold_values])],summary.loc[name,["contrast_mean","log_loss_mean"]].astype(float),atol=1e-14,rtol=0.)
    baseline_folds=[score(ctx.train[ctx.train.fold.eq(f)],stored[stored.model.eq("M0") & stored.fold.eq(f)][[*data.KEYS,"p_positive"]])["cultural_contrast"] for f in range(1,ctx.config["cv_folds"]+1)]
    for name in stored.model.unique():
        wins=sum(score(ctx.train[ctx.train.fold.eq(f)],stored[stored.model.eq(name) & stored.fold.eq(f)][[*data.KEYS,"p_positive"]])["cultural_contrast"]>base for f,base in enumerate(baseline_folds,1))
        if wins!=summary.loc[name,"fold_wins"]:raise ValueError("Fold wins replay differs")
    original=pd.read_csv(args.exp018_dir/"verification/seed_predictions.csv",float_precision="round_trip")
    original=original[original.model.isin(["M1","M3"])].copy();original["model"]=original.model.map(REUSE)
    for fold in range(1,ctx.config["cv_folds"]+1):
        for old,name in (("M1","C0"),("M3","C1")):
            for seed in ctx.config["seeds"]:
                curve=pd.read_csv(args.exp018_dir/"work"/f"fold_{fold}"/"seeds"/f"{old}_{seed}"/"curves.csv")
                curves.append(curve.assign(fold=fold,seed=seed,model=name))
    seeds=pd.concat([original,*seed_rows],ignore_index=True);seed_metrics=[]
    base=ctx.predictions[ctx.predictions.model.eq("M0")][[*data.KEYS,"p_positive"]]
    for (name,seed),pred in seeds.groupby(["model","seed"]):
        pred=pred[[*data.KEYS,"p_positive"]]
        for model,values in ((name,pred),("E"+name[1:],blend(base,pred,.5))):
            metric=score(ctx.train,values);seed_metrics.append({"model":model,"seed":seed,"contrast":metric["cultural_contrast"],"log_loss":metric["log_loss"]})
    count=preserve_parent(args,ctx.identity)
    output=args.run_dir/"verification"
    if not output.exists():
        with new_output(output) as stage:
            gradient_frame=summarize_gradients(gradients);gradient_frame.to_csv(stage/"gradients.csv",index=False)
            pd.DataFrame(seed_metrics).to_csv(stage/"seed_metrics.csv",index=False)
            pd.concat(curves).to_csv(stage/"all_training_curves.csv",index=False)
            json_output(research_assessment(summary,gradient_frame),stage/"research_assessment.json")
            seal(stage,ctx.identity,{"checkpoints_replayed":ctx.config["cv_folds"]*len(ctx.config["seeds"]),"parent_files_matched":count,
                                   "cv_manifest_sha256":file_sha256(args.run_dir/"cv/run.json")})
    else:read_sealed(output,ctx.identity)
    result=choose(summary,ctx.config,True)
    output=args.run_dir/"decision"
    if not output.exists():
        with new_output(output) as stage:
            seal(stage,ctx.identity,{**result,"cv_manifest_sha256":file_sha256(args.run_dir/"cv/run.json"),
                                   "verification_manifest_sha256":file_sha256(args.run_dir/"verification/run.json")})
    return result


def run_verify(args):
    ctx=context019(args)
    directory=args.run_dir/"verification_attempts";directory.mkdir(parents=True,exist_ok=True)
    attempts=sorted(directory.glob("*.json"));number=int(attempts[-1].stem)+1 if attempts else 1
    path=directory/f"{number:08}.json"
    with path.open("x") as handle:json.dump({"status":"STARTED"},handle)
    try:
        result=_verify(args,ctx)
        json_output({"status":"PASS","cv_manifest_sha256":file_sha256(args.run_dir/"cv/run.json")},path)
        verified_decision(args)
    except BaseException as error:
        json_output({"status":"FAILED","error":str(error)},path);raise
    print(f"EXP-019 verified competition decision {result['status']}",flush=True)


def verified_decision(args):
    ctx=context019(args);result=read_sealed(args.run_dir/"decision",ctx.identity)
    attempts=sorted((args.run_dir/"verification_attempts").glob("*.json"))
    if not attempts:raise StageBlocked("Verify must complete before confirmation")
    latest=json.loads(attempts[-1].read_text());cv=read_sealed(args.run_dir/"cv",ctx.identity)
    if latest["status"]!="PASS" or latest.get("cv_manifest_sha256")!=file_sha256(args.run_dir/"cv/run.json"):
        raise StageBlocked("Latest Verify failed/interrupted; rerun Verify")
    read_sealed(args.run_dir/"verification",ctx.identity)
    for name,digest in cv["seed_manifests"].items():
        fold,name=name.split("/");path=args.run_dir/"work"/f"fold_{fold}"/name
        read_sealed(path,ctx.identity)
        if file_sha256(path/"run.json")!=digest:raise ValueError("Verified C2 checkpoint changed")
    for stage in ("cv","verification"):
        if result[stage+"_manifest_sha256"]!=file_sha256(args.run_dir/stage/"run.json"):
            raise ValueError("Decision evidence changed")
    summary=pd.read_csv(args.run_dir/"cv/summary.csv",index_col="model",float_precision="round_trip")
    if any(result[k]!=v for k,v in choose(summary,ctx.config,True).items()):raise ValueError("Frozen promotion decision differs")
    preserve_parent(args,ctx.identity)
    return result
