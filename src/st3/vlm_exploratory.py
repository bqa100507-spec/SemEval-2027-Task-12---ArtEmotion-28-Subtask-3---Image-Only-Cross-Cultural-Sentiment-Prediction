"""Two-epoch, no-refit EXP-026 screening. Never enters the scientific fold runner."""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .caption_data import inner_partition, require_isolation
from .data import KEYS
from .metrics import score
from .submission import validate_predictions
from .vlm_semantics import atomic_json, file_hash, fingerprint
from .vlm_supervised import frozen_digest
from .vlm_supervised_training import Trainer, model_state, load_state


def estimate_cost(inner_pairs, validation_pairs, held_pairs, paintings, protocol):
    """Pilot-based projection only; cold and fully cached endpoints, not new benchmarks."""
    p=protocol["pilot_timing_reference"]
    validation_rate=p["validation_pairs"]/p["L0_validation_seconds"]
    training=protocol["max_epochs"]*inner_pairs/p["L0_training_pairs_per_second"]
    validation=protocol["max_epochs"]*validation_pairs/validation_rate
    prediction=held_pairs/validation_rate
    # Use the measured frozen training rate even for its cheaper forward-only work.
    frozen=2*(protocol["max_epochs"]*(inner_pairs+validation_pairs)+held_pairs)/p["frozen_training_pairs_per_second"]
    cold_cache=paintings*p["cache_seconds"]/p["cache_paintings"]
    warm=p["compute_margin"]*(training+validation+prediction+frozen+p["checkpoint_allowance_seconds"])
    warm+=p["setup_allowance_seconds"]+p["export_allowance_seconds"]
    return {"projection_not_benchmark":True,"L0_training_seconds":training,
            "L0_validation_seconds":validation,"L0_held_prediction_seconds":prediction,
            "fully_cached_total_seconds":warm,"cold_cache_total_seconds":warm+p["compute_margin"]*cold_cache,
            "caveat":"Pilot scaling estimate; downloads, Drive I/O and full-fold sequence lengths may differ. Hard allocation still applies."}


def save(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".pending.pt")
    torch.save(payload,temporary);temporary.replace(path)
    if payload.get("complete"):atomic_json(path.with_suffix(".sha256.json"),{"sha256":file_hash(path)})


def read_complete(path, expected):
    seal=path.with_suffix(".sha256.json")
    if not seal.is_file() or json.loads(seal.read_text())["sha256"]!=file_hash(path):
        raise ValueError("Exploratory model checksum missing/different; preserve artifacts and inspect")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("identity")!=expected or not payload.get("complete"):
        raise ValueError("Exploratory checkpoint identity differs; never reuse pilot/scientific fits")
    return {**payload,"checkpoint_sha256":file_hash(path)}


def reset(backend, initial):
    current=dict(backend.model.named_parameters())
    with torch.no_grad():
        for name,value in initial.items():current[name].copy_(value.to(current[name].device))


def train_selected(path, identity, backend, cache, config, protocol, condition, languages, inner, validation, initial, check):
    """Persist current epoch for resume and the true best state for held-out inference."""
    require_isolation(inner,validation)
    expected={"run":identity,"stage":"exploratory-fold","condition":condition,
              "training_paintings":sorted(inner.painting.unique()),"validation_paintings":sorted(validation.painting.unique())}
    if path.exists():return read_complete(path,expected)
    reset(backend,initial)
    trainer=Trainer(backend,cache,config,languages,condition)
    progress=path.with_name(path.stem+"_progress.pt")
    first=0;curves=[];best=float("inf");best_state=None;selected=0
    if progress.exists():
        saved=torch.load(progress,map_location="cpu",weights_only=True)
        if saved["identity"]!=expected:raise ValueError("Exploratory epoch identity differs")
        load_state(backend,trainer.head,saved["state"])
        trainer.optimizer.load_state_dict(saved["optimizer"]);trainer.scaler.load_state_dict(saved["scaler"])
        first=saved["epoch"];curves=saved["curves"];best=saved["best_loss"]
        best_state=saved["best_state"];selected=saved["selected_epoch"]
        trainer.steps,trainer.clipped=saved["steps"],saved["clipped"]
    for epoch in range(first,protocol["max_epochs"]):
        check();metrics=trainer.epoch(inner,epoch,check)
        check();tick=time.perf_counter();loss=trainer.validate(validation)
        if not np.isfinite(loss):raise ValueError("Nonfinite exploratory validation loss")
        row={"phase":"exploratory_selection","epoch":epoch+1,**metrics,
             "validation_BCE_nats":loss,"validation_seconds":time.perf_counter()-tick}
        curves.append(row);print(json.dumps({"condition":condition,**row}),flush=True)
        current=model_state(backend,trainer.head)
        if condition!="L0":current["adapters"]={}
        if loss<best:best=loss;best_state=current;selected=epoch+1
        save(progress,{"identity":expected,"epoch":epoch+1,"state":current,
                       "optimizer":trainer.optimizer.state_dict(),"scaler":trainer.scaler.state_dict(),
                       "steps":trainer.steps,"clipped":trainer.clipped,"curves":curves,
                       "best_loss":best,"best_state":best_state,"selected_epoch":selected})
    if best_state is None or not 1<=selected<=protocol["max_epochs"]:
        raise ValueError("No valid selected exploratory state")
    if condition=="L0":
        if any(r["LoRA_gradient_norm_min"]<=0 for r in curves):raise ValueError("Zero LoRA gradient in exploratory fit")
        update=sum(float((best_state["adapters"][name]-value).float().square().sum()) for name,value in initial.items())**.5
        if not np.isfinite(update) or update==0:raise ValueError("Selected LoRA adapters did not update")
    else:update=0.
    payload={"identity":expected,"complete":True,"condition":condition,"state":best_state,
             "selected_epoch":selected,"validation_BCE_nats":best,"curves":curves,"refit":False,
             "LoRA_update_norm":update}
    save(path,payload)
    return read_complete(path,expected)


def held_predictions(path, payload, identity, backend, cache, config, condition, languages, queries, initial, check):
    """Predict each selected model once; exact compatible persisted predictions resume."""
    expected={"identity":identity,"model":condition,"selected_epoch":payload["selected_epoch"],
              "checkpoint_sha256":payload["checkpoint_sha256"],"keys":queries[KEYS].to_dict("records")}
    receipt=path.with_suffix(".json")
    if path.exists() or receipt.exists():
        if not path.exists() or not receipt.exists():raise ValueError("Incomplete exploratory predictions; preserve and inspect")
        saved=json.loads(receipt.read_text())
        if saved.get("identity")!=expected or saved.get("sha256")!=file_hash(path):
            raise ValueError("Exploratory prediction identity/checksum differs")
        predictions=pd.read_csv(path,float_precision="round_trip")
    else:
        check();reset(backend,initial)
        trainer=Trainer(backend,cache,config,languages,condition);load_state(backend,trainer.head,payload["state"])
        # No labels/votes reach inference. No second held prediction for replay/shuffle.
        predictions=queries[KEYS].assign(p_positive=np.round(trainer.predict(queries),6))
        validate_predictions(predictions,queries)
        temporary=path.with_suffix(".pending.csv")
        predictions.to_csv(temporary,index=False,float_format="%.6f",lineterminator="\n");temporary.replace(path)
        atomic_json(receipt,{"identity":expected,"sha256":file_hash(path)})
    validate_predictions(predictions,queries)
    return predictions.reset_index(drop=True)


def run(args, config, protocol, backend, cache, train, reference, report, check, identity):
    """No pilot, no fresh refit and no call to the original scientific runner."""
    if protocol["fold"]!=1 or protocol["max_epochs"]!=2 or protocol["refit"] is not False:
        raise ValueError("Exploratory stage is fixed to fold1, at most2 epochs and no refit")
    if any((args.run_dir/name).exists() for name in ("pilot_report.json","one-fold_report.json","technical_report.json","fold_1")):
        raise ValueError("Use a separate exploratory run directory; historical scientific/pilot artifacts are protected")
    identity={**identity,"stage":"exploratory-fold","protocol":protocol,
              "exploratory_source":file_hash(Path(__file__))}
    run_identity=fingerprint(identity)
    identity_path=args.run_dir/"exploratory_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError("Exploratory run source/config/input/runtime identity differs; choose a fresh run directory")
    atomic_json(identity_path,identity)
    # Revoke stale success before a new attempt; final success is only republished after verification/scoring.
    completed=args.run_dir/"exploratory_complete.json"
    completed.unlink(missing_ok=True)
    fitting=train[train.fold.ne(1)];held=train[train.fold.eq(1)]
    inner,validation=inner_partition(fitting,config,1);require_isolation(inner,validation,held)
    languages=sorted(inner.language.unique())
    queries=train[["painting","image","image_sha256","language"]]
    held_queries=held[queries.columns]
    atomic_json(args.run_dir/"exploratory_partitions.json",{name:sorted(frame.painting.unique())
                for name,frame in (("inner",inner),("validation",validation),("held",held))})
    report.update(run_identity=run_identity,scope="exploratory screening only",refit=False,
                  comparison_warning=protocol["comparison"],maximum_epochs=2,
                  fitting_counts={name:{"paintings":int(frame.painting.nunique()),"pairs":len(frame)}
                                  for name,frame in (("VLM_inner",inner),("validation",validation),("held",held),("E50L_outer",fitting))},
                  estimated_cost=estimate_cost(len(inner),len(validation),len(held),train.painting.nunique(),protocol))
    backend.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    initial={name:p.detach().cpu().clone() for name,p in backend.model.named_parameters() if p.requires_grad}
    frozen_before=frozen_digest(backend.model)
    from PIL import Image
    row=inner.iloc[0];vision=cache.vision(row.image,backend)
    with Image.open(cache.inputs/"artwork/images"/row.image) as image:
        report["official_cached_state_parity"]=backend.parity(image.convert("RGB"),vision)
    check();tick=time.perf_counter();cache.prepare(queries,backend,check)
    report["frozen_cache_preparation_seconds"]=time.perf_counter()-tick
    predictions={};models={}
    directory=args.run_dir/"exploratory_fold_1";directory.mkdir(parents=True,exist_ok=True)
    # Finish all selection before consuming any held-out prediction/score.
    for condition in ("F0_image","F0","L0"):
        check();models[condition]=train_selected(directory/(condition+".pt"),identity,backend,cache,config,protocol,
                                                 condition,languages,inner,validation,initial,check)
    for condition,payload in models.items():
        predictions[condition]=held_predictions(directory/(condition+"_predictions.csv"),payload,run_identity,
                                                backend,cache,config,condition,languages,held_queries,initial,check)
    if frozen_digest(backend.model)!=frozen_before:raise ValueError("Frozen pretrained/vision weights changed")
    predictions["E50+L"]=held_queries[KEYS].merge(reference[KEYS+["p_positive"]],on=KEYS,validate="one_to_one")
    rows=[]
    for condition,prediction in predictions.items():
        result=score(held,prediction)
        rows.append({"model":condition,"fold":1,"contrast":result["cultural_contrast"],"macro_log_loss":result["log_loss"],
                     "selected_epoch":models[condition]["selected_epoch"] if condition in models else None,
                     "fitting_paintings":int((fitting if condition=="E50+L" else inner).painting.nunique()),
                     "fitting_pairs":len(fitting if condition=="E50+L" else inner),"refitted":condition=="E50+L"})
    table=pd.DataFrame(rows);table.to_csv(args.run_dir/"exploratory_metrics.csv",index=False)
    report.update(status="EXPLORATORY_FOLD_COMPLETE",scientific_fold_completed=False,
                  exploratory_fold_completed=True,fold=1,metrics=rows,
                  selected_epochs={name:p["selected_epoch"] for name,p in models.items()},
                  automatic_promotion=False,full_CV_automatically_launched=False)
    atomic_json(completed,{"run_identity":run_identity,"metrics_sha256":file_hash(args.run_dir/"exploratory_metrics.csv"),
                          "status":"EXPLORATORY_FOLD_COMPLETE","screening_only":True})
    print(protocol["comparison"],flush=True);print(table.to_string(index=False),flush=True)
