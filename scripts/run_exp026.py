"""Direct-state Qwen supervision: bounded technical check, pilot, or one P2 fold."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from st3.vlm_semantics import atomic_json,file_hash,fingerprint


def check_assets(model_path,config):
    for name,expected in config["vlm"]["assets_sha256"].items():
        if file_hash(model_path/name)!=expected:raise ValueError("Pinned checkpoint differs: "+name)
    import hashlib
    for name,expected in config["vlm"]["assets_git_sha1"].items():
        raw=(model_path/name).read_bytes()
        if hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()!=expected:
            raise ValueError("Pinned processor/config differs: "+name)


def build_backend(args,config):
    import platform,torch
    if platform.python_version()!=config["runtime"]["python"]:raise RuntimeError("Pinned Python3.12.10 required")
    for name,expected in config["runtime"].items():
        if name!="python" and importlib.metadata.version(name)!=expected:raise RuntimeError("Pinned dependency differs: "+name)
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; use the explicitly authorized Colab L4 notebook")
    if args.stage!="technical" and "L4" not in torch.cuda.get_device_name():
        raise RuntimeError("Pilot/one-fold profile requires explicitly authorized Colab L4; local4060 authorization covers technical checks only")
    if args.stage=="technical":torch.cuda.set_per_process_memory_fraction(config["technical_vram_fraction"])
    torch.manual_seed(config["seed"]);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model_path=args.model_path
    if model_path is None:
        from huggingface_hub import snapshot_download
        model_path=Path(snapshot_download(config["vlm"]["model"],revision=config["vlm"]["revision"],local_dir=args.model_cache,
                            allow_patterns=["*.json","*.safetensors","*.txt","LICENSE","README.md"]))
    check_assets(model_path,config)
    from st3.vlm_semantics import load_vlm
    from st3.vlm_supervised import QwenStates,attach_adapters
    processor,model=load_vlm(config,model_path)
    if model.config.hidden_size!=2048 or model.config.num_hidden_layers!=36:raise ValueError("Unexpected pinned Qwen architecture")
    names=attach_adapters(model,config)
    return QwenStates(model,processor,config),names


def technical(backend,cache,train,config,report,check):
    import numpy as np
    import torch
    from PIL import Image
    from st3.caption_data import inner_partition
    from st3.vlm_supervised import adapters_disabled,adapter_parameters,frozen_digest,make_head,vote_loss
    from st3.vlm_supervised_training import Trainer,model_state
    fitting=train[train.fold.ne(config["fold"])];inner,_=inner_partition(fitting,config,config["fold"])
    painting=sorted(inner.painting.unique())[0]
    pair=inner[inner.painting.eq(painting)].sort_values("language").iloc[:1]
    image_name=pair.iloc[0].image
    report.update(artworks_used=1,painting=painting,target_language=pair.iloc[0].language,scope="technical only; no pilot/held-out score")
    check();vision=cache.vision(image_name,backend)
    with Image.open(cache.inputs/"artwork/images"/image_name) as image:
        report["official_cached_state_parity"]=backend.parity(image.convert("RGB"),vision)
    check();backend.model.eval()
    with torch.no_grad():
        with adapters_disabled(backend.model):original=backend.hidden([vision],[pair.iloc[0].language])
        zero=backend.hidden([vision],[pair.iloc[0].language])
    report["zero_LoRA_max_abs_error"]=float((original-zero).abs().max())
    if not torch.equal(original,zero):raise ValueError("Zero-initialized adapters changed frozen states")
    direct=zero.float().cpu();saved=cache.hidden(image_name,pair.iloc[0].language,backend)[None]
    cache.hidden(image_name,None,backend)
    if not torch.equal(direct,saved):raise ValueError("Frozen cached/direct decoder feature mismatch")
    report["cached_hidden_exact"]=True
    report["frozen_weights_before"]=frozen_digest(backend.model)
    initial=[p.detach().clone() for p in adapter_parameters(backend.model)]
    backend.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    trainer=Trainer(backend,cache,config,sorted(fitting.language.unique()),"L0")
    initial_head={k:v.detach().clone() for k,v in trainer.head.state_dict().items()}
    diagnostics=[]
    for step in range(config["technical_optimizer_steps"]):
        check();backend.model.model.train();trainer.head.train();trainer.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16):
            hidden=trainer.hidden(pair,train=True)
            if not hidden.requires_grad:raise ValueError("LoRA graph disconnected")
            logits=trainer.head(hidden,trainer.head.indices(pair.language,trainer.device))
            loss=vote_loss(logits,torch.tensor(pair.k.to_numpy(),device="cuda"),torch.tensor(pair.n.to_numpy(),device="cuda"))
        trainer.scaler.scale(loss).backward();trainer.scaler.unscale_(trainer.optimizer)
        gradient=lambda parameters:float(torch.sqrt(sum((p.grad.float().square().sum() for p in parameters if p.grad is not None),torch.zeros((),device="cuda"))))
        head_gradient=gradient(trainer.head.parameters());adapter_gradient=gradient(adapter_parameters(backend.model))
        norm=torch.nn.utils.clip_grad_norm_(trainer.parameters,config["optimizer"]["gradient_clip"])
        if not torch.isfinite(norm):raise ValueError("Nonfinite technical gradient")
        trainer.scaler.step(trainer.optimizer);trainer.scaler.update()
        diagnostics.append({"step":step+1,"BCE_nats":float(loss.detach()),"head_gradient_norm":head_gradient,
                            "LoRA_gradient_norm":adapter_gradient,"preclip_gradient_norm":float(norm),"AMP_scale":trainer.scaler.get_scale()})
    report["gradient_steps"]=diagnostics
    updates=[float((p.detach()-before).float().norm()) for p,before in zip(adapter_parameters(backend.model),initial)]
    report["LoRA_update_norm"]=float(np.linalg.norm(updates))
    report["head_update_norm"]=float(torch.sqrt(sum((v-initial_head[k]).float().square().sum() for k,v in trainer.head.state_dict().items())))
    report["frozen_weights_after"]=frozen_digest(backend.model)
    if report["frozen_weights_before"]!=report["frozen_weights_after"]:raise ValueError("Pretrained/vision weights changed")
    if not report["LoRA_update_norm"] or not all(r["LoRA_gradient_norm"]>0 and r["head_gradient_norm"]>0 for r in diagnostics):
        raise ValueError("LoRA/head did not receive real finite BCE updates")
    backend.model.model.eval()
    with torch.no_grad():changed=backend.hidden([vision],[pair.iloc[0].language])
    report["adapted_hidden_shift_norm"]=float((changed.float()-original.float()).norm())
    report["trainable_adapter_parameters"]=sum(p.numel() for p in adapter_parameters(backend.model))
    report["trainable_head_parameters"]=sum(p.numel() for p in trainer.head.parameters())
    report["vision_has_gradients"]=any(p.grad is not None for p in backend.model.visual.parameters())
    if report["vision_has_gradients"]:raise ValueError("Frozen vision tower received gradients")
    # Small local diagnostic checkpoint only; never reused as a scientific warm start.
    temporary=Path(report["run_dir"])/"technical_state.pending.pt"
    torch.save(model_state(backend,trainer.head),temporary);temporary.replace(temporary.with_name("technical_state.pt"))
    report["status"]="TECHNICAL_PASS"


def worker(args,config):
    import torch
    from st3.vlm_supervised_training import load_training,FeatureCache
    started=time.monotonic();args.run_dir.mkdir(parents=True,exist_ok=True)
    args.started_monotonic=started
    report={"stage":args.stage,"status":"RUNNING","run_dir":str(args.run_dir),"production":"E50+L unchanged",
            "scientific_fold_completed":False,"GPU":torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    def check():
        if time.monotonic()-started>args.max_seconds-15:raise TimeoutError("Allocation exhausted; completed cache/checkpoint boundaries persist")
    try:
        train,reference=load_training(args.inputs,config)
        if args.stage=="exploratory-fold":
            protocol=json.loads((ROOT/"configs/exp026_exploratory.json").read_text(encoding="utf-8"))
            import hashlib
            if hashlib.sha256((ROOT/"configs/exp026.json").read_bytes().replace(b"\r\n",b"\n")).hexdigest()!=protocol["base_config_sha256"]:
                raise ValueError("Original EXP-026 config differs; exploratory stage preserves the base settings")
        backend,names=build_backend(args,config)
        report["LoRA_target_modules"]=names
        report["model_load_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        report["model_load_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        cache=FeatureCache(args.cache,args.inputs,config,train)
        if args.stage=="technical":technical(backend,cache,train,config,report,check)
        elif args.stage=="exploratory-fold":
            from st3.vlm_exploratory import run
            run(args,config,protocol,backend,cache,train,reference,report,check,source_identity(config,args,backend))
        else:run_research(args,config,backend,cache,train,reference,report,check)
    except torch.cuda.OutOfMemoryError as error:report.update(status="CUDA_OOM",error=str(error),retry_attempted=False)
    except Exception as error:report.update(status="STOPPED_TIME_LIMIT" if isinstance(error,TimeoutError) else "FAILED_TECHNICAL",error=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        if torch.cuda.is_initialized():
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          total_VRAM_bytes=torch.cuda.get_device_properties(0).total_memory)
        atomic_json(args.run_dir/(args.stage+"_report.json"),report)
        print(json.dumps(report,indent=2),flush=True)
    return 0 if report["status"] in ("TECHNICAL_PASS","PILOT_PASS","ONE_FOLD_COMPLETE","EXPLORATORY_FOLD_COMPLETE") else 2


def save_checkpoint(path,payload):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(".pending.pt");torch.save(payload,pending);pending.replace(path)
    if payload.get("complete"):
        atomic_json(path.with_suffix(".sha256.json"),{"sha256":file_hash(path)})


def source_identity(config,args,backend):
    return {"config":config,"inputs":config["input_sha256"],"GPU":__import__("torch").cuda.get_device_name(),
            "source":{name:file_hash(ROOT/name) for name in ("scripts/run_exp026.py","src/st3/vlm_supervised.py","src/st3/vlm_supervised_training.py",
                      "src/st3/vlm_semantics.py","src/st3/caption_data.py","src/st3/ablation.py","src/st3/metrics.py","src/st3/data.py",
                      "requirements-exp026.lock.txt")}}


def reset_adapters(backend,initial):
    import torch
    parameters=dict(backend.model.named_parameters())
    with torch.no_grad():
        for name,value in initial.items():parameters[name].copy_(value.to(parameters[name].device))


def train_complete(path,identity,backend,cache,config,condition,languages,inner,validation,fitting,initial,check,*,pilot=False):
    import torch
    from st3.vlm_supervised_training import Trainer,model_state,load_state
    expected={"identity":identity,"condition":condition,"pilot":pilot,
              "paintings":{name:sorted(frame.painting.unique()) for name,frame in (("inner",inner),("validation",validation),("refit",fitting))}}
    if path.exists():
        seal=path.with_suffix(".sha256.json")
        if not seal.exists() or json.loads(seal.read_text()).get("sha256")!=file_hash(path):
            raise ValueError("Completed checkpoint checksum missing/different; preserve artifacts and inspect")
        payload=torch.load(path,map_location="cpu",weights_only=True)
        if payload.get("identity")!=expected or not payload.get("complete"):raise ValueError("Incompatible/incomplete model checkpoint; preserve it and use a new run directory")
        return payload
    reset_adapters(backend,initial)
    trainer=Trainer(backend,cache,config,languages,condition)
    curves=[];best=float("inf");best_epoch=0;stale=0
    selection_path=path.with_name(path.stem+"_selection.pt")
    first=0
    if selection_path.exists():
        resumed=torch.load(selection_path,map_location="cpu",weights_only=True)
        if resumed["identity"]!=expected:raise ValueError("Selection checkpoint identity differs")
        load_state(backend,trainer.head,resumed["state"]);trainer.optimizer.load_state_dict(resumed["optimizer"])
        trainer.scaler.load_state_dict(resumed["scaler"])
        first=resumed["epoch"];curves=resumed["curves"];best=resumed["best"];best_epoch=resumed["best_epoch"];stale=resumed["stale"]
    epochs=config["pilot"]["max_epochs"] if pilot else config["max_epochs"]
    for epoch in range(first,epochs):
        if not pilot and stale>=config["patience"]:break
        check();metrics=trainer.epoch(inner,epoch,check)
        tick=time.perf_counter();validation_loss=trainer.validate(validation)
        row={"phase":"pilot" if pilot else "selection","epoch":epoch+1,**metrics,
             "validation_BCE_nats":validation_loss,"validation_seconds":time.perf_counter()-tick}
        curves.append(row);print(json.dumps({"condition":condition,**row}),flush=True)
        if validation_loss<best:best=validation_loss;best_epoch=epoch+1;stale=0
        else:stale+=1
        save_checkpoint(selection_path,{"identity":expected,"epoch":epoch+1,"curves":curves,"best":best,"best_epoch":best_epoch,
                        "stale":stale,"state":model_state(backend,trainer.head),"optimizer":trainer.optimizer.state_dict(),"scaler":trainer.scaler.state_dict()})
    if pilot:
        state=model_state(backend,trainer.head)
    else:
        # Fresh head + original zero-B adapters; no pilot/selection warm start.
        reset_adapters(backend,initial)
        trainer=Trainer(backend,cache,config,languages,condition)
        refit_path=path.with_name(path.stem+"_refit.pt");first=0
        if refit_path.exists():
            resumed=torch.load(refit_path,map_location="cpu",weights_only=True)
            if resumed["identity"]!=expected or resumed["selected_epoch"]!=best_epoch:raise ValueError("Refit checkpoint identity/epoch differs")
            load_state(backend,trainer.head,resumed["state"]);trainer.optimizer.load_state_dict(resumed["optimizer"])
            trainer.scaler.load_state_dict(resumed["scaler"]);first=resumed["epoch"]
            curves=resumed["curves"]
        for epoch in range(first,best_epoch):
            check();metrics=trainer.epoch(fitting,epoch,check)
            curves.append({"phase":"refit","epoch":epoch+1,**metrics})
            save_checkpoint(refit_path,{"identity":expected,"epoch":epoch+1,"selected_epoch":best_epoch,"curves":curves,
                       "state":model_state(backend,trainer.head),"optimizer":trainer.optimizer.state_dict(),"scaler":trainer.scaler.state_dict()})
        state=model_state(backend,trainer.head)
    # Frozen controls contain no learned adapters, even when sharing a runtime.
    if condition!="L0":state["adapters"]={}
    payload={"identity":expected,"condition":condition,"complete":True,"state":state,"curves":curves,"selected_epoch":best_epoch}
    save_checkpoint(path,payload)
    return payload


def run_research(args,config,backend,cache,train,reference,report,check):
    import numpy as np
    import pandas as pd
    import torch
    from st3.caption_data import inner_partition,require_isolation
    from st3.metrics import score
    from st3.vlm_supervised import adapter_parameters,frozen_digest
    from st3.vlm_supervised_training import Trainer,load_state
    from st3.ablation import shuffle_images
    backend.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    initial={name:p.detach().cpu().clone() for name,p in backend.model.named_parameters() if p.requires_grad}
    identity=source_identity(config,args,backend)
    report["run_identity"]=fingerprint(identity)
    fitting,held=train[train.fold.ne(config["fold"])],train[train.fold.eq(config["fold"])]
    inner,validation=inner_partition(fitting,config,config["fold"]);require_isolation(inner,validation,held)
    languages=sorted(fitting.language.unique())
    def choose(frame,count,seed):
        names=sorted(frame.painting.unique(),key=lambda name:fingerprint([seed,name]))[:count]
        return frame[frame.painting.isin(names)]
    pilot_inner=choose(inner,config["pilot"]["training_paintings"],config["seed"])
    pilot_val=choose(validation,config["pilot"]["validation_paintings"],config["seed"]+1)
    atomic_json(args.run_dir/"pilot_partitions.json",{"inner_training_paintings":sorted(pilot_inner.painting.unique()),
                "inner_validation_paintings":sorted(pilot_val.painting.unique()),"outer_held_paintings":sorted(held.painting.unique())})
    from PIL import Image
    image=pilot_inner.iloc[0].image;vision=cache.vision(image,backend)
    with Image.open(cache.inputs/"artwork/images"/image) as opened:report["official_cached_state_parity"]=backend.parity(opened.convert("RGB"),vision)
    check();tick=time.perf_counter();cache.prepare(pd.concat([pilot_inner,pilot_val]),backend,check)
    report["pilot_cache_seconds"]=time.perf_counter()-tick
    previous=args.run_dir/"pilot_report.json"
    if previous.exists():
        old=json.loads(previous.read_text())
        if old.get("run_identity")==report["run_identity"]:report["pilot_cache_seconds"]=max(report["pilot_cache_seconds"],old.get("pilot_cache_seconds",0))
    frozen_before=frozen_digest(backend.model)
    pilot_results={}
    for condition in ("F0_image","F0","L0"):
        check();payload=train_complete(args.run_dir/"pilot"/(condition+".pt"),identity,backend,cache,config,condition,languages,
                                      pilot_inner,pilot_val,pilot_inner,initial,check,pilot=True)
        pilot_results[condition]={"curves":payload["curves"],"selected_epoch":payload["selected_epoch"]}
        if condition=="L0":
            reset_adapters(backend,initial)
            diagnostic=Trainer(backend,cache,config,languages,"L0");load_state(backend,diagnostic.head,payload["state"])
            update=sum(float((dict(backend.model.named_parameters())[name].detach().cpu()-value).square().sum()) for name,value in initial.items())**.5
            if update==0 or any(row["LoRA_gradient_norm_min"]<=0 for row in payload["curves"]):raise ValueError("Pilot LoRA graph/update is zero")
            report["pilot_LoRA_update_norm"]=update
    if frozen_digest(backend.model)!=frozen_before:raise ValueError("Frozen pretrained/vision weights changed during pilot")
    report.update(status="PILOT_PASS",pilot_results=pilot_results,pilot_training_paintings=pilot_inner.painting.nunique(),
                  pilot_validation_paintings=pilot_val.painting.nunique(),pilot_held_out_performance=False)
    lora_curves=pilot_results["L0"]["curves"]
    slowest_rate=min(row["pairs_per_second"] for row in lora_curves)
    # Full max-selection+refit allowance, measured on real complete pilot epochs.
    estimate=1.25*(config["max_epochs"]*(len(inner)+len(fitting))/slowest_rate
                     +config["max_epochs"]*(len(inner)+len(fitting))*sum(1/min(row["pairs_per_second"] for row in pilot_results[name]["curves"]) for name in ("F0_image","F0"))
                     +config["max_epochs"]*sum(max(row["validation_seconds"] for row in values["curves"]) for values in pilot_results.values())*len(validation)/len(pilot_val)
                     +report["pilot_cache_seconds"]*train.painting.nunique()/(pilot_inner.painting.nunique()+pilot_val.painting.nunique())+120)
    report["one_fold_conservative_estimate_seconds"]=estimate
    report["full_CV_recommendation"]="Do not launch five-fold QLoRA; inspect measured one-fold results first."
    atomic_json(args.run_dir/"pilot_report.json",report)
    if args.stage=="pilot":return
    if estimate>args.max_seconds-(time.monotonic()-args.started_monotonic)-15:
        # Outer supervisor enforces the actual hard deadline too.
        raise TimeoutError("Measured one-fold estimate exceeds allocation; pilot is preserved. Increase only an explicitly authorized allocation, not scientific settings.")
    reset_adapters(backend,initial)
    tick=time.perf_counter();cache.prepare(train,backend,check);report["full_frozen_cache_seconds"]=time.perf_counter()-tick
    predictions=[]
    for condition in ("F0_image","F0","L0"):
        check();payload=train_complete(args.run_dir/f"fold_{config['fold']}"/(condition+".pt"),identity,backend,cache,config,condition,languages,
                                      inner,validation,fitting,initial,check)
        trainer=Trainer(backend,cache,config,languages,condition);load_state(backend,trainer.head,payload["state"])
        q=trainer.predict(held)
        # Replay the published state before reporting scientific predictions.
        replay=torch.load(args.run_dir/f"fold_{config['fold']}"/(condition+".pt"),map_location="cpu",weights_only=True)
        reset_adapters(backend,initial)
        replay_trainer=Trainer(backend,cache,config,languages,condition)
        load_state(backend,replay_trainer.head,replay["state"])
        if not np.array_equal(np.round(q,6),np.round(replay_trainer.predict(held),6)):
            raise ValueError("Published checkpoint prediction replay failed")
        trainer=replay_trainer
        predictions.append(held[["painting","language"]].assign(p_positive=np.round(q,6),model=condition))
        # Lightweight image-use check on the same fixed held-out subset.
        subset=choose(held,config["image_shuffle_paintings"],config["seed"]+2)
        shuffled,mapping=shuffle_images(subset,config["seed"])
        if condition!="L0":cache.prepare(shuffled,backend,check,neutral=condition=="F0_image")
        correct=trainer.predict(subset);wrong=trainer.predict(shuffled)
        atomic_json(args.run_dir/(condition+"_image_shuffle.json"),{"paintings":subset.painting.nunique(),"pairs":len(subset),
                     "mean_abs_probability_change":float(np.mean(np.abs(correct-wrong))),"original_variance":float(np.var(correct)),
                     "shuffled_variance":float(np.var(wrong)),"diagnostic_only":True})
        mapping.to_csv(args.run_dir/(condition+"_image_shuffle_mapping.csv"),index=False)
    baseline=held[["painting","language"]].merge(reference[["painting","language","p_positive"]],on=["painting","language"],validate="one_to_one")
    predictions.append(baseline.assign(model="E50+L"))
    all_predictions=pd.concat(predictions,ignore_index=True)
    all_predictions.to_csv(args.run_dir/"predictions.csv",index=False,float_format="%.6f",lineterminator="\n")
    metrics=[]
    for name,pred in all_predictions.groupby("model"):
        result=score(held,pred[["painting","language","p_positive"]])
        metrics.append({"model":name,"fold":config["fold"],"contrast":result["cultural_contrast"],"macro_log_loss":result["log_loss"]})
    pd.DataFrame(metrics).to_csv(args.run_dir/"one_fold_metrics.csv",index=False)
    report.update(status="ONE_FOLD_COMPLETE",scientific_fold_completed=True,fold=config["fold"],one_fold_metrics=metrics,
                  held_paintings=held.painting.nunique(),held_pairs=len(held),full_CV_automatically_launched=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["technical","pilot","one-fold","exploratory-fold"],default="technical")
    parser.add_argument("--inputs",type=Path,required=True);parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True);parser.add_argument("--model-path",type=Path)
    parser.add_argument("--model-cache",type=Path,default=Path("cache/exp026_model"))
    parser.add_argument("--max-seconds",type=float,default=300.)
    parser.add_argument("--authorize-gpu",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.authorize_gpu:parser.error("Explicit --authorize-gpu is required; no GPU stage is authorized by default")
    import math
    if not math.isfinite(args.max_seconds) or args.max_seconds<=15:parser.error("Choose a finite allocation greater than15 seconds")
    if args.stage=="technical":args.max_seconds=min(args.max_seconds,300.)
    if args.stage=="exploratory-fold":args.max_seconds=min(args.max_seconds,7200.)
    config=json.loads((ROOT/"configs/exp026.json").read_text(encoding="utf-8"))
    if args.worker:return worker(args,config)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-u",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"]
    log=args.run_dir/(args.stage+"_"+str(time.time_ns())+".log")
    try:
        with log.open("w") as handle:result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,timeout=args.max_seconds)
        print(log.read_text(errors="replace")[-16000:]);return result.returncode
    except subprocess.TimeoutExpired:
        atomic_json(args.run_dir/(args.stage+"_report.json"),{"status":"STOPPED_TIME_LIMIT","hard_seconds":args.max_seconds,
                     "stage":args.stage,"complete_scientific_result":False,"log":str(log),"production":"E50+L unchanged"})
        print("Hard allocation reached; GPU worker stopped. No retry. "+str(log));return 2


if __name__=="__main__":raise SystemExit(main())
