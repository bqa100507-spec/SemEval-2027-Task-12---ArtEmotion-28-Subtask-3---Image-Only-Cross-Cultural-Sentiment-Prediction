"""Future inner-only matched pilot. No outer prediction/scoring implementation."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from .contrast_qlora import ContrastTrainer,objective,two_pass_backward,rng_state,restore_rng
from .vlm_exploratory import reset
from .vlm_supervised import adapter_parameters,frozen_digest
from .vlm_supervised_training import model_state,load_state
from .vlm_semantics import atomic_json,file_hash,fingerprint


def save_checkpoint(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(".pending-"+path.name);torch.save(payload,temp);temp.replace(path)
    atomic_json(path.with_suffix(".sha256.json"),{"sha256":file_hash(path)})


def load_checkpoint(path,identity):
    seal=path.with_suffix(".sha256.json")
    if not seal.exists() or json.loads(seal.read_text())["sha256"]!=file_hash(path):raise ValueError("EXP-027 checkpoint checksum mismatch")
    result=torch.load(path,map_location="cpu",weights_only=True)
    if result["identity"]!=identity:raise ValueError("EXP-027 source/config/input/partition/variant identity mismatch")
    return result


def train_variant(path,identity,backend,cache,config,variant,languages,fit,val,initial,check,*,device=None):
    expected={"run":identity,"variant":variant,"fit":sorted(fit.painting.unique()),"validation":sorted(val.painting.unique())}
    objective_config={"mode":variant["mode"],"lambda":variant["lambda"],"epsilon":config.get("pearson_epsilon",.01)}
    def trainer():return ContrastTrainer(backend,cache,config,languages,"L0",objective_config=objective_config,device=device)
    def replay(payload):
        reset(backend,initial);current=trainer();load_state(backend,current.head,payload["state"])
        q=np.round(current.predict(val[["painting","image","language"]]),6)
        if q.tolist()!=payload["validation_prediction_six_decimals"]:raise ValueError("EXP-027 independent selected-checkpoint replay failed")
        return payload
    if path.exists():
        payload=load_checkpoint(path,expected)
        if not payload.get("complete"):raise ValueError("Incomplete selected model")
        return replay(payload)
    reset(backend,initial);current=trainer();progress=path.with_name(path.stem+"_epoch.pt")
    epoch=0;curves=[];best=float("inf");best_state=None;selected=0
    devices=[current.device.index or 0] if current.device.type=="cuda" else []
    if progress.exists():
        old=load_checkpoint(progress,expected)
        load_state(backend,current.head,old["state"]);current.optimizer.load_state_dict(old["optimizer"])
        current.scaler.load_state_dict(old["scaler"]);current.steps=old["steps"];current.clipped=old["clipped"]
        epoch=old["epoch"];curves=old["curves"];best=old["best_loss"];best_state=old["best_state"];selected=old["selected_epoch"]
        restore_rng(old["rng"])
    for i in range(epoch,config["max_epochs"]):
        check();metrics=current.epoch(fit,i,check);check();tick=time.perf_counter();loss=current.validate(val)
        row={**metrics,"epoch":i+1,"validation_BCE_nats":loss,"validation_seconds":time.perf_counter()-tick}
        curves.append(row);print(json.dumps({"variant":variant["name"],**row}),flush=True)
        state=model_state(backend,current.head)
        if loss<best:best=loss;selected=i+1;best_state=state
        save_checkpoint(progress,{"identity":expected,"state":state,"epoch":i+1,"curves":curves,"best_state":best_state,
                                   "best_loss":best,"selected_epoch":selected,"optimizer":current.optimizer.state_dict(),
                                   "scaler":current.scaler.state_dict(),"steps":current.steps,"clipped":current.clipped,"rng":rng_state(devices)})
    if best_state is None:raise ValueError("No finite selected pilot state")
    update=sum(float((best_state["adapters"][name]-value).float().square().sum()) for name,value in initial.items())**.5
    if update<=0 or not np.isfinite(update) or any(r["LoRA_gradient_norm_min"]<=0 for r in curves):raise ValueError("Pilot LoRA updates/gradients are zero or nonfinite")
    reset(backend,initial);selected_trainer=trainer();load_state(backend,selected_trainer.head,best_state)
    q=np.round(selected_trainer.predict(val[["painting","image","language"]]),6).tolist()
    payload={"identity":expected,"complete":True,"state":best_state,"selected_epoch":selected,"curves":curves,
             "validation_BCE_nats":best,"validation_prediction_six_decimals":q,"LoRA_update_norm":update,"refit":False}
    # Replay before publishing a successful selected checkpoint.
    replay(payload);save_checkpoint(path,payload)
    return replay(load_checkpoint(path,expected))


def gradient_preflight(backend,cache,config,languages,fit,protocol,check):
    """One complete 3–8-language inner-training painting; forward/gradient check, no update."""
    groups=[frame for _,frame in fit.groupby("painting",sort=True) if 3<=len(frame)<=8 and (frame.k/frame.n).nunique()>1]
    if not groups:raise ValueError("No small complete painting group for bounded gradient preflight")
    batch=groups[0].sort_values("language");variant=protocol["variants"][-1]
    t=ContrastTrainer(backend,cache,config,languages,"L0",objective_config={"mode":"bce_pearson","lambda":variant["lambda"],"epsilon":protocol["pearson_epsilon"]})
    rng=rng_state([0]);k=torch.tensor(batch.k.to_numpy(),device="cuda");n=torch.tensor(batch.n.to_numpy(),device="cuda")
    def forward(ids):
        rows=batch.iloc[ids]
        with torch.autocast("cuda",dtype=torch.float16):
            return t.head(t.hidden(rows,train=True),t.head.indices(rows.language,t.device)).float()
    try:
        check();backend.model.model.train();z=forward(list(range(len(batch))))
        loss,components=objective(z,k,n,batch.painting.tolist(),"bce_pearson",variant["lambda"],protocol["pearson_epsilon"])
        component_gradients=[]
        for part in (components["bce"],variant["lambda"]*components["pearson"]):
            values=torch.autograd.grad(t.scaler.scale(part),t.parameters,retain_graph=True,allow_unused=True)
            component_gradients.append([torch.zeros_like(p,dtype=torch.float32) if g is None else g.detach().float()/t.scaler.get_scale() for p,g in zip(t.parameters,values)])
        split=len(list(t.head.parameters()));ratios={}
        for name,part in (("head",slice(None,split)),("LoRA",slice(split,None))):
            a=torch.cat([g.ravel() for g in component_gradients[0][part]])
            b=torch.cat([g.ravel() for g in component_gradients[1][part]])
            ratios[name]={"BCE_gradient_norm":float(a.norm()),"scaled_aux_gradient_norm":float(b.norm()),
                          "auxiliary_to_BCE_ratio":float(b.norm()/a.norm().clamp_min(1e-30))}
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or a.norm()==0 or b.norm()==0:
                raise ValueError("GPU preflight has zero/nonfinite BCE or auxiliary head/LoRA gradients")
        t.scaler.scale(loss).backward()
        ref=[None if p.grad is None else p.grad.detach().float().clone()/t.scaler.get_scale() for p in t.parameters]
        t.optimizer.zero_grad(set_to_none=True);restore_rng(rng)
        two_pass_backward(forward,list(range(len(batch))),k,n,batch.painting.tolist(),2,variant["lambda"],
                          scaler=t.scaler,epsilon=protocol["pearson_epsilon"],modules=(backend.model,t.head),devices=[0],check=check)
        differences=[]
        for p,expected in zip(t.parameters,ref):
            if expected is None:
                if p.grad is not None and p.grad.count_nonzero():raise ValueError("Preflight unexpected gradient")
                continue
            actual=p.grad.float()/t.scaler.get_scale()
            if not torch.allclose(actual,expected,atol=5e-4,rtol=2e-3):raise ValueError("AMP parameter-gradient preflight differs from unsplit reference")
            differences.append(float((actual-expected).abs().max()))
        return {"status":"PASS","painting_languages":len(batch),"microbatch":2,"max_abs_parameter_gradient_error":max(differences),
                "boundary":"AMP tolerance audit; not bitwise floating-point equivalence. Two-pass same-chunk logits must match exactly.","optimizer_updates":0,
                "component_gradient_diagnostics":ratios}
    finally:t.optimizer.zero_grad(set_to_none=True);restore_rng(rng)


def run_pilot(args,config,protocol,backend,cache,fit,val,parts,report,check,identity):
    previous=args.run_dir/"pilot_report.json"
    if (args.run_dir/"exploratory_identity.json").exists() or (previous.exists() and json.loads(previous.read_text()).get("experiment")!="EXP-027"):
        raise ValueError("Use a new EXP-027 run directory")
    identity={**identity,"protocol":protocol,"partitions":{k:sorted(v.painting.unique()) for k,v in parts.items()}}
    id_file=args.run_dir/"identity.json"
    if id_file.exists() and json.loads(id_file.read_text())!=identity:raise ValueError("Run identity differs; preserve artifacts and use new RunDir")
    atomic_json(id_file,identity)
    (args.run_dir/"PILOT_COMPLETE.json").unlink(missing_ok=True)
    backend.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    initial={name:p.detach().cpu().clone() for name,p in backend.model.named_parameters() if p.requires_grad}
    languages=sorted(fit.language.unique());before=frozen_digest(backend.model)
    tick=time.perf_counter()
    for image in sorted(set(fit.image)|set(val.image)):check();cache.vision(image,backend)
    report["vision_cache_seconds"]=time.perf_counter()-tick
    from PIL import Image
    row=fit.iloc[0]
    with Image.open(cache.inputs/"artwork/images"/row.image) as image:report["processor_parity"]=backend.parity(image.convert("RGB"),cache.vision(row.image,backend))
    report["gradient_preflight"]=gradient_preflight(backend,cache,config,languages,fit,protocol,check)
    results={}
    for variant in protocol["variants"]:
        check();payload=train_variant(args.run_dir/"models"/(variant["name"]+".pt"),identity,backend,cache,config,variant,languages,fit,val,initial,check)
        results[variant["name"]]={k:payload[k] for k in ("selected_epoch","validation_BCE_nats","LoRA_update_norm","curves")}
    if frozen_digest(backend.model)!=before:raise ValueError("Frozen Qwen pretrained/vision weights changed")
    report.update(status="GPU_PILOT_COMPLETE",variants=results,outer_evaluation_performed=False,full_cv_performed=False,
                  pilot_train_paintings=int(fit.painting.nunique()),pilot_val_paintings=int(val.painting.nunique()),
                  scientific_conclusion="Inner-only feasibility/comparison; no outer Contrast or promotion conclusion")
    atomic_json(args.run_dir/"PILOT_COMPLETE.json",{"identity":fingerprint(identity),"status":"GPU_PILOT_COMPLETE","no_outer_evaluation":True})
