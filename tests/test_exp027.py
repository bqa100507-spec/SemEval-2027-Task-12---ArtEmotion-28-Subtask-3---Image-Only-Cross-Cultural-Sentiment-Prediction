"""EXP-027 isolation, native BCE parity and safe pilot persistence on tiny CPU fixtures."""
from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from st3.contrast_qlora import ContrastTrainer
from st3.exp027_data import filter_votes,partitions,load_config
from st3.exp027_pilot import train_variant
from st3.vlm_supervised import attach_adapters
from st3.vlm_supervised_training import Trainer
from test_vlm_supervised import fixture,config

ROOT=Path(__file__).resolve().parents[1]


def cpu_backend(tmp_path,monkeypatch):
    frame,backend,cache=fixture(tmp_path);c=config();attach_adapters(backend.model,c)
    frozen_hidden=backend.hidden
    backend.hidden=lambda visions,languages,train=False:backend.model(frozen_hidden(visions,languages,train))
    cache.prepare(frame,backend,lambda:None)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(torch,"autocast",lambda *a,**k:nullcontext())
    scaler=torch.amp.GradScaler
    monkeypatch.setattr(torch.amp,"GradScaler",lambda *a,**k:scaler("cuda",enabled=False))
    return frame,backend,cache,c


def test_native_bce_epoch_parameter_optimizer_and_vote_parity(tmp_path,monkeypatch):
    frame,backend,cache,c=cpu_backend(tmp_path,monkeypatch)
    initial=copy.deepcopy(backend.model.state_dict())
    a=Trainer(backend,cache,c,["A","B"],"L0",device="cpu")
    before={k:v.clone() for k,v in a.head.state_dict().items()}
    metrics=a.epoch(frame,0,lambda:None)
    model={k:v.clone() for k,v in backend.model.state_dict().items()};head={k:v.clone() for k,v in a.head.state_dict().items()}
    backend.model.load_state_dict(initial)
    b=ContrastTrainer(backend,cache,c,["A","B"],"L0",device="cpu",objective_config={"mode":"bce_only","lambda":0.,"epsilon":.01})
    assert all(torch.equal(v,b.head.state_dict()[k]) for k,v in before.items())
    result=b.epoch(frame,0,lambda:None)
    assert result["binary_BCE_nats"]==metrics["binary_BCE_nats"] and a.steps==b.steps==4
    for key,value in model.items():torch.testing.assert_close(value,backend.model.state_dict()[key],rtol=0,atol=0)
    for key,value in head.items():torch.testing.assert_close(value,b.head.state_dict()[key],rtol=0,atol=0)
    assert a.optimizer.state_dict()["param_groups"]==b.optimizer.state_dict()["param_groups"]
    for key,state in a.optimizer.state_dict()["state"].items():
        for field,value in state.items():torch.testing.assert_close(value,b.optimizer.state_dict()["state"][key][field],rtol=0,atol=0)


def test_label_isolation_before_target_indexing_and_invalid_counts():
    class Poison(dict):
        def __getitem__(self,key):
            if key in ("k","n"):pytest.fail("Forbidden outer/locked target was accessed")
            return super().__getitem__(key)
    row=dict(painting="allowed",language="L",image="i",image_sha256="s",fold=3,split="train",k=2,n=5)
    result=filter_votes([Poison(painting="held"),row,Poison(painting="sealed")],{"allowed"})
    assert result.painting.tolist()==["allowed"]
    with pytest.raises(ValueError):filter_votes([{**row,"n":0}],{"allowed"})


def test_original_painting_disjoint_metadata_and_repeatable_subset():
    c,p=load_config(ROOT)
    meta=pd.DataFrame([{"painting":f"p{i}","image":f"p{i}.jpg","image_sha256":f"hash{i}","fold":i%5+1} for i in range(1335)])
    a=partitions(meta,c,p);b=partitions(meta,c,p)
    assert all(x.equals(y) for x,y in zip(a,b))
    inner,val,pi,pv,held=a
    assert set(held.fold)=={2} and len(pi)==192 and len(pv)==48
    assert not set(inner.painting)&set(held.painting) and not set(val.painting)&set(inner.painting)
    assert set(pi.painting)<=set(inner.painting) and set(pv.painting)<=set(val.painting)


def test_selected_checkpoint_replay_resume_and_corruption(tmp_path,monkeypatch):
    frame,backend,cache,c=cpu_backend(tmp_path,monkeypatch)
    initial={name:p.detach().clone() for name,p in backend.model.named_parameters() if p.requires_grad}
    fit=frame[frame.painting.ne("p7")];val=frame[frame.painting.eq("p7")]
    # Synthetic two-language data has no Pearson-eligible group, but exercises the same real parameter/optimizer path.
    variant={"name":"P","mode":"bce_pearson","lambda":.00196}
    args=({"source":"fixture"},backend,cache,c,variant,["A","B"],fit,val,initial,lambda:None)
    result=train_variant(tmp_path/"model.pt",*args,device="cpu")
    assert result["complete"] and result["LoRA_update_norm"]>0
    monkeypatch.setattr(ContrastTrainer,"epoch",lambda *a:pytest.fail("Completed model must not retrain"))
    assert train_variant(tmp_path/"model.pt",*args,device="cpu")["selected_epoch"]==result["selected_epoch"]
    with pytest.raises(ValueError,match="identity"):train_variant(tmp_path/"model.pt",{"source":"different"},*args[1:],device="cpu")
    (tmp_path/"model.pt").write_bytes((tmp_path/"model.pt").read_bytes()+b"corrupt")
    with pytest.raises(ValueError,match="checksum"):train_variant(tmp_path/"model.pt",*args,device="cpu")


def test_cli_no_default_authorization_no_outer_endpoint_and_bounded_budget(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("exp027_runner",ROOT/"scripts/run_exp027.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    base=["run_exp027.py","--inputs",str(tmp_path),"--cache",str(tmp_path/"cache"),"--run-dir",str(tmp_path/"run")]
    monkeypatch.setattr(sys,"argv",base)
    with pytest.raises(SystemExit):m.main()
    monkeypatch.setattr(sys,"argv",base+["--authorize-gpu","--stage","outer-fold"])
    with pytest.raises(SystemExit):m.main()
    def timeout(*a,**k):assert k["timeout"]==7200;raise m.subprocess.TimeoutExpired("CPU_MOCK",7200)
    monkeypatch.setattr(m.subprocess,"run",timeout);monkeypatch.setattr(sys,"argv",base+["--authorize-gpu","--max-seconds","99999"])
    assert m.main()==2
    assert json.loads((tmp_path/"run/pilot_report.json").read_text())["status"]=="STOPPED_BUDGET"


def test_interrupted_completed_epoch_resumes_exact_parameters(tmp_path,monkeypatch):
    frame,backend,cache,c=cpu_backend(tmp_path,monkeypatch)
    initial={name:p.detach().clone() for name,p in backend.model.named_parameters() if p.requires_grad}
    fit=frame[frame.painting.ne("p7")];val=frame[frame.painting.eq("p7")]
    args=({"source":"fixture"},backend,cache,c,{"name":"BCE","mode":"bce_only","lambda":0.},["A","B"],fit,val,initial,lambda:None)
    full=train_variant(tmp_path/"full.pt",*args,device="cpu")
    original=ContrastTrainer.epoch
    def interrupt(self,rows,epoch,check):
        if epoch==1:raise TimeoutError("CPU mock interruption")
        return original(self,rows,epoch,check)
    monkeypatch.setattr(ContrastTrainer,"epoch",interrupt)
    with pytest.raises(TimeoutError):train_variant(tmp_path/"resume.pt",*args,device="cpu")
    calls=[]
    def resumed(self,rows,epoch,check):calls.append(epoch);return original(self,rows,epoch,check)
    monkeypatch.setattr(ContrastTrainer,"epoch",resumed)
    actual=train_variant(tmp_path/"resume.pt",*args,device="cpu")
    assert calls==[1] and actual["selected_epoch"]==full["selected_epoch"]
    for section in ("head","adapters"):
        for key,value in full["state"][section].items():torch.testing.assert_close(actual["state"][section][key],value,rtol=0,atol=0)


def test_cuda_oom_stops_once_and_persists_failure_without_retry(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("exp027_oom",ROOT/"scripts/run_exp027.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,"read_inputs",lambda *a:(None,None,None))
    calls=[]
    def oom(*a):calls.append(1);raise torch.cuda.OutOfMemoryError("CPU mock, no GPU")
    monkeypatch.setattr(m,"build_backend",oom);monkeypatch.setattr(torch.cuda,"is_initialized",lambda:False)
    args=SimpleNamespace(run_dir=tmp_path,inputs=tmp_path,max_seconds=300)
    assert m.worker(args)==2 and calls==[1]
    result=json.loads((tmp_path/"pilot_report.json").read_text())
    assert result["status"]=="CUDA_OOM" and result["retry_attempted"] is False
