import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from st3.vlm_supervised import (DecoderLoRA,SentimentHead,attach_adapters,adapter_parameters,
                                frozen_digest,make_head,vote_loss)
from st3.vlm_supervised_training import FeatureCache,Trainer,windows

ROOT=Path(__file__).resolve().parents[1]


def config():
    c=json.loads((ROOT/"configs/exp026.json").read_text(encoding="utf-8"))
    c.update(hidden_size=8,decoder_layers=2,head_width=4,microbatch_pairs=2,effective_painting_batch=2,max_epochs=2,patience=1)
    c["lora"]["expected_modules"]=4;c["pilot"]["max_epochs"]=2
    return c


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList()
        for _ in range(2):
            layer=nn.Module();layer.self_attn=nn.Module()
            layer.self_attn.q_proj=nn.Linear(8,8);layer.self_attn.v_proj=nn.Linear(8,8)
            self.model.layers.append(layer)
        self.visual=nn.Linear(8,8)

    def forward(self,x):
        for layer in self.model.layers:x=torch.tanh(layer.self_attn.q_proj(x)+layer.self_attn.v_proj(x))
        return x


def test_zero_adapters_real_gradients_updates_and_base_freeze():
    torch.manual_seed(1);model=TinyModel();x=torch.randn(3,8)
    original=model(x).detach().clone();names=attach_adapters(model,config())
    assert len(names)==4 and not any(name.startswith("visual") for name in names)
    torch.testing.assert_close(model(x),original,rtol=0,atol=0)
    before=frozen_digest(model);initial=[p.detach().clone() for p in adapter_parameters(model)]
    optimizer=torch.optim.AdamW(adapter_parameters(model),lr=.01)
    for step in range(3):
        optimizer.zero_grad();output=model(x);assert output.requires_grad
        output.square().mean().backward()
        if step==0:assert all(float(m.A.grad.abs().sum())==0 for m in model.modules() if isinstance(m,DecoderLoRA))
        assert sum(float(p.grad.abs().sum()) for p in adapter_parameters(model))>0
        optimizer.step()
    assert sum(float((p-old).abs().sum()) for p,old in zip(adapter_parameters(model),initial))>0
    assert frozen_digest(model)==before
    assert all(p.grad is None for p in model.visual.parameters())


def test_adapter_matching_and_rng_cannot_change_head_initialization():
    c=config();torch.manual_seed(18);model=TinyModel();state=torch.random.get_rng_state().clone()
    attach_adapters(model,c)
    torch.testing.assert_close(torch.random.get_rng_state(),state)
    a,b=make_head(["A","B"],c),make_head(["A","B"],c)
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    bad=config();bad["lora"]["expected_modules"]=5
    with pytest.raises(ValueError):attach_adapters(TinyModel(),bad)


def test_count_weighted_bce_and_accumulation_match_full_objective():
    logits=torch.tensor([-.7,.4,1.2],requires_grad=True);k=torch.tensor([1.,4.,9.]);n=torch.tensor([5.,7.,10.])
    expected=-(k*torch.nn.functional.logsigmoid(logits)+(n-k)*torch.nn.functional.logsigmoid(-logits)).sum()/n.sum()
    actual=vote_loss(logits,k,n);torch.testing.assert_close(actual,expected)
    full_grad=torch.autograd.grad(actual,logits,retain_graph=True)[0]
    split=sum(vote_loss(logits[i:i+1],k[i:i+1],n[i:i+1],denominator=n.sum()) for i in range(3))
    torch.testing.assert_close(torch.autograd.grad(split,logits)[0],full_grad)
    with pytest.raises(ValueError):vote_loss(logits,k,n*0)
    with pytest.raises(ValueError):vote_loss(logits,k*float("nan"),n)
    with pytest.raises(ValueError):vote_loss(logits,k,n,denominator=0)


def test_unknown_language_uses_shared_head_only():
    head=make_head(["A"],config());h=torch.randn(2,8)
    with torch.no_grad():head.language_bias.weight.fill_(9);head.language_visual.weight.fill_(3)
    q=head(h,head.indices(["unseen","unseen"],"cpu"))
    torch.testing.assert_close(q,head.shared(head.visual(h)).squeeze(-1))


class FakeBackend:
    def __init__(self,c):self.config=c;self.model=TinyModel().requires_grad_(False);self.calls=0
    def vision(self,image):
        self.calls+=1;return {"tokens":torch.ones(2,8)*np.asarray(image).mean()/255,"grid":torch.tensor([1,2,4])}
    def hidden(self,visions,languages,train=False):
        return torch.stack([v["tokens"].mean(0)+(0 if language is None else len(language)/10) for v,language in zip(visions,languages)])


def fixture(tmp_path):
    from PIL import Image
    folder=tmp_path/"artwork/images";folder.mkdir(parents=True)
    from st3.vlm_semantics import file_hash
    rows=[]
    for i in range(8):
        name=f"p{i}.jpg";Image.new("RGB",(4,4),(i*20+1,30,40)).save(folder/name)
        for language,k in [("A",1),("B",4)]:rows.append(dict(painting=f"p{i}",image=name,image_sha256=file_hash(folder/name),language=language,k=k,n=5))
    frame=pd.DataFrame(rows);backend=FakeBackend(config());cache=FeatureCache(tmp_path/"cache",tmp_path,config(),frame)
    return frame,backend,cache


def test_image_level_cache_integrity_reuse_and_grouped_rows(tmp_path):
    frame,backend,cache=fixture(tmp_path)
    cache.prepare(frame,backend,lambda:None)
    assert backend.calls==8  # One vision forward per image, despite two language rows.
    cache.prepare(frame,backend,lambda:None);assert backend.calls==8
    assert not torch.equal(cache.hidden("p0.jpg",None,None),cache.hidden("p0.jpg","A",None))
    all_rows=pd.concat(list(windows(frame,config(),2027)))
    assert len(all_rows)==len(frame) and not all_rows[["painting","language"]].duplicated().any()
    shuffled=frame.sample(frac=1.,random_state=4)
    assert [list(v.painting) for v in windows(frame,config(),2027)]==[list(v.painting) for v in windows(shuffled,config(),2027)]
    with pytest.raises(ValueError):cache.path("sealed_confirmation.jpg")
    path=cache.path("p0.jpg",None)
    with np.load(path,allow_pickle=False) as values:saved={k:values[k] for k in values.files}
    saved["neutral"]=saved["neutral"]+1;np.savez(path,**saved)
    with pytest.raises(ValueError,match="Corrupt"):cache.read(path)


def runner():
    spec=importlib.util.spec_from_file_location("exp026_cpu_runner",ROOT/"scripts/run_exp026.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_cpu_frozen_training_selection_refit_and_exact_completed_resume(tmp_path,monkeypatch):
    frame,backend,cache=fixture(tmp_path);cache.prepare(frame,backend,lambda:None)
    c=config();module=runner();training=frame[~frame.painting.isin(["p6","p7"])]
    inner=training[training.painting.ne("p5")];validation=training[training.painting.eq("p5")]
    path=tmp_path/"model.pt";identity={"source":"fixture","config":c}
    first=module.train_complete(path,identity,backend,cache,c,"F0",["A","B"],inner,validation,training,{},lambda:None)
    assert first["complete"] and first["selected_epoch"] in (1,2)
    assert any(row["phase"]=="refit" for row in first["curves"])
    monkeypatch.setattr(Trainer,"epoch",lambda *args:pytest.fail("Completed model must not retrain"))
    second=module.train_complete(path,identity,backend,cache,c,"F0",["A","B"],inner,validation,training,{},lambda:None)
    assert all(torch.equal(first["state"]["head"][k],second["state"]["head"][k]) for k in first["state"]["head"])
    with pytest.raises(ValueError):module.train_complete(path,{"source":"changed"},backend,cache,c,"F0",["A","B"],inner,validation,training,{},lambda:None)
    path.write_bytes(path.read_bytes()+b"corrupt")
    with pytest.raises(ValueError,match="checksum"):
        module.train_complete(path,identity,backend,cache,c,"F0",["A","B"],inner,validation,training,{},lambda:None)


def test_completed_epoch_resume_reproduces_frozen_model(tmp_path,monkeypatch):
    frame,backend,cache=fixture(tmp_path);cache.prepare(frame,backend,lambda:None)
    c=config();module=runner();inner=frame[~frame.painting.isin(["p6","p7"])];validation=frame[frame.painting.eq("p6")]
    identity={"source":"fixture","config":c};args=(identity,backend,cache,c,"F0",["A","B"],inner,validation,inner,{},lambda:None)
    uninterrupted=module.train_complete(tmp_path/"reference.pt",*args)
    original=Trainer.epoch;calls=[]
    def interrupt(self,frame,epoch,check):
        if epoch==1 and not calls:
            calls.append(epoch);raise TimeoutError("CPU fixture interruption")
        return original(self,frame,epoch,check)
    monkeypatch.setattr(Trainer,"epoch",interrupt)
    with pytest.raises(TimeoutError):module.train_complete(tmp_path/"resumed.pt",*args)
    assert (tmp_path/"resumed_selection.pt").exists() and not (tmp_path/"resumed.pt").exists()
    monkeypatch.setattr(Trainer,"epoch",original)
    resumed=module.train_complete(tmp_path/"resumed.pt",*args)
    assert uninterrupted["selected_epoch"]==resumed["selected_epoch"]
    for key,value in uninterrupted["state"]["head"].items():
        torch.testing.assert_close(value,resumed["state"]["head"][key],rtol=0,atol=0)


def test_oom_stops_without_retry_and_preserves_report(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import st3.vlm_supervised_training as training
    module=runner();monkeypatch.setattr(training,"load_training",lambda *args:(None,None))
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:False)
    calls=[]
    def oom(*args):calls.append(1);raise torch.cuda.OutOfMemoryError("CPU MOCK OOM")
    monkeypatch.setattr(module,"build_backend",oom)
    args=SimpleNamespace(run_dir=tmp_path/"run",stage="technical",inputs=tmp_path,max_seconds=300)
    assert module.worker(args,config())==2 and len(calls)==1
    report=json.loads((args.run_dir/"technical_report.json").read_text())
    assert report["status"]=="CUDA_OOM" and report["retry_attempted"] is False


def test_technical_parent_hard_caps_300_seconds_without_running_gpu(tmp_path,monkeypatch):
    module=runner()
    monkeypatch.setattr(sys,"argv",["run_exp026.py","--inputs",str(tmp_path),"--cache",str(tmp_path/"cache"),"--run-dir",str(tmp_path/"run"),"--authorize-gpu","--max-seconds","999"])
    def timeout(*args,**kwargs):
        assert kwargs["timeout"]==300
        raise module.subprocess.TimeoutExpired("CPU_MOCK_ONLY",300)
    monkeypatch.setattr(module.subprocess,"run",timeout)
    assert module.main()==2
    result=json.loads((tmp_path/"run/technical_report.json").read_text())
    assert result["status"]=="STOPPED_TIME_LIMIT" and result["hard_seconds"]==300


def test_gpu_authorization_and_no_automatic_full_cv(tmp_path,monkeypatch):
    module=runner()
    monkeypatch.setattr(sys,"argv",["run_exp026.py","--inputs",str(tmp_path),"--cache",str(tmp_path),"--run-dir",str(tmp_path)])
    with pytest.raises(SystemExit):module.main()
    monkeypatch.setattr(sys,"argv",["run_exp026.py","--stage","full-cv","--inputs",str(tmp_path),"--cache",str(tmp_path),"--run-dir",str(tmp_path),"--authorize-gpu"])
    with pytest.raises(SystemExit):module.main()
