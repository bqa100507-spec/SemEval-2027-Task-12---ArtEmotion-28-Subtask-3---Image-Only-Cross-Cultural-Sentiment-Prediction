"""Tiny CPU and mocked screening checks; no real VLM, cloud or GPU training."""

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from st3 import vlm_exploratory as exp
from st3.vlm_supervised import attach_adapters
from st3.vlm_supervised_training import Trainer
from test_vlm_supervised import config,fixture,runner

ROOT=Path(__file__).resolve().parents[1]


def protocol():return json.loads((ROOT/"configs/exp026_exploratory.json").read_text())


def test_best_epoch_is_restored_without_refit_or_held_selection(tmp_path,monkeypatch):
    frame,backend,cache=fixture(tmp_path);cache.prepare(frame,backend,lambda:None)
    inner=frame[frame.painting.ne("p7")];validation=frame[frame.painting.eq("p7")]
    losses=iter([.3,.5]);monkeypatch.setattr(Trainer,"validate",lambda *args:next(losses))
    path=tmp_path/"F0.pt"
    args=({"source":"fixture"},backend,cache,config(),protocol(),"F0",["A","B"],inner,validation,{},lambda:None)
    payload=exp.train_selected(path,*args)
    progress=torch.load(tmp_path/"F0_progress.pt",weights_only=True)
    assert payload["selected_epoch"]==1 and payload["validation_BCE_nats"]==.3 and not payload["refit"]
    assert len(payload["curves"])==2 and all(row["phase"]=="exploratory_selection" for row in payload["curves"])
    assert all(torch.equal(v,progress["best_state"]["head"][k]) for k,v in payload["state"]["head"].items())
    assert any(not torch.equal(v,progress["state"]["head"][k]) for k,v in payload["state"]["head"].items())
    monkeypatch.setattr(Trainer,"epoch",lambda *args:pytest.fail("Selected completion must not retrain"))
    assert exp.train_selected(path,*args)["selected_epoch"]==1
    with pytest.raises(ValueError,match="identity"):
        exp.train_selected(path,{"source":"changed"},*args[1:])


def test_epoch_resume_preserves_best_state_and_does_not_restart(tmp_path,monkeypatch):
    frame,backend,cache=fixture(tmp_path);cache.prepare(frame,backend,lambda:None)
    inner=frame[frame.painting.ne("p7")];validation=frame[frame.painting.eq("p7")]
    args=({"source":"fixture"},backend,cache,config(),protocol(),"F0",["A","B"],inner,validation,{},lambda:None)
    full=exp.train_selected(tmp_path/"full.pt",*args)
    original=Trainer.epoch
    def interrupt(self,frame,epoch,check):
        if epoch==1:raise TimeoutError("CPU fixture interrupted")
        return original(self,frame,epoch,check)
    monkeypatch.setattr(Trainer,"epoch",interrupt)
    with pytest.raises(TimeoutError):exp.train_selected(tmp_path/"resume.pt",*args)
    assert (tmp_path/"resume_progress.pt").exists() and not (tmp_path/"resume.pt").exists()
    calls=[]
    def resumed(self,frame,epoch,check):calls.append(epoch);return original(self,frame,epoch,check)
    monkeypatch.setattr(Trainer,"epoch",resumed)
    actual=exp.train_selected(tmp_path/"resume.pt",*args)
    assert calls==[1] and actual["selected_epoch"]==full["selected_epoch"]
    for key,value in full["state"]["head"].items():
        torch.testing.assert_close(value,actual["state"]["head"][key],rtol=0,atol=0)


def test_held_predictions_once_label_free_and_resume_validates_checkpoint(tmp_path,monkeypatch):
    frame,backend,cache=fixture(tmp_path);cache.prepare(frame,backend,lambda:None)
    selected=exp.train_selected(tmp_path/"model.pt",{"source":"fixture"},backend,cache,config(),protocol(),"F0",["A","B"],
                                frame[frame.painting.ne("p7")],frame[frame.painting.eq("p7")],{},lambda:None)
    queries=frame[frame.painting.eq("p7")][["painting","image","image_sha256","language"]]
    calls=[]
    def predict(self,rows):
        assert "k" not in rows and "n" not in rows
        calls.append(1);return np.array([.2,.8])
    monkeypatch.setattr(Trainer,"predict",predict)
    path=tmp_path/"held.csv"
    args=(selected,"source",backend,cache,config(),"F0",["A","B"],queries,{},lambda:None)
    first=exp.held_predictions(path,*args);again=exp.held_predictions(path,*args)
    assert calls==[1] and first.equals(again)
    changed={**selected,"checkpoint_sha256":"different"}
    with pytest.raises(ValueError,match="checksum"):
        exp.held_predictions(path,changed,*args[1:])


def test_full_screen_path_uses_inner_only_all_conditions_and_same_fold_reference(tmp_path,monkeypatch):
    frame,backend,cache=fixture(tmp_path);c=config()
    frame["fold"]=np.where(frame.painting.isin(["p0","p1"]),1,2)
    attach_adapters(backend.model,c)
    initial={name:p.detach().clone() for name,p in backend.model.named_parameters() if p.requires_grad}
    frozen_hidden=backend.hidden
    backend.hidden=lambda visions,languages,train=False:backend.model(frozen_hidden(visions,languages,train))
    backend.model.model.gradient_checkpointing_enable=lambda **kwargs:None
    backend.parity=lambda *args:{"status":"CPU_FIXTURE_PASS"}
    class CPUTrainer(Trainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs,device="cpu")
            self.scaler=torch.amp.GradScaler("cuda",enabled=False)
        def predict(self,rows):
            if set(rows.painting)<=set(["p0","p1"]):
                assert "k" not in rows and "n" not in rows
                held_calls.append(self.condition)
            return super().predict(rows)
    monkeypatch.setattr(exp,"Trainer",CPUTrainer)
    monkeypatch.setattr(torch,"autocast",lambda *args,**kwargs:nullcontext())
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    scaler=torch.amp.GradScaler
    monkeypatch.setattr(torch.amp,"GradScaler",lambda *a,**k:scaler("cuda",enabled=False))
    held_calls=[];report={};run_dir=tmp_path/"exploratory";run_dir.mkdir()
    args=SimpleNamespace(run_dir=run_dir)
    reference=frame[["painting","language"]].assign(p_positive=.6)
    exp.run(args,c,protocol(),backend,cache,frame,reference,report,lambda:None,{"source":"CPU_FIXTURE"})
    assert held_calls==["F0_image","F0","L0"]
    assert report["status"]=="EXPLORATORY_FOLD_COMPLETE" and not report["scientific_fold_completed"]
    assert not report["refit"] and not report["automatic_promotion"]
    assert {r["model"] for r in report["metrics"]}=={"F0_image","F0","L0","E50+L"}
    assert report["fitting_counts"]["VLM_inner"]["paintings"]<report["fitting_counts"]["E50L_outer"]["paintings"]
    partitions=json.loads((run_dir/"exploratory_partitions.json").read_text())
    for part in ["inner","validation"]:assert not set(partitions[part])&set(partitions["held"])
    # Completed run verifies artifacts, rescoring the same persisted predictions without fitting or predicting again.
    monkeypatch.setattr(CPUTrainer,"epoch",lambda *args:pytest.fail("No retraining on resume"))
    monkeypatch.setattr(CPUTrainer,"predict",lambda *args:pytest.fail("No held re-prediction on resume"))
    # A new process would load original zero-B adapters; simulate that initial state here.
    exp.reset(backend,initial)
    exp.run(args,c,protocol(),backend,cache,frame,reference,report,lambda:None,{"source":"CPU_FIXTURE"})


def test_cli_exploratory_has_hard_two_hour_cap_without_gpu(tmp_path,monkeypatch):
    module=runner()
    monkeypatch.setattr(sys,"argv",["run_exp026.py","--stage","exploratory-fold","--inputs",str(tmp_path),"--cache",str(tmp_path/"cache"),
                                  "--run-dir",str(tmp_path/"run"),"--max-seconds","999999","--authorize-gpu"])
    def stop(*args,**kwargs):
        assert kwargs["timeout"]==7200
        raise module.subprocess.TimeoutExpired("CPU_MOCK",7200)
    monkeypatch.setattr(module.subprocess,"run",stop)
    assert module.main()==2
    report=json.loads((tmp_path/"run/exploratory-fold_report.json").read_text())
    assert report["hard_seconds"]==7200


def test_worker_dispatch_never_enters_scientific_or_pilot_runner(tmp_path,monkeypatch):
    import st3.vlm_supervised_training as training
    module=runner();calls=[]
    monkeypatch.setattr(training,"load_training",lambda *args:(None,None))
    monkeypatch.setattr(training,"FeatureCache",lambda *args:None)
    monkeypatch.setattr(module,"build_backend",lambda *args:(None,[]))
    monkeypatch.setattr(module,"run_research",lambda *args:pytest.fail("Expensive scientific/pilot runner must not be invoked"))
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:False)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda:"CPU_MOCK")
    for name in ("max_memory_allocated","max_memory_reserved","reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda,name,lambda:0)
    def screening(args,c,p,backend,cache,train,reference,report,check,identity):
        calls.append(p["stage"]);report["status"]="EXPLORATORY_FOLD_COMPLETE"
    monkeypatch.setattr(exp,"run",screening)
    args=SimpleNamespace(inputs=tmp_path,cache=tmp_path/"cache",run_dir=tmp_path/"run",stage="exploratory-fold",max_seconds=6000)
    assert module.worker(args,config())==0 and calls==["exploratory-fold"]


def test_projection_preserved_base_settings_and_feature_cache_implementation():
    import hashlib
    p=protocol()
    original=(ROOT/"configs/exp026.json").read_bytes().replace(b"\r\n",b"\n")
    assert hashlib.sha256(original).hexdigest()==p["base_config_sha256"]
    # The feature identity/extraction files are unchanged, allowing exact cache reuse.
    original_hashes={"src/st3/vlm_supervised.py":"551e2dd8471cb276bd8a6603c5c28b0a6ec6a47c971c9a52e423d3f0d8faaf24",
                     "src/st3/vlm_supervised_training.py":"4732de0f025543689eb5a9eab6140a8fe4475b06e4e444337261657ec802ca3e"}
    for name,expected in original_hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()==expected
    estimate=exp.estimate_cost(9759,1077,2764,1335,p)
    assert 2800<estimate["L0_training_seconds"]<3000
    assert 4500<estimate["fully_cached_total_seconds"]<estimate["cold_cache_total_seconds"]<7200
