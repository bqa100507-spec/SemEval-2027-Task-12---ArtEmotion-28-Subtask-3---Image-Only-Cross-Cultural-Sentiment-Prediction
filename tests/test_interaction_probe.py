import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from st3 import interaction_probe as probe
from st3 import probe_inference as inference
from st3.vlm_semantics import file_hash, fingerprint, atomic_json
from st3.vlm_supervised import attach_adapters, make_head
from st3.vlm_supervised_training import model_state
from test_vlm_supervised import TinyModel, config

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    c=json.loads((ROOT/"configs/task031.json").read_text());c["bootstrap_replicates"]=100
    queries=[dict(painting=p,target_language=l,condition=v,prompt_language=None if v=="B_neutral_prompt" else l,
                  head_language="A" if v=="C_fixed_head" else "B" if v=="D_crossed_head" and l=="A" else "A" if v=="D_crossed_head" else l)
             for p,l,v in itertools.product(["I","J"],["A","B"],c["conditions"])]
    targets=[dict(painting=p,language=l,k=k,n=10000) for p,l,k in [("I","A",9000),("I","B",1000),("J","A",1000),("J","B",9000)]]
    m=dict(config=c,queries=queries,images={"I":{},"J":{}},rectangles=[dict(pair_id="pair00",I="I",J="J",a="A",b="B")],targets_sha256=fingerprint(targets),checkpoint_sha256={})
    rows=[dict(q,model=model,adapter_state=state,logit=(2. if q["painting"]=="I" else -2.)*(1 if q["target_language"]=="A" else -1))
          for q,model,state in itertools.product(queries,probe.MODELS,c["adapter_states"])]
    return m,targets,rows


def test_raw_factorial_additive_interaction_sign_and_sigmoid_warning():
    additive={(p,l):x+y for (p,x),(l,y) in itertools.product([("I",2.),("J",0.)],[("A",1.),("B",-.5)])}
    assert probe.mixed(additive,"I","J","A","B")==0
    probabilities={k:float(torch.sigmoid(torch.tensor(v))) for k,v in additive.items()}
    assert abs(probe.mixed(probabilities,"I","J","A","B"))>.01
    interaction={k:v+(3 if k==("I","A") else 0) for k,v in additive.items()}
    assert probe.mixed(interaction,"I","J","A","B")==3
    assert probe.mixed(interaction,"J","I","A","B")==-3
    assert probe.mixed(interaction,"I","J","B","A")==-3
    assert probe.mixed(interaction,"J","I","B","A")==3


def test_exact_keys_prevent_ordering_corruption_and_nonfinite():
    m,t,r=fixture();a=probe.analyze(r,m,t)[1];r.reverse()
    assert a==probe.analyze(r,m,t)[1]
    for bad in (r[:-1],r+[r[0]], [dict(x,head_language="WRONG") for x in r], [dict(x,logit=float("nan")) for x in r]):
        with pytest.raises(ValueError):probe.validate_rows(bad,m)
    with pytest.raises(ValueError,match="targets changed"):probe.analyze(r,m,t[::-1])


def test_strength_ties_noise_orders_and_clustered_comparisons():
    m,t,r=fixture();frame,result=probe.analyze(r,m,t);s=result["summaries"][0]
    assert s["mean_abs_D"]==8 and s["resolved_target_sign_agreement"]==1
    assert s["language_order_reversal_fraction"]==1
    assert s["pair_cluster_95_CI_mean_abs_D"]==[8.,8.]
    assert all(x["mean_abs_D_difference"]==0 for x in result["paired_interaction_comparisons"])
    for x in r:x["logit"]=.0001
    for x in t:x["k"]=5000
    m["targets_sha256"]=fingerprint(t)
    s=probe.analyze(r,m,t)[1]["summaries"][0]
    assert s["near_zero_fraction"]==1 and s["non_ties"]==0 and s["D_target_correlation"] is None
    assert s["resolved_target_sign_agreement"] is None and s["language_order_reversal_fraction"] is None


def test_sampling_has_no_target_dependence_and_excludes_forbidden_populations(tmp_path,monkeypatch):
    c=json.loads((ROOT/"configs/task031.json").read_text());c["painting_pairs"]=2
    training=tmp_path/"training";training.mkdir()
    ids=["p0","p1","p2","p3","old_fold1","outer2","validation","locked"]
    pd.DataFrame([dict(painting=p,role="holdout" if p=="locked" else "development",fold=1 if p=="old_fold1" else 2 if p=="outer2" else 3,image=p+".jpg",image_sha256=p) for p in ids]).to_csv(training/"protocol.csv",index=False)
    rows=[dict(painting=p,language=l,k=2,n=5) for p,l in itertools.product(ids,["A","B","C"])]
    pd.DataFrame(rows).to_csv(training/"pairs.csv",index=False)
    for name in probe.MODELS:(tmp_path/"models").mkdir(exist_ok=True);(tmp_path/"models"/(name+".pt")).write_bytes(b"fixture")
    def fake(*a):return {"inputs":{}},{name:{"identity":{"fit":ids[:6]},"state":{"languages":["A","B","C"]}} for name in probe.MODELS}
    monkeypatch.setattr(probe,"checkpoints",fake)
    original=probe.csv.DictReader
    class Poison(dict):
        def __getitem__(self,key):
            if self.get("painting") in ids[4:] and key in ("k","n"):pytest.fail("Forbidden vote indexed")
            return super().__getitem__(key)
    monkeypatch.setattr(probe.csv,"DictReader",lambda f:(Poison(r) for r in original(f)))
    a,t=probe.make_manifest(tmp_path,tmp_path,ROOT,c)
    assert set(a["images"])==set(ids[:4])
    for row in rows:row["k"]=3
    pd.DataFrame(rows[::-1]).to_csv(training/"pairs.csv",index=False)
    b,u=probe.make_manifest(tmp_path,tmp_path,ROOT,c)
    assert a["rectangles"]==b["rectangles"] and a["queries"]==b["queries"]
    assert t!=u and a["targets_sha256"]!=b["targets_sha256"]


def test_pinned_checkpoint_corruption_fails_before_loading(tmp_path):
    expected=json.loads((ROOT/"docs/exp027_closure_evidence.json").read_text())["files_sha256"]
    (tmp_path/"identity.json").write_text('{"source_revision":"wrong"}')
    with pytest.raises(ValueError,match="identity differs"):probe.checkpoints(tmp_path,ROOT)


def tiny_backend():
    c=config();model=TinyModel();attach_adapters(model,c)
    def hidden(visions,languages,train=False):
        assert train is False and not torch.is_grad_enabled()
        values=torch.stack([v["tokens"] + (0 if l is None else .1 if l=="A" else -.1) for v,l in zip(visions,languages)])
        return model(values)
    backend=SimpleNamespace(model=model,device="cpu",hidden=hidden)
    head=make_head(["A","B"],c)
    with torch.no_grad():head.language_visual.weight.fill_(.1)
    state=model_state(backend,head)
    inference.load_saved(backend,head,state)
    return backend,head,state


def test_authentic_image_prompt_head_paths_frozen_without_optimizer_or_backward(monkeypatch):
    backend,head,state=tiny_backend();m,_,_=fixture();queries=[q for q in m["queries"] if q["painting"]=="I"]
    monkeypatch.setattr(torch.optim,"AdamW",lambda *a,**k:pytest.fail("No optimizer allowed"))
    monkeypatch.setattr(torch.Tensor,"backward",lambda *a,**k:pytest.fail("No backward allowed"))
    before=copy.deepcopy(backend.model.state_dict())
    a=inference.query_image(backend,head,{"tokens":torch.arange(8).float()},queries,"trained")
    assert a==inference.query_image(backend,head,{"tokens":torch.arange(8).float()},queries,"trained")
    b=inference.query_image(backend,head,{"tokens":torch.zeros(8)},queries,"trained")
    assert [r["logit"] for r in a]!=[r["logit"] for r in b]
    assert all(torch.equal(v,backend.model.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None and not p.requires_grad for p in backend.model.parameters())
    bad=copy.deepcopy(state);bad["adapters"].pop(next(iter(bad["adapters"])))
    with pytest.raises(ValueError,match="names differ"):inference.load_saved(backend,head,bad)


def test_cache_resume_integrity_source_and_offline_replay(tmp_path):
    m,t,rows=fixture();identity=dict(manifest_sha256=fingerprint(m),source_sha256={},checkpoint_sha256={})
    atomic_json(tmp_path/"identity.json",identity)
    diagnostics=dict(exact_determinism_checks={str(i):True for i in range(6)},official_cache_parity={"status":"PASS"},fresh_vision_cache_exact=True)
    atomic_json(tmp_path/"inference_diagnostics.json",diagnostics)
    paths=[]
    for model,state,p in itertools.product(probe.MODELS,m["config"]["adapter_states"],m["images"]):
        path=tmp_path/"logits"/model/state/(p+".json");paths.append(path)
        group=[r for r in rows if (r["model"],r["adapter_state"],r["painting"])==(model,state,p)]
        key=inference.batch_identity(identity,model,state,p)
        inference.write_batch(path,key,group);assert inference.read_batch(path,key)==group
    result=inference.verify_outputs(tmp_path,m,t,ROOT)
    assert result==inference.verify_outputs(tmp_path,m,t,ROOT)
    assert json.loads((tmp_path/"verification.json").read_text())["status"]=="PASS"
    paths[0].write_bytes(paths[0].read_bytes()+b"corruption")
    with pytest.raises(ValueError,match="corrupt"):inference.verify_outputs(tmp_path,m,t,ROOT)
    assert json.loads((tmp_path/"verification.json").read_text())["mechanism_decision"]=="INVALID_UNTIL_PASS"


def test_cli_authorization_limits_and_no_outer_training_endpoint(tmp_path,monkeypatch):
    import importlib.util,sys
    spec=importlib.util.spec_from_file_location("task031",ROOT/"scripts/run_task031.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=["probe","--stage","probe","--inputs",str(tmp_path),"--checkpoints",str(tmp_path),"--run-dir",str(tmp_path)]
    for extra in ([],["--authorize-gpu","--max-seconds","nan"],["--authorize-gpu","--max-seconds","1801"],["--stage","train"]):
        monkeypatch.setattr(sys,"argv",base+extra)
        with pytest.raises(SystemExit):module.main()
    def timeout(*a,**k):raise module.subprocess.TimeoutExpired("CPU mock",100)
    monkeypatch.setattr(module.subprocess,"run",timeout);monkeypatch.setattr(sys,"argv",base+["--authorize-gpu","--max-seconds","100"])
    assert module.main()==2
    assert json.loads((tmp_path/"probe_report.json").read_text())["status"]=="STOPPED_BUDGET"
