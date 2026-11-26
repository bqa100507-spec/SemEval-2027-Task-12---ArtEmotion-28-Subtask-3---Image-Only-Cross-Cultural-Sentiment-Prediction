import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from st3 import data, gradient_confirmation as confirmation, gradient_experiment as experiment
from st3 import gradient_workflow as workflow
from st3.baseline import file_sha256
from st3.caption_student import concatenate_features, predict_student
from st3.fusion import blend
from st3.gradient_analysis import BLOCKS, decompose, deterministic_batches, diagnostic
from st3.gradient_training import TrainingRecorder, matched_factory, state_hash
from st3.polarity_student import polarity_epoch, polarity_rows, seed_polarity_model
from st3.polarity_targets import condition_targets
from st3.polarity_workflow import BudgetStop, ExecutionBudget, StageBlocked, read_sealed, seal
from test_polarity_captions import CONFIG, tiny_workflow
from st3 import polarity_experiment as parent_experiment


def test_gradient_decomposition_alignment_orthogonality_and_zero():
    b=torch.tensor([3.,0.]); t=torch.tensor([-2.,2.])
    r=decompose(b,t,1.,.5)
    assert r["cosine"]==pytest.approx(-1/np.sqrt(2))
    assert r["orthogonal_fraction"]==pytest.approx(1/np.sqrt(2))
    assert r["opposing_fraction"]==pytest.approx(1/np.sqrt(2))
    assert r["combined_norm"]==pytest.approx(np.sqrt(5))
    assert r["post_clip_combined_norm"]==pytest.approx(np.sqrt(5)/2)
    zero=decompose(torch.zeros(2),t,0.)
    assert zero["zero_bce"] and zero["cosine"] is None and zero["scaled_ratio"] is None
    with pytest.raises(ValueError):decompose(b,torch.tensor([float('nan'),0.]),1.)


def test_deterministic_batches_do_not_touch_training_rng():
    torch.manual_seed(9);before=torch.get_rng_state().clone()
    left=deterministic_batches(100,16,3,192027)
    right=deterministic_batches(100,16,3,192027)
    assert torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(left,right))
    assert len(set(torch.cat(left).tolist()))==48
    assert not torch.equal(left[0],deterministic_batches(100,16,5,192027)[0])


@pytest.fixture
def diagnostic_fixture():
    torch.set_num_threads(1)
    rng=np.random.default_rng(19)
    frame=pd.DataFrame({"painting":[f"p{i}" for i in range(8)],"image":[f"i{i}" for i in range(8)],"language":"Thai","n":5,"k":2})
    features={name:rng.normal(size=2048).astype(np.float32) for name in frame.image}
    vectors=rng.normal(size=(8,2,64)).astype(np.float32);vectors/=np.linalg.norm(vectors,axis=-1,keepdims=True)
    masks=np.ones((8,2),dtype=bool);masks[0,1]=False;vectors[0,1]=0
    targets={key:(v,m) for key,v,m in zip(frame[data.KEYS].itertuples(index=False,name=None),vectors,masks)}
    weights={key:1. for key in targets}
    rows=polarity_rows(frame,features,["Thai"],torch.device("cpu"),targets,weights)
    return rows,{**CONFIG,"batch_size":4,"text_weight":1.}


def test_matched_initialization_and_lambda_configuration():
    cfg=json.loads((Path(__file__).resolve().parents[1]/"configs/exp019.json").read_text())
    assert cfg["text_weights"]=={"C0":0.,"C1":.1,"C2":1.}
    first,rng1,_=matched_factory(["Thai"],2027,{**CONFIG,"text_weight":1.},torch.device("cpu"))
    second,rng2,_=seed_polarity_model(["Thai"],2027,CONFIG,torch.device("cpu"))
    assert state_hash(first)==state_hash(second)
    assert torch.equal(rng1.get_state(),rng2.get_state())


def test_feature_gradient_blocks_tenfold_scaling_controlled_rng_and_clipping(diagnostic_fixture):
    rows,cfg=diagnostic_fixture
    model,_,_=matched_factory(["Thai"],2027,cfg,torch.device("cpu"));model.train()
    original=state_hash(model);rng=torch.get_rng_state().clone();batch=torch.arange(4)
    low=diagnostic(model,rows,batch,.1,1.)
    high=diagnostic(model,rows,batch,1.,1.)
    assert state_hash(model)==original and torch.equal(rng,torch.get_rng_state()) and model.training
    assert all(p.grad is None for p in model.parameters())
    assert BLOCKS==(("clip_b",0,512),("siglip2",512,1280),("clip_l",1280,2048))
    for group in high["groups"]:
        a,b=low["groups"][group],high["groups"][group]
        assert b["scaled_ratio"]==pytest.approx(10*a["scaled_ratio"])
        assert b["cosine"]==pytest.approx(a["cosine"])
    clipped=diagnostic(model,rows,batch,1.,1e-6)
    assert clipped["clipped"] and clipped["clip_factor"]<1.
    visual=clipped["groups"]["first_visual"]
    assert visual["post_clip_combined_norm"]==pytest.approx(visual["combined_norm"]*clipped["clip_factor"])
    assert "post_clip_combined_norm" not in clipped["groups"]["clip_b"]


def test_telemetry_preserves_exact_original_optimizer_and_rng(diagnostic_fixture,tmp_path):
    rows,cfg=diagnostic_fixture
    old,old_rng,old_opt=seed_polarity_model(["Thai"],2027,cfg,torch.device("cpu"))
    expected=[]
    for _ in range(2):expected.append(polarity_epoch(old,rows,old_rng,old_opt,cfg,1.))
    new,new_rng,new_opt=matched_factory(["Thai"],2027,cfg,torch.device("cpu"))
    recorder=TrainingRecorder(tmp_path,ExecutionBudget(60),1)
    actual=[]
    for epoch in range(1,3):
        recorder.progress("refit",epoch)
        actual.append(recorder.epoch_function(new,rows,new_rng,new_opt,cfg,1.))
    assert actual==expected and state_hash(new)==state_hash(old)
    assert torch.equal(old_rng.get_state(),new_rng.get_state())
    assert (tmp_path/"early.pt").exists()
    assert all(row["mean_post_clip_norm"]<=cfg["gradient_clip"] for row in recorder.records)


def test_auxiliary_zero_masks_and_inference_ignore_text(diagnostic_fixture):
    rows,cfg=diagnostic_fixture
    model,_,_=matched_factory(["Thai"],2027,cfg,torch.device("cpu"));model.eval()
    with torch.no_grad():before=model(rows["features"],rows["language"])[0]
    rows["text"].fill_(float('nan'))
    with torch.no_grad():after=model(rows["features"],rows["language"])[0]
    assert torch.equal(before,after)
    rows["text"].zero_();rows["text_mask"].zero_()
    result=diagnostic(model,rows,torch.arange(4),1.,1.)
    assert result["groups"]["first_visual"]["zero_text"]


def test_missing_parent_reports_precise_recovery(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,"config019",lambda:{"parent_config_sha256":file_sha256(Path("configs/exp018.json"))})
    with pytest.raises(StageBlocked,match="run_exp018.ps1 -Stage CV"):
        workflow.validate_parent(SimpleNamespace(exp018_dir=tmp_path))


def test_original_source_mutation_is_rejected(tmp_path,monkeypatch):
    (tmp_path/"decision").mkdir();(tmp_path/"decision/run.json").write_text(json.dumps({"identity":{"source_sha256":{"old.py":"original"}}}))
    monkeypatch.setattr(workflow,"config019",lambda:{"parent_config_sha256":file_sha256(Path("configs/exp018.json"))})
    monkeypatch.setattr(workflow,"fingerprint",lambda *a:{"source_sha256":{"old.py":"changed"}})
    with pytest.raises(ValueError,match="implementation changed"):
        workflow.validate_parent(SimpleNamespace(exp018_dir=tmp_path))


@pytest.fixture
def gradient_workflow(tiny_workflow,monkeypatch):
    old_args,old_ctx,_=tiny_workflow
    parent_experiment.run_pilot(old_args);parent_experiment.run_cv(old_args);parent_experiment.run_verify(old_args)
    args=SimpleNamespace(**{**vars(old_args),"run_dir":old_args.run_dir.parent/"new", "exp018_dir":old_args.run_dir})
    protocol={"text_weights":{"C0":0.,"C1":.1,"C2":1.},"diagnostic_folds":[1],"diagnostic_batches":3,
              "diagnostic_seed":192027,"early_refit_epoch":1,"pilot_epochs":2,"historical_margin":1.5,
              "diagnostic_verification_allowance_seconds":300,"codex_budget_seconds":900,"bootstrap_replicates":50,"bootstrap_seed":19}
    predictions=pd.read_csv(old_args.run_dir/"cv/predictions.csv",float_precision="round_trip")
    identity={"fixture":19,"runtime_and_source":{"device":"cpu"}}
    ctx=SimpleNamespace(identity=identity,parent_identity=old_ctx.identity,config={**old_ctx.config,"text_weight":1.},
                        protocol=protocol,train=old_ctx.train,predictions=predictions,banks=old_ctx.banks)
    monkeypatch.setattr(experiment,"context019",lambda *a,**k:ctx)
    monkeypatch.setattr(workflow,"context019",lambda *a,**k:ctx)
    monkeypatch.setattr(confirmation,"context019",lambda *a,**k:ctx)
    preflight=args.run_dir/"preflight";preflight.mkdir(parents=True)
    hashes={str(p.resolve()):file_sha256(p) for p in old_args.run_dir.rglob("*") if p.is_file()}
    (preflight/"parent_files_sha256.json").write_text(json.dumps(hashes))
    seal(preflight,identity,{"status":"INPUTS_VERIFIED"})
    return args,ctx


def test_tiny_end_to_end_reuse_rounding_resume_and_locked_no_go(gradient_workflow,monkeypatch):
    args,ctx=gradient_workflow
    experiment.run_gradient_pilot(args);experiment.run_cv(args)
    paths=list((args.run_dir/"work").rglob("student.pt"));assert len(paths)==2
    hashes={p:file_sha256(p) for p in paths}
    monkeypatch.setattr(experiment,"fit_student",lambda *a,**k:pytest.fail("Refit completed seed"))
    experiment.run_cv(args);experiment.run_verify(args)
    assert all(file_sha256(p)==digest for p,digest in hashes.items())
    result=experiment.verified_decision(args);assert result["status"]=="COMPETITION_NO_GO"
    values=pd.read_csv(args.run_dir/"cv/predictions.csv",float_precision="round_trip")
    for old,new in experiment.REUSE.items():
        pd.testing.assert_frame_equal(values[values.model.eq(new)][data.KEYS+["p_positive"]].reset_index(drop=True),ctx.predictions[ctx.predictions.model.eq(old)][data.KEYS+["p_positive"]].reset_index(drop=True))
    actual=values[values.model.eq("E2")][data.KEYS+["p_positive"]]
    expected=blend(values[values.model.eq("M0")][data.KEYS+["p_positive"]],values[values.model.eq("C2")][data.KEYS+["p_positive"]],.5)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True))
    monkeypatch.setattr(confirmation,"old_inputs",lambda *a:pytest.fail("Opened holdout"))
    with pytest.raises(StageBlocked,match="sealed"):confirmation.run_confirm(args)
    with pytest.raises(StageBlocked,match="sealed"):confirmation.run_dev(args)
    assert not (args.run_dir/"confirm").exists() and not (args.run_dir/"dev").exists()
    next(iter(hashes)).write_bytes(b"broken")
    with pytest.raises(ValueError,match="checksum"):experiment.run_cv(args)


def test_interrupted_seed_preserves_completed_models_and_replays_later(gradient_workflow,monkeypatch):
    args,ctx=gradient_workflow;experiment.run_gradient_pilot(args)
    class StopSecond:
        def check(self,phase="stage",epoch=0):
            if "fold 2" in phase:raise BudgetStop("fixture stop")
    monkeypatch.setattr(experiment,"budget019",lambda *a:StopSecond())
    with pytest.raises(BudgetStop):experiment.run_cv(args)
    path=args.run_dir/"work/fold_1/C2_2027/student.pt";digest=file_sha256(path)
    assert not (args.run_dir/"cv").exists()
    monkeypatch.setattr(experiment,"budget019",lambda *a:ExecutionBudget(60))
    experiment.run_cv(args);assert file_sha256(path)==digest
    experiment.run_verify(args)
    monkeypatch.setattr(experiment,"_verify",lambda *a:(_ for _ in ()).throw(BudgetStop("later replay")))
    with pytest.raises(BudgetStop):experiment.run_verify(args)
    with pytest.raises(StageBlocked,match="Latest Verify"):experiment.verified_decision(args)


def test_preservation_checks_detect_mutation(gradient_workflow):
    args,ctx=gradient_workflow
    assert workflow.preserve_parent(args,ctx.identity)>0
    (args.exp018_dir/"cv/summary.csv").write_text("changed")
    with pytest.raises(ValueError,match="changed"):workflow.preserve_parent(args,ctx.identity)


def test_gate_is_frozen_and_incomplete_is_not_no_go():
    summary=pd.DataFrame({"pooled_contrast":[.44,.441,.443],"log_loss_mean":[.71,.705,.715],"fold_wins":[0,3,3],"folds":5},index=["M0","E0","E2"])
    assert experiment.choose(summary,CONFIG,False)["status"]=="INCOMPLETE"
    assert experiment.choose(summary,CONFIG,True)["status"]=="PASS"
    summary.loc["E2","pooled_contrast"]-=.000001
    assert experiment.choose(summary,CONFIG,True)["status"]=="COMPETITION_NO_GO"


def test_research_value_independent_of_competition_gate():
    summary=pd.DataFrame({"pooled_contrast":[.44,.443]},index=["C1","C2"])
    gradients=pd.DataFrame({"checkpoint":"selected_final_refit","group":"shared_representation","condition":["C1","C2"],"scaled_ratio":[.01,.1],"cosine":[-.1,.1]})
    assert experiment.research_assessment(summary,gradients)["status"]=="RESEARCH_PROMISING"
