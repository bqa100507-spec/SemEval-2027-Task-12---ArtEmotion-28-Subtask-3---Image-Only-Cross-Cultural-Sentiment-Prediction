import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from st3 import data,emotion_confirmation as confirmation,emotion_experiment as experiment
from st3 import emotion_verification as verification,emotion_workflow as workflow
from st3.baseline import file_sha256
from st3.cli import json_output
from st3.emotion_student import EmotionStudent,emotion_diagnostic,emotion_rows,seed_model
from st3.emotion_targets import EMOTIONS,POLARITY,condition_targets,emotion_means,normalize,project_targets
from st3.fusion import blend
from st3.gradient_training import state_hash
from st3.polarity_student import balanced_channels,cosine_loss
from st3.polarity_workflow import BudgetStop,ExecutionBudget,StageBlocked,read_sealed,seal
from st3 import polarity_workflow as parent_workflow
from st3.caption_data import inner_partition
from test_polarity_captions import tiny_workflow
from test_polarity_captions import CONFIG


def test_exact_eight_mapping_and_original_polarities():
    assert len(set(EMOTIONS))==8
    assert set(EMOTIONS[:4])==data.POSITIVE and set(EMOTIONS[4:])==data.NEGATIVE
    assert POLARITY.tolist()==[0,0,0,0,1,1,1,1]
    assert not {"other","something else"}&set(EMOTIONS)


def test_means_keep_duplicates_and_emotion_associations_and_exclude_ambiguous():
    captions=pd.DataFrame({"painting":"p","language":"Thai","split":"train","valid_caption":True,
        "emotion":["awe","awe","sadness","other"],"binary":[True,True,True,False],"caption_id":["a","b","c","x"]})
    pairs=pd.DataFrame({"painting":["p"],"language":["Thai"],"k":[3],"n":[5]})
    vectors={"a":np.array([1.,0.],dtype=np.float32),"b":np.array([0.,1.],dtype=np.float32),"c":np.array([-1.,0.],dtype=np.float32)}
    _,means,counts,_=emotion_means(captions,pairs,vectors)
    np.testing.assert_array_equal(counts,[[0,2,0,0,0,0,0,1]])
    np.testing.assert_array_equal(means[0,1],[.5,.5]);np.testing.assert_array_equal(means[0,7],[-1.,0.])
    duplicate=pd.concat([captions.iloc[:1],captions],ignore_index=True)
    _,values,counts,_=emotion_means(duplicate,pairs,vectors)
    assert counts[0,1]==3
    np.testing.assert_allclose(values[0,1],[2/3,1/3])
    with pytest.raises(ValueError,match="official training"):
        emotion_means(captions.assign(split="val"),pairs,vectors)


def test_shared_pca_no_fitting_and_identical_emotion_masks(tmp_path,monkeypatch):
    records=pd.DataFrame({"painting":["p"],"language":["Thai"]})
    means=np.zeros((1,8,2),dtype=np.float32);means[0,0]=[1,0];means[0,1]=[0,1]
    counts=np.zeros((1,8),dtype=int);counts[0,:2]=1
    raw=np.array([[[.5,.5],[1.,0.],[0.,0.]]],dtype=np.float32)
    pca=tmp_path/"pca.npz";np.savez(pca,components=np.eye(2,dtype=np.float32),mean=np.zeros(2,dtype=np.float32))
    reference={"records":records,"targets":raw.copy(),"valid":np.array([[True,True,False]])}
    from sklearn.decomposition import PCA
    monkeypatch.setattr(PCA,"fit",lambda *a,**k:pytest.fail("Fitted new emotion PCA"))
    bundle=project_targets(records,means,counts,records,raw,records,pca,reference)
    assert bundle["valid"].tolist()==[[True,True,False,False,False,False,False,False]]
    np.testing.assert_array_equal(bundle["A8"][0,:2],[[1,0],[1,0]])
    np.testing.assert_array_equal(bundle["E8"][0,:2],[[1,0],[0,1]])
    a,e=condition_targets(bundle,"A8"),condition_targets(bundle,"E8")
    assert a[1]==e[1] and np.array_equal(a[0][("p","Thai")][1],e[0][("p","Thai")][1])
    means[0,1]=0
    degenerate=project_targets(records,means,counts,records,raw,records,pca,reference)
    assert not degenerate["valid"][0,1]
    assert not np.count_nonzero(degenerate["A8"][0,1])


def test_channel_average_painting_weighting_and_missing_targets():
    losses=torch.tensor([[1.,0.,0.,0.,0.,0.,0.,0.],[1.]*8])
    masks=torch.tensor([[1,0,0,0,0,0,0,0],[1]*8],dtype=torch.bool)
    assert balanced_channels(losses,masks,torch.tensor([1.,1.])).item()==1.
    records=pd.DataFrame({"painting":["p","p","q"],"language":["a","b","a"]})
    bundle={"records":records,"A8":np.zeros((3,8,2)),"valid":np.ones((3,8),dtype=bool)}
    _,weights=condition_targets(bundle,"A8")
    assert weights=={("p","a"):.5,("p","b"):.5,("q","a"):1.}
    pred=torch.randn(2,8,4,requires_grad=True);pred=torch.nn.functional.normalize(pred,dim=-1)
    target=torch.randn(2,8,4,requires_grad=True)
    loss=cosine_loss(pred,target,masks,torch.ones(2));grad=torch.autograd.grad(loss,target)[0]
    assert not torch.count_nonzero(grad[~masks])


@pytest.fixture
def diagnostic_rows():
    torch.set_num_threads(1);rng=np.random.default_rng(20)
    frame=pd.DataFrame({"painting":[f"p{i}" for i in range(8)],"image":[f"i{i}" for i in range(8)],"language":"Thai","k":2,"n":5})
    visual={name:rng.normal(size=2048).astype(np.float32) for name in frame.image}
    values=rng.normal(size=(8,8,64)).astype(np.float32);values/=np.linalg.norm(values,axis=-1,keepdims=True)
    masks=np.ones((8,8),dtype=bool);masks[:,6]=False;values[:,6]=0
    keys=list(frame[data.KEYS].itertuples(index=False,name=None))
    targets={key:(v,m) for key,v,m in zip(keys,values,masks)};weights={key:1. for key in keys}
    return emotion_rows(frame,visual,["Thai"],torch.device("cpu"),targets,weights)


def test_matched_model_and_query_gradients_json_masks_collapse_and_inference(diagnostic_rows):
    a,ra,oa=seed_model(["Thai"],2027,CONFIG,torch.device("cpu"))
    b,rb,ob=seed_model(["Thai"],2027,CONFIG,torch.device("cpu"))
    assert state_hash(a)==state_hash(b) and torch.equal(ra.get_state(),rb.get_state())
    assert oa.param_groups[0]["lr"]==ob.param_groups[0]["lr"]==.001
    a.eval();b.eval();rows=diagnostic_rows
    with torch.no_grad():
        x=a(rows["features"],rows["language"])[0];y=b(rows["features"],rows["language"])[0]
        assert torch.equal(x,y)
    rng=torch.get_rng_state().clone();result=emotion_diagnostic(a,rows,torch.arange(8),CONFIG,.99)
    assert torch.equal(rng,torch.get_rng_state()) and result["finite"]
    assert result["query_embedding_gradient_norms"][6]==0
    assert all(n>0 for i,n in enumerate(result["query_embedding_gradient_norms"]) if i!=6)
    assert result["missing_target_gradient_max"]==0
    assert result["predicted_query_cosine"][6][6] is None
    assert result["emotion_gradient_cosine"][6][6] is None
    json_output(result) # no NaN/Infinity in sparse channel diagnostics
    rows["text"].fill_(float('nan'))
    with torch.no_grad():assert torch.equal(x,a(rows["features"],rows["language"])[0])


def test_frozen_gate_and_selection_tie_rule():
    table=pd.DataFrame({"pooled_contrast":[.44,.441,.443,.444],"log_loss_mean":[.71,.707,.713,.714],"fold_wins":[0,3,3,4],"folds":5},index=["R0","RB1","EA8","EE8"])
    assert experiment.choose(table,CONFIG,False)["status"]=="INCOMPLETE"
    assert experiment.choose(table,CONFIG,True)["model"]=="EE8"
    table.loc[["EA8","EE8"],"pooled_contrast"]=.4429
    assert experiment.choose(table,CONFIG,True)["status"]=="COMPETITION_NO_GO"


@pytest.fixture
def emotion_workflow(tiny_workflow,monkeypatch):
    a18,c18,raw=tiny_workflow
    parent_predictions=[];parent_seeds=[];historical_fits=[]
    # Frozen synthetic references are fixtures, not replayed research results.
    # Only the new A8/E8 models are actually trained in this integration test.
    for fold in range(1,3):
        fit=c18.train[c18.train.fold.ne(fold)];held=c18.train[c18.train.fold.eq(fold)]
        inner,_=inner_partition(fit,c18.config,fold)
        parent_workflow.geometry(a18,c18,raw,fit,inner,a18.run_dir/"work"/f"fold_{fold}"/"geometry")
        p0=held[data.KEYS].assign(p_positive=held.p.to_numpy())
        p1=held[data.KEYS].assign(p_positive=np.round(.8*held.p.to_numpy()+.1,6))
        p3=held[data.KEYS].assign(p_positive=np.round(.82*held.p.to_numpy()+.09,6))
        for name,pred in (("M0",p0),("M1",p1),("M3",p3),("B1",blend(p0,p1,.5)),("B3",blend(p0,p3,.5))):
            parent_predictions.append(pred.assign(model=name,fold=fold))
        for name,pred in (("M1",p1),("M3",p3)):
            parent_seeds.append(pred.assign(model=name,fold=fold,seed=2027))
        historical_fits.append({"model":"M3","fold":fold,"seed":2027,"fit_seconds":.01,"selection_epochs":1,"selected_epoch":1})
    for directory in (a18.run_dir/"cv",a18.run_dir/"verification"):directory.mkdir()
    (a18.run_dir/"cv/fits.json").write_text(json.dumps(historical_fits))
    pd.concat(parent_seeds).to_csv(a18.run_dir/"verification/seed_predictions.csv",index=False)
    old19=a18.run_dir.parent/"old19";(old19/"cv").mkdir(parents=True)
    (old19/"cv/fits.json").write_text(json.dumps(historical_fits))
    args=SimpleNamespace(**{**vars(a18),"run_dir":a18.run_dir.parent/"emotion", "exp018_dir":a18.run_dir,
                           "exp019_dir":old19,"data_dir":a18.run_dir.parent/"data"})
    protocol={"near_identical_cosine":.99,"subgroup_polarity_similarity":.95,"high_truncation_rate":.5,
        "collapsed_prediction_cosine":.99,"pilot_warmup_epochs":0,"pilot_timed_epochs":1,"historical_margin":1.5,
        "diagnostic_verification_allowance_seconds":300,"bootstrap_seed":20,"bootstrap_replicates":50}
    identity={"fixture":20,"runtime_and_source":{"device":"cpu"}}
    ctx=SimpleNamespace(config={**c18.config,"text_weight":.1},protocol=protocol,identity=identity,parent_identity=c18.identity,
        train=c18.train,predictions=pd.concat(parent_predictions),banks=c18.banks,previous_decision={"status":"COMPETITION_NO_GO"})
    for module in (workflow,experiment,verification,confirmation):
        monkeypatch.setattr(module,"context020",lambda *a,**k:ctx)
    captions=[];vectors={};rng=np.random.default_rng(20);i=0
    for row in ctx.train.itertuples():
        emotions=[EMOTIONS[j%4] for j in range(row.k)]+[EMOTIONS[4+j%4] for j in range(row.n-row.k)]
        for emotion in emotions:
            key=f"caption{i}";value=rng.normal(size=5).astype(np.float32);value/=np.linalg.norm(value);vectors[key]=value
            captions.append({"annotation_id":i,"painting":row.painting,"language":row.language,"emotion":emotion,
                "caption_id":key,"split":"train","valid_caption":True,"binary":True});i+=1
    captions=pd.DataFrame(captions)
    monkeypatch.setattr(workflow,"old_inputs",lambda *a:(ctx.train,ctx.train,{}))
    monkeypatch.setattr(workflow,"load_teacher",lambda *a,**k:(vectors,{}))
    monkeypatch.setattr(workflow,"training_captions",lambda *a,**k:captions)
    token=args.exp017_dir/"text/clip";token.mkdir(parents=True)
    captions.assign(tokens=10,truncated=False).to_csv(token/"tokens.csv",index=False)
    preflight=args.run_dir/"preflight";preflight.mkdir(parents=True)
    hashes={str(p.resolve()):file_sha256(p) for root in (args.exp018_dir,args.exp019_dir) for p in root.rglob("*") if p.is_file()}
    (preflight/"protected_file_sha256.json").write_text(json.dumps(hashes));seal(preflight,identity,{"seconds":.01})
    return args,ctx


def test_tiny_full_pipeline_audit_matched_training_replay_resume_and_guards(emotion_workflow,monkeypatch):
    args,ctx=emotion_workflow
    workflow.run_audit(args);experiment.run_pilot(args);experiment.run_cv(args)
    paths=list((args.run_dir/"work").rglob("student.pt"));assert len(paths)==4
    hashes={p:file_sha256(p) for p in paths}
    monkeypatch.setattr(experiment,"fit_student",lambda *a,**k:pytest.fail("Retrained valid historical/new seed"))
    experiment.run_cv(args);verification.run_verify(args)
    assert all(file_sha256(p)==h for p,h in hashes.items())
    decision=verification.verified_decision(args);assert decision["status"]=="COMPETITION_NO_GO"
    frame=pd.read_csv(args.run_dir/"cv/predictions.csv",float_precision="round_trip")
    for old,new in experiment.REFERENCES.items():
        pd.testing.assert_frame_equal(frame[frame.model.eq(new)][data.KEYS+["p_positive"]].reset_index(drop=True),ctx.predictions[ctx.predictions.model.eq(old)][data.KEYS+["p_positive"]].reset_index(drop=True))
    expected=blend(frame[frame.model.eq("R0")][data.KEYS+["p_positive"]],frame[frame.model.eq("E8")][data.KEYS+["p_positive"]],.5)
    pd.testing.assert_frame_equal(expected.reset_index(drop=True),frame[frame.model.eq("EE8")][data.KEYS+["p_positive"]].reset_index(drop=True))
    monkeypatch.setattr(confirmation,"old_inputs",lambda *a:pytest.fail("Opened locked set"))
    with pytest.raises(StageBlocked,match="sealed"):confirmation.run_confirm(args)
    with pytest.raises(StageBlocked,match="sealed"):confirmation.run_dev(args)
    assert read_sealed(args.run_dir/"verification",ctx.identity)["checkpoints_replayed"]==4
    assert read_sealed(args.run_dir/"verification",ctx.identity)["target_partitions_replayed"]==4
    next(iter(hashes)).write_bytes(b"broken")
    with pytest.raises(ValueError,match="checksum"):experiment.run_cv(args)


def test_budget_resume_and_failed_later_verification_blocks_previous_success(emotion_workflow,monkeypatch):
    args,ctx=emotion_workflow;workflow.run_audit(args);experiment.run_pilot(args)
    class StopE8:
        def check(self,phase="",epoch=0):
            if "E8" in phase:raise BudgetStop("fixture interruption")
    original=experiment.stage_budget
    monkeypatch.setattr(experiment,"stage_budget",lambda *a:StopE8())
    with pytest.raises(BudgetStop):experiment.run_cv(args)
    path=args.run_dir/"work/fold_1/A8_2027/student.pt";digest=file_sha256(path)
    monkeypatch.setattr(experiment,"stage_budget",original)
    experiment.run_cv(args);assert file_sha256(path)==digest
    verification.run_verify(args)
    monkeypatch.setattr(verification,"_verify",lambda *a:(_ for _ in ()).throw(BudgetStop("replay interrupted")))
    with pytest.raises(BudgetStop):verification.run_verify(args)
    with pytest.raises(StageBlocked,match="Latest replay"):verification.verified_decision(args)


def test_research_interpretation_is_independent_of_promotion():
    summary=pd.DataFrame({"pooled_contrast":[.44,.442,.4405]},index=["A8","E8","R1"])
    folds=pd.DataFrame({"fold":[1,1,2,2,3,3],"model":["A8","E8"]*3,"contrast":[.44,.442,.44,.443,.44,.441]})
    result=experiment.research_assessment(summary,folds,{"E8_minus_A8":{"ci95":[.0001,.004]}})
    assert result["status"]=="RESEARCH_PROMISING"
    summary.loc["E8","pooled_contrast"]=.4405
    assert experiment.research_assessment(summary,folds,{"E8_minus_A8":{"ci95":[-.001,.003]}})["status"]=="RESEARCH_INCONCLUSIVE"
