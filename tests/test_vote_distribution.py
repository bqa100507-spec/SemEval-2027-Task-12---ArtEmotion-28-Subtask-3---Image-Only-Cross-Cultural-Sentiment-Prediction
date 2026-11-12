import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from st3 import data
from st3.vote_audit import ordering_audit,reliability_summary
from st3.vote_inputs import COUNT_COLUMNS,EMOTIONS,development_votes,empirical_distributions
from st3.vote_model import LOG_EIGHT,batch_gradients,seed_model,train_epoch,vote_losses,vote_rows
from st3 import vote_audit as audit_module,vote_experiment as experiment,vote_verification as verification
from st3.gradient_training import state_hash
from st3.polarity_student import polarity_epoch
from st3.polarity_workflow import BudgetStop,ExecutionBudget,StageBlocked,read_sealed,seal
from st3.caption_student import concatenate_features
from st3.baseline import file_sha256
from st3.fusion import blend
from test_polarity_captions import tiny_workflow

CONFIG=json.loads((Path(__file__).resolve().parents[1]/"configs/exp021.json").read_text())


def tiny_votes(tmp_path):
    root=tmp_path/"annotations";root.mkdir()
    protocol=pd.DataFrame({"painting":["train","locked"],"image":["train.jpg","locked.jpg"],
        "role":["development","holdout"],"fold":[1,0],"image_sha256":["a","b"]})
    records=[]
    for language,emotions in (("Thai",["awe","awe","amusement","fear","sadness","other","something else"]),
                               ("Tamil",["anger"]*5),("Korean",["excitement"]*5)):
        for emotion in emotions:records.append(["train",language,"train.jpg","train",emotion,"ignored caption"])
    records+=[["locked","Thai","locked.jpg","train","FORBIDDEN_EMOTION","must not be retained"],
              ["train","Thai","train.jpg","val","FORBIDDEN_EMOTION","validation"]]
    with (root/"artelingo28_train_val.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle);writer.writerow(data.ANNOTATION_COLUMNS+["caption"]);writer.writerows(records)
    return protocol


def test_annotation_only_eight_vote_reconstruction_preserves_repeated_rows(tmp_path):
    protocol=tiny_votes(tmp_path);pairs,votes=development_votes(tmp_path,protocol)
    assert set(EMOTIONS[:4])==data.POSITIVE and set(EMOTIONS[4:])==data.NEGATIVE
    assert pairs.n.tolist()==[5,5,5] and len(votes)==15
    thai=pairs[pairs.language.eq("Thai")].iloc[0]
    assert thai.count_awe==2 and thai.k==3 and thai.p==.6
    assert "caption" not in votes and set(votes.painting)=={"train"}
    d,y=empirical_distributions(pairs)
    np.testing.assert_allclose(d.sum(1),1.)
    np.testing.assert_array_equal(y,pairs.p)
    with pytest.raises(ValueError,match="Only original training"):
        empirical_distributions(pairs.assign(split="val"))
    with pytest.raises(ValueError,match="Invalid"):
        empirical_distributions(pairs.assign(n=6))


def test_beta_audit_reproducible_and_never_changes_empirical_targets(tmp_path):
    pairs,_=development_votes(tmp_path,tiny_votes(tmp_path));before=pairs.copy(deep=True)
    cfg={**CONFIG,"posterior_samples":1024}
    first,paintings=ordering_audit(pairs,cfg)
    second,_=ordering_audit(pairs.sample(frac=1,random_state=21),cfg)
    pd.testing.assert_frame_equal(first,second);pd.testing.assert_frame_equal(pairs,before)
    assert first.confidence.between(.5,1.).all()
    assert first[first.abs_difference.eq(1)].confidence.min()>=.95
    summary=reliability_summary(pairs,first,paintings,cfg)
    assert summary["n5_pairs"]==3 and summary["contrast_eligible_paintings"]==1
    assert "noise ceiling" in summary["assumptions"] and summary["training_policy"].startswith("Empirical")


def test_near_tie_boundary_uses_exact_vote_fraction():
    frame=pd.DataFrame({"painting":"p","language":["a","b"],"n":[5,5],"k":[4,3]})
    rows,_=ordering_audit(frame,{**CONFIG,"posterior_samples":128})
    assert rows.iloc[0].abs_difference==.2 and bool(rows.iloc[0].near_tie)


def test_natural_log_vote_objectives_and_fixed_normalization():
    counts=torch.tensor([[2.,1.,0.,0.,0.,0.,1.,1.],[0.,0.,0.,0.,3.,0.,0.,2.]],dtype=torch.float64)
    binary=torch.tensor([.3,-.4],dtype=torch.float64,requires_grad=True)
    logits=torch.arange(16,dtype=torch.float64).reshape(2,8)/7;logits.requires_grad_(True)
    result=vote_losses(binary,logits,counts,.1)
    expected=-(counts*torch.log_softmax(logits,1)).sum()/counts.sum()
    y=counts[:,:4].sum(1)/counts.sum(1)
    bce=(torch.nn.functional.binary_cross_entropy_with_logits(binary,y,reduction='none')*counts.sum(1)).sum()/counts.sum()
    torch.testing.assert_close(result['ce'],expected);torch.testing.assert_close(result['bce'],bce)
    torch.testing.assert_close(result['total'],bce+.1*expected/LOG_EIGHT)
    uniform=vote_losses(torch.zeros(2,dtype=torch.float64),torch.zeros(2,8,dtype=torch.float64),counts,.1)
    assert float(uniform['normalized_ce'])==pytest.approx(1.)
    with pytest.raises(ValueError,match="eight auxiliary"):vote_losses(binary,None,counts,.1)


@pytest.fixture
def model_rows():
    torch.set_num_threads(1);rng=np.random.default_rng(21)
    frame=pd.DataFrame({'painting':[f'p{i}' for i in range(8)],'image':[f'i{i}' for i in range(8)],'language':'Thai','n':5,'k':3,'p':.6,'split':'train'})
    for column in COUNT_COLUMNS:frame[column]=0
    frame[COUNT_COLUMNS[1]]=3;frame[COUNT_COLUMNS[7]]=2
    visual={name:rng.normal(size=2048).astype(np.float32) for name in frame.image}
    cfg={**CONFIG['training'],'batch_size':4}
    return frame,visual,cfg,vote_rows(frame,visual,['Thai'],torch.device('cpu'))


def test_matched_initialization_gradient_flow_and_exact_bce_rng(model_rows):
    frame,visual,cfg,rows=model_rows
    a,ra,oa=seed_model(['Thai'],2027,cfg,torch.device('cpu'))
    b,rb,ob=seed_model(['Thai'],2027,cfg,torch.device('cpu'))
    assert state_hash(a)==state_hash(b) and torch.equal(ra.get_state(),rb.get_state())
    before=torch.get_rng_state().clone();weights=state_hash(b)
    result=batch_gradients(b,rows,torch.arange(4),.1,1.)
    assert torch.equal(before,torch.get_rng_state()) and state_hash(b)==weights
    assert result['all_logits_valid_gradients'] and result['groups']['shared_representation']['text_norm']>0
    assert all(p.grad is None for p in b.parameters())
    # Independent old BCE-only optimizer loop must produce identical updates.
    a,ra,oa=seed_model(['Thai'],2027,cfg,torch.device('cpu'))
    polarity_epoch(a,rows,ra,oa,cfg,0.)
    b,rb,ob=seed_model(['Thai'],2027,cfg,torch.device('cpu'))
    train_epoch(b,rows,rb,ob,cfg,0.)
    assert state_hash(a)==state_hash(b) and torch.equal(ra.get_state(),rb.get_state())


def test_inference_needs_only_image_language_and_returns_binary_sigmoid(model_rows,monkeypatch):
    frame,visual,cfg,rows=model_rows
    model,_,_=seed_model(['Thai'],2027,cfg,torch.device('cpu'));model.eval()
    ctx=SimpleNamespace(visual=visual,device=torch.device('cpu'),training=cfg)
    with torch.no_grad():expected=model(rows['features'],rows['language'],auxiliary=False)[0].sigmoid().numpy()
    def forbidden(*a):pytest.fail('Auxiliary head ran during sentiment inference')
    monkeypatch.setattr(model.auxiliary,'forward',forbidden)
    values=experiment.predict_binary(model,frame[data.KEYS+['image']],ctx)
    # Full-batch reference versus four-row inference batches can differ by one
    # float32 ulp; this checks the sigmoid path, not cross-batch bit parity.
    np.testing.assert_allclose(values,expected.astype(np.float64),atol=1e-7,rtol=0.)
    unknown=frame[data.KEYS+['image']].assign(language='Unseen')
    assert np.isfinite(experiment.predict_binary(model,unknown,ctx)).all()


def test_rare_stopping_languages_are_retained_without_training_vocabulary_leakage(model_rows):
    frame,visual,cfg,rows=model_rows
    rare=frame.assign(language='OnlyInStopping')
    with pytest.raises(ValueError,match='fitting language'):
        vote_rows(rare,visual,['Thai'],torch.device('cpu'))
    validation=vote_rows(rare,visual,['Thai'],torch.device('cpu'),require_known=False)
    assert len(validation['counts'])==len(rare) and bool(validation['language'].eq(-1).all())


def test_frozen_screen_and_competition_gates():
    table=pd.DataFrame({'pooled_contrast':[.44,.441],'pooled_log_loss':[.71,.71]},index=['D0','D8'])
    folds=pd.DataFrame({'fold':[1,1,3,3],'model':['D0','D8']*2,'contrast':[.44,.442,.44,.44]})
    assert verification.screening_choice(table,folds,CONFIG,False)['status']=='INCOMPLETE'
    assert verification.screening_choice(table,folds,CONFIG,True)['status']=='PILOT_PROMISING'
    assert verification.screening_choice(table,folds,CONFIG,True,True)['status']=='PILOT_UNPROMISING'
    table.loc['D8','pooled_contrast']=.4405
    assert verification.screening_choice(table,folds,CONFIG,True)['status']=='PILOT_INCONCLUSIVE'
    table.loc['D8','pooled_contrast']=.435
    assert verification.screening_choice(table,folds,CONFIG,True)['status']=='PILOT_UNPROMISING'
    assert CONFIG['screen_folds']==[1,3] and CONFIG['screen_seeds']==[2027,2028]


@pytest.fixture
def vote_workflow(tiny_workflow,monkeypatch):
    old,previous,_=tiny_workflow
    train=previous.train.copy()
    for column in COUNT_COLUMNS:train[column]=0
    train[COUNT_COLUMNS[2]]=train.k;train[COUNT_COLUMNS[7]]=train.n-train.k
    args=SimpleNamespace(**{**vars(old),'run_dir':old.run_dir.parent/'vote','exp018_dir':old.run_dir,
        'exp019_dir':old.run_dir.parent/'old19','exp020_dir':old.run_dir.parent/'old20'})
    cfg={**CONFIG,'training':{**CONFIG['training'],'cv_folds':2,'seeds':[2027,2028],'max_epochs':1,'patience':1},
         'screen_folds':[1,2],'screen_seeds':[2027],'pilot_warmup_epochs':0,'pilot_timed_epochs':1,'bootstrap_replicates':50,'posterior_samples':128}
    aid={'fixture':'audit21'};identity={'fixture':'training21'}
    ctx=SimpleNamespace(config=cfg,training={**cfg['training'],'bootstrap_replicates':50,'bootstrap_seed':21},
        identity=identity,audit_identity=aid,train=train,reference=previous.reference,device=torch.device('cpu'),visual=concatenate_features(previous.banks))
    for module in (experiment,verification):monkeypatch.setattr(module,'context',lambda *a:ctx)
    monkeypatch.setattr(audit_module,'input_context',lambda *a:SimpleNamespace(config=cfg,identity=aid,train=train))
    for directory,models in ((args.exp018_dir,['M1','M3']),(args.exp019_dir,['C2']),(args.exp020_dir,['A8','E8'])):
        (directory/'cv').mkdir(parents=True,exist_ok=True)
        (directory/'cv/fits.json').write_text(json.dumps([{'model':m,'fold':f,'seed':2027,'fit_seconds':.01,'selection_epochs':1,'selected_epoch':1} for m in models for f in [1,2]]))
    stage=args.run_dir/'preflight';stage.mkdir(parents=True)
    protected={str(p.resolve()):file_sha256(p) for root in (args.exp017_dir,args.exp018_dir,args.exp019_dir,args.exp020_dir) for p in root.rglob('*') if p.is_file()}
    (stage/'protected_sha256.json').write_text(json.dumps(protected));seal(stage,aid,{'seconds':.01})
    audit_module.run_audit(args)
    return args,ctx


def test_screen_resume_verify_gate_and_full_cv_reuse(vote_workflow,monkeypatch):
    args,ctx=vote_workflow;experiment.run_pilot(args)
    with pytest.raises(StageBlocked):experiment.run_study(args,screen=False)
    experiment.run_study(args,screen=True)
    paths=list((args.run_dir/'work').rglob('student.pt'));assert len(paths)==4
    old={p:file_sha256(p) for p in paths}
    original=verification.screening_choice
    # Synthetic resource-allocation outcome solely to exercise full-CV reuse.
    def promising(*a,**k):return {**original(*a,**k),'status':'PILOT_PROMISING'}
    monkeypatch.setattr(verification,'screening_choice',promising)
    verification.run_verify(args,screen=True)
    experiment.run_study(args,screen=False)
    assert len(list((args.run_dir/'work').rglob('student.pt')))==8
    assert all(file_sha256(p)==digest for p,digest in old.items())
    monkeypatch.setattr(experiment,'fit_model',lambda *a:pytest.fail('Refitted completed checkpoint'))
    experiment.run_study(args,screen=True);experiment.run_study(args,screen=False)
    verification.run_verify(args,screen=False)
    assert verification.verified_decision(args,screen=False)['status'] in {'PASS','COMPETITION_NO_GO'}
    assert read_sealed(args.run_dir/'verification',ctx.identity)['checkpoints_replayed']==8
    stored=pd.read_csv(args.run_dir/'cv/predictions.csv',float_precision='round_trip')
    base=stored[stored.model.eq('R0')][data.KEYS+['p_positive']]
    expected=blend(base,stored[stored.model.eq('D8')][data.KEYS+['p_positive']],.5)
    pd.testing.assert_frame_equal(expected.reset_index(drop=True),stored[stored.model.eq('ED8')][data.KEYS+['p_positive']].reset_index(drop=True))
    assert not (args.run_dir/'confirm').exists() and not (args.run_dir/'dev').exists()


def test_interrupted_screen_retains_seeds_and_failed_replay_blocks_cv(vote_workflow,monkeypatch):
    args,ctx=vote_workflow;experiment.run_pilot(args)
    original=experiment.fit_seed
    def stop(*a,**kw):
        if a[4]=='D8':raise BudgetStop('fixture interruption')
        return original(*a,**kw)
    monkeypatch.setattr(experiment,'fit_seed',stop)
    with pytest.raises(BudgetStop):experiment.run_study(args,screen=True)
    path=args.run_dir/'work/fold_1/D0_2027/student.pt';digest=file_sha256(path)
    monkeypatch.setattr(experiment,'fit_seed',original);experiment.run_study(args,screen=True)
    assert file_sha256(path)==digest
    verification.run_verify(args,screen=True)
    monkeypatch.setattr(verification,'_verify',lambda *a:(_ for _ in ()).throw(BudgetStop('later replay')))
    with pytest.raises(BudgetStop):verification.run_verify(args,screen=True)
    with pytest.raises(StageBlocked,match='Latest replay'):verification.screen_decision(args)


def test_nonpromising_screen_blocks_full_fitting_and_corruption_is_rejected(vote_workflow,monkeypatch):
    args,ctx=vote_workflow;experiment.run_pilot(args);experiment.run_study(args,screen=True)
    monkeypatch.setattr(verification,'screen_decision',lambda *a:{'status':'PILOT_INCONCLUSIVE'})
    with pytest.raises(StageBlocked,match='prohibited'):experiment.run_study(args,screen=False)
    assert not (args.run_dir/'cv').exists()
    next((args.run_dir/'work').rglob('student.pt')).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):experiment.run_study(args,screen=True)
