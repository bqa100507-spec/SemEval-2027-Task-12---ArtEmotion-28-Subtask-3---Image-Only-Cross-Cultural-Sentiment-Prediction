"""Tiny local HF visual transformer; never downloads pretrained weights."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection

from st3 import data
from st3.caption_data import require_isolation
from st3.gradient_training import state_hash
from st3.polarity_workflow import ExecutionBudget,StageBlocked,read_sealed,seal
from st3.visual_feasibility import choose_microbatch,runtime_projection
from st3.visual_gates import competition_gate,screening_gate
from st3.visual_groups import PaintingGroups,fusion,grouped_step,make_head,train_epoch
from st3.visual_lora import VisualLoRA,adapter_state,load_visual
from st3.visual_lora_audit import gradient_audit
from st3.visual_training import load_checkpoint,predict,save_checkpoint,update_norm
from st3.vote_inputs import COUNT_COLUMNS,empirical_distributions
from st3.vote_model import vote_losses

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'configs/exp022.json').read_text())


@pytest.fixture
def tiny():
    torch.set_num_threads(1);torch.manual_seed(45)
    cfg=copy.deepcopy(CONFIG);cfg['clip'].update(vision_width=16);cfg['expected_adapter_parameters']=6144
    model=CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=12,
        num_attention_heads=4,image_size=32,patch_size=16,projection_dim=512)).eval()
    pixels=torch.randn(4,3,32,32)
    with torch.no_grad():original=torch.nn.functional.normalize(model(pixels).image_embeds,dim=1)
    vision=VisualLoRA(model,cfg);records=[]
    for i in range(4):
        for j in range(i%3+1):
            counts=np.zeros(8,dtype=int);counts[i%8]=j+2;counts[(i+4)%8]=4
            n=counts.sum();k=counts[:4].sum()
            records.append({'painting':str(i),'image':f'{i}.jpg','language':f'L{j}','n':n,'k':k,'p':k/n,'split':'train',
                            **dict(zip(COUNT_COLUMNS,counts))})
    frame=pd.DataFrame(records)
    banks=[dict(zip(frame.image.unique(),original.numpy()))]
    banks.extend({f'{i}.jpg':torch.nn.functional.normalize(torch.randn(768),dim=0).numpy() for i in range(4)} for _ in range(2))
    calls=[]
    def reader(names,device):
        calls.extend(names);return pixels[[int(n.split('.')[0]) for n in names]].to(device)
    ctx=SimpleNamespace(config=cfg,device=torch.device('cpu'),banks=banks,pixels=reader)
    return ctx,vision,frame,original,pixels,calls


def test_pins_and_missing_original_checkpoint_are_fail_closed(tmp_path):
    assert CONFIG['clip']['revision']=='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
    assert CONFIG['rank']==8 and CONFIG['alpha']==16 and CONFIG['clip']['image_size']==224
    assert CONFIG['expected_adapter_parameters']==294912 and CONFIG['targets']==['q_proj','v_proj']
    with pytest.raises(FileNotFoundError,match='pinned CLIP-B'):load_visual(tmp_path,CONFIG,'cpu')


def test_zero_lora_exact_visual_only_geometry_and_trainable_names(tiny):
    ctx,vision,_,original,pixels,_=tiny
    torch.testing.assert_close(vision(pixels),original,atol=1e-7,rtol=1e-6)
    assert len(vision.targets)==24 and len(vision.adapters())==48
    assert sum(p.numel() for p in vision.adapters().values())==6144
    assert all('vision_model.encoder.layers.' in n and '.self_attn.' in n for n in vision.targets)
    assert {n for n,p in vision.named_parameters() if p.requires_grad}==set(vision.adapters())
    assert not any('text' in n for n,_ in vision.named_modules())
    torch.testing.assert_close(vision(pixels).norm(dim=1),torch.ones(4))


def test_fusion_exact_and_gradients_only_in_live_clip(tiny):
    _,vision,_,_,pixels,_=tiny
    clip=vision(pixels);sig=torch.randn(4,768);large=torch.randn(4,768)
    value=fusion(clip,sig,large)
    assert value.shape==(4,2048) and value.requires_grad
    torch.testing.assert_close(value,torch.cat([clip,sig,large],1)/np.float32(np.sqrt(3)))
    value.square().sum().backward()
    assert sig.grad is None and large.grad is None
    assert all(p.grad is None for n,p in vision.named_parameters() if n not in vision.adapters())
    with pytest.raises(ValueError,match='detached'):fusion(clip,sig.requires_grad_(),large)


@pytest.mark.parametrize('condition',['F0','F8','L0','L8'])
def test_real_gradient_update_and_pretrained_immutability(tiny,condition):
    ctx,vision,frame,_,_,_=tiny;rows=PaintingGroups(frame,sorted(frame.language.unique()))
    head,opt,scaler=make_head(ctx,vision,sorted(frame.language.unique()),2027,condition)
    base=vision.frozen_digest();initial=adapter_state(vision)
    first=gradient_audit(ctx,vision,head,rows,np.array([0,1]),condition)
    for _ in range(3):grouped_step(ctx,vision,head,rows,np.arange(4),2,condition,opt,scaler)
    last=gradient_audit(ctx,vision,head,rows,np.array([0,1]),condition)
    assert vision.frozen_digest()==base
    delta=update_norm(vision,initial)
    if condition.startswith('L'):
        assert delta>0 and last['clip_features_require_grad']
        assert last['lora']['bce_norm']>0 and last['lora']['text_norm']>0
        assert all(v==0 for n,v in first['per_adapter_bce_norms'].items() if n.endswith('.a'))
        assert any(v>0 for n,v in last['per_adapter_bce_norms'].items() if n.endswith('.a'))
        assert last['lora']['scaled_text_norm']>0 if condition=='L8' else last['lora']['scaled_text_norm']==0
    else:assert delta==0 and last['lora'] is None


def test_matched_initialization_batches_and_inference_contract(tiny):
    ctx,vision,frame,_,_,_=tiny;hashes=[];predictions=[];orders=[]
    for condition in ('F0','F8','L0','L8'):
        head,opt,scaler=make_head(ctx,vision,sorted(frame.language.unique()),2027,condition)
        hashes.append(state_hash(head));rows=PaintingGroups(frame,head.languages)
        orders.append(rows.order(2027,1));predictions.append(predict(ctx,vision,head,frame,condition,2)[0])
    assert len(set(hashes))==1
    for order in orders:np.testing.assert_array_equal(order,orders[0])
    for pred in predictions:np.testing.assert_allclose(pred.p_positive,predictions[0].p_positive,atol=1e-6,rtol=0.)
    # Removing every target column leaves prediction unchanged. Unknown language uses shared head.
    head.eval();unseen=frame[data.KEYS+['image']].copy();unseen.language='UNKNOWN'
    unseen=unseen.drop_duplicates(data.KEYS)
    p,_=predict(ctx,vision,head,unseen,'L0',2)
    renamed=unseen.copy();renamed.painting='renamed-'+renamed.painting
    q,_=predict(ctx,vision,head,renamed,'L0',2)
    np.testing.assert_array_equal(p.p_positive,q.p_positive)
    assert p.p_positive.std()>0


def test_grouped_each_image_once_and_all_language_rows_once(tiny):
    ctx,vision,frame,_,_,calls=tiny;head,opt,scaler=make_head(ctx,vision,sorted(frame.language.unique()),2027,'L8')
    rows=PaintingGroups(frame,head.languages)
    info=train_epoch(ctx,vision,head,rows,2,'L8',opt,scaler,2027,1,ExecutionBudget(60))
    assert len(calls)==4 and len(set(calls))==4 and info['language_pairs']==len(frame)
    batch=rows.batch(np.array([0,1,2,3]),ctx.device)
    assert sorted(batch['rows'])==list(range(len(frame)))
    z=torch.randn(4,2048,requires_grad=True);z[batch['gather']].sum().backward()
    np.testing.assert_array_equal(z.grad[:,0].numpy(),[len(g) for g in rows.groups])


def test_accumulation_vote_denominator_matches_full_batch(tiny):
    ctx,vision,frame,_,_,_=tiny;ctx.config['training']['dropout']=0.
    states=[]
    for micro in (4,2,1):
        head,opt,scaler=make_head(ctx,vision,sorted(frame.language.unique()),2027,'F8')
        # Compare the accumulated gradient with SGD; Adam's near-zero division
        # amplifies harmless reduction roundoff and is not a weighting test.
        opt=torch.optim.SGD(head.parameters(),lr=.001)
        rows=PaintingGroups(frame,head.languages)
        grouped_step(ctx,vision,head,rows,np.arange(4),micro,'F8',opt,scaler)
        states.append({k:v.detach().clone() for k,v in head.state_dict().items()})
    for state in states[1:]:
        for k in state:torch.testing.assert_close(state[k],states[0][k],atol=2e-7,rtol=1e-4)
    d,y=empirical_distributions(frame);counts=torch.tensor(frame[COUNT_COLUMNS].to_numpy(),dtype=torch.float32)
    b=torch.randn(len(frame));e=torch.randn(len(frame),8);loss=vote_losses(b,e,counts,.1)
    expected=(torch.nn.functional.binary_cross_entropy_with_logits(b,torch.tensor(y,dtype=torch.float32),reduction='none')*counts.sum(1)).sum()/counts.sum()
    torch.testing.assert_close(loss['bce'],expected)
    torch.testing.assert_close(loss['total'],expected+.1*loss['ce']/np.log(8))


@pytest.mark.parametrize('negative',[False,True])
def test_auxiliary_polarity_targets_reach_visual_adapters(tiny,negative):
    ctx,vision,frame,_,_,_=tiny;counts=np.zeros((len(frame),8));counts[:,7 if negative else 0]=5
    frame=frame.copy();frame[COUNT_COLUMNS]=counts.astype(int);frame['n']=5;frame['k']=0 if negative else 5;frame['p']=0. if negative else 1.
    head,_,_=make_head(ctx,vision,sorted(frame.language.unique()),2027,'L8')
    audit=gradient_audit(ctx,vision,head,PaintingGroups(frame,head.languages),np.array([0,1]),'L8')
    assert audit['lora']['bce_norm']>0 and audit['lora']['text_norm']>0


def test_memory_sequence_margin_and_oom_are_explicit():
    attempted=[]
    def attempt(n):
        attempted.append(n)
        if n==4:raise torch.cuda.OutOfMemoryError('fixture')
        return {'peak_cuda_reserved_bytes':95 if n==2 else 60}
    size,trials=choose_microbatch([4,2,1],attempt,100,20)
    assert size==1 and attempted==[4,2,1]
    assert [r['status'] for r in trials]==['CUDA_OOM','INSUFFICIENT_MARGIN','PASS']
    size,trials=choose_microbatch([4,2,1],lambda n:{'peak_cuda_reserved_bytes':99},100,20)
    assert size is None and all(r['status']!='PASS' for r in trials)


def test_amp_retry_preserves_batch_rng_and_never_silently_skips(tiny,monkeypatch):
    from st3 import visual_groups as groups
    ctx,*_=tiny;ctx.config['amp_max_retries']=1;seen=[]
    def attempt(*args):
        seen.append(float(torch.rand(())))
        return None if len(seen)==1 else {'finite':True}
    monkeypatch.setattr(groups,'_grouped_attempt',attempt)
    result=groups.grouped_step(ctx,None,None,None,None,4,'F0',None,None)
    assert result['amp_retries']==1 and seen[0]==seen[1]
    monkeypatch.setattr(groups,'_grouped_attempt',lambda *args:None)
    with pytest.raises(ValueError,match='bounded AMP'):groups.grouped_step(ctx,None,None,None,None,4,'F0',None,None)


def test_production_reference_exact_when_local_artifact_is_present():
    from st3.baseline import file_sha256
    path=Path(__file__).resolve().parents[1]/'runs/exp010_clip_large/dev/predictions/E50+L/predictions.csv'
    if not path.exists():pytest.skip('Local protected production artifact is not distributed in Git')
    assert len(pd.read_csv(path))==849
    assert file_sha256(path)=='162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3'


def gate_fixture():
    names=['R0','F0','F8','L0','L8','EF0','EF8','EL0','EL8']
    summary=pd.DataFrame({'pooled_contrast':[.45]*9,'pooled_log_loss':[.72]*9,'log_loss_mean':[.72]*9,'fold_wins':[3]*9},index=names)
    folds=pd.DataFrame([{'fold':f,'model':n,'contrast':.45} for f in (1,3) for n in names])
    return summary,folds


def test_frozen_screen_gates_distinguish_backbone_auxiliary_and_inconclusive():
    s,f=gate_fixture();assert screening_gate(s,f,CONFIG,verified=True,healthy=True)['status']=='SCREEN_INCONCLUSIVE'
    s.loc['L0','pooled_contrast']=.452;s.loc['L8','pooled_contrast']=.454
    f.loc[f.model.eq('L0'),'contrast']=.452;f.loc[f.model.eq('L8'),'contrast']=.454
    result=screening_gate(s,f,CONFIG,verified=True,healthy=True)
    assert result['lora_promising'] and result['auxiliary_promising']
    assert not screening_gate(s,f,CONFIG,verified=False,healthy=True)['lora_promising']
    s.loc[['L0','L8'],'pooled_contrast']=.42;s.loc[['L0','L8'],'pooled_log_loss']=.75
    assert screening_gate(s,f,CONFIG,verified=True,healthy=True)['status']=='SCREEN_UNPROMISING'
    assert competition_gate(s,CONFIG['training'],verified=True,healthy=True)['status']=='COMPETITION_NO_GO'


def test_checkpoint_resume_replay_and_tamper_detection(tiny,tmp_path):
    ctx,vision,frame,_,_,_=tiny;head,opt,scaler=make_head(ctx,vision,sorted(frame.language.unique()),2027,'L8')
    rows=PaintingGroups(frame,head.languages)
    for _ in range(2):grouped_step(ctx,vision,head,rows,np.arange(4),2,'L8',opt,scaler)
    expected=predict(ctx,vision,head,frame,'L8',2)[0]
    stage=tmp_path/'model';stage.mkdir();save_checkpoint(stage/'student.pt',head,vision)
    seal(stage,{'fixture':True},{'complete':True});read_sealed(stage,{'fixture':True})
    restored=load_checkpoint(ctx,vision,stage/'student.pt',2027,'L8')
    pd.testing.assert_frame_equal(predict(ctx,vision,restored,frame,'L8',2)[0],expected,check_exact=True)
    (stage/'partial.tmp').write_text('partial')
    with pytest.raises(ValueError,match='partial or extra'):read_sealed(stage,{'fixture':True})
    with pytest.raises(ValueError):require_isolation(frame,frame)


def test_tiny_nested_miniscreen_reuses_four_fits_and_failed_replay_revokes_gate(tiny,tmp_path,monkeypatch):
    from st3 import visual_experiment as study,visual_verification as verify,visual_lora_inputs as input_module
    ctx,vision,_,original,_,_=tiny
    # Unwrap a fresh copy for the loader; this fixture is random tiny HF CLIP,
    # not a pretrained model or a full dataset training job.
    raw=copy.deepcopy(vision.model)
    for layer in raw.vision_model.encoder.layers:
        layer.self_attn.q_proj=layer.self_attn.q_proj.base
        layer.self_attn.v_proj=layer.self_attn.v_proj.base
    records=[]
    for i in range(100):
        for language in range(3):
            k=(i+language)%6;counts=[k,0,0,0,5-k,0,0,0]
            records.append({'painting':f'p{i:03}','image':f'{i%4}.jpg','language':f'L{language}',
                'fold':i%5+1,'n':5,'k':k,'p':k/5,'split':'train',**dict(zip(COUNT_COLUMNS,counts))})
    ctx.train=pd.DataFrame(records)
    # Isolation is by actual painting AND image. Give each synthetic painting an
    # independent lookup name while deliberately reusing four numerical tensors.
    ctx.train['image']=ctx.train.painting+'.jpg'
    ctx.train['image_sha256']=ctx.train.painting
    for bank in ctx.banks:
        for i in range(100):bank[f'p{i:03}.jpg']=bank[f'{i%4}.jpg']
    old_reader=ctx.pixels
    ctx.pixels=lambda names,device:old_reader([f'{int(n[1:4])%4}.jpg' for n in names],device)
    ctx.reference=ctx.train[data.KEYS+['fold']].assign(p_positive=.3+ctx.train.language.str[-1].astype(int)*.15)
    ctx.config['training'].update(max_epochs=1,patience=1,bootstrap_replicates=32)
    ctx.config['bootstrap_replicates']=32;ctx.identity={'synthetic_fixture':True};ctx.input_dir=tmp_path/'inputs';ctx.input_dir.mkdir()
    historical=pd.concat([ctx.reference.assign(model=m,seed=2027) for m in ('D0','D8')])
    historical.to_csv(ctx.input_dir/'historical_seeds.csv',index=False)
    args=SimpleNamespace(run_dir=tmp_path,allow_full_cv=False,max_seconds=120.)
    pre=tmp_path/'preflight';pre.mkdir();(pre/'historical_sha256.json').write_text('{}')
    seal(pre,ctx.identity,{'status':'PASS','microbatch':4,'base_sha256':vision.frozen_digest()})
    pilot=tmp_path/'pilot';pilot.mkdir();seal(pilot,ctx.identity,{'status':'PILOT_PASS','runtime_projection':{
        'miniscreen_seconds':1.,'screen_seconds':4.,'local_screen_within_gpu_budget':True}})
    monkeypatch.setattr(input_module,'inputs',lambda a:ctx);monkeypatch.setattr(verify,'inputs',lambda a:ctx)
    loader=lambda a:(ctx,copy.deepcopy(raw))
    monkeypatch.setattr(study,'load_context',loader);monkeypatch.setattr(verify,'load_context',loader)
    study.run_study(args,'mini');assert len(list((tmp_path/'work').glob('fold_*/*/run.json')))==4
    monkeypatch.setattr(study,'fit_model',lambda *a:pytest.fail('Completed model must be reused'))
    study.run_study(args,'mini');verify.run_verify(args,'mini')
    decision=verify.verified_decision(args,'mini',ctx)
    assert decision['status'] in ('MINISCREEN_SANITY_PASS','MINISCREEN_STOP_REVIEW')
    prediction=next((tmp_path/'work').glob('fold_*/*/predictions.csv'))
    prediction.write_bytes(prediction.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='checksum mismatch'):verify.run_verify(args,'mini')
    with pytest.raises(StageBlocked,match='Latest replay'):verify.verified_decision(args,'mini',ctx)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA-only tiny FP16 head test')
def test_tiny_cuda_mixed_precision_finite(tiny):
    ctx,vision,frame,_,_,_=tiny;ctx.device=torch.device('cuda');vision.to(ctx.device)
    head,opt,scaler=make_head(ctx,vision,sorted(frame.language.unique()),2027,'L8')
    metric=grouped_step(ctx,vision,head,PaintingGroups(frame,head.languages),np.arange(4),2,'L8',opt,scaler)
    assert np.isfinite(metric['pre_clip_norm'])
