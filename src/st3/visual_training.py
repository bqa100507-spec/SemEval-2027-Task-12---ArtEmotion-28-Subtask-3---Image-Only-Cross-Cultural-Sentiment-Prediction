"""Nested stopping/refit and pixel-only prediction for the matched experiment."""

import time

import numpy as np
import torch

from . import data
from .caption_data import inner_partition,require_isolation
from .gradient_training import state_hash
from .emotion_experiment import memory_report
from .visual_groups import PaintingGroups,features,make_head,train_epoch
from .visual_lora import adapter_state,load_adapter
from .visual_lora_audit import feature_comparison,gradient_audit
from .vote_model import vote_losses

CONDITIONS=('F0','F8','L0','L8')


@torch.inference_mode()
def predict(ctx,vision,head,frame,condition,microbatch,*,with_loss=False):
    # Rejects every prospective input other than identity lookup and language.
    rows=PaintingGroups(frame if with_loss else frame[data.KEYS+['image']],head.languages,labels=with_loss)
    head.eval();values=np.empty(len(rows.frame),dtype=np.float64);votes=0.;numerator=0.;start=time.perf_counter()
    for at in range(0,len(rows.groups),microbatch):
        group=rows.batch(np.arange(at,min(at+microbatch,len(rows.groups))),ctx.device)
        z,_,_=features(ctx,vision,group['names'],condition.startswith('L'))
        logits,aux=head(z[group['gather']],group['language'],auxiliary=with_loss)
        values[group['rows']]=logits.sigmoid().cpu().numpy()
        if with_loss:
            n=float(group['counts'].sum());votes+=n
            numerator+=float(vote_losses(logits,aux,group['counts'],.1)['bce'])*n
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Invalid inference probabilities')
    return rows.frame[data.KEYS].assign(p_positive=values),{'bce':numerator/votes if votes else None,'seconds':time.perf_counter()-start}


def update_norm(vision,initial):
    return float(torch.stack([(p.detach().cpu()-initial[n]).norm() for n,p in vision.adapters().items()]).norm())


def fit_model(ctx,vision,fit,held,fold,seed,condition,microbatch,budget):
    cfg=ctx.config['training'];inner,stopping=inner_partition(fit,cfg,fold)
    require_isolation(inner,stopping,held);data.require_same_pairs(fit,__import__('pandas').concat([inner,stopping]))
    base=vision.frozen_digest();start=time.perf_counter();curves=[];initial={};best=float('inf');selected=0
    if ctx.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    head,optimizer,scaler=make_head(ctx,vision,sorted(inner.language.unique()),seed,condition)
    initial['selection']=state_hash(head)
    rows=PaintingGroups(inner,head.languages)
    diagnostic_indices=rows.order(seed,0)[:microbatch]
    gradients={'selection_initial':gradient_audit(ctx,vision,head,rows,diagnostic_indices,condition)}
    for epoch in range(1,cfg['max_epochs']+1):
        metric=train_epoch(ctx,vision,head,rows,microbatch,condition,optimizer,scaler,seed,epoch,budget)
        _,val=predict(ctx,vision,head,stopping,condition,microbatch,with_loss=True)
        if not np.isfinite(val['bce']):raise ValueError('Nonfinite stopping BCE')
        curves.append({'phase':'selection','epoch':epoch,**metric,'validation_bce':val['bce'],'validation_seconds':val['seconds']})
        if val['bce']<best:best=val['bce'];selected=epoch
        if epoch-selected>=cfg['patience']:break
    selection_epochs=epoch
    del optimizer,scaler,head
    head,optimizer,scaler=make_head(ctx,vision,sorted(fit.language.unique()),seed,condition)
    initial['refit']=state_hash(head);adapter_initial=adapter_state(vision)
    rows=PaintingGroups(fit,head.languages);indices=rows.order(seed,0)[:microbatch]
    for epoch in range(1,selected+1):
        metric=train_epoch(ctx,vision,head,rows,microbatch,condition,optimizer,scaler,seed,epoch,budget)
        curves.append({'phase':'refit','epoch':epoch,**metric})
    gradients['refit_final']=gradient_audit(ctx,vision,head,rows,indices,condition)
    change=update_norm(vision,adapter_initial)
    if condition.startswith('L') and change<=0:raise ValueError('Zero LoRA updates block publication')
    if condition.startswith('F') and change!=0:raise ValueError('Frozen adapters changed')
    if base!=vision.frozen_digest():raise ValueError('Pretrained vision/projection weights changed')
    names=rows.batch(indices,ctx.device)['names']
    with torch.no_grad():
        live=vision(ctx.pixels(names,ctx.device))
    shift=feature_comparison(live,np.stack([ctx.banks[0][n] for n in names]))
    if not all(bool(torch.isfinite(p).all()) for p in head.parameters()):raise ValueError('Nonfinite task head')
    info={'selected_epoch':selected,'selection_epochs':selection_epochs,'best_validation_bce':best,
        'fit_seconds':time.perf_counter()-start,'initialization_sha256':initial,'base_sha256':base,
        'adapter_update_norm':change,'representation_shift':shift,'memory':memory_report(),
        'fit_paintings':sorted(fit.painting.unique()),'inner_paintings':sorted(inner.painting.unique()),
        'stopping_paintings':sorted(stopping.painting.unique()),'held_paintings':sorted(held.painting.unique()),
        'refit_diagnostic_indices':indices.tolist(),'unseen_language_pairs':int((~held.language.isin(head.languages)).sum())}
    return head,curves,gradients,info


def save_checkpoint(path,head,vision):
    torch.save({'languages':head.languages,'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},
                'adapters':adapter_state(vision)},path)


def load_checkpoint(ctx,vision,path,seed,condition):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    head,optimizer,scaler=make_head(ctx,vision,payload['languages'],seed,condition)
    del optimizer,scaler
    head.load_state_dict(payload['head'],strict=True);load_adapter(vision,payload['adapters'])
    return head.eval()
