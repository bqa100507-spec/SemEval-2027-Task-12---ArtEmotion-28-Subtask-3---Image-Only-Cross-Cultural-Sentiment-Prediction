"""Unique-painting batches and vote-normalized microbatch accumulation."""

import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .data import image_path
from .vote_inputs import COUNT_COLUMNS,empirical_distributions
from .vote_model import seed_model,vote_losses


class PaintingGroups:
    def __init__(self,frame,languages,*,labels=True):
        self.frame=frame.sort_values(['painting','language']).reset_index(drop=True)
        if self.frame.groupby('painting').image.nunique().gt(1).any():raise ValueError('Painting maps to multiple images')
        self.groups=list(self.frame.groupby('painting',sort=True).indices.values())
        self.images=[self.frame.image.iloc[rows[0]] for rows in self.groups]
        self.languages=pd.Index(languages).get_indexer(self.frame.language)
        self.labels=labels
        if labels:
            empirical_distributions(self.frame)
            self.counts=self.frame[COUNT_COLUMNS].to_numpy(dtype=np.float32)
            self.votes=np.array([self.counts[rows].sum() for rows in self.groups])

    def batch(self,indices,device):
        groups=[self.groups[int(i)] for i in indices];rows=np.concatenate(groups)
        gather=np.repeat(np.arange(len(groups)),[len(g) for g in groups])
        return {'names':[self.images[int(i)] for i in indices],'rows':rows,
            'gather':torch.as_tensor(gather,device=device),'language':torch.as_tensor(self.languages[rows],device=device),
            'counts':torch.as_tensor(self.counts[rows],device=device) if self.labels else None}

    def order(self,seed,epoch):
        return np.random.default_rng(seed+1009*epoch).permutation(len(self.groups))


class PixelReader:
    def __init__(self,root,processor):self.root=root;self.processor=processor
    def __call__(self,names,device):
        images=[]
        for name in names:
            with Image.open(image_path(self.root,name)) as image:images.append(image.convert('RGB'))
        return self.processor(images=images,return_tensors='pt')['pixel_values'].to(device)


def fusion(clip,siglip,large):
    if clip.shape[1]!=512 or siglip.shape[1]!=768 or large.shape[1]!=768:raise ValueError('Expected 512+768+768 features')
    if siglip.requires_grad or large.requires_grad:raise ValueError('Frozen feature blocks must be detached')
    return torch.cat([clip,siglip,large],dim=1)/np.sqrt(3.)


def features(ctx,vision,names,adapted):
    tick=time.perf_counter()
    pixels=ctx.pixels(names,ctx.device) if adapted else None
    loading=time.perf_counter()-tick;tick=time.perf_counter()
    clip=vision(pixels) if adapted else torch.as_tensor(np.stack([ctx.banks[0][n] for n in names]),device=ctx.device)
    if adapted and torch.is_grad_enabled() and not clip.requires_grad:raise ValueError('Adapted image graph was detached')
    siglip=torch.as_tensor(np.stack([ctx.banks[1][n] for n in names]),device=ctx.device).detach()
    large=torch.as_tensor(np.stack([ctx.banks[2][n] for n in names]),device=ctx.device).detach()
    if ctx.device.type=='cuda':torch.cuda.synchronize()
    return fusion(clip,siglip,large),loading,time.perf_counter()-tick


def make_head(ctx,vision,languages,seed,condition):
    vision.reset(seed,condition.startswith('L'))
    head,_,_=seed_model(languages,seed,ctx.config['training'],ctx.device)
    # seed_model resets global RNG AFTER adapter reset; adapter allocation cannot
    # perturb the matched task head or head dropout sequence.
    groups=[{'params':list(head.parameters()),'lr':ctx.config['head_lr']}]
    if condition.startswith('L'):groups.append({'params':list(vision.adapters().values()),'lr':ctx.config['lora_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=ctx.config['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=ctx.device.type=='cuda',init_scale=ctx.config['amp_initial_scale'])
    return head,optimizer,scaler


def _grouped_attempt(ctx,vision,head,rows,paintings,microbatch,condition,optimizer,scaler):
    head.train();vision.model.eval();optimizer.zero_grad(set_to_none=True)
    total_votes=float(rows.votes[paintings].sum());totals=np.zeros(3)
    timing={key:0. for key in ('image_loading','encoder_forward','head_forward','head_backward','encoder_backward','backward')}
    for offset in range(0,len(paintings),microbatch):
        group=rows.batch(paintings[offset:offset+microbatch],ctx.device)
        vectors,loading,forward=features(ctx,vision,group['names'],condition.startswith('L'))
        timing['image_loading']+=loading;timing['encoder_forward']+=forward;tick=time.perf_counter()
        with torch.autocast(device_type=ctx.device.type,dtype=torch.float16,enabled=ctx.device.type=='cuda'):
            binary,aux=head(vectors[group['gather']],group['language'],auxiliary=condition.endswith('8'))
        losses=vote_losses(binary.float(),aux.float() if aux is not None else None,group['counts'],.1 if condition.endswith('8') else 0.)
        weight=float(group['counts'].sum())/total_votes
        if ctx.device.type=='cuda':torch.cuda.synchronize()
        timing['head_forward']+=time.perf_counter()-tick;tick=time.perf_counter()
        boundary=[]
        def mark_encoder_backward(gradient):
            if ctx.device.type=='cuda':torch.cuda.synchronize()
            boundary.append(time.perf_counter())
            return gradient
        hook=vectors.register_hook(mark_encoder_backward) if vectors.requires_grad else None
        try:scaler.scale(losses['total']*weight).backward()
        finally:
            if hook is not None:hook.remove()
        if ctx.device.type=='cuda':torch.cuda.synchronize()
        end=time.perf_counter();timing['backward']+=end-tick
        timing['head_backward']+=(boundary[0] if boundary else end)-tick
        timing['encoder_backward']+=end-boundary[0] if boundary else 0.
        totals+=float(group['counts'].sum())*np.array([float(losses[k].detach()) for k in ('bce','ce','normalized_ce')])
    scaler.unscale_(optimizer)
    parameters=[p for g in optimizer.param_groups for p in g['params']]
    norm=torch.nn.utils.clip_grad_norm_(parameters,ctx.config['gradient_clip'],error_if_nonfinite=False)
    if not bool(torch.isfinite(norm)):
        if not scaler.is_enabled():raise ValueError('Nonfinite FP32 gradients')
        scaler.step(optimizer);scaler.update()  # found_inf skips the unsafe update
        return None
    scaler.step(optimizer);scaler.update()
    return {'bce_sum':totals[0],'ce_sum':totals[1],'normalized_ce_sum':totals[2],'votes':total_votes,
        'pre_clip_norm':float(norm),'clipped':bool(norm>ctx.config['gradient_clip']),**timing}


def grouped_step(ctx,vision,head,rows,paintings,microbatch,condition,optimizer,scaler):
    # Retry AMP overflow with the SAME dropout RNG and effective painting batch.
    # No row is silently skipped and no optimizer update occurs on overflow.
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(ctx.device) if ctx.device.type=='cuda' else None
    started=time.perf_counter()
    for retry in range(ctx.config['amp_max_retries']+1):
        if retry:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,ctx.device)
        result=_grouped_attempt(ctx,vision,head,rows,paintings,microbatch,condition,optimizer,scaler)
        if result is not None:
            return {**result,'amp_retries':retry,'step_seconds':time.perf_counter()-started}
    raise ValueError('Repeated nonfinite gradients despite bounded AMP scale backoff')


def train_epoch(ctx,vision,head,rows,microbatch,condition,optimizer,scaler,seed,epoch,budget):
    start=time.perf_counter();order=rows.order(seed,epoch);traces=[];seen=[]
    for at in range(0,len(order),ctx.config['effective_paintings']):
        budget.check(f'{condition} epoch {epoch}',epoch) if at==0 else None
        selected=order[at:at+ctx.config['effective_paintings']]
        traces.append(grouped_step(ctx,vision,head,rows,selected,microbatch,condition,optimizer,scaler));seen.extend(selected.tolist())
        if at%128==0 and at:budget.check(f'{condition} training boundary',-1)
    if sorted(seen)!=list(range(len(rows.groups))):raise ValueError('Painting coverage is not exactly one pass')
    votes=sum(t['votes'] for t in traces)
    return {'train_bce':sum(t['bce_sum'] for t in traces)/votes,'train_ce':sum(t['ce_sum'] for t in traces)/votes,
        'clipping_frequency':float(np.mean([t['clipped'] for t in traces])),
        'amp_retries':sum(t['amp_retries'] for t in traces),
        'seconds':time.perf_counter()-start,'paintings':len(seen),'language_pairs':len(rows.frame),
        **{key:sum(t[key] for t in traces) for key in ('image_loading','encoder_forward','head_forward','head_backward','encoder_backward','backward')}}
