"""Paired full-epoch execution benchmark, never outer-fold scoring/training."""
import hashlib
import time

import numpy as np
import torch

from .cli import json_output
from .contrast_study import new_output
from .emotion_experiment import memory_report
from .pixel_cache import RamPixels
from .polarity_workflow import ExecutionBudget,read_sealed,seal
from .visual_feasibility import load_context,representative,runtime_projection
from .visual_groups import PaintingGroups,make_head,grouped_step,train_epoch,features
from .visual_lora import VisualLoRA,adapter_state
from .visual_training import predict,update_norm
from .vote_model import vote_losses


def fingerprint(value):
    digest=hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            digest.update(str(item.dtype).encode());digest.update(str(tuple(item.shape)).encode());digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=str):digest.update(str(key).encode());visit(item[key])
        elif isinstance(item,(tuple,list)):
            for part in item:visit(part)
        else:digest.update(repr(item).encode())
    visit(value);return digest.hexdigest()


def rng():return [torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []]


def state(vision,head,optimizer,scaler):
    return fingerprint([head.state_dict(),adapter_state(vision),
        {n:p.grad for n,p in head.named_parameters()},
        {n:p.grad for n,p in vision.adapters().items()},optimizer.state_dict(),scaler.state_dict(),rng()])


def loss_gradients(ctx,vision,head,rows,selected):
    head.eval();batch=rows.batch(selected,ctx.device)
    z,_,_=features(ctx,vision,batch['names'],True)
    binary,aux=head(z[batch['gather']],batch['language'],auxiliary=True)
    losses=vote_losses(binary,aux,batch['counts'],.1)
    objects=tuple(head.parameters())+tuple(vision.adapters().values())
    parts=[]
    for name in ('bce','normalized_ce'):
        grads=torch.autograd.grad(losses[name],objects,retain_graph=name=='bce',allow_unused=True)
        parts.append(fingerprint([losses[name],grads]))
    return parts


def run_pixel_benchmark(args):
    ctx,model=load_context(args);output=args.run_dir/'pixel_benchmark'
    if output.exists():read_sealed(output,ctx.identity);print('Reused sealed pixel benchmark');return
    budget=ExecutionBudget(args.max_seconds or 900.)
    original=ctx.pixels.reader if isinstance(ctx.pixels,RamPixels) else ctx.pixels
    before=fingerprint(rng())
    cached=ctx.pixels if isinstance(ctx.pixels,RamPixels) else RamPixels(original,ctx.train.image.unique(),
        max_bytes=int(args.pixel_cache_max_gib*1024**3),margin_bytes=int(args.pixel_cache_margin_gib*1024**3))
    if before!=fingerprint(rng()):raise ValueError('Cache construction changed Torch RNG')
    # Every allowed painting checked, including each original batch boundary.
    for at in range(0,len(cached.names),4):
        names=cached.names[at:at+4]
        if not torch.equal(original.cpu(names),cached.cpu(names)):raise ValueError('Pixel tensor parity failed')
    for name in cached.names[:16]:
        if not torch.equal(original.cpu([name]),cached.cpu([name])):raise ValueError('Singleton preprocessing parity failed')
    vision=VisualLoRA(model,ctx.config);base_hash=vision.frozen_digest()
    inner,stopping=representative(ctx);languages=sorted(inner.language.unique());rows=PaintingGroups(inner,languages)
    selected=rows.order(2027,0)[:4];audits={}
    for condition in ('L0','L8'):
        signatures=[];loss_signatures=[]
        for reader in (original,cached):
            ctx.pixels=reader;head,opt,scaler=make_head(ctx,vision,languages,2027,condition)
            loss_signatures.append(loss_gradients(ctx,vision,head,rows,selected))
            initial=adapter_state(vision)
            for _ in range(3):grouped_step(ctx,vision,head,rows,selected,4,condition,opt,scaler)
            if update_norm(vision,initial)<=0:raise ValueError('Zero adapter update')
            signatures.append(state(vision,head,opt,scaler))
            del head,opt,scaler
        if signatures[0]!=signatures[1] or loss_signatures[0]!=loss_signatures[1]:raise ValueError('Gradient/optimizer/RNG parity failed')
        audits[condition]={'three_step_state_equal':True,'bce_ce_gradients_equal':True,'state_sha256':signatures[0]}
    records={mode:{} for mode in ('off','ram')};epoch_signatures={}
    # Counterbalanced order; reset identical heads/adapters/batches for EVERY fit.
    for repeat in range(2):
        for mode in (('off','ram') if repeat==0 else ('ram','off')):
            reader=original if mode=='off' else cached;reader.profile=True;ctx.pixels=reader
            for condition in ('F0','F8','L0','L8'):
                budget.check('paired full epoch',-1)
                head,opt,scaler=make_head(ctx,vision,languages,2027,condition);reader.reset_stats()
                if ctx.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
                metric=train_epoch(ctx,vision,head,rows,4,condition,opt,scaler,2027,1,budget)
                profile=dict(reader.stats);signature=state(vision,head,opt,scaler)
                if condition in epoch_signatures and signature!=epoch_signatures[condition]:raise ValueError('Full-epoch parameters/gradients/optimizer/RNG differ')
                epoch_signatures[condition]=signature
                _,validation=predict(ctx,vision,head,stopping,condition,4,with_loss=True)
                records[mode].setdefault(condition,[]).append({'epoch':metric,'validation':validation,
                    'diagnostic_seconds':0.,'pixel_profile':profile,'state_sha256':signature,'memory':memory_report()})
                print(f'Pixel benchmark {mode} {condition} repeat {repeat+1}: {metric["seconds"]:.3f}s, {961/metric["seconds"]:.2f} paintings/s',flush=True)
                del head,opt,scaler
    if base_hash!=vision.frozen_digest():raise ValueError('Frozen base weights changed')
    projections={};speedup={}
    for mode in records:
        aggregate={}
        for condition,values in records[mode].items():
            aggregate[condition]={**values[0],'epoch':{**values[0]['epoch'],
                'seconds':float(np.median([v['epoch']['seconds'] for v in values]))},
                'validation':{'seconds':float(np.median([v['validation']['seconds'] for v in values]))}}
        projections[mode]=runtime_projection(aggregate,ctx.historical,ctx.config)
    for condition in ('L0','L8'):
        a=np.median([v['epoch']['seconds'] for v in records['off'][condition]])
        b=np.median([v['epoch']['seconds'] for v in records['ram'][condition]])
        speedup[condition]={'original_seconds':float(a),'cached_seconds':float(b),'speedup':float(a/b)}
    with new_output(output) as stage:
        json_output(records,stage/'epochs.json');json_output(projections,stage/'runtime_projection.json')
        seal(stage,ctx.identity,{'status':'PASS','gpu':ctx.identity['gpu'],'all_pixels_bitwise_equal':True,
            'painting_count':len(cached.names),'cache_bytes':cached.bytes,'cache_build_seconds':cached.build_seconds,
            'gradient_audits':audits,'full_epoch_state_equal':True,'epoch_signatures':epoch_signatures,'speedup':speedup,
            'base_sha256':base_hash,'outer_fold_scoring':False})
    print('Measured paired speedup:',speedup,flush=True)
    print('Screen forecasts (hours):',{k:v['screen_gpu_hours'] for k,v in projections.items()},flush=True)
