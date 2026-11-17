"""Painting-complete Pearson VJPs with bounded activation memory; no encoder changes."""

from contextlib import contextmanager
import copy
import random

import numpy as np
import torch

from .vlm_supervised import adapter_parameters, vote_loss
from .vlm_supervised_training import Trainer, windows


def pearson_loss(logits, positive, votes, groups, epsilon=.01):
    """EXP-004 surrogate, painting-balanced, with exact scorer target eligibility.

    Aggregate in float64 for the small group vectors. Epsilon smooths the predicted
    centered norm only; this is intentionally not the official unsmoothed scorer.
    """
    if logits.ndim!=1 or positive.shape!=logits.shape or votes.shape!=logits.shape or len(groups)!=len(logits):
        raise ValueError("Pearson rows/counts/group dimensions differ")
    if not np.isfinite(epsilon) or epsilon<=0:raise ValueError("Positive finite Pearson epsilon required")
    # Same count validity as BCE. Invalid targets fail rather than silently changing supervision.
    if not torch.isfinite(logits).all() or not torch.isfinite(positive).all() or not torch.isfinite(votes).all():
        raise ValueError("Nonfinite Pearson input")
    if (votes<=0).any() or (positive<0).any() or (positive>votes).any():raise ValueError("Invalid votes")
    q=torch.sigmoid(logits.double());target=positive.double()/votes.double();losses=[]
    for group in dict.fromkeys(groups):
        ids=torch.tensor([i for i,g in enumerate(groups) if g==group],device=logits.device)
        ids=ids[votes[ids]>=5]
        if len(ids)<3:continue
        y=target[ids]
        if bool(y.max()==y.min()):continue
        yc=y-y.mean();qc=q[ids]-q[ids].mean()
        correlation=(qc*(yc/torch.linalg.vector_norm(yc))).sum()/torch.sqrt(qc.square().sum()+epsilon**2)
        losses.append(1-correlation)
    # Keep a connected zero for windows with no eligible paintings.
    result=torch.stack(losses).mean() if losses else logits.double().sum()*0
    if not torch.isfinite(result):raise ValueError("Nonfinite Pearson objective")
    return result,len(losses)


def objective(logits,positive,votes,groups,mode,weight,epsilon=.01):
    if mode not in ("bce_only","bce_pearson") or not np.isfinite(weight) or weight<0:
        raise ValueError("Unknown objective or invalid lambda")
    bce=vote_loss(logits,positive,votes)
    if mode=="bce_only":
        if weight!=0:raise ValueError("BCE-only requires lambda zero")
        pearson=logits.double().sum()*0;eligible=0
    else:pearson,eligible=pearson_loss(logits,positive,votes,groups,epsilon)
    return bce+weight*pearson,{"bce":bce,"pearson":pearson,"eligible":eligible}


def rng_state(devices=()):
    legacy=np.random.get_state()
    return {"torch":torch.get_rng_state().clone(),"cuda":{int(d):torch.cuda.get_rng_state(d).cpu() for d in devices},
            "python":random.getstate(),"numpy":[legacy[0],legacy[1].tolist(),legacy[2],legacy[3],legacy[4]]}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    for device,value in state["cuda"].items():torch.cuda.set_rng_state(value,device)
    random.setstate(state["python"])
    a=state["numpy"];np.random.set_state((a[0],np.asarray(a[1],dtype=np.uint32),a[2],a[3],a[4]))


def buffers(modules):
    return [(b,b.detach().clone()) for module in modules for b in module.buffers()]


def assert_buffers(saved):
    if any(not torch.equal(buffer,value) for buffer,value in saved):
        raise RuntimeError("Mutable training buffers are unsupported by exact two-pass replay")


def two_pass_backward(forward, rows, positive, votes, groups, microbatch, weight, *, epsilon=.01,
                      scaler=None, modules=(), devices=(), check=lambda:None):
    """Exact chain-rule VJP for the SAME microbatched stochastic forward function.

    Pass1 is grad-enabled (same forward kernels), but each micrograph is discarded
    immediately. Only detached logits/RNG states survive. Pass2 replays each chunk,
    rejects unequal logits or mutated buffers, and injects d(full-window loss)/dz.
    No optimizer step is permitted between passes. Memory holds one micrograph.
    """
    if microbatch<1 or not len(rows):raise ValueError("Nonempty batch and positive microbatch required")
    start=rng_state(devices);saved_buffers=buffers(modules);chunks=[];states=[];end=None
    versions=[(p,p._version) for m in modules for p in m.parameters()]
    def unchanged():
        assert_buffers(saved_buffers)
        if any(p._version!=version for p,version in versions):raise RuntimeError("Parameters changed during two-pass forward/replay")
    try:
        for first in range(0,len(rows),microbatch):
            check();states.append(rng_state(devices))
            with torch.enable_grad():z=forward(rows[first:first+microbatch])
            if not z.requires_grad:raise RuntimeError("Model logit graph is disconnected")
            chunks.append(z.detach().clone());del z
            unchanged()
        end=rng_state(devices)
        z=torch.cat(chunks).detach().requires_grad_(True)
        total,parts=objective(z,positive,votes,groups,"bce_pearson",weight,epsilon)
        gradient=torch.autograd.grad(total,z)[0].detach()
        if not torch.isfinite(gradient).all():raise RuntimeError("Nonfinite complete-group logit gradient")
        for i,first in enumerate(range(0,len(rows),microbatch)):
            check();restore_rng(states[i])
            replay=forward(rows[first:first+microbatch])
            if not torch.equal(replay.detach(),chunks[i]):
                raise RuntimeError("Two-pass logits differ; exact replay is not established")
            unchanged()
            # AMP is applied once to this VJP, never to the precomputed gradient.
            local=(replay*gradient[first:first+len(replay)]).sum()
            (scaler.scale(local) if scaler is not None else local).backward()
            del replay,local
            unchanged()
        return {"bce":float(parts["bce"].detach()),"pearson":float(parts["pearson"].detach()),
                "eligible":parts["eligible"],"total":float(total.detach()),"exact_logit_replay":True,
                "forward_passes":2}
    except Exception:
        # Never leave a partially accumulated gradient that could be stepped after a failed audit.
        for module in modules:
            for parameter in module.parameters():parameter.grad=None
        with torch.no_grad():
            for buffer,value in saved_buffers:buffer.copy_(value)
        raise
    finally:restore_rng(end if end is not None else start)


class ContrastTrainer(Trainer):
    """Same head/LoRA optimizer, window ordering and vote denominator as EXP-026."""
    def __init__(self,*args,objective_config,**kwargs):
        super().__init__(*args,**kwargs)
        self.objective_config=copy.deepcopy(objective_config)
        if self.condition!="L0":raise ValueError("EXP-027 compares adapted BCE and adapted BCE+Pearson only")

    def epoch(self,frame,epoch,check):
        c=self.objective_config
        if c["mode"]=="bce_only":
            if c["lambda"]!=0:raise ValueError("BCE-only requires lambda zero")
            return {**super().epoch(frame,epoch,check),"objective_mode":"bce_only","forward_passes":1}
        if c["mode"]!="bce_pearson" or c["lambda"]<=0:raise ValueError("Invalid Pearson treatment")
        import time
        self.head.train();self.backend.model.model.train();tick=time.perf_counter()
        bce_sum=0.;votes=0.;pairs=0;auxiliary=[];heads=[];adapters=[];eligible=0
        for frame_batch in windows(frame,self.config,self.config["seed"]+epoch):
            check();self.optimizer.zero_grad(set_to_none=True)
            rows=list(range(len(frame_batch)))
            def forward(ids):
                batch=frame_batch.iloc[ids]
                with torch.autocast("cuda",dtype=torch.float16,enabled=self.device.type=="cuda"):
                    hidden=self.hidden(batch,train=True)
                    if not hidden.requires_grad:raise RuntimeError("Adapted hidden graph is disconnected")
                    return self.head(hidden,self.head.indices(batch.language,self.device)).float()
            result=two_pass_backward(forward,rows,torch.tensor(frame_batch.k.to_numpy(),device=self.device),
                         torch.tensor(frame_batch.n.to_numpy(),device=self.device),frame_batch.painting.tolist(),
                         self.config["microbatch_pairs"],c["lambda"],epsilon=c["epsilon"],scaler=self.scaler,
                         modules=(self.backend.model,self.head),devices=([self.device.index or 0] if self.device.type=="cuda" else []),check=check)
            self.scaler.unscale_(self.optimizer)
            norm=lambda ps:float(torch.sqrt(sum((p.grad.float().square().sum() for p in ps if p.grad is not None),torch.zeros((),device=self.device))))
            heads.append(norm(self.head.parameters()));adapters.append(norm(adapter_parameters(self.backend.model)))
            total=torch.nn.utils.clip_grad_norm_(self.parameters,self.config["optimizer"]["gradient_clip"])
            if not torch.isfinite(total):raise RuntimeError("Nonfinite complete-window parameter gradient")
            self.clipped+=int(float(total)>self.config["optimizer"]["gradient_clip"])
            self.scaler.step(self.optimizer);self.scaler.update();self.steps+=1
            count=float(frame_batch.n.sum());bce_sum+=result["bce"]*count;votes+=count;pairs+=len(frame_batch)
            auxiliary.append(result["pearson"]);eligible+=result["eligible"]
        if self.device.type=="cuda":torch.cuda.synchronize()
        elapsed=time.perf_counter()-tick
        return {"binary_BCE_nats":bce_sum/votes,"pearson_mean_per_step":float(np.mean(auxiliary)),"eligible_paintings":eligible,
                "pairs":pairs,"paintings":int(frame.painting.nunique()),"seconds":elapsed,"pairs_per_second":pairs/elapsed,
                "head_gradient_norm_min":min(heads),"LoRA_gradient_norm_min":min(adapters),
                "clipping_frequency":self.clipped/self.steps,"AMP_scale":self.scaler.get_scale(),
                "objective_mode":"bce_pearson","forward_passes":2,"exact_logit_replay":True}
