"""Reusable empirical vote objectives and matched binary/eight-way students."""

import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .caption_student import CaptionStudent,make_rows
from .gradient_analysis import decompose
from .vote_inputs import COUNT_COLUMNS,empirical_distributions

LOG_EIGHT=math.log(8.)


def vote_losses(binary_logits,emotion_logits,counts,coefficient):
    """Natural-log vote likelihoods; independent of any encoder architecture."""
    if counts.ndim!=2 or counts.shape[1]!=8 or binary_logits.shape!=counts.shape[:1]:
        raise ValueError("Expected N binary logits and Nx8 empirical vote counts")
    if (emotion_logits is not None and emotion_logits.shape!=counts.shape) or (coefficient>0 and emotion_logits is None):
        raise ValueError("D8 requires eight auxiliary logits per example")
    n=counts.sum(1);d=counts/n[:,None];y=counts[:,:4].sum(1)/n
    bce=(F.binary_cross_entropy_with_logits(binary_logits,y,reduction="none")*n).sum()/n.sum()
    ce=bce.new_zeros(()) if emotion_logits is None else (-(d*F.log_softmax(emotion_logits,dim=1)).sum(1)*n).sum()/n.sum()
    normalized=ce/LOG_EIGHT
    return {"bce":bce,"ce":ce,"normalized_ce":normalized,"total":bce+coefficient*normalized}


class VoteStudent(CaptionStudent):
    def __init__(self,languages,config):
        super().__init__(languages,{**config,"target_dimensions":64})
        self.auxiliary=nn.Linear(2*config["hidden_width"],8)

    def forward(self,images,language,*,auxiliary=False):
        hidden=self.visual(images);known=language.ge(0);index=language.clamp_min(0)
        embedding=self.language_visual(index)*known[:,None]
        offset=self.language_bias(index).squeeze(-1)
        binary=self.shared(hidden).squeeze(-1)+torch.where(known,offset+(embedding*hidden).sum(-1)/self.scale,0.)
        emotions=self.auxiliary(torch.cat([hidden,embedding],dim=1)) if auxiliary else None
        return binary,emotions


def seed_model(languages,seed,config,device):
    torch.manual_seed(seed);model=VoteStudent(languages,config).to(device)
    rng=torch.Generator(device=device).manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    return model,rng,optimizer


def vote_rows(frame,features,languages,device,*,training=True):
    rows=make_rows(frame,features,languages,device,training=training)
    if training:
        empirical_distributions(frame)
        if bool(rows["language"].lt(0).any()):raise ValueError("A training language ID was silently dropped")
        rows["counts"]=torch.as_tensor(frame[COUNT_COLUMNS].to_numpy(dtype=np.float32),device=device)
    return rows


def batch_gradients(model,rows,batch,coefficient,clip_limit):
    mode=model.training;captured=[]
    hook=model.visual.register_forward_hook(lambda m,i,o:captured.append(o))
    try:
        model.eval();batch=batch.to(rows["features"].device)
        z=rows["features"][rows["image"][batch]].detach().clone().requires_grad_(True)
        binary,emotions=model(z,rows["language"][batch],auxiliary=True)
        loss=vote_losses(binary,emotions,rows["counts"][batch],coefficient)
        parameters=tuple(model.parameters());objects=(*parameters,captured[0],z,emotions)
        b=torch.autograd.grad(loss["bce"],objects,retain_graph=True,allow_unused=True)
        a=torch.autograd.grad(loss["ce"],objects,allow_unused=True)
        bg=[torch.zeros_like(p) if g is None else g for p,g in zip(objects,b)]
        ag=[torch.zeros_like(p) if g is None else g for p,g in zip(objects,a)]
        weight=coefficient/LOG_EIGHT
        norm=float(torch.stack([(x+weight*y).norm() for x,y in zip(bg[:len(parameters)],ag[:len(parameters)])]).norm())
        factor=min(1.,clip_limit/(norm+1e-6))
        names=[name for name,_ in model.named_parameters()]
        groups={"first_visual":[i for i,n in enumerate(names) if n.startswith("visual.0.")],
                "language_interaction":[i for i,n in enumerate(names) if n.startswith(("language_visual.","language_bias."))]}
        result={}
        for name,indices in groups.items():
            bv=torch.cat([bg[i].flatten() for i in indices]);av=torch.cat([ag[i].flatten() for i in indices])
            result[name]=decompose(bv,av,weight,factor)
            result[name]["normalized_ce_gradient_norm"]=result[name]["text_norm"]/LOG_EIGHT
        for name,idx in (("shared_representation",-3),("feature_input",-2)):
            result[name]=decompose(bg[idx],ag[idx],weight)
            result[name]["normalized_ce_gradient_norm"]=result[name]["text_norm"]/LOG_EIGHT
            for key in list(result[name]):
                if key.startswith("post_clip_"):del result[name][key]
        logits_gradient=ag[-1].norm(dim=0)
        if not all(bool(torch.isfinite(x).all()) for x in (*bg,*ag)) or not np.isfinite(norm):raise ValueError("Nonfinite vote gradients")
        return {"bce":float(loss["bce"].detach()),"ce":float(loss["ce"].detach()),
            "normalized_ce":float(loss["normalized_ce"].detach()),"coefficient":coefficient,
            "all_logits_gradient_norms":logits_gradient.tolist(),"all_logits_valid_gradients":bool(logits_gradient.gt(0).all()),
            "global_combined_norm":norm,"clip_factor":factor,"groups":result,"finite":True,
            "gradient_names":"text_norm is unnormalized eight-way CE; scaled_text_norm is coefficient*CE/ln(8)",
            "scope":"Training-only head and feature-input gradients, not pretrained encoder parameter gradients"}
    finally:hook.remove();model.train(mode)


def train_epoch(model,rows,rng,optimizer,config,coefficient):
    started=time.perf_counter();model.train();totals=np.zeros(3);votes=0.;clipped=0;norms=[]
    order=torch.randperm(len(rows["image"]),device=rows["features"].device,generator=rng)
    for batch in order.split(config["batch_size"]):
        optimizer.zero_grad(set_to_none=True)
        binary,emotions=model(rows["features"][rows["image"][batch]],rows["language"][batch],auxiliary=coefficient>0)
        losses=vote_losses(binary,emotions,rows["counts"][batch],coefficient)
        losses["total"].backward();norm=nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip"])
        if not bool(torch.isfinite(norm)):raise ValueError("Nonfinite training gradient")
        optimizer.step();n=float(rows["counts"][batch].sum())
        totals+=n*np.array([float(losses[k].detach()) for k in ("bce","ce","normalized_ce")]);votes+=n
        norms.append(float(norm));clipped+=int(norm>config["gradient_clip"])
    return {"train_bce":totals[0]/votes,"train_ce":totals[1]/votes,"train_normalized_ce":totals[2]/votes,
        "clipping_frequency":clipped/len(norms),"pre_clip_norm_mean":float(np.mean(norms)),
        "post_clip_norm_mean":float(np.mean(np.minimum(norms,config["gradient_clip"]))),
        "epoch_training_seconds":time.perf_counter()-started}


@torch.inference_mode()
def evaluate(model,rows,batch_size):
    model.eval();totals=np.zeros(3);votes=0.;probabilities=[];binary_parts=[]
    for batch in torch.arange(len(rows["image"]),device=rows["features"].device).split(batch_size):
        binary,emotions=model(rows["features"][rows["image"][batch]],rows["language"][batch],auxiliary=True)
        loss=vote_losses(binary,emotions,rows["counts"][batch],.1);n=float(rows["counts"][batch].sum())
        totals+=n*np.array([float(loss[k]) for k in ("bce","ce","normalized_ce")]);votes+=n
        probabilities.append(emotions.softmax(1).cpu().numpy());binary_parts.append(binary.sigmoid().cpu().numpy())
    predicted=np.concatenate(probabilities);binary=np.concatenate(binary_parts)
    return {"bce":totals[0]/votes,"ce":totals[1]/votes,"normalized_ce":totals[2]/votes,
            "binary_aux_mae":float(np.mean(np.abs(binary-predicted[:,:4].sum(1))))},predicted


def prior_diagnostic(frame,predicted,training):
    """Compare to training-only language priors; never an inference feature."""
    results=[]
    for language,group in frame.assign(row=np.arange(len(frame))).groupby("language"):
        fit=training[training.language.eq(language)];counts=fit[COUNT_COLUMNS].sum().to_numpy(dtype=float)
        if counts.sum()==0:continue
        prior=counts/counts.sum();values=predicted[group.row]
        target=group[COUNT_COLUMNS].to_numpy(dtype=float);n=target.sum(1)
        ce_prior=float(-(target*np.log(np.clip(prior,1e-12,1.))).sum()/n.sum())
        ce_model=float(-(target*np.log(np.clip(values,1e-12,1.))).sum()/n.sum())
        within=float(values.std(axis=0).mean());distance=float(np.abs(values-prior).sum(1).mean())
        eligible=group.image.nunique()>=2
        results.append({"language":language,"pairs":len(group),"variation_eligible":bool(eligible),"within_language_probability_std":within,
            "mean_l1_from_training_prior":distance,"prior_ce_nats":ce_prior,"model_ce_nats":ce_model,
            "prior_like_diagnostic":bool(eligible and within<1e-4 and ce_model>=ce_prior-.001)})
    return results
