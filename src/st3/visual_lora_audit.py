"""Independent BCE/CE LoRA gradients and numerical image representation audits."""

import numpy as np
import torch

from .gradient_analysis import decompose
from .visual_groups import features
from .vote_model import LOG_EIGHT,vote_losses


def gradient_audit(ctx,vision,head,rows,indices,condition):
    head.eval();group=rows.batch(indices,ctx.device)
    vectors,_,_=features(ctx,vision,group['names'],condition.startswith('L'))
    binary,aux=head(vectors[group['gather']],group['language'],auxiliary=True)
    loss=vote_losses(binary.float(),aux.float(),group['counts'],.1 if condition.endswith('8') else 0.)
    hp=tuple(head.parameters());ap=tuple(vision.adapters().values()) if condition.startswith('L') else ()
    objects=(*hp,*ap)
    b=torch.autograd.grad(loss['bce'],objects,retain_graph=True,allow_unused=True)
    a=torch.autograd.grad(loss['normalized_ce'],objects,allow_unused=True)
    def vector(values,parameters):return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,values)])
    bv,av=vector(b[:len(hp)],hp),vector(a[:len(hp)],hp)
    if not bool(torch.isfinite(torch.cat([bv,av])).all()):raise ValueError('Nonfinite head gradients')
    result={'bce':float(loss['bce'].detach()),'ce':float(loss['ce'].detach()),'normalized_ce':float(loss['normalized_ce'].detach()),
        'head':decompose(bv,av,.1 if condition.endswith('8') else 0.),'clip_features_require_grad':vectors.requires_grad,
        'lora':None,'per_adapter_bce_norms':{},'per_adapter_ce_norms':{}}
    if ap:
        gb,ga=vector(b[len(hp):],ap),vector(a[len(hp):],ap)
        result['lora']=decompose(gb,ga,.1 if condition.endswith('8') else 0.)
        for name,g1,g2 in zip(vision.adapters(),b[len(hp):],a[len(hp):]):
            result['per_adapter_bce_norms'][name]=float(g1.norm()) if g1 is not None else 0.
            result['per_adapter_ce_norms'][name]=float(g2.norm()) if g2 is not None else 0.
        if result['lora']['zero_bce'] or result['lora']['zero_text']:raise ValueError('Disconnected BCE or CE LoRA gradient')
    if any(p.grad is not None for name,p in vision.named_parameters() if name not in vision.adapters()):
        raise ValueError('Pretrained visual weights received gradients')
    for group in (result['head'],result['lora']):
        if group is not None:
            for key in list(group):
                if key.startswith('post_clip_'):del group[key]
    result['finite']=True
    result['gradient_names']='text_norm is CE8/ln(8); scaled_text_norm is coefficient*CE8/ln(8); eval-mode gradients before clipping'
    return result


def feature_comparison(actual,reference):
    a=actual.detach().cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
    b=reference.detach().cpu().numpy() if torch.is_tensor(reference) else np.asarray(reference)
    if a.shape!=b.shape or a.shape[1]!=512 or not np.isfinite(a).all():raise ValueError('Invalid CLIP feature geometry')
    return {'shape':list(a.shape),'norm_max_error':float(np.max(np.abs(np.linalg.norm(a,axis=1)-1.))),
        'cosine_min':float(np.min(np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)))),
        'max_abs_error':float(np.max(np.abs(a-b))),'mean_l2_shift':float(np.linalg.norm(a-b,axis=1).mean())}
