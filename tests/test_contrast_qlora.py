"""Synthetic CPU checks for complete-painting gradients; never downloads a VLM."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from st3.contrast import PaintingPearson
from st3.contrast_qlora import objective,pearson_loss,two_pass_backward,rng_state,restore_rng
from st3.vlm_supervised import DecoderLoRA,adapter_parameters,vote_loss


class Toy(nn.Module):
    def __init__(self,dropout=0.):
        super().__init__();self.lora=DecoderLoRA(nn.Linear(4,5),2,4)
        self.head=nn.Linear(5,1);self.dropout=nn.Dropout(dropout)
    def forward(self,x):return self.head(self.dropout(torch.tanh(self.lora(x)))).squeeze(-1)


def fixture():
    torch.manual_seed(2717)
    model=Toy();x=torch.randn(18,4)
    k=torch.tensor([0,2,1,3,4,0,1,2,3,4,5,1,2,4,0,3,5,2]);n=torch.ones(18)*5
    groups=["a"]*2+["b"]*3+["c"]*7+["d"]+["e"]*5
    return model,x,k,n,groups


def test_pearson_values_and_derivatives_match_exp004():
    z=torch.tensor([-.8,.1,.4,1.1,-.4,.7],dtype=torch.float64,requires_grad=True)
    k=torch.tensor([0.,2.,5.,1.,3.,4.]);n=torch.ones(6)*5;groups=[0,0,0,1,1,1]
    actual,count=pearson_loss(z,k,n,groups,.01)
    reference=PaintingPearson(k.double().numpy().reshape(2,3),n.double().numpy().reshape(2,3),.01)
    p=torch.sigmoid(z).detach().numpy().reshape(2,3);loss,gradient=reference(p)
    assert count==2 and float(actual.detach())==pytest.approx(loss,abs=1e-12)
    expected=torch.tensor((gradient*p*(1-p)).ravel())
    torch.testing.assert_close(torch.autograd.grad(actual,z)[0],expected,rtol=1e-12,atol=1e-12)


def test_eligibility_constant_and_near_constant_predictions():
    z=torch.zeros(8,dtype=torch.float64,requires_grad=True);k=torch.tensor([0.,2.,4.,1.,1.,1.,0.,5.]);n=torch.ones(8)*5
    loss,count=pearson_loss(z,k,n,[0,0,0,1,1,1,2,2])
    assert count==1 and float(loss.detach())==1
    g=torch.autograd.grad(loss,z)[0];assert torch.isfinite(g).all() and float(g[:3].norm())>0
    assert not g[3:].any()
    z=(torch.arange(8,dtype=torch.float64)*1e-10).requires_grad_()
    assert torch.autograd.gradcheck(lambda q:pearson_loss(q,k,n,[0,0,0,1,1,1,2,2])[0],(z,),eps=1e-6,atol=1e-5)
    zero,count=pearson_loss(z,torch.ones(8),n,[0]*3+[1]*3+[2]*2)
    assert count==0 and float(zero.detach())==0 and not torch.autograd.grad(zero,z)[0].any()
    # A row below the five-vote eligibility threshold cannot turn a two-language group into three.
    loss,count=pearson_loss(z[:3],k[:3],torch.tensor([5.,5.,4.]),[0]*3)
    assert count==0


@pytest.mark.parametrize("bad",["nan","counts","epsilon","shape"])
def test_invalid_inputs_fail(bad):
    z=torch.ones(3,requires_grad=True);k=torch.tensor([0.,2.,3.]);n=torch.ones(3)*5
    if bad=="nan":z=torch.tensor([0.,float("nan"),1.],requires_grad=True)
    if bad=="counts":n[0]=0
    with pytest.raises(ValueError):pearson_loss(z,k,n,[0]*3,epsilon=0 if bad=="epsilon" else .01) if bad!="shape" else pearson_loss(z,k[:2],n,[0]*3)


@pytest.mark.parametrize("microbatch",[1,2,4,8,18])
def test_two_pass_matches_unsplit_parameter_gradients_and_updates(microbatch):
    model,x,k,n,groups=fixture();reference=copy.deepcopy(model)
    loss,_=objective(reference(x),k,n,groups,"bce_pearson",.00196)
    loss.backward()
    result=two_pass_backward(lambda ids:model(x[ids]),list(range(18)),k,n,groups,microbatch,.00196,modules=(model,))
    assert result["exact_logit_replay"] and result["eligible"]==3
    for (name,a),(_,b) in zip(model.named_parameters(),reference.named_parameters()):
        if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=2e-7,msg=name)
        else:assert a.grad is None
    for network in (model,reference):torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],lr=.001).step()
    for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,rtol=1e-6,atol=2e-7)


def test_two_pass_replays_stochastic_draws_and_preserves_rng_state():
    torch.manual_seed(2);random.seed(2);np.random.seed(2)
    model,x,k,n,groups=fixture();model.dropout.p=.3;reference=copy.deepcopy(model)
    def forward(m,ids):return m(x[ids])*(.9+.1*random.random()+.01*np.random.random())
    initial=rng_state()
    z=torch.cat([forward(reference,list(range(i,min(i+4,18)))) for i in range(0,18,4)])
    total,_=objective(z,k,n,groups,"bce_pearson",.00196);total.backward();expected_rng=rng_state()
    restore_rng(initial)
    two_pass_backward(lambda ids:forward(model,ids),list(range(18)),k,n,groups,4,.00196,modules=(model,))
    after=rng_state();assert torch.equal(after["torch"],expected_rng["torch"])
    assert after["numpy"]==expected_rng["numpy"] and after["python"]==expected_rng["python"]
    for a,b in zip(model.parameters(),reference.parameters()):
        if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,rtol=2e-6,atol=2e-7)


def test_nonzero_head_and_lora_updates_base_frozen():
    model,x,k,n,groups=fixture();base={name:p.clone() for name,p in model.named_parameters() if not p.requires_grad}
    initial=[p.detach().clone() for p in adapter_parameters(model)];optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    for _ in range(3):
        optimizer.zero_grad();two_pass_backward(lambda ids:model(x[ids]),list(range(18)),k,n,groups,3,.00196,modules=(model,))
        assert float(model.head.weight.grad.norm())>0 and sum(float(p.grad.norm()) for p in adapter_parameters(model))>0
        optimizer.step()
    assert sum(float((p-before).norm()) for p,before in zip(adapter_parameters(model),initial))>0
    assert all(torch.equal(dict(model.named_parameters())[name],value) for name,value in base.items())


def test_state_mutation_and_mismatched_logits_block_partial_gradients():
    model,x,k,n,groups=fixture();model.register_buffer("counter",torch.zeros(()))
    def mutable(ids):model.counter.add_(1);return model(x[ids])
    with pytest.raises(RuntimeError,match="buffers"):two_pass_backward(mutable,list(range(18)),k,n,groups,4,.001,modules=(model,))
    assert model.counter==0 and all(p.grad is None for p in model.parameters())
    calls=[]
    def changed(ids):calls.append(1);return model(x[ids])+(0 if len(calls)<=5 else .1)
    with pytest.raises(RuntimeError,match="logits differ"):two_pass_backward(changed,list(range(18)),k,n,groups,4,.001,modules=(model,))
    assert all(p.grad is None for p in model.parameters())


def test_full_window_vote_normalization_and_zero_lambda():
    _,x,k,n,groups=fixture();z=x[:,0].clone().requires_grad_()
    control,_=objective(z,k,n,groups,"bce_only",0.)
    torch.testing.assert_close(control.float(),vote_loss(z,k,n),rtol=0,atol=0)
    expected=torch.autograd.grad(vote_loss(z,k,n),z,retain_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(control,z)[0],expected,rtol=0,atol=0)
