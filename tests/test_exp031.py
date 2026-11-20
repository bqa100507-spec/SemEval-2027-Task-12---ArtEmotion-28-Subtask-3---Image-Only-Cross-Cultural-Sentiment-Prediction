"""CPU-only EXP031 invariants; no pretrained CUDA model or held-out gold."""
import numpy as np
import pandas as pd
import pytest
import torch

from st3.exp031 import evidence_rank_loss, grouped_objective, painting_batches, training_priors
from st3.exp031_model import HierarchicalHead


def test_train_only_priors_and_no_unseen_fallback():
    train = pd.DataFrame({"language": ["B", "A", "A"], "k": [2, 1, 2], "n": [5, 5, 5]})
    languages, logits = training_priors(train)
    assert languages == ["A", "B"]
    torch.testing.assert_close(logits.sigmoid(), torch.tensor([3.5 / 11, 2.5 / 6]))
    assert (logits.abs() < 10).all()


def test_joint_head_probabilities_preserve_binary_and_conditionals():
    torch.manual_seed(3)
    head = HierarchicalHead(hidden_size=8, prior_logits=torch.tensor([0., -0.5, 1.]),
                            rank=4)
    neutral = torch.randn(8)
    conditioned = torch.randn(3, 8)
    logp, binary, interaction = head(neutral, conditioned, torch.tensor([0, 1, 2]))
    probs = logp.exp()
    torch.testing.assert_close(probs.sum(1), torch.ones(3))
    torch.testing.assert_close(probs[:, :4].sum(1), binary.sigmoid())
    torch.testing.assert_close(interaction, torch.zeros(3))
    loss = -logp[:, 0].mean()
    loss.backward()
    assert head.language.weight.grad is not None
    assert head.language.weight.grad.abs().sum() > 0
    assert head.emotions[-1].weight.grad.abs().sum() > 0


def test_rank_gradients_are_within_one_painting_and_ignore_exact_ties():
    counts = torch.tensor([[5., 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 5., 0, 0, 0],
                           [5., 0, 0, 0, 0, 0, 0, 0]])
    correct = torch.tensor([3., -2., 2.], requires_grad=True)
    wrong = torch.tensor([-3., 2., -2.], requires_grad=True)
    good = evidence_rank_loss(correct, counts)
    bad = evidence_rank_loss(wrong, counts)
    assert good.item() < bad.item()
    bad.backward()
    assert wrong.grad is not None and torch.isfinite(wrong.grad).all()
    assert evidence_rank_loss(torch.tensor([0., 0.]),
                              counts[[0, 2]]).item() == 0


def test_rank_downweights_unreliable_small_differences():
    near = torch.tensor([[3., 0, 0, 0, 2., 0, 0, 0],
                         [2., 0, 0, 0, 3., 0, 0, 0]])
    logits = torch.tensor([10., -10.], requires_grad=True)
    ranking = evidence_rank_loss(logits, near)
    assert ranking.item() == 0
    ranking.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_complete_painting_batches_deterministic_without_pair_leakage():
    frame = pd.DataFrame({"painting": ["p", "p", "q", "r", "r", "r"],
                          "language": ["A", "B", "A", "A", "B", "C"]})
    batches = list(painting_batches(frame, seed=42, epoch=2, max_pairs=2))
    paintings = [group.painting.iloc[0] for batch in batches for group in batch]
    assert set(paintings) == {"p", "q", "r"}
    assert len(paintings) == len(set(paintings))
    assert sum(len(group) for batch in batches for group in batch) == len(frame)
    assert [group.painting.iloc[0] for batch in painting_batches(
        frame, seed=42, epoch=2, max_pairs=2) for group in batch] == paintings


def test_complete_objective_backward_finite():
    torch.manual_seed(4)
    head = HierarchicalHead(8, torch.tensor([-.4, .1, .3]), rank=4)
    neutral = torch.randn(8)
    conditioned = torch.randn(3, 8)
    logits, binary, interaction = head(neutral, conditioned, torch.tensor([0, 1, 2]))
    counts = torch.tensor([[4., 1, 0, 0, 0, 0, 0, 1],
                           [1., 0, 0, 0, 3, 1, 0, 0],
                           [0, 3, 0, 1, 0, 0, 0, 1]]).float()
    config = {"ranking_min_z": 1., "ranking_z_ramp": 1.5}
    ce, bce, rank, reg = grouped_objective(logits, binary, interaction, counts, config)
    total = (ce + .5 * bce) / counts.sum() + .15 * rank + .01 * reg
    assert torch.isfinite(total)
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in head.parameters() if p is not head.cultural[-1].weight
               and p is not head.cultural[-1].bias)  # zero-initialized interaction projection learns after embedding update
    assert head.cultural[-1].weight.grad is not None


def test_no_out_of_group_rank_supervision():
    counts_a = torch.tensor([[5., 0, 0, 0, 0, 0, 0, 0]])
    counts_b = torch.tensor([[0, 0, 0, 0, 5., 0, 0, 0]])
    assert evidence_rank_loss(torch.tensor([10.]), counts_a) == 0
    assert evidence_rank_loss(torch.tensor([-10.]), counts_b) == 0
