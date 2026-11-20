"""CPU-only EXP031 invariants; no pretrained CUDA model or held-out gold."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from st3.exp031 import binary_terms, conditional_ce, evidence_rank_loss, painting_batches, training_priors
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
    binary, interaction = head.binary(neutral, torch.tensor([0, 1, 2]))
    logp = head.distribution(binary, head.conditionals(conditioned))
    torch.testing.assert_close(binary, head.prior_logits, atol=0, rtol=0)
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


def test_no_out_of_group_rank_supervision():
    counts_a = torch.tensor([[5., 0, 0, 0, 0, 0, 0, 0]])
    counts_b = torch.tensor([[0, 0, 0, 0, 5., 0, 0, 0]])
    assert evidence_rank_loss(torch.tensor([10.]), counts_a) == 0
    assert evidence_rank_loss(torch.tensor([-10.]), counts_b) == 0

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_exp031 import CFG, backward_painting, completed_epochs, publish_epoch, restore


def test_weak_evidence_cannot_cancel_its_own_shrinkage():
    counts = torch.tensor([[3., 0, 0, 0, 2, 0, 0, 0], [1., 0, 0, 0, 4, 0, 0, 0]])
    z = torch.zeros(2, requires_grad=True)
    weak = evidence_rank_loss(z, counts)
    strong = evidence_rank_loss(z, counts * 20)
    assert 0 < weak < strong
    grad, = torch.autograd.grad(weak, z)
    assert grad[0] < 0 < grad[1]
    torch.testing.assert_close(grad.sum(), torch.tensor(0.))


class TinyModel(torch.nn.Module):
    """Same decoder parameters receive gradients from both neutral and language paths."""
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(5, 8)
        self.head = HierarchicalHead(8, torch.tensor([-.5, .5, .7, -.4]), rank=3)
        self.head.anchor.copy_(self.head.normalize(self.decoder(torch.zeros(5)).detach()))
        self.features = torch.randn(4, 5)

    def binary(self, image, ids):
        return self.head.binary(self.decoder(image), torch.tensor(ids))

    def conditionals(self, image, ids):
        return self.head.conditionals(self.decoder(image + self.features[ids]))


def fixture_groups():
    counts = torch.tensor([[5., 0, 0, 0, 0, 0, 0, 0],
                           [0., 0, 0, 0, 3, 2, 0, 0],
                           [0., 3, 0, 0, 0, 0, 2, 0]])
    return [(torch.randn(5), [0, 1, 2], counts), (torch.randn(5), [0, 1, 3], counts * 2)]


def test_streaming_backward_matches_joint_loss_and_gradient():
    torch.manual_seed(2031)
    model = TinyModel()
    with torch.no_grad():
        model.head.language.weight.normal_(std=.1)
    groups = fixture_groups()
    votes = sum(int(group[2].sum()) for group in groups)
    loss = 0.
    for image, ids, counts in groups:
        binary, interaction = model.binary(image, ids)
        conditionals = model.conditionals(image, ids)
        ce = -(model.head.distribution(binary, conditionals) * counts).sum()
        bce, rank, reg = binary_terms(binary, interaction, counts, CFG)
        loss += ((ce + CFG["binary_weight"] * bce) / votes
                 + (CFG["ranking_weight"] * rank + CFG["interaction_regularization"] * reg) / len(groups))
    loss.backward()
    expected = {k: p.grad.detach().clone() for k, p in model.named_parameters()}
    for width in (1, 2, 3):
        model.zero_grad()
        actual = 0.
        for image, ids, counts in groups:
            ce, bce, rank, reg, diag = backward_painting(
                model, image, ids, counts, votes, len(groups), CFG | {"microbatch_pairs": width}, diagnose=True)
            assert diag["weighted_binary_logit_norm"] > 0 and diag["weighted_rank_logit_norm"] > 0
            actual += ((ce + CFG["binary_weight"] * bce) / votes
                       + (CFG["ranking_weight"] * rank + CFG["interaction_regularization"] * reg) / len(groups))
        assert actual == pytest.approx(loss.item(), abs=2e-7)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.grad, expected[name], atol=1e-7, rtol=1e-5)


def test_interaction_is_anchored_and_learns_after_first_update():
    torch.manual_seed(2032)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["head_lr"])
    image, ids, counts = fixture_groups()[0]
    for step in range(2):
        optimizer.zero_grad()
        binary, interaction = model.binary(image, ids)
        subset, _ = model.binary(image, [1])
        torch.testing.assert_close(binary[1], subset[0])
        torch.testing.assert_close(interaction.mean(), torch.tensor(0.), atol=1e-7, rtol=0)
        backward_painting(model, image, ids, counts, int(counts.sum()), 1, CFG)
        assert model.head.language.weight.grad.norm() > 0
        assert bool(model.head.visual.weight.grad.norm() > 0) == bool(step)
        optimizer.step()
    # Use the original reference state rather than the updated decoder.
    with patch.object(model.head, "normalize", return_value=model.head.anchor):
        _, at_anchor = model.head.binary(model.decoder(torch.zeros(5)), torch.tensor(ids))
    torch.testing.assert_close(at_anchor, torch.zeros_like(at_anchor), atol=0, rtol=0)


def test_epoch_resume_matches_next_adam_update_and_checks_seal(tmp_path):
    from st3.experiment_workflow import StopRun
    from st3.metrics import score

    torch.manual_seed(2033)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["head_lr"])
    groups = fixture_groups()

    def step(net, opt):
        opt.zero_grad()
        image, ids, counts = groups[torch.randint(len(groups), ()).item()]
        backward_painting(net, image, ids, counts, int(counts.sum()), 1, CFG)
        opt.step()

    step(model, optimizer)
    gold = pd.DataFrame({"painting": [f"p{i}" for i in range(20) for _ in range(3)],
                         "language": ["A", "B", "C"] * 20, "k": [1, 2, 4] * 20, "n": [5] * 60})
    pred = gold[["painting", "language"]].assign(p_positive=[.2, .5, .7] * 20)
    metrics = score(gold, pred)
    row = {"epoch": 1, "contrast": metrics["cultural_contrast"], "macro_log_loss_bits": metrics["log_loss"]}
    identity = {"source": "test"}
    path = tmp_path / "epoch-01"
    publish_epoch(path, identity | {"epoch": 1},
                  {"epoch": 1, "weights": dict(model.named_parameters()), "anchor": model.head.anchor,
                   "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(), "cuda_rng": None},
                  pred, row)
    assert completed_epochs(tmp_path, identity, gold) == [row]
    step(model, optimizer)
    torch.manual_seed(2033)
    resumed = TinyModel()
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=CFG["head_lr"])
    with patch("torch.cuda.is_available", return_value=False):
        restore(path, dict(resumed.named_parameters()), resumed_opt, 1, resumed.head.anchor)
    step(resumed, resumed_opt)
    for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(StopRun, match="identity"):
        completed_epochs(tmp_path, {"source": "other"}, gold)
    (path / "resume.pt").write_bytes(b"corrupt")
    with pytest.raises(StopRun, match="corruption"):
        completed_epochs(tmp_path, identity, gold)


def test_saved_validation_replay_rejects_changed_prediction():
    from types import SimpleNamespace
    from run_exp031 import verify_prediction_replay

    model = TinyModel()
    frame = pd.DataFrame({"painting": ["p", "p"], "image": ["img", "img"], "language": [0, 1]})
    vision = torch.ones(5)
    with torch.no_grad():
        binary, _ = model.binary(vision, frame.language.tolist())
    expected = frame[["painting", "language"]].assign(p_positive=binary.sigmoid().tolist())
    args = model, lambda name: vision, frame
    budget = SimpleNamespace(check=lambda: None)
    verify_prediction_replay(*args, expected.iloc[::-1], budget)
    with pytest.raises(AssertionError):
        verify_prediction_replay(*args, expected.assign(p_positive=0.01), budget)
