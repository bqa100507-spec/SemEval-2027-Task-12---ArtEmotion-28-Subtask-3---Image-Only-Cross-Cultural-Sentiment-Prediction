import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from st3.exp030 import EMOTIONS, best_epoch, count_votes, loss_sums, positive_probability

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_exp030 import CFG, completed_epochs, ordered_indices, publish_epoch


def test_count_objective_matches_individual_votes_and_accumulation():
    logits = torch.tensor([[1., 0., -1., 2., 0., 1., -4., 0.],
                           [2., 1., 0., -1., 2., 1., 1., 0.]], requires_grad=True)
    counts = torch.tensor([[1, 0, 0, 0, 3, 0, 0, 1], [3, 2, 0, 1, 0, 0, 0, 1]])
    ce, bce = loss_sums(logits, counts)
    indices = torch.arange(8).repeat_interleave(counts.flatten().int().reshape(-1)[:8])
    torch.testing.assert_close(ce, F.cross_entropy(logits[:1].expand(len(indices), -1), indices, reduction="sum")
                               + loss_sums(logits[1:], counts[1:])[0])
    p = positive_probability(logits)
    torch.testing.assert_close(bce, F.binary_cross_entropy(p, counts[:, :4].sum(1) / counts.sum(1),
                                                          reduction="none").mul(counts.sum(1)).sum())
    full = torch.autograd.grad((ce + .5 * bce) / counts.sum(), logits)[0]
    parts = sum((a + .5 * b) / counts.sum() for a, b in
                (loss_sums(logits[i:i+1], counts[i:i+1]) for i in range(len(logits))))
    torch.testing.assert_close(torch.autograd.grad(parts, logits)[0], full)
    extreme = torch.tensor([[1000.] * 4 + [-1000.] * 4])
    assert torch.isfinite(torch.stack(loss_sums(extreme, counts[:1]))).all()


def test_votes_ignore_outside_populations_and_ambiguous_labels(tmp_path):
    path = tmp_path / "votes.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["painting", "language", "image_name", "split", "emotion"])
        writer.writerows([["p", "english", "a.jpg", "train", e] for e in
                          ("amusement", "awe", "anger", "anger", "fear", "other")])
        writer.writerow(["p", "english", "a.jpg", "val", "invalid"])
        writer.writerow(["sealed", "english", "b.jpg", "train", "invalid"])
    pairs = pd.DataFrame({"painting": ["p"], "language": ["English"], "image": ["a.jpg"], "k": [2], "n": [5]})
    assert count_votes(path, pairs)[list(EMOTIONS)].iloc[0].tolist() == [1, 1, 0, 0, 2, 0, 1, 0]
    with pytest.raises(ValueError, match="counts"):
        count_votes(path, pairs.assign(k=3))


def test_epoch_order_covers_every_pair_exactly_once():
    frame = pd.DataFrame({"painting": ["p", "p", "q", "r", "r", "r"]})
    order = ordered_indices(frame, 1)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(frame)))
    np.testing.assert_array_equal(order, ordered_indices(frame, 1))


def test_checkpoint_seal_and_selection(tmp_path):
    from st3.metrics import score
    from st3.experiment_workflow import StopRun

    val = pd.DataFrame({"painting": [f"p{i}" for i in range(20) for _ in range(3)],
                        "language": ["A", "B", "C"] * 20, "k": [1, 2, 4] * 20, "n": [5] * 60})
    pred = val[["painting", "language"]].assign(p_positive=[.2, .5, .7] * 20)
    m = score(val, pred)
    row = {"epoch": 1, "contrast": m["cultural_contrast"], "macro_log_loss_bits": m["log_loss"]}
    identity = {"source": "fixture"}
    directory = tmp_path / "epoch-01"
    publish_epoch(directory, identity | {"epoch": 1}, {"weights": {"head": torch.zeros(2)}}, pred, row)
    assert completed_epochs(tmp_path, identity, val) == [row]
    assert best_epoch([row, row | {"epoch": 2, "macro_log_loss_bits": 0}]) == 1
    with pytest.raises(StopRun, match="identity"):
        completed_epochs(tmp_path, {"source": "different"}, val)
    (directory / "predictions.csv").write_text("corrupt")
    with pytest.raises(StopRun, match="corruption"):
        completed_epochs(tmp_path, identity, val)


def test_resume_restores_optimizer_and_rng_for_next_update(tmp_path):
    from run_exp030 import restore
    from unittest.mock import patch

    torch.manual_seed(3)
    head = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.AdamW([head], lr=.001, foreach=False, fused=False)

    def step(w, optim):
        optim.zero_grad()
        (w * torch.rand_like(w)).sum().backward()
        optim.step()

    step(head, optimizer)
    torch.save({"epoch": 1, "weights": {"head": head.detach()}, "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(), "cuda_rng": None}, tmp_path / "resume.pt")
    step(head, optimizer)
    resumed = torch.nn.Parameter(torch.zeros(2))
    restarted = torch.optim.AdamW([resumed], lr=.001, foreach=False, fused=False)
    with patch("torch.cuda.is_available", return_value=False):
        restore(tmp_path, {"head": resumed}, restarted, 1)
    step(resumed, restarted)
    torch.testing.assert_close(resumed, head, atol=0, rtol=0)
