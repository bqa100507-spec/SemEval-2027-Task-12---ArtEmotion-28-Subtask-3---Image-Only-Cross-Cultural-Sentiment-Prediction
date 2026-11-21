import importlib
from pathlib import Path

import pandas as pd
import pytest
import torch
from scipy.special import expit


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_competition_direct_qwen_pilot")


def test_optimizer_checkpoint_is_portable_and_bound_to_source(tmp_path, pilot):
    module = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(module.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    module(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    trainable = pilot.trainables(module)
    target = tmp_path / "progress_0128"
    identity = {"arm": "qlora", "epoch": 1, "run": "pinned"}
    pilot.save_sealed(target, identity, {"trainable": trainable, "optimizer": optimizer.state_dict(),
                                         "scaler": scaler.state_dict()}, torch)
    replay = pilot.load_sealed(target, identity, torch)
    restored = torch.nn.Linear(3, 1)
    pilot.restore(restored, replay["trainable"])
    exact = torch.optim.AdamW(restored.parameters())
    exact.load_state_dict(replay["optimizer"])
    for model, opt in ((module, optimizer), (restored, exact)):
        opt.zero_grad()
        model(torch.ones(1, 3)).sum().backward()
        opt.step()
    torch.testing.assert_close(module.weight, restored.weight, atol=0, rtol=0)
    with pytest.raises(ValueError, match="different frozen identity"):
        pilot.load_sealed(target, identity | {"epoch": 2}, torch)
    with (target / "state.pt").open("ab") as out:
        out.write(b"changed")
    with pytest.raises(ValueError, match="hashed"):
        pilot.load_sealed(target, identity, torch)


def test_validation_replay_uses_same_keys_and_vote_weighted_logits(tmp_path, pilot):
    gold = pd.DataFrame({"painting": ["x", "x"], "language": ["English", "Thai"], "k": [5, 0], "n": [5, 12]})
    pred = gold[["painting", "language"]].iloc[::-1].assign(logit=[-1., 2.])
    pred["p_positive"] = expit(pred.logit)
    path = tmp_path / "validation"
    identity = {"arm": "frozen", "checkpoint_sha256": "selected"}
    pilot.validation_seal(path, identity, pred, gold)
    result = pilot.read_validation(path, identity, gold)
    expected = (5 * torch.nn.functional.softplus(torch.tensor(-2.)) +
                12 * torch.nn.functional.softplus(torch.tensor(-1.))) / 17
    assert pilot.exact_bce(gold, result) == pytest.approx(float(expected))
    with pytest.raises(ValueError, match="coverage"):
        pilot.read_validation(path, identity, gold.iloc[:1])
    with pytest.raises(ValueError):
        pilot.exact_bce(gold, pred.assign(language="English"))


def test_order_is_fixed_across_arms_and_does_not_repeat_paintings(pilot):
    names = ["p3", "p1", "p2", "p4"]
    one = pilot.epoch_order(names, 204202, 1)
    assert one == pilot.epoch_order(reversed(names), 204202, 1)
    assert set(one) == set(names) and len(one) == len(names)
    assert one != pilot.epoch_order(names, 204202, 2)
