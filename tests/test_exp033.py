import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from st3.exp033 import Adapter, SpatialHead, patch_coordinates, prefix, suffix
from st3.exp033_io import fuse, publish_checkpoint, read_checkpoint
from st3.exp033_train import Bank, Panel, Trainer, inner_split, select_epoch

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/exp033.json").read_text())


@pytest.fixture(autouse=True)
def small_torch_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Attention(nn.Module):
    def forward(self, x, attention_mask):
        legal = attention_mask[:, 0, 0] == 0
        return (x * legal[..., None]).sum(1, keepdim=True) / legal.sum(1)[:, None, None], None


class Layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm1, self.layer_norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.self_attn = Attention()
        self.mlp = nn.Linear(dim, dim)

    def forward(self, tokens, attention_mask):
        y = tokens + self.self_attn(self.layer_norm1(tokens), attention_mask)[0]
        return (y + self.mlp(self.layer_norm2(y)),)


class Vision(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([Layer(dim) for _ in range(3)])
        self.post_layernorm = nn.LayerNorm(dim)

    def embeddings(self, pixels, shapes):
        return pixels


def fixture():
    rule = CONFIG | {"width": 8, "adapter_width": 3, "query_width": 4, "step_images": 4}
    torch.manual_seed(41)
    vision = Vision(8).eval().requires_grad_(False)
    raw = torch.randn(12, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * 12, dtype=torch.bool)
    shapes = torch.tensor([[2, 2]] * 12)
    pre = prefix(vision, raw, mask, shapes, 2)
    tokens = suffix(vision, pre, mask, 2)
    bank = Bank({"prefix": pre.numpy(), "tokens": tokens.numpy(), "mask": mask.numpy(),
                 "shapes": shapes.numpy(), "global": tokens[:, :4].mean(1).numpy()}, "cpu")
    frame = pd.DataFrame([(f"p{i}", language, (i + j) % 5, 5)
                          for i in range(12) for j, language in enumerate("en fr vi".split())],
                         columns=["painting", "language", "k", "n"])
    index = {f"p{i}": i for i in range(12)}
    return rule, vision, bank, frame, index


def test_masked_spatial_alignment_and_initial_pooling():
    rule, _, bank, _, _ = fixture()
    data = bank.take([0, 1])
    head = SpatialHead(8, ["en", "fr"], rule)
    fields = ("tokens", "global", "mask", "shapes")
    args = [data[name] for name in fields]
    b = head.representation(*args, "B")
    assert b.shape == (2, 40)
    torch.testing.assert_close(head.representation(*args, "C"), b, atol=0, rtol=0)
    assert not torch.allclose(head.representation(*args, "A"), b)
    args[0][:, 4:] = float("nan")
    torch.testing.assert_close(head.representation(*args, "C"), b, atol=0, rtol=0)
    coords = patch_coordinates(data["mask"], data["shapes"])
    torch.testing.assert_close(coords[0, :4], torch.tensor([[.25, .25], [.25, .75], [.75, .25], [.75, .75]]))
    with pytest.raises(ValueError, match="contiguous"):
        patch_coordinates(~data["mask"], data["shapes"])


def test_prefix_and_zero_adapter_parity():
    rule, vision, bank, _, _ = fixture()
    data = bank.take([0, 1], ("prefix", "tokens", "mask"))
    adapters = nn.ModuleList([Adapter(8, 3), Adapter(8, 3)])
    adapted = suffix(vision, data["prefix"], data["mask"], rule["tail_blocks"], adapters)
    torch.testing.assert_close(adapted, data["tokens"], atol=0, rtol=0)


def test_grouped_accumulation_and_adaptation_gradients():
    rule, vision, bank, frame, index = fixture()
    panel = Panel(frame, index)
    left, right = [Trainer(vision, bank, panel, rule) for _ in range(2)]
    for worker, size in ((left, 4), (right, 2)):
        receipt = worker.train_epoch("D", 4, size)
        assert receipt["trainable_parameters"]["adapter"] == 96
        assert receipt["mean_module_gradient_norms"]["adapter"] > 0
        assert not any(p.grad is not None for p in vision.parameters())
        assert all(a.down.weight.grad.norm() > 0 for a in worker.adapters)
    for p, q in zip(left.head.parameters(), right.head.parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=1e-4)
    for a, b in zip(left.adapters.parameters(), right.adapters.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)


def test_shared_warmup_and_exact_checkpoint_replay(tmp_path):
    rule, vision, bank, frame, index = fixture()
    panel = Panel(frame, index)
    worker, adapted = [Trainer(vision, bank, panel, rule) for _ in range(2)]
    worker.train_epoch("C", 1, 2)
    publish_checkpoint(tmp_path, {"condition": "C"}, worker.state(1, {"inner": None}))
    initial = read_checkpoint(tmp_path, {"condition": "C"})
    adapted.restore(initial)
    inputs = frame[["painting", "language"]]
    assert worker.predict(inputs, index, "C", 1, 2).equals(adapted.predict(inputs, index, "D", 1, 2))
    adapted.train_epoch("D", 4, 2)
    a = copy.deepcopy(adapted.state(4, {}))
    adapted.restore(initial)
    adapted.train_epoch("D", 4, 2)
    for group in ("adapters", "head"):
        for key, tensor in adapted.state(4, {})[group].items():
            torch.testing.assert_close(a[group][key], tensor, atol=0, rtol=0)
    with pytest.raises(ValueError, match="identity"):
        read_checkpoint(tmp_path, {"condition": "D"})
    (tmp_path / "state.pt").write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash"):
        read_checkpoint(tmp_path, {"condition": "C"})


def test_vote_isolation_checkpoint_selection_and_rounding():
    _, _, _, frame, _ = fixture()
    fit, held = inner_split(frame, .2, 2033)
    assert set(fit.painting).isdisjoint(held.painting)
    assert len(fit) + len(held) == len(frame)
    c = {6: {"log_loss": .7, "cultural_contrast": .4}, 9: {"log_loss": .71, "cultural_contrast": .5}}
    d = {6: {"log_loss": .702, "cultural_contrast": .4}, 9: {"log_loss": .7, "cultural_contrast": .5}}
    assert select_epoch(c, d, .005) == 6
    side = pd.DataFrame({"painting": ["p"], "language": ["vi"], "clip": [.1], "large": [.3]})
    assert fuse(side, np.array([.5])).p_positive.iat[0] == pytest.approx(.3)


def test_export_readback_and_tamper(tmp_path):
    import runpy
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/exp033_artifacts.py"))
    source = tmp_path / "run"
    source.mkdir()
    publish_checkpoint(source / "epoch_03", {"phase": "C"}, {"weights": torch.zeros(3)})
    archive = tmp_path / "run.zip"
    api["export"](source, archive)
    restored = tmp_path / "restored"
    api["restore"](archive, restored, api["digest"](archive))
    torch.testing.assert_close(read_checkpoint(restored / "epoch_03", {"phase": "C"})["weights"], torch.zeros(3))
    with pytest.raises(ValueError, match="SHA-256"):
        api["restore"](archive, tmp_path / "wrong", "0" * 64)


def test_five_fold_evaluation_emits_separate_scopes_and_gate():
    from st3.exp033_metrics import report
    frame = pd.DataFrame([(f"p{i:03}", language, value, 5, i // 20 + 1)
                          for i in range(100) for language, value in zip("abc", (1, 2, 4))],
                         columns=["painting", "language", "k", "n", "fold"])
    perfect = frame[["painting", "language"]].assign(p_positive=frame.k / frame.n)
    other = perfect.assign(p_positive=(1 - perfect["p_positive"]))
    solo = dict(A=other, B=other, C=other, D=perfect, probe=perfect)
    result = report(frame, solo, solo | {"E50+L": other, "ridge": other}, CONFIG | {"bootstrap": 128})
    assert result["status"] == "EXPLORATORY_D_GO"
    assert result["paired"]["standalone"]["D-C"]["gain"] == pytest.approx(2)
    assert result["paired"]["fusion"]["probe-B"]["fold_wins"] == 5


def test_complete_fold_checks_all_four_conditions_and_independent_probe(tmp_path):
    import runpy
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_exp033.py"))
    rule, vision, bank, _, _ = fixture()
    rule = rule | {"warmup_epochs": 1, "checkpoints": [2], "step_images": 16}
    n = 110
    bank = Bank({k: v[np.arange(n) % len(v)] for k, v in bank.arrays.items()}, "cpu")
    keys = {f"p{i:03}": i for i in range(n)}
    data = pd.DataFrame([(name, language, k, 5, 2 if i < 100 else 1)
                         for name, i in keys.items() for language, k in zip("abc", (0, 2, 4))],
                        columns=["painting", "language", "k", "n", "fold"])
    api["complete_fold"](tmp_path, 1, {"test": "tiny"}, data, bank, keys, vision, rule, 4, float("inf"))
    result = json.loads((tmp_path / "fold_1/outer_seal.json").read_text())
    assert result["epoch"] == 2
    assert set(result["prediction_sha256"]) == {"A", "B", "C", "D", "probe"}
    assert all(len(pd.read_csv(tmp_path / "fold_1" / f"{name}.csv")) == 30
               for name in result["prediction_sha256"])
    d = json.loads((tmp_path / "fold_1/outer/D/curves.json").read_text())
    assert d["shared_warmup_epochs"] == 1
    assert d["epochs"]["2"]["trainable_parameters"]["adapter"] == 96


def test_notebook_configuration_has_no_implicit_gpu_execution():
    ipynb = Path(__file__).resolve().parents[1] / "notebooks/EXP033_Kaggle.ipynb"
    if not ipynb.exists():
        pytest.skip("Pinned notebook is delivered in a separate packaging commit")
    state = {}
    content = json.loads(ipynb.read_text(encoding="utf-8"))
    for cell in content["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(ipynb), "exec"), state)
    assert not state["AUTHORIZE_STAGE"]
    assert not state["AUTHORIZE_FULL_CV"]
