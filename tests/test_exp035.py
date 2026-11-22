from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from st3.e50_heads import read_json
from st3.exp035 import (ResidualHead, add_merger_lora, frozen_feature, image_hidden, objective,
                       odds_update, select_languages, update)
from st3.exp035_protocol import paired_epoch_curves, select_epoch, split_groups

CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/exp035.json")


def test_count_weight_centering_and_exact_inference():
    q = torch.tensor([0., .23, .61, 1.], dtype=torch.float64)
    z = torch.zeros(4, requires_grad=True)
    assert torch.equal(odds_update(q, z), q)
    z = torch.tensor([.1, -.3, .25, -.4], requires_grad=True)
    k, n = torch.tensor([0, 1, 4, 6]), torch.tensor([5, 5, 7, 6])
    loss, (bce, centered, penalty) = objective(q, z, k, n, n.sum(), CONFIG)
    expected = F.binary_cross_entropy_with_logits((torch.logit(q.clamp(1e-6, 1-1e-6)) + z).float(),
                                                  k/n, weight=n, reduction="sum") / n.sum()
    torch.testing.assert_close(bce, expected)
    p = odds_update(q, z)
    torch.testing.assert_close(centered, ((p - p.mean()) - (k/n - (k/n).mean())).square().mean())
    loss.backward()
    assert z.grad.isfinite().all()
    assert select_languages(torch.tensor([.2, .3]), torch.tensor([1, -1])).tolist() == pytest.approx([.3, 0.])


def test_partitioning_is_grouped_and_checkpoint_selection_is_inner_only():
    data = pd.DataFrame({"painting": np.repeat(list("abcdefgh"), 3), "language": list("xyz") * 8})
    held = []
    for fit, val in split_groups(data, 42, 3):
        assert not set(fit.painting) & set(val.painting)
        assert val.groupby("painting").size().eq(3).all()
        held.extend(val.index)
    assert sorted(held) == list(data.index)

    def record(epoch, contrast, loss):
        axis = {"prior": {"log_loss": .75}, "candidate": {"cultural_contrast": contrast, "log_loss": loss}}
        return {"epoch": epoch, "inner": dict(full=axis, test_like=axis)}
    row, ok = select_epoch([record(1, .42, .751), record(2, .45, .77), record(3, .44, .754)], .005)
    assert row["epoch"] == 3 and ok


def test_corrected_ridge_offsets_reject_query_leakage_before_fitting():
    from st3.exp035_protocol import corrected_offsets

    frame = pd.DataFrame({"painting": ["a", "b"], "language": ["x", "x"], "k": [1, 3], "n": [5, 5]})
    with pytest.raises(ValueError, match="disjoint"):
        corrected_offsets(frame, frame[["painting", "language"]], {}, {}, {}, 1)
    with pytest.raises(ValueError, match="disjoint"):
        corrected_offsets(frame.iloc[:1], frame.iloc[1:], {}, {}, {}, 1)


def test_frozen_features_preserve_dtype_values_and_head_gradients():
    cache, calls = {}, []
    state = torch.rand(1, 8, dtype=torch.float16, requires_grad=True)

    def live():
        calls.append(1)
        return state

    cached = frozen_feature(cache, "a", live, "cpu")
    assert torch.equal(cached, state) and cached.dtype == state.dtype
    assert not cached.requires_grad
    assert torch.equal(frozen_feature(cache, "a", live, "cpu"), state)
    assert len(calls) == 1
    head = ResidualHead(8, 3, CONFIG)
    from_live = torch.autograd.grad(head(state.detach()).sum(), tuple(head.parameters()))
    from_cache = torch.autograd.grad(head(cached).sum(), tuple(head.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(from_live, from_cache, strict=True))


def test_paired_epoch_curves_report_signed_deltas_and_check_reference():
    a = {"cultural_contrast": .4, "log_loss": .8}
    b = {"cultural_contrast": .5, "log_loss": .7}
    c = {"cultural_contrast": .3, "log_loss": .9}
    arms = {name: {"history": [{"epoch": 1, "inner": {scope: {"prior": a, "candidate": candidate}
                                                    for scope in ("full", "test_like")}}]}
            for name, candidate in (("frozen", b), ("merger", c))}
    result = paired_epoch_curves(arms)[0]["inner"]["full"]
    assert result["cultural_contrast_delta"] == pytest.approx(
        {"B_minus_A": .1, "C_minus_A": -.1, "C_minus_B": -.2})
    assert result["log_loss_delta"]["C_minus_B"] == pytest.approx(.2)
    arms["merger"]["history"][0]["inner"]["full"]["prior"] = b
    with pytest.raises(ValueError, match="identical Ridge"):
        paired_epoch_curves(arms)


def test_real_module_layout_and_delayed_merger_gradients_on_tiny_qwen():
    qwen = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
    text = {"hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
            "num_hidden_layers": 3, "head_dim": 16, "vocab_size": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]}}
    vision = {"depth": 3, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32,
              "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "deepstack_visual_indexes": [0, 1, 2]}
    cfg = qwen.Qwen3VLConfig(text_config=text, vision_config=vision, image_token_id=50, video_token_id=51,
                             vision_start_token_id=52, vision_end_token_id=53)
    model = qwen.Qwen3VLModel(cfg).eval().requires_grad_(False)
    encoded = {"input_ids": torch.tensor([[52, 50, 53, 21, 11]]), "attention_mask": torch.ones(1, 5, dtype=torch.long),
               "pixel_values": torch.randn(4, 24), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    baseline = image_hidden(model, encoded)
    adapter = add_merger_lora(model, CONFIG)
    assert torch.equal(baseline, image_hidden(model, encoded))
    head = ResidualHead(32, 3, CONFIG)
    params = [*head.parameters(), adapter.left, adapter.right]
    opt = torch.optim.AdamW(params, lr=.001)
    scale = torch.amp.GradScaler("cpu", enabled=False)
    args = (torch.tensor([.45, .6, .7]), torch.arange(3), torch.tensor([4, 1, 5]), torch.tensor([5, 5, 6]))
    initial = adapter.right.detach().clone()
    trace = [update(head, adapter, opt, scale, image_hidden(model, encoded), *args, 16., CONFIG) for _ in range(3)]
    assert trace[0]["adapter_gradient"] == 0
    assert all(x["adapter_gradient"] > 0 for x in trace[1:])
    assert not torch.equal(adapter.right, initial)
    assert adapter.left.grad.norm() > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
