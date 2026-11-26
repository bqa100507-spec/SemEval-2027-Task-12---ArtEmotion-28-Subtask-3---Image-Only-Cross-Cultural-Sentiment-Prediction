"""CPU checks for native Qwen conditioning, selective gradients and vote weighting."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from st3.competition_direct_qwen import CulturalQwen, LowRankUpdate, count_bce, last_valid, optimizer_for, train_step
from st3.e50_heads import read_json

RULE = read_json(Path(__file__).resolve().parents[1] / "configs/competition_direct_qwen.json")


def test_count_objective_matches_expanded_votes_and_exact_chunk_gradients():
    logits = torch.tensor([-.8, .25, 3.], requires_grad=True)
    positive, total = torch.tensor([2, 7, 5]), torch.tensor([5, 11, 5])
    expanded = torch.repeat_interleave(logits, total)
    targets = torch.cat([torch.cat((torch.ones(k), torch.zeros(n - k))) for k, n in zip(positive, total, strict=True)])
    target = F.binary_cross_entropy_with_logits(expanded, targets)
    full = count_bce(logits, positive, total)
    torch.testing.assert_close(full, target)
    split = sum(count_bce(z, k, n, vote_mass=total.sum()) for z, k, n in
                zip(logits.split(1), positive.split(1), total.split(1), strict=True))
    torch.testing.assert_close(torch.autograd.grad(full, logits)[0], torch.autograd.grad(split, logits)[0])
    with pytest.raises(ValueError, match="integer votes"):
        count_bce(logits, torch.tensor([2., 1.5, 0.]), total)
    with pytest.raises(ValueError, match="integer votes"):
        count_bce(logits, positive, torch.tensor([4, 11, 5]))


def test_last_token_selection_respects_padding_and_rejects_empty_rows():
    hidden = torch.arange(12).reshape(2, 3, 2)
    assert last_valid(hidden, torch.tensor([[0, 1, 1], [1, 0, 0]])).tolist() == [[4, 5], [6, 7]]
    with pytest.raises(ValueError, match="nonempty"):
        last_valid(hidden, torch.tensor([[0, 0, 0], [1, 0, 0]]))


def test_lora_zero_init_penalty_and_frozen_weights():
    rng = torch.Generator().manual_seed(5)
    base = nn.Linear(8, 10)
    layer = LowRankUpdate(base, 2, 4, rng)
    x = torch.randn(4, 8)
    initial = base(x)
    assert torch.equal(layer(x), initial)
    layer(x).square().mean().backward()
    assert layer.right.grad.norm() > 0
    assert layer.left.grad.norm() == 0
    assert base.weight.grad is None
    with torch.no_grad():
        layer.right.copy_(torch.randn_like(layer.right))
    torch.testing.assert_close(layer.squared_delta(), ((layer.right @ layer.left) * layer.scale).square().sum())


def test_native_tiny_qwen_multimodal_path_and_selective_train_step():
    qwen = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
    txt = {"hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
           "num_hidden_layers": 3, "head_dim": 16, "vocab_size": 64,
           "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]}}
    vis = {"depth": 3, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32,
           "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "deepstack_visual_indexes": [0, 1, 2]}
    cfg = qwen.Qwen3VLConfig(text_config=txt, vision_config=vis, image_token_id=50, video_token_id=51,
                             vision_start_token_id=52, vision_end_token_id=53)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        model = qwen.Qwen3VLModel(cfg).eval()
        pixels = torch.randn(8, 24)
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    rows = [0, 0, 1, 1]
    ids = torch.tensor([[0, 52, 50, 53, 21, 11], [52, 50, 53, 22, 23, 11]] * 2)
    mask = ids.ne(0).long()
    batch = {"input_ids": ids, "attention_mask": mask}
    rule = RULE | {"decoder_layers": 3, "lora_tail_layers": 1, "lora_rank": 2}
    frozen = CulturalQwen(model, rule, adapted=False)
    vision = frozen.encode_vision({"pixel_values": pixels, "image_grid_thw": grid})
    with torch.no_grad():
        native = model(**batch, pixel_values=pixels.reshape(2, 4, 24)[rows].reshape(-1, 24),
                       image_grid_thw=grid[rows], use_cache=False).last_hidden_state
        features = frozen.features(vision, rows, batch)
        torch.testing.assert_close(features, last_valid(native, mask))
        assert not torch.equal(features[0], features[1])
        assert not torch.equal(features[0], features[2])
        assert not torch.equal(features[0] - features[1], features[2] - features[3])
        start = frozen.logits(features)
        assert torch.isfinite(start.sigmoid()).all()
    wrong_rows = [0, 0, 1, 2]
    with pytest.raises(ValueError, match="valid artwork"):
        frozen.features(vision, wrong_rows, batch)

    control_opt = optimizer_for(frozen, rule)
    positive = torch.tensor([0., 2., 4., 6.])
    total = torch.tensor([5., 6., 7., 8.])
    scale = torch.amp.GradScaler("cpu", enabled=False)
    step = train_step(frozen, control_opt, scale, rule, [(lambda: frozen.logits(features), positive, total)], total.sum())
    assert step["head_grad_norm"] > 0 and step["adapter_grad_norm"] == 0
    assert all(p.grad is None for p in model.parameters())

    adapted = CulturalQwen(model, rule, adapted=True)
    assert list(adapted.adapters) == ["language_model.layers.2.self_attn.q_proj", "language_model.layers.2.self_attn.v_proj"]
    torch.testing.assert_close(adapted(vision, rows, batch), start, atol=0, rtol=0)
    old = {key: p.detach().clone() for key, p in adapted.named_parameters() if not p.requires_grad}
    train = optimizer_for(adapted, rule)
    for _ in range(2):
        step = train_step(adapted, train, scale, rule, [(lambda: adapted(vision, rows, batch), positive, total)], total.sum())
        assert step["adapter_grad_norm"] > 0 and step["head_grad_norm"] > 0
    assert all(p.grad is not None and p.grad.norm() > 0 for p in adapted.adapter_parameters())
    assert all(torch.equal(p, old[key]) and p.grad is None for key, p in adapted.named_parameters() if key in old)
