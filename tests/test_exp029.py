import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from st3.exp029 import (VisualCorrection, baseline_frame, crossfit_partitions,
                       prefix_mask_positions, shuffled_images, visual_vectors)
from st3.vlm_supervised import DecoderLoRA, vote_loss


def test_crossfit_isolates_all_validation_images():
    train = pd.DataFrame([(f"p{i}", f"i{i}", f"h{i}", lang)
                          for i in range(8) for lang in ["A", "B"]],
                         columns=["painting", "image", "image_sha256", "language"])
    val = train.iloc[:2].assign(painting="held", image="held", image_sha256="held")
    seen = []
    for fit, query in crossfit_partitions(train, val, 4, 2027):
        assert len(query) == 4
        assert set(fit.painting).isdisjoint(query.painting)
        assert set(fit.painting).isdisjoint(val.painting)
        seen.extend(query.index)
    assert sorted(seen) == list(train.index)
    with pytest.raises(ValueError, match="overlap"):
        list(crossfit_partitions(train, val.assign(image_sha256="h1"), 4, 2027))


def test_cache_fusion_shuffle_and_baseline_coverage():
    rows = pd.DataFrame({"painting": ["p", "p", "q"], "image": ["x", "x", "y"], "language": ["A", "B", "A"]})
    v = np.eye(2, 768, dtype=np.float32)
    blocks = {k: dict(zip(["x", "y"], v)) for k in ["SigLIP2-Base", "CLIP-L"]}
    vectors = visual_vectors(blocks, rows)
    assert vectors["x"].shape == (1536,)
    np.testing.assert_allclose(np.linalg.norm(vectors["x"]), 1., atol=1e-7)
    assert shuffled_images(rows, 2027) == ["y", "y", "x"]
    pred = rows[["painting", "language"]].assign(p_positive=[.2, .3, .5])
    assert baseline_frame(rows, pred.iloc[::-1]).p_positive.tolist() == [.2, .3, .5]
    with pytest.raises(ValueError, match="coverage"):
        baseline_frame(rows, pred.iloc[:-1])
    with pytest.raises(ValueError, match="undefined"):
        baseline_frame(rows, pred.assign(p_positive=1.))


def test_prefix_mask_axes_and_designated_readout():
    mask, positions, last = prefix_mask_positions(torch.tensor([[1, 1, 1], [1, 1, 0]]), 4)
    assert mask.tolist() == [[1]*7, [1]*6 + [0]]
    assert positions.tolist() == [[list(range(7)), list(range(6)) + [0]]]*3
    assert last.tolist() == [6, 5]


def test_real_qwen_decoder_forward_and_backward_on_cpu():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.51.3":
        pytest.skip("Exact 4.51.3 implementation test")
    from transformers import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    torch.manual_seed(5)
    cfg = Qwen2_5_VLConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                           rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]})
    cfg._attn_implementation = "sdpa"
    decoder = Qwen2_5_VLModel(cfg).requires_grad_(False)
    attn = decoder.layers[0].self_attn
    attn.q_proj = DecoderLoRA(attn.q_proj, 8, 16)
    attn.v_proj = DecoderLoRA(attn.v_proj, 8, 16)
    module = VisualCorrection(32, {"visual_tokens": 4, "projector_width": 16, "residual_scale": .1})
    ids, mask = torch.tensor([[2, 5, 8], [2, 4, 0]]), torch.tensor([[1, 1, 1], [1, 1, 0]])
    x, base = torch.rand(2, 1536), torch.tensor([.4, .7])
    with torch.no_grad():
        text_reference = decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        _, rope, _ = prefix_mask_positions(mask, 0)
        direct = decoder(inputs_embeds=decoder.get_input_embeddings()(ids), attention_mask=mask,
                         position_ids=rope, use_cache=False).last_hidden_state
        torch.testing.assert_close(direct[mask.bool()], text_reference[mask.bool()], rtol=0, atol=0)
        torch.testing.assert_close(module(decoder, x, ids, mask, base), torch.logit(base), rtol=0, atol=0)
        torch.testing.assert_close(module.states(decoder, x[1:], ids[1:, :2], mask[1:, :2]),
                                   module.states(decoder, x, ids, mask)[1:], rtol=1e-5, atol=1e-6)
    decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decoder.train()
    parameters = [*module.parameters(), *[p for p in decoder.parameters() if p.requires_grad]]
    opt = torch.optim.AdamW(parameters, lr=.005)
    k, n = torch.tensor([1., 5.]), torch.tensor([5., 7.])
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        logits = module(decoder, x, ids, mask, base)
        loss = vote_loss(logits, k, n)
        torch.testing.assert_close(loss, (F.softplus(-logits)*k + F.softplus(logits)*(n-k)).sum()/n.sum())
        loss.backward()
        opt.step()
    for p in (module.projector[0].weight, attn.q_proj.B, attn.v_proj.B):
        assert torch.isfinite(p.grad).all() and p.grad.norm() > 0
    assert attn.v_proj.B.norm() > 0
    assert all(p.grad is None for p in decoder.parameters() if not p.requires_grad)
    decoder.eval()
    with torch.no_grad():
        assert not torch.equal(module(decoder, x, ids, mask, base),
                               module(decoder, x.flip(0), ids, mask, base))
