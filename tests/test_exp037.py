import ast
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from st3.exp036_io import export, restore, sha, write
from st3.exp037 import EMOTIONS, FusionHeads, Progressive, blends, counts_for, image_tokens, losses
from st3.exp037_io import safe_members, unpack_inputs, verify_inputs

ROOT = Path(__file__).resolve().parents[1]
CFG = {"fusion_width": 8, "fusion_heads": 2, "fusion_blocks": 2, "dropout": 0.,
       "qwen_width": 32, "qwen_layers": 3, "qwen_tail": 2, "qwen_rank": 2, "qwen_alpha": 4, "seed": 19}
VISUAL = {"hidden_size": 12, "layers": 2, "adapter_layers": 1,
          "adapter_modules": ["self_attn.q_proj", "self_attn.v_proj"], "rank": 2, "scale": 2}


def test_token_gather_excludes_padding_prompt_and_retains_gradients():
    state = torch.randn(2, 6, 32, requires_grad=True)
    ids = torch.tensor([[50, 52, 50, 53, 20, 21], [52, 50, 50, 53, 20, 0]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
    layout = torch.tensor([[1, 2, 2], [1, 2, 4]])
    context, padding = image_tokens(state, ids, mask, 50, layout, 2)
    assert padding.tolist() == [[False, True], [False, False]]
    torch.testing.assert_close(context[0, 0], state[0, 2])
    heads = FusionHeads(12, 2, CFG, auxiliary=True)
    patches, pool = torch.randn(2, 3, 12), torch.randn(2, 12)
    fused = heads(patches, pool, context, padding)
    altered = torch.where(padding.unsqueeze(2), 1e5, context)
    torch.testing.assert_close(heads(patches, pool, altered, padding), fused, atol=0., rtol=0.)
    fused[:, 12].sum().backward()
    assert torch.equal(state.grad.norm(dim=-1).ne(0), ids.eq(50) & mask.bool())
    with pytest.raises(ValueError, match="grid"):
        image_tokens(state, ids, torch.ones_like(mask), 50, layout, 2)


def test_vote_losses_and_microbatch_normalization():
    annotations = pd.DataFrame(
        [("x", "a", "train", emotion) for emotion in EMOTIONS] +
        [("y", "b", "train", EMOTIONS[0])]*5 + [("x", "a", "train", "other")],
        columns=["painting", "language", "split", "emotion"],
    )
    pairs = pd.DataFrame([("x", "a", 4, 8), ("y", "b", 5, 5)], columns=["painting", "language", "k", "n"])
    counts = counts_for(annotations, pairs)
    assert counts.tolist() == [[1]*8, [5]+[0]*7]
    with pytest.raises(ValueError, match="training annotations"):
        counts_for(annotations.assign(split="val"), pairs)
    heads = FusionHeads(12, 2, CFG, auxiliary=True)
    with torch.no_grad():
        for parameter in heads.parameters():
            parameter.zero_()
    z = torch.randn(2, 20)
    batch = (*[torch.arange(2)]*2, torch.tensor([4., 5.]), torch.tensor([8., 5.]),
             torch.as_tensor(counts), torch.ones(2, 12), torch.tensor([8., 0.]))
    full = losses(heads, z, batch, 13., 8.)
    split = sum(losses(heads, z, tuple(x[i:i+1] for x in batch), 13., 8.) for i in range(2))
    torch.testing.assert_close(full, split)
    torch.testing.assert_close(full, torch.tensor([np.log(2), np.log(8), 1.], dtype=torch.float32))


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(12, 12)
            layer.self_attn.v_proj = nn.Linear(12, 12)
            self.encoder.layers.append(layer)

    def forward(self, pixel_values):
        h = pixel_values
        for layer in self.encoder.layers:
            h = layer.self_attn.q_proj(h) + layer.self_attn.v_proj(h)
        return SimpleNamespace(last_hidden_state=h, pooler_output=h.mean(1))


def test_native_tiny_qwen_stage_isolation_and_inference_restore():
    torch.manual_seed(7)
    qwen = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
    txt = {"hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
           "num_hidden_layers": 3, "head_dim": 16, "vocab_size": 64,
           "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]}}
    vis = {"depth": 3, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32,
           "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "deepstack_visual_indexes": [0, 1, 2]}
    cfg = qwen.Qwen3VLConfig(text_config=txt, vision_config=vis, image_token_id=50, video_token_id=51,
                             vision_start_token_id=52, vision_end_token_id=53)
    decoder = qwen.Qwen3VLModel(cfg)
    decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = Progressive(TinyVision(), decoder, VISUAL, CFG, 2)
    # Variable native image token counts in the same forward.
    ids = torch.tensor([[0, 52, 50, 53, 21, 10], [52, 50, 50, 53, 21, 10]])
    pixels = torch.randn(2, 3, 12)
    encoded = dict(input_ids=ids, attention_mask=ids.ne(0).long(), pixel_values=torch.randn(12, 24),
                   image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 4]]))
    groups = torch.tensor([0, 0, 1])
    languages = torch.tensor([0, 1, -1])
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if n not in model.learned()}
    assert len(model.qwen_targets) == 4
    for stage, active in (("A", {"fusion"}), ("B", {"fusion", "vision"}), ("C", {"fusion", "vision", "qwen"})):
        model.stage(stage)
        model.train()
        assert active == {name for name, group in model.groups().items() if all(p.requires_grad for p in group)}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
        before = {name: [p.detach().clone() for p in group] for name, group in model.groups().items()}
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            h = model.features(pixels, encoded)
            # Test gradients from sentiment and a nonzero feature-level supervision signal.
            (model.heads.logits(h, groups, languages).sum() + h[:, 12].sum()).backward()
            optimizer.step()
        for name, group in model.groups().items():
            assert any(not torch.equal(old, p) for old, p in zip(before[name], group)) == (name in active)
    assert all(torch.equal(p, frozen[n]) and p.grad is None for n,p in model.named_parameters() if n in frozen)
    model.eval()
    reference = model.heads.logits(model.features(pixels, encoded), groups, languages).detach()
    retained = model.snapshot(inference=True)
    assert all("caption" not in name and "emotion" not in name for name in retained)
    with torch.no_grad():
        model.heads.shared.weight.zero_()
    model.restore(retained, inference=True)
    torch.testing.assert_close(model.heads.logits(model.features(pixels, encoded), groups, languages), reference,
                               atol=0., rtol=0.)


@pytest.mark.parametrize("names", [["A.jpg", "a.JPG"], ["p/A.jpg", "P/b.jpg"], ["NUL.jpg"], ["ok."],
                                  ["../x.jpg"], ["bad's.jpg"], ["a", "a/b"], ["/a"], ["a//b"], ["a\\b"]])
def test_kaggle_filename_rejections(names):
    with pytest.raises(ValueError):
        safe_members(names)


def test_intact_and_recursively_extracted_kaggle_input(tmp_path):
    root = tmp_path / "input/owner/dataset/version/exp037_inputs"
    (root / "artwork").mkdir(parents=True)
    temp_file = root / "artwork/file"
    temp_file.write_bytes(b"art")
    physical = f"artwork/{sha(temp_file)}.jpg"
    temp_file.rename(root / physical)
    logical = "Portrait/artist's-painting.jpg"
    manifest = {"files": {physical: sha(root / physical)}, "images": {logical: physical}}
    write(root / "exp037_manifest.json", manifest)
    pin = sha(root / "exp037_manifest.json")
    extracted = next((tmp_path / "input").rglob("exp037_manifest.json")).parent
    assert verify_inputs(extracted, pin) == manifest
    zip_path = tmp_path / "exp037_inputs.zip"
    with ZipFile(zip_path, "w") as z:
        for entry in (physical, "exp037_manifest.json"):
            z.write(root / entry, "exp037_inputs/"+entry)
    mounted = unpack_inputs(zip_path, tmp_path / "restored", pin, sha(zip_path))
    assert verify_inputs(mounted, pin) == manifest
    (mounted / physical).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="corrupt"):
        verify_inputs(mounted, pin)


def test_checkpoint_roundtrip_restores_exact_optimizer_schedule_and_rng(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    worker = importlib.import_module("run_exp037")
    monkeypatch.setattr(worker, "DEVICE", torch.device("cpu"))
    p = nn.Parameter(torch.tensor([1.]))
    model = SimpleNamespace(snapshot=lambda: {"p": p.detach().clone()})
    opt = torch.optim.AdamW([p], lr=.001)
    schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.8)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    def step():
        opt.zero_grad(set_to_none=True)
        (p * torch.rand(1)).sum().backward()
        opt.step()
        schedule.step()

    identity = {"source": "pinned"}
    root = tmp_path / "run"
    root.mkdir()
    step()
    boundary = {"stage": "B", "epoch": 2, "step": 24}
    worker.save_boundary(root, identity, model, opt, schedule, scaler, boundary)
    step()
    expected, expected_lr = p.detach().clone(), opt.param_groups[0]["lr"]
    archive = tmp_path / "progress.zip"
    export(archive, root, list(worker.exportable(root)), "handoff")
    restored = tmp_path / "fresh"
    restored.mkdir()
    restore(archive, restored, "handoff")
    state = worker.latest(restored, identity)
    assert state["boundary"] == boundary
    with torch.no_grad():
        p.copy_(state["model"]["p"])
    opt.load_state_dict(state["optimizer"])
    schedule.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    worker.restore_rng(state["rng"], worker.DEVICE)
    step()
    torch.testing.assert_close(p, expected, atol=0., rtol=0.)
    assert opt.param_groups[0]["lr"] == expected_lr
    with pytest.raises(ValueError, match="identity"):
        worker.latest(restored, {"source": "different"})


def test_blend_retains_incumbent_and_validates_new_candidate(tmp_path):
    from st3.submission import pack, validate_archive

    gold = pd.DataFrame([(str(i), l, k, 10) for i in range(25) for l,k in zip("abcde", [1,3,5,7,9])],
                        columns=["painting", "language", "k", "n"])
    parent = gold[["painting", "language"]].assign(p_positive=gold.k / 10)
    new = parent.assign(p_positive=1-parent.p_positive)
    config = {"ensemble_grid": [0, .2, .5, 1], "test_like_excluded": ["a"]}
    table, best, output = blends(gold, parent.iloc[:, :2], parent, new, config)
    assert len(table) == 4 and best["alpha"] == 0
    csv = tmp_path / "predictions.csv"
    output.to_csv(csv, index=False)
    archive = tmp_path / "submission.zip"
    pack(csv, archive, parent)
    pd.testing.assert_frame_equal(validate_archive(archive, parent), parent, check_exact=True)


def test_kaggle_notebook_discovers_both_readonly_layouts_and_pins_worker(tmp_path):
    notebook = json.loads((ROOT / "notebooks/EXP037_Kaggle.ipynb").read_text(encoding="utf-8"))
    code = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    assert list(code) == ["owner-settings", "verify-inputs", "run", "outputs"]
    for label, source in code.items():
        compile(source, label, "exec")
    state = {}
    verify = ast.parse(code["verify-inputs"])
    safe_prefix = ast.Module(body=verify.body[:next(i for i, node in enumerate(verify.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "STARTED")],
        type_ignores=[])
    exec(compile(safe_prefix, "verify-helpers", "exec"), state)
    assert len(state["WORKER_SOURCE"]) == 40
    assert len(state["MANIFEST_SHA256"]) == len(state["ARCHIVE_SHA256"]) == 64
    source = state["WORKER_SOURCE"]
    extracted = tmp_path / "layout-a/owner/dataset/exp037_inputs"
    (extracted / "runtime").mkdir(parents=True)
    (extracted / "runtime/example.py").write_text("x = 1", encoding="utf-8")
    manifest = {"source": source, "files": {"runtime/example.py": sha(extracted / "runtime/example.py")}}
    write(extracted / "exp037_manifest.json", manifest)
    checksum = sha(extracted / "exp037_manifest.json")
    unmodified = tmp_path / "layout-b/owner/dataset/exp037_inputs.zip"
    unmodified.parent.mkdir(parents=True)
    with ZipFile(unmodified, "w") as z:
        for member in ["exp037_manifest.json", "runtime/example.py"]:
            z.write(extracted / member, "exp037_inputs/"+member)
    locate = state["locate_inputs"]
    assert locate(tmp_path / "layout-a", tmp_path / "scratch", checksum, sha(unmodified), source) == (extracted, manifest)
    mounted, result = locate(tmp_path / "layout-b", tmp_path / "scratch", checksum, sha(unmodified), source)
    assert result == manifest and mounted.parent.parent == tmp_path / "scratch"
    with pytest.raises(ValueError, match="different worker"):
        locate(tmp_path / "layout-a", tmp_path / "scratch", checksum, sha(unmodified), "wrong")
