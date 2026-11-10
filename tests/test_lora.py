import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from st3 import data, lora_training, submission
from st3.baseline import file_sha256
from st3.clip_lora import ClipTail, adapter_parameters, frozen_digest, load_adapter, save_adapter, vision_prefix
from st3.e50_heads import fit_frozen_head
from st3.fusion import blend
from st3.lora_study import choose_lora, run_lora_dev
from st3.lora_training import FrozenReadout, PaintingRows, check_readout, step, train_phase
from st3.nonlinear import early_split, vote_loss
from st3.residual_crossfit import disjoint

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "exp014.json").read_text())


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, name, nn.Linear(8, 8))

    def forward(self, x, _mask, _causal):
        attn = self.self_attn
        return (x + attn.out_proj(attn.q_proj(x) + attn.k_proj(x) + attn.v_proj(x)),)


def toy_encoder():
    torch.manual_seed(12)
    model = nn.Module()
    model.vision_model = nn.Module()
    vision = model.vision_model
    vision.embeddings = nn.Identity()
    vision.pre_layrnorm, vision.post_layernorm = nn.LayerNorm(8), nn.LayerNorm(8)
    vision.encoder = nn.Module()
    vision.encoder.layers = nn.ModuleList(ToyBlock() for _ in range(4))
    model.visual_projection = nn.Linear(8, 6, bias=False)
    return model


@pytest.mark.parametrize("checkpointing", [False, True])
def test_only_final_two_query_and_value_layers_train_and_reload(tmp_path, checkpointing):
    model = toy_encoder()
    x = torch.randn(3, 4, 8)
    prefix = vision_prefix(model, x, 2).detach()
    original = prefix
    for layer in model.vision_model.encoder.layers[-2:]:
        original = layer(original, None, None)[0]
    expected = nn.functional.normalize(model.visual_projection(model.vision_model.post_layernorm(original[:, 0])), dim=-1)
    tail = ClipTail(model, {**CONFIG, "gradient_checkpointing": checkpointing})
    assert tail.targets == [f"vision_model.encoder.layers.{i}.self_attn.{key}" for i in (2, 3) for key in ("q_proj", "v_proj")]
    params = adapter_parameters(tail)
    assert len(params) == 8 and sum(p.numel() for p in params.values()) == 256
    assert all(name.endswith((".a", ".b")) and ("q_proj" in name or "v_proj" in name) for name in params)
    torch.testing.assert_close(tail(prefix), expected, rtol=0, atol=0)
    frozen = frozen_digest(tail)
    optimizer = torch.optim.AdamW(params.values(), lr=.01)
    for i in range(2):
        optimizer.zero_grad()
        tail(prefix)[:, 0].sum().backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert all((p.grad.norm() == 0 if i == 0 and name.endswith(".a") else p.grad.norm() > 0)
                   for name, p in params.items())
        optimizer.step()
    trained = tail(prefix).detach()
    assert not torch.equal(trained, expected)
    file = tmp_path / "adapter.npz"
    save_adapter(tail, file)
    tail.reset(CONFIG["seed"])
    torch.testing.assert_close(tail(prefix), expected, rtol=0, atol=0)
    load_adapter(tail, file)
    torch.testing.assert_close(tail(prefix), trained, rtol=0, atol=0)
    assert frozen_digest(tail) == frozen


def training_fixture():
    rows = [(f"p{i}", f"{i}.png", lang, i % 4, i % 3 + 5, "train")
            for i in range(12) for lang in ("Tamil", "Yoruba")]
    frame = pd.DataFrame(rows, columns=["painting", "image", "language", "k", "n", "split"])
    return frame, {f"{i}.png": np.eye(1, 4, k=i % 4, dtype=np.float32)[0] for i in range(12)}


def test_sklearn_readout_equivalence_and_grouped_early_stopping():
    frame, vectors = training_fixture()
    partition, held = frame.iloc[:20], frame.iloc[20:]
    fit, stop = early_split(partition, held, CONFIG, 1)
    disjoint(fit, stop, held)
    assert stop["painting"].nunique() == 1
    assert set(pd.concat([fit, stop]).index) == set(partition.index)
    _, replay = early_split(partition, held, CONFIG, 1)
    pd.testing.assert_frame_equal(stop, replay)
    with pytest.raises(ValueError, match="overlapping image"):
        early_split(partition, held.assign(image=partition.iloc[0]["image"]), CONFIG, 1)
    languages = sorted(fit["language"].unique())
    clf = fit_frozen_head(fit, vectors, languages, CONFIG["seed"])
    readout = FrozenReadout(clf, languages, width=4)
    assert not list(readout.parameters())
    query = stop.copy()
    query.iloc[0, query.columns.get_loc("language")] = "Kazakh"
    assert check_readout(clf, readout, query, vectors, torch.device("cpu"), 1e-6) < 1e-6


class ScalarTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.2))

    def reset(self, _seed):
        with torch.no_grad():
            self.weight.fill_(.2)

    def forward(self, images):
        return self.weight * images


def scalar_head(features, image, _language):
    return features[image, 0]


def test_one_image_per_painting_and_vote_normalized_accumulation():
    frame, _ = training_fixture()
    frame = frame.iloc[:8]
    index = {f"{i}.png": i for i in range(4)}
    bank = torch.arange(1., 5.).reshape(-1, 1)
    rows = PaintingRows(frame, index, ["Tamil", "Yoruba"], torch.device("cpu"))
    images, pair_image, *_ = rows.batch([0, 2])
    assert images.tolist() == [0, 2] and pair_image.tolist() == [0, 0, 1, 1]
    a, b = ScalarTail(), ScalarTail()
    optimizer = torch.optim.SGD(a.parameters(), lr=.1)
    config = {**CONFIG, "gradient_clip": 100.}
    actual, _ = step(a, scalar_head, bank, rows, [np.array([0]), np.array([1, 2, 3])], optimizer, config)
    image, pair, lang, targets, counts = rows.batch([0, 1, 2, 3])
    loss = vote_loss(scalar_head(b(bank[image]), pair, lang), targets, counts)
    (loss / counts.sum()).backward()
    assert float(actual) == pytest.approx(float(loss.detach()))
    torch.testing.assert_close(a.weight, b.weight - .1 * b.weight.grad)


def test_early_stopping_refit_and_seed_replay(monkeypatch):
    frame, _ = training_fixture()
    rows = PaintingRows(frame, {f"{i}.png": i for i in range(12)}, ["Tamil", "Yoruba"], torch.device("cpu"))
    tail = ScalarTail()
    bank = torch.ones((12, 1))
    outcomes = []
    for _ in range(2):
        validation = iter([.8, .7, .72, .73])
        monkeypatch.setattr(lora_training, "validation_loss", lambda *args: next(validation))
        curve, selected = train_phase(tail, scalar_head, bank, rows, CONFIG, 6, rows)
        assert selected == 2 and len(curve) == 4
        refit, _ = train_phase(tail, scalar_head, bank, rows, CONFIG, selected)
        assert len(refit) == 2 and all(row["validation_bce"] is None for row in refit)
        outcomes.append((float(tail.weight.detach()), [row["train_bce"] for row in refit]))
    assert outcomes[0] == outcomes[1]


def test_only_m2_can_promote_and_requires_all_four_checks():
    summary = pd.DataFrame({"pooled_contrast": [.44, .47, .444], "log_loss_mean": [.72, .72, .724],
                            "fold_wins": [0, 5, 4], "folds": 5, "converged": 5, "scored": 5, "loss_scored": 5},
                           index=["M0", "M1", "M2"])
    assert choose_lora(summary, CONFIG, 5)["model"] == "M2"
    for column, invalid in (("fold_wins", 3), ("converged", 4), ("pooled_contrast", .4429), ("log_loss_mean", .7251)):
        changed = summary.copy()
        changed.loc["M2", column] = invalid
        assert choose_lora(changed, CONFIG, 5)["model"] == "M0"


def test_fixed_blend_zip_and_no_go_keep_original_artifacts(tmp_path):
    original = tmp_path / submission.FILENAME
    inputs = pd.DataFrame({"painting": ["p", "p"], "language": ["Yoruba", "Tamil"]})
    e50 = inputs.assign(p_positive=[.2, .3])
    e50.to_csv(original, index=False)
    archived = file_sha256(original)
    adapted = inputs.iloc[::-1].assign(p_positive=[.5, .4])
    blend_csv = tmp_path / "new" / submission.FILENAME
    blend_csv.parent.mkdir()
    blend(e50, adapted, .5).to_csv(blend_csv, index=False)
    archive = blend_csv.parent / "submission.zip"
    submission.pack(blend_csv, archive, inputs)
    assert submission.validate_archive(archive, inputs)["p_positive"].tolist() == [.3, .4]

    cv = tmp_path / "cv"
    cv.mkdir()
    summary = pd.DataFrame({"pooled_contrast": .44, "log_loss_mean": .72, "fold_wins": 0, "folds": 5,
                            "converged": 5, "scored": 5, "loss_scored": 5}, index=pd.Index(["M0", "M1", "M2"], name="model"))
    result = choose_lora(summary, CONFIG, 5)
    summary.to_csv(cv / "summary.csv")
    (cv / "run.json").write_text(json.dumps({"config": CONFIG, "exp002_config": {"cv_folds": 5}}))
    (cv / "fold_runs.json").write_text("[]")
    result["sha256"] = {name: file_sha256(cv / name) for name in ("run.json", "summary.csv", "fold_runs.json")}
    (cv / "selection.json").write_text(json.dumps(result))
    run_lora_dev(SimpleNamespace(run_dir=tmp_path), None)
    assert not (tmp_path / "dev").exists() and file_sha256(original) == archived
