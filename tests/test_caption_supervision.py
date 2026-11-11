import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from st3 import caption_comparison as comparison, data
from st3.caption_data import (allowed_captions, audit_captions, lock_protocol, require_isolation,
                              text_digest, training_captions)
from st3.caption_student import (CaptionStudent, make_rows, predict_student, seed_model, text_loss, train_epoch)
from st3.caption_targets import fit_targets, raw_targets
from st3.caption_text import load_teacher, load_text_model, masked_mean, teacher_texts, text_projection, tokenize_diagnostics
from st3.fusion import blend

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/exp017.json").read_text())


def caption_frame():
    return pd.DataFrame({"painting": ["a", "a", "a", "a", "b"], "language": "Tamil",
                         "split": "train", "image_name": ["a.png"] * 4 + ["b.png"],
                         "emotion": ["awe", "awe", "sadness", "other", "awe"],
                         "caption": ["same", "same", "different", "other", "held"],
                         "caption_id": [text_digest(s) for s in ("same", "same", "different", "other", "held")],
                         "valid_caption": True, "binary": [True, True, True, False, True]})


def test_official_train_only_caption_identity_and_emotion_audit(tmp_path):
    (tmp_path / "annotations").mkdir()
    frame = caption_frame().drop(columns=["valid_caption", "binary", "caption_id"])
    frame.loc[4, ["split", "caption"]] = ["val", "FORBIDDEN VALIDATION CAPTION"]
    frame.to_csv(tmp_path / data.ANNOTATIONS, index=False)
    result = training_captions(tmp_path)
    assert len(result) == 4 and result.caption.tolist() == ["same", "same", "different", "other"]
    assert result.binary.tolist() == [True, True, True, False]
    pairs = pd.DataFrame({"painting": ["a"], "language": ["Tamil"]})
    summary, languages = audit_captions(result, pairs)
    assert summary["emotion_counts"] == {"awe": 2, "sadness": 1, "other": 1}
    assert languages.iloc[0]["emotion_awe"] == 2 and summary["duplicate_annotation_rows"] == 1


def test_targets_keep_duplicate_votes_and_exclude_other_and_held_text():
    frame = caption_frame()
    pairs = pd.DataFrame({"painting": ["a"], "language": ["Tamil"]})
    vectors = {text_digest("same"): np.array([1., 0.], dtype=np.float32),
               text_digest("different"): np.array([0., 1.], dtype=np.float32)}
    records, values = raw_targets(frame, pairs, vectors)
    assert records.captions.tolist() == [3] and records.positive.tolist() == [2]
    np.testing.assert_allclose(values, [[2/3, 1/3]])
    assert allowed_captions(frame, pairs).caption.tolist() == ["same", "same", "different"]
    with pytest.raises(ValueError, match="training captions"):
        allowed_captions(frame.assign(split="val"), pairs)


def test_locked_protocol_is_deterministic_and_groups_image_content(tmp_path):
    (tmp_path / "images").mkdir()
    rows = []
    for i in range(11):
        name = f"{i}.png"
        (tmp_path / "images" / name).write_bytes(str(i if i != 1 else 0).encode())
        rows.append({"painting": f"p{i}", "image_name": name, "split": "train" if i < 10 else "val"})
    annotations = pd.DataFrame(rows)
    train = annotations[annotations.split.eq("train")].rename(columns={"image_name": "image"})
    first = lock_protocol(train, tmp_path, annotations, CONFIG)
    second = lock_protocol(train.sample(frac=1, random_state=1), tmp_path, annotations, CONFIG)
    pd.testing.assert_frame_equal(first, second)
    assert first.set_index("painting").loc[["p0", "p1"], ["role", "fold"]].drop_duplicates().shape[0] == 1
    (tmp_path / "images/10.png").write_bytes(b"0")
    with pytest.raises(ValueError, match="image_sha256"):
        lock_protocol(train, tmp_path, annotations, CONFIG)


@pytest.mark.parametrize("column", ["painting", "image", "image_sha256"])
def test_split_integrity_rejects_each_identity_overlap(column):
    with pytest.raises(ValueError, match=column):
        require_isolation(pd.DataFrame({column: ["same"]}), pd.DataFrame({column: ["same"]}))


def test_clip_projected_embedding_and_e5_masked_mean():
    class Model:
        def __call__(self, **inputs):
            return SimpleNamespace(text_embeds=torch.tensor([[3., 4.]]),
                                   pooler_output=torch.tensor([[1., 0.]]),
                                   last_hidden_state=torch.tensor([[[2., 0.], [0., 2.], [100., 100.]]]))
    inputs = {"attention_mask": torch.tensor([[1, 1, 0]])}
    torch.testing.assert_close(text_projection(Model(), inputs, "clip"), torch.tensor([[.6, .8]]))
    torch.testing.assert_close(masked_mean(Model()().last_hidden_state, inputs["attention_mask"]), torch.tensor([[1., 1.]]))
    torch.testing.assert_close(text_projection(Model(), inputs, "e5"), torch.tensor([[2**-.5, 2**-.5]]))


def test_tokenizer_original_text_prefix_and_truncation_diagnostics():
    raw = ["原始 caption", "long"]
    assert teacher_texts(raw, "clip") == raw
    assert teacher_texts(raw, "e5") == ["passage: 原始 caption", "passage: long"]
    class Tokenizer:
        unk_token_id = 99
        def __call__(self, texts, **kwargs):
            assert texts == raw and kwargs == {"truncation": False, "add_special_tokens": True}
            return {"input_ids": [[0, 99, 2], list(range(80))]}
    diagnostic = tokenize_diagnostics(Tokenizer(), raw, "clip", 77)
    assert diagnostic.truncated.tolist() == [False, True]
    assert diagnostic.unknown_tokens.tolist() == [1, 0]


def test_clip_loader_requires_the_trained_text_projection(monkeypatch):
    import sys

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert kwargs["use_safetensors"] and kwargs["attn_implementation"] == "eager"
            return object(), {"missing_keys": ["text_projection.weight"], "mismatched_keys": [], "unexpected_keys": []}
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=Loader, CLIPTextModelWithProjection=Loader))
    with pytest.raises(ValueError, match="Incomplete text teacher"):
        load_text_model("fixture", "clip")


def test_text_cache_pins_coverage_and_payload_checksum(tmp_path):
    np.savez(tmp_path / "features.npz", keys=["a"], features=np.array([[1., 0.]], dtype=np.float32))
    from st3.baseline import file_sha256
    config = {"model": "fixture", "width": 2}
    audit = {"config_sha256": "config", "protocol_sha256": "protocol"}
    manifest = {**audit, "identity": config, "fit_paintings": ["train"],
                "features_sha256": file_sha256(tmp_path / "features.npz")}
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    assert set(load_teacher(tmp_path, config, audit, ["train"])[0]) == {"a"}
    with pytest.raises(ValueError, match="coverage"):
        load_teacher(tmp_path, config, audit, ["held"])
    with (tmp_path / "features.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_teacher(tmp_path, config, audit, ["train"])


def test_partition_only_pca_and_degenerate_mask():
    frame = pd.DataFrame({"painting": list("abcH"), "language": "Tamil", "split": "train", "valid_caption": True,
                          "binary": True, "emotion": "awe", "caption_id": list("abcH")})
    pairs = frame.iloc[:3][data.KEYS]
    vectors = dict(zip("abc", np.eye(3, dtype=np.float32)))
    result = fit_targets(frame, pairs, vectors, {**CONFIG, "target_dimensions": 2})
    targets, weights, pca, _, metadata = result
    np.testing.assert_allclose(pca.mean_, np.ones(3) / 3)
    assert metadata["fit_paintings"] == list("abc") and not metadata["whiten"]
    assert all(np.isclose(np.linalg.norm(v), 1) for v in targets.values()) and list(weights.values()) == [1., 1., 1.]
    identical = dict.fromkeys("abc", np.array([1., 0., 0.], dtype=np.float32))
    with pytest.warns(RuntimeWarning):
        masked = fit_targets(frame, pairs, identical, {**CONFIG, "target_dimensions": 2})
    assert not masked[0] and masked[-1]["degenerate_targets"] == 3


def test_text_loss_is_dimension_normalized_and_ignores_missing_targets():
    predicted = torch.tensor([[1., 1.], [100., 100.], [3., 3.]], requires_grad=True)
    value = text_loss(predicted, torch.zeros_like(predicted), torch.tensor([.5, 0., 1.]))
    assert value.item() == pytest.approx((.5 + 9.) / 1.5)
    value.backward()
    assert torch.equal(predicted.grad[1], torch.zeros(2))


def test_zero_auxiliary_weight_matches_bce_parameters_exactly():
    cfg = {**CONFIG, "text_weight": 0., "batch_size": 2}
    frame = pd.DataFrame({"painting": ["a", "b"], "image": ["a", "b"], "language": "Tamil", "k": [2, 4], "n": 5})
    features = {name: np.ones(2048, dtype=np.float32) * value for name, value in (("a", .1), ("b", .2))}
    targets = {(name, "Tamil"): np.ones(64, dtype=np.float32) / 8 for name in ("a", "b")}
    states = []
    for supervised in (False, True):
        rows = make_rows(frame, features, ["Tamil"], "cpu", targets if supervised else None,
                         {key: 1. for key in targets})
        model, rng, optimizer = seed_model(["Tamil"], 2027, cfg, "cpu")
        train_epoch(model, rows, rng, optimizer, cfg, 0.)
        states.append(model.state_dict())
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


def test_inference_ignores_caption_and_label_fields_and_unknown_language_falls_back():
    model = CaptionStudent(["Tamil"], CONFIG)
    inputs = pd.DataFrame({"painting": ["a", "b"], "language": ["Tamil", "Kazakh"], "image": ["a", "a"]})
    baseline = inputs[data.KEYS].assign(p_positive=[.3, .777777])
    features = {"a": np.zeros(2048, dtype=np.float32)}
    first = predict_student(model, inputs, features, baseline, "cpu", 4)
    second = predict_student(model, inputs.assign(caption=object(), k=-100), features, baseline, "cpu", 4)
    np.testing.assert_array_equal(first, second)
    assert first[1] == .777777


def test_nested_baseline_reconstruction_and_single_blend_rounding(monkeypatch):
    held = pd.DataFrame({"painting": ["held"], "image": ["held.png"], "language": "Tamil"})
    fit = held.assign(painting="train", image="train.png")
    parts = iter([.1, .100001, .100001])
    monkeypatch.setattr(comparison, "fit_head", lambda *args: (held[data.KEYS].assign(p_positive=next(parts)), {"converged": True}))
    baseline, _ = comparison.refit_baseline(fit, held, [{}, {}, {}])
    assert baseline.p_positive.tolist() == [.1]
    assert blend(baseline, baseline.assign(p_positive=.100001), .5).p_positive.tolist() == [.1]


def test_gate_requires_incremental_gain_and_completed_conditions():
    table = pd.DataFrame({"pooled_contrast": [.44, .442, .4435, .446], "fold_wins": [0, 3, 3, 3],
                          "log_loss_mean": .72, "folds": 5}, index=["M0", "B1", "B2", "B3"])
    assert comparison.select_caption(table, CONFIG, True)["model"] == "B3"
    table.loc["B3", "log_loss_mean"] = .726
    assert comparison.select_caption(table, CONFIG, True)["model"] == "M0"
    assert comparison.select_caption(table, CONFIG, False)["status"] == "NO_GO"


def test_modified_decision_cannot_open_confirmation(tmp_path, monkeypatch):
    from st3 import caption_study
    from st3.baseline import file_sha256

    directory = tmp_path / "cv"
    directory.mkdir()
    table = pd.DataFrame({"pooled_contrast": .44, "fold_wins": 0, "log_loss_mean": .72, "folds": 5},
                         index=pd.Index(["M0", "B1", "B2", "B3"], name="model"))
    table.to_csv(directory / "summary.csv")
    fits = [{"fold": fold, "model": model, "seed": seed} for fold in range(1, 6)
            for model in ("M1", "M2", "M3") for seed in CONFIG["seeds"]]
    (directory / "fits.json").write_text(json.dumps(fits))
    (directory / "run.json").write_text("{}")
    decision = {**comparison.select_caption(table, CONFIG, True), "config_sha256": "frozen",
                "sha256": {name: file_sha256(directory / name) for name in ("summary.csv", "fits.json", "run.json")}}
    decision["model"], decision["status"] = "B2", "PASS"
    (directory / "selection.json").write_text(json.dumps(decision))
    monkeypatch.setattr(caption_study, "audited_inputs", lambda *args: (CONFIG, None, {"config_sha256": "frozen"}))
    with pytest.raises(ValueError, match="frozen CV gate"):
        caption_study.checked_decision(SimpleNamespace(run_dir=tmp_path), "cv")


def test_no_go_keeps_holdout_and_development_closed(monkeypatch):
    monkeypatch.setattr(comparison, "checked_decision", lambda *args: {"status": "NO_GO"})
    monkeypatch.setattr(comparison, "audited_inputs", lambda *args: pytest.fail("Opened data after NO-GO"))
    comparison.run_caption_holdout(None)
    comparison.run_caption_dev(None)


def test_confirmed_candidate_compares_both_baseline_and_matched_control(tmp_path, monkeypatch):
    rows = [{"painting": f"held-{i}", "language": language, "image": f"{i}.png", "role": "holdout",
             "k": k, "n": 5, "p": k / 5, "split": "train"}
            for i in range(20) for language, k in (("Tamil", 1), ("Korean", 2), ("Thai", 4))]
    held = pd.DataFrame(rows)
    fit = held.iloc[:3].assign(painting="fit", image="fit.png", role="development")
    train = pd.concat([fit, held], ignore_index=True)
    baseline = held[data.KEYS].assign(p_positive=np.tile([.35, .65, .55], 20))
    predictions = {"M0": baseline, "B1": baseline.assign(p_positive=np.tile([.3, .4, .5], 20)),
                   "B2": held[data.KEYS].assign(p_positive=np.tile([.2, .4, .8], 20))}
    monkeypatch.setattr(comparison, "checked_decision", lambda *args: {"status": "PASS", "model": "B2"})
    monkeypatch.setattr(comparison, "audited_inputs", lambda *args: (CONFIG, train, {"config_sha256": "frozen"}))
    monkeypatch.setattr(comparison, "load_vectors", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(comparison, "image_banks", lambda *args: [])
    monkeypatch.setattr(comparison, "training_captions", lambda *args: caption_frame().assign(painting="fit"))
    monkeypatch.setattr(comparison, "refit_baseline", lambda *args: (baseline, []))
    monkeypatch.setattr(comparison, "train_conditions", lambda *args: (predictions, []))
    comparison.run_caption_holdout(SimpleNamespace(run_dir=tmp_path, device="cpu", data_dir=tmp_path))
    decision = json.loads((tmp_path / "holdout/selection.json").read_text())
    assert decision["status"] == "PASS" and decision["gain_over_B1"] > 0
    assert (tmp_path / "holdout/matched_control/metrics.json").is_file()
