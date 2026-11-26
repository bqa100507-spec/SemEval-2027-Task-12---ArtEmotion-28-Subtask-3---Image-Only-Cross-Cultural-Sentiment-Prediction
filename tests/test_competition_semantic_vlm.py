"""CPU regressions for structured artwork semantics and fold-local prediction."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from st3 import competition_semantic_vlm as pipeline
from st3.baseline import file_sha256
from st3.competition_semantic_generation import CONFIG, TRACK1, cache_identity, parse_description, read_record, store_record
from st3.competition_semantic_vlm import evaluate, field_terms, predict_head, semantic_features
from st3.data import KEYS
from st3.e50_heads import read_json

RULE = read_json(CONFIG)


def description():
    return {"entities": "two figures", "scene": "an indoor meeting", "style_atmosphere": "soft light",
            "visual_cues": "faces turned away", "uncertain_context": "uncertain: the doorway may frame a departure"}


def test_structured_description_rejects_missing_or_unhedged_fields():
    fields = RULE["fields"]
    raw = json.dumps(description())
    assert parse_description(raw, fields) == description()
    with pytest.raises(ValueError, match="Duplicate"):
        parse_description(raw[:-1] + ', "entities": "duplicate"}', fields)
    with pytest.raises(ValueError, match="marked uncertain"):
        parse_description(json.dumps(description() | {"uncertain_context": "This depicts a particular ritual"}), fields)
    with pytest.raises(ValueError, match="field set"):
        parse_description(json.dumps(description() | {"language": "English"}), fields)
    with pytest.raises(ValueError, match="short strings"):
        parse_description(json.dumps(description() | {"entities": ["figures", "flags"]}), fields)


def test_cached_description_rejects_wrong_artwork_or_derived_fields(tmp_path):
    record = {"image_sha256": "art", "identity": cache_identity(RULE),
              "raw": json.dumps(description()), "description": description()}
    store_record(tmp_path, record)
    path = tmp_path / "art.json"
    assert read_record(path, "art", RULE) == record
    with pytest.raises(ValueError, match="identity mismatch"):
        read_record(path, "different", RULE)
    record["description"]["entities"] = "different from raw"
    store_record(tmp_path, record)
    with pytest.raises(ValueError, match="differs from its raw output"):
        read_record(path, "art", RULE)


def test_semantic_projection_uses_only_fitting_documents():
    fit = pd.Series({f"p{i}": description() | {"entities": "two walkers" if i % 2 else "stone bridge"}
                     for i in range(8)})
    query = pd.Series({"a": description(), "b": description()})
    modified = query.map(lambda d: {key: "heldonly" for key in d})
    first, original = semantic_features(fit, query, RULE)
    second, changed = semantic_features(fit, modified, RULE)
    np.testing.assert_array_equal(first.loc[fit.index], second.loc[fit.index])
    assert original["vocabulary_size"] == changed["vocabulary_size"]
    assert changed["zero_query_vectors"] == 2
    assert "entities:stone_bridge" in set(field_terms(fit.iloc[0]))
    with pytest.raises(ValueError, match="disjoint"):
        semantic_features(fit, fit.iloc[:2], RULE)


def test_vote_head_accepts_only_image_and_language_keys():
    train = pd.DataFrame([(p, l, k, 5) for p in ("a", "b") for l, k in (("Thai", 2), ("Korean", 4))],
                         columns=[*KEYS, "k", "n"])
    vectors = pd.DataFrame([[0., 1.], [1., 0.], [1., 0.]], index=["a", "b", "c"])
    query = pd.DataFrame([("c", "Thai")], columns=KEYS)
    track1 = read_json(TRACK1)
    result, receipt = predict_head(train, query, vectors, track1)
    assert receipt["converged"] and 0 <= result.p_positive.iloc[0] <= 1
    with pytest.raises(ValueError, match="keys only"):
        predict_head(train, query.assign(k=2), vectors, track1)


def test_equal_candidate_does_not_displace_corrected_ridge():
    gold = pd.DataFrame([(f"p{i}", l, 1 + i // 20, k, 5) for i in range(40)
                         for l, k in (("Thai", 2), ("Korean", 4), ("Hindi", 3), ("English", 1))],
                        columns=[*KEYS, "fold", "k", "n"])
    q = gold.k / gold.n
    result = evaluate(gold.assign(corrected_ridge=q, semantic=q, visual_semantic=q), RULE, ["English"])
    assert result["decision"] == "RETAIN CORRECTED RIDGE"
    assert result["filtered"]["pooled"]["semantic"]["log_loss_languages"] == 3


def test_cv_replays_sealed_cache_and_writes_aligned_predictions(tmp_path, monkeypatch):
    paintings = [f"p{i:02}" for i in range(30)]
    scope = pd.DataFrame({"painting": paintings, "image_sha256": paintings})
    gold = pd.DataFrame([(p, lang, i // 6 + 1, (i + j) % 6, 5)
                         for i, p in enumerate(paintings)
                         for j, lang in enumerate(["Thai", "Korean", "Hindi", "English"])],
                        columns=[*KEYS, "fold", "k", "n"])
    rng = np.random.default_rng(8)
    features = {name: pd.DataFrame(rng.normal(size=(30, 3)), index=paintings) for name in pipeline.ENCODERS}
    monkeypatch.setattr(pipeline, "training_inventory", lambda *_: scope)
    monkeypatch.setattr(pipeline, "load_p2", lambda *_: (gold, features, None))
    reference = tmp_path / "reference"
    reference.mkdir()
    prior = gold[[*KEYS, "fold"]].assign(corrected=0.2 + gold.k * 0.12)
    prior.sample(frac=1, random_state=8).to_csv(reference / "oof.csv", index=False)
    (reference / "metrics.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(run_dir=tmp_path, inputs=tmp_path, reference_dir=reference)
    config = RULE | {"incumbent_oof_sha256": file_sha256(reference / "oof.csv"),
                     "incumbent_metrics_sha256": file_sha256(reference / "metrics.json")}
    track1 = read_json(TRACK1)
    with pytest.raises(FileNotFoundError, match="30 P2 paintings. 0 present"):
        pipeline.run_cv(args, config, track1)
    identity = cache_identity(config)
    hashes = {}
    for i, p in enumerate(paintings):
        item = description() | {"entities": ["sunlit river", "stone bridge", "dark trees"][i % 3]}
        store_record(tmp_path / "descriptions", {"image_sha256": p, "identity": identity,
                                                   "raw": json.dumps(item), "description": item})
        hashes[p] = file_sha256(tmp_path / "descriptions" / f"{p}.json")
    (tmp_path / "extraction.json").write_text(json.dumps({"identity": identity, "images": hashes}), encoding="utf-8")
    pipeline.run_cv(args, config, track1)
    result = pd.read_csv(tmp_path / "cv/oof.csv")
    pd.testing.assert_frame_equal(result[[*KEYS, "fold"]], gold[[*KEYS, "fold"]])
    np.testing.assert_allclose(result.corrected_ridge, prior.corrected, atol=0, rtol=1e-15)
    assert result[list(pipeline.ARMS)].ge(0).all().all() and result[list(pipeline.ARMS)].le(1).all().all()
    assert all(file_sha256(tmp_path / "cv" / name) == digest
               for name, digest in read_json(tmp_path / "cv/seal.json").items())
