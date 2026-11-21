"""CPU regressions for structured artwork semantics and fold-local prediction."""

import json

import numpy as np
import pandas as pd
import pytest

from st3.competition_semantic_generation import CONFIG, cache_identity, parse_description, read_record, store_record
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
    from st3.competition_semantic_generation import TRACK1

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
