"""The recovery parser never invents missing semantic data."""

import json

import pandas as pd
import pytest

from st3.baseline import file_sha256
from st3.competition_semantic_generation import CONFIG
from st3.competition_semantic_recovery import completion, parse_compact, read_sample
from st3.e50_heads import read_json

FIELDS = read_json(CONFIG)["fields"]
ITEM = {"e": "woman, tree", "s": "garden path", "a": "muted green palette", "v": "downcast eyes",
        "u": "uncertain: posture suggests fatigue"}


def test_compact_entities_are_canonicalized_without_adding_semantics():
    compact = parse_compact(json.dumps(ITEM), FIELDS)
    assert set(compact) == set(FIELDS)
    assert compact["entities"] == "woman, tree"
    assert compact["uncertain_context"] == ITEM["u"]
    assert parse_compact(json.dumps(ITEM | {"e": ["woman", "tree"]}), FIELDS) == compact


@pytest.mark.parametrize("raw", [
    json.dumps(ITEM)[:-1],
    json.dumps(ITEM)[:-1] + ', "e":"repeated"}',
    json.dumps(ITEM | {"u": "she is grieving"}),
    json.dumps(ITEM | {"v": ""}),
    json.dumps(ITEM | {"e": ["objects", None]}),
    json.dumps(ITEM | {"s": ["garden path"]}),
    json.dumps(ITEM | {"probability": 0.75}),
])
def test_compact_does_not_repair_bad_json_or_guess_missing_fields(raw):
    with pytest.raises(ValueError):
        parse_compact(raw, FIELDS)


def test_batch_padding_is_excluded_from_per_artwork_token_counts():
    assert completion([1, 2, 8, 0, 0], {8, 9}, 5) == ([1, 2, 8], True)
    assert completion([1, 2, 3], {8, 9}, 3) == ([1, 2, 3], False)


def test_sample_is_bound_to_historical_artwork_and_hash(tmp_path):
    entry = {"painting": "p", "image": "p.jpg", "image_sha256": "hash"}
    historical = tmp_path / "preflight.json"
    historical.write_text(json.dumps({"records": [entry]}), encoding="utf-8")
    config = {"historical_preflight_sha256": file_sha256(historical)}
    scope = pd.DataFrame([entry])
    assert read_sample(historical, scope, config) == [entry]
    with pytest.raises(ValueError, match="artwork digest/path"):
        read_sample(historical, scope.assign(image_sha256="changed"), config)
    with pytest.raises(ValueError, match="hash"):
        read_sample(historical, scope, config | {"historical_preflight_sha256": "changed"})
