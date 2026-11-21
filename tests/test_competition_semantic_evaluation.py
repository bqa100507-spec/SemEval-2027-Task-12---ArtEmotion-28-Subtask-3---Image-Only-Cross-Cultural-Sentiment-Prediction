import json

import pandas as pd
import pytest

from st3.competition_semantic_evaluation import charge, save_batch, validate_record, verify_batch
from st3.competition_semantic_generation import CONFIG
from st3.competition_semantic_recovery import parse_compact
from st3.competition_semantic_vlm import semantic_features
from st3.e50_heads import read_json

BASE = read_json(CONFIG)
RECIPE = {"max_new_tokens": 72}
IMAGE = {"painting": "a", "image": "style/a.jpg", "image_sha256": "artwork"}
RAW = '{"e":"woman, tree","s":"garden path","a":"muted green","v":"downcast eyes","u":"none"}'


def test_exact_document_and_explicit_missing_document():
    record = IMAGE | {"raw": RAW, "output_tokens": 50, "description": parse_compact(RAW, BASE["fields"])}
    assert validate_record(record, IMAGE, BASE["fields"], 72) == record["description"]
    bad = IMAGE | {"raw": RAW[:-1], "output_tokens": 72, "parse_error": "Incomplete or malformed compact JSON"}
    assert validate_record(bad, IMAGE, BASE["fields"], 72) == {}
    with pytest.raises(ValueError, match="does not reproduce"):
        validate_record(bad | {"description": record["description"]}, IMAGE, BASE["fields"], 72)


def test_atomic_cache_rejects_tampering_and_changed_identity(tmp_path):
    record = IMAGE | {"raw": RAW, "output_tokens": 50, "description": parse_compact(RAW, BASE["fields"])}
    target = tmp_path / "batches/0000.json"
    save_batch(target, {"run": "frozen"}, {"records": [record]})
    assert verify_batch(target, [IMAGE], {"run": "frozen"}, BASE, RECIPE) == {"a": record["description"]}
    with pytest.raises(ValueError, match="Corrupt or incompatible"):
        verify_batch(target, [IMAGE], {"run": "other"}, BASE, RECIPE)
    item = read_json(target)
    item["result"]["records"][0]["raw"] = "tampered"
    target.write_text(json.dumps(item), encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt or incompatible"):
        verify_batch(target, [IMAGE], {"run": "frozen"}, BASE, RECIPE)


def test_cumulative_gpu_budget_charges_interrupted_worker_and_clamps_at_cap():
    ledger = {"limit_seconds": 12600, "charged_seconds": 1800, "active_since": 10000}
    assert charge(ledger, 13600) == 5400
    assert charge(ledger, 100000) == 12600
    assert charge(ledger | {"active_since": None}, 100000) == 1800


def test_invalid_semantics_remains_a_zero_vector_without_dropping_the_painting():
    main = parse_compact(RAW, BASE["fields"])
    fit = pd.Series({f"p{i}": main | {"entities": "woman tree" if i % 2 else "bridge river"} for i in range(8)})
    descriptors, info = semantic_features(fit, pd.Series({"missing": {}, "valid": main}), BASE)
    assert info["zero_query_vectors"] == 1
    assert (descriptors.loc["missing"] == 0).all()
    assert descriptors.loc["valid"].abs().sum() > 0
