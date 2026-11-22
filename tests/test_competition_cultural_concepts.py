import numpy as np
import pandas as pd
import pytest

from st3.competition_cultural_concepts import CONFIG, concept_hits, feature_matrix, fit_fold, mismatch_map
from st3.e50_heads import read_json


RULE = read_json(CONFIG)


def test_observable_fields_do_not_accept_interpretation_or_negated_fragments():
    record = {"entities": "horse, not a tree, bayonet-shaped", "scene": "crowd protest, maybe flowers",
              "style_atmosphere": "gloomy shadow", "visual_cues": "flags; hands",
              "uncertain_context": "possible: a boat, mourning"}
    assert set(concept_hits(record, RULE)) == {"horse", "crowd", "flag", "hand"}
    assert concept_hits({}, RULE) == []
    np.testing.assert_array_equal(feature_matrix([[], ["horse"]], ["horse"]), [[0], [1]])


def test_mismatch_uses_a_different_supported_language_without_erasing_support():
    support = np.array([[1, 0], [1, 1], [0, 1]], dtype=bool)
    votes = np.array([[200, 0], [100, 300], [0, 200]])
    actual = mismatch_map(support, votes)
    np.testing.assert_array_equal(actual, [[1, 0], [0, 2], [2, 1]])


def test_fold_excludes_held_votes_and_keeps_static_prior_under_mismatch():
    groups = ("Thai", "Hindi")
    train = pd.DataFrame([{"painting": f"p{i}", "language": lang, "n": 20,
                           "k": (16 if (i % 2 == j) else 4)} for i in range(20) for j, lang in enumerate(groups)])
    query = pd.DataFrame([(p, lang) for p in ("q", "unknown") for lang in groups], columns=["painting", "language"])
    hits = {f"p{i}": ["horse"] if i % 2 == 0 else ["tree"] for i in range(20)} | {"q": ["horse"], "unknown": ["flag"]}
    rule = RULE | {"minimum_concept_paintings": 9, "minimum_concept_language_paintings": 6, "minimum_concept_language_votes": 90}
    out, details = fit_fold(train, query, hits, rule)
    assert "flag" not in details["fitted_vocabulary"]
    assert details["supported_concept_language_cells"] == 4
    assert out.loc[0, "concept_culture"] > out.loc[1, "concept_culture"]
    assert out.loc[0, "concept_mismatched"] < out.loc[1, "concept_mismatched"]
    assert out.loc[0:1, "concept_global"].nunique() == 1
    np.testing.assert_array_equal(out.loc[2:, "concept_culture"], out.loc[2:, "concept_mismatched"])
    with pytest.raises(ValueError, match="painting-disjoint"):
        fit_fold(train, query.assign(painting="p0"), hits, rule)
    with pytest.raises(ValueError, match="query keys only"):
        fit_fold(train, query.assign(k=20), hits, rule)
