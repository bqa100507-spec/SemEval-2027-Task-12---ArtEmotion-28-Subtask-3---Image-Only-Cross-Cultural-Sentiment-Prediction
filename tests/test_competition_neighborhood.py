"""CPU tests for a training-only local cultural residual bank."""

import numpy as np
import pandas as pd
import pytest

from st3 import competition_neighborhood as experiment
from st3.competition_stack import ENCODERS
from st3.data import KEYS
from st3.e50_heads import read_json

RULE = read_json(experiment.CONFIG)
TRACK1 = read_json(experiment.TRACK1_CONFIG)


def test_centering_count_shrinkage_and_unseen_language_fallback():
    train = pd.DataFrame([("a", "English", 4, 5), ("a", "Thai", 2, 10), ("sparse", "English", 1, 5)],
                         columns=KEYS + ["k", "n"])
    inner = train[KEYS].assign(p_positive=.5).iloc[::-1]
    visual = pd.DataFrame([[1., 0.], [0., 1.], [1., 0.]], index=["a", "sparse", "test"])
    bank = experiment.make_bank(train, inner, visual, 2)
    assert list(bank.visual.index) == ["a"]
    np.testing.assert_allclose(bank.numerator.loc["a", ["English", "Thai"]], [2., -2.])
    np.testing.assert_allclose(bank.numerator.sum(axis=1), 0, atol=1e-12)
    keys = pd.DataFrame([("test", "English"), ("test", "Thai"), ("test", "Kazakh")], columns=KEYS)
    base = keys.assign(p_positive=[.4, .6, .2]).iloc[::-1]
    result = experiment.predict_correction(bank, keys, visual, base, RULE)
    np.testing.assert_array_equal(result.weighted_neighbor_votes, [5, 10, 0])
    np.testing.assert_array_equal(result.corrected, [.48, .533333, .2])
    assert result[KEYS].equals(keys)
    with pytest.raises(ValueError, match="disjoint"):
        experiment.predict_correction(bank, keys.assign(painting="a"), visual, base, RULE)
    with pytest.raises(ValueError, match="keys"):
        experiment.predict_correction(bank, keys.assign(k=1), visual, base, RULE)


def test_neighbor_ties_use_stable_training_painting_order():
    train = pd.DataFrame([("b", "English", 2, 5), ("a", "English", 4, 5),
                          ("a", "Thai", 1, 5), ("b", "Thai", 3, 5)], columns=KEYS + ["k", "n"])
    visual = pd.DataFrame([[1.], [1.], [1.]], index=["b", "a", "query"])
    bank = experiment.make_bank(train, train[KEYS].assign(p_positive=.5), visual, 2)
    keys = pd.DataFrame([("query", "English")], columns=KEYS)
    result = experiment.predict_correction(bank, keys, visual, keys.assign(p_positive=.5), RULE | {"neighbors": 1})
    assert result.corrected.iloc[0] == .56


def test_ridge_crossfit_queries_disjoint_paintings_once(monkeypatch):
    train = pd.DataFrame([(f"p{i}", l, 3, 5) for i in range(9) for l in ("English", "Thai")],
                         columns=KEYS + ["k", "n"])
    queried = set()

    def train_head(fitting, _, config):
        return set(fitting.painting), {}

    def predict(training, keys, vectors):
        groups = set(keys.painting)
        assert not groups & (training | queried)
        queried.update(groups)
        return keys.assign(**dict.fromkeys(ENCODERS, .37))

    monkeypatch.setattr(experiment, "fit_heads", train_head)
    monkeypatch.setattr(experiment, "component_predictions", predict)
    oof, receipts = experiment.fitting_predictions(train, {}, TRACK1, 1)
    assert queried == set(train.painting)
    assert all(r["fit_paintings"] == 6 and r["query_paintings"] == 3 for r in receipts["inner_folds"])
    np.testing.assert_array_equal(oof.p_positive, np.full(len(train), .37))


def test_official_filtered_score_on_identical_predictions_keeps_ridge():
    gold = pd.DataFrame([(f"p{i:02d}", name, i // 20 + 1, k, 5) for i in range(40)
                         for name, k in [("English", 5), ("Thai", 1), ("Tamil", 3), ("Korean", 4)]],
                        columns=KEYS + ["fold", "k", "n"])
    q = gold.k / gold.n
    output = gold[KEYS].assign(ridge=q, corrected=q)
    metrics, paintings = experiment.selection_metrics(gold, output, RULE, TRACK1["test_like_excluded"])
    assert metrics["recommendation"] == "RETAIN RIDGE"
    assert metrics["full"]["pooled"]["ridge"]["log_loss_languages"] == 4
    assert metrics["filtered"]["pooled"]["ridge"]["log_loss_languages"] == 3
    assert metrics["filtered"]["corrected_minus_ridge"]["bootstrap_95"] == [0., 0.]
    assert len(paintings) == 80
