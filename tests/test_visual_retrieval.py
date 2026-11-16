import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from st3.visual_retrieval import Parameters, RetrievalMemory, predict_pairs, visual_features


def fixture():
    train = pd.DataFrame([
        ("a", "A", "a.jpg", 5, 4, "train"), ("a", "B", "a.jpg", 10, 2, "train"),
        ("b", "A", "b.jpg", 5, 1, "train"), ("b", "B", "b.jpg", 5, 4, "train")],
        columns=["painting", "language", "image", "n", "k", "split"])
    features = {"a.jpg": np.array([1., 0.]), "b.jpg": np.array([0., 1.]), "q.jpg": np.array([1., 0.])}
    return train, features


def test_weighted_neighbor_formulas_centering_and_confidence():
    train, features = fixture()
    memory = RetrievalMemory(train, features)
    params = Parameters(1, .1, 5.)
    query = np.array([[1., 0.]])
    prior = memory.prior
    uncentered = memory.predict(query, ["A", "B"], "uncentered", params)[0]
    np.testing.assert_allclose(uncentered, (np.array([4., 2.]) + 5 * prior) / (np.array([5., 10.]) + 5))
    image_mean = (6 + 5 * memory.global_prior) / (15 + 5)
    deviation = (np.array([4., 2.]) - np.array([5., 10.]) * .4 + 5 * (prior - memory.global_prior)) / (np.array([5., 10.]) + 5)
    np.testing.assert_allclose(memory.predict(query, ["A", "B"], "centered", params)[0], image_mean + deviation)
    assert not np.allclose(uncentered, image_mean + deviation)
    # Vote-weighted residuals center to zero separately for every memory painting.
    np.testing.assert_allclose(memory.residual_votes.sum(axis=1), 0., atol=1e-12)


def test_grouping_order_and_inference_without_votes_or_captions():
    train, features = fixture()
    query = pd.DataFrame([("q", "B", "q.jpg"), ("q", "A", "q.jpg")], columns=["painting", "language", "image"])
    params = Parameters(2, .1, 5.)
    first = RetrievalMemory(train, features)
    second = RetrievalMemory(train.sample(frac=1., random_state=3), features)
    for method in ("prior", "uncentered", "centered"):
        a = predict_pairs(first, query, features, method, params)
        b = predict_pairs(second, query.assign(n=999, k=0, caption="forbidden"), features, method, params)
        pd.testing.assert_frame_equal(a, b)
    # Image pixels matter for retrieval; filenames/painting IDs are only joins.
    original = first.predict(np.array([[1., 0.], [0., 1.]]), ["A", "B"], "centered", Parameters(1, .1, 5.))
    assert not np.allclose(original[0], original[1])


def test_sparse_languages_fallback_and_prior_control():
    train, features = fixture()
    train = train[~(train.painting.eq("a") & train.language.eq("B"))]
    memory = RetrievalMemory(train, features)
    for method in ("prior", "uncentered", "centered"):
        values = memory.predict(np.eye(2), ["B", "UNSEEN"], method, Parameters(1, .1, 5.))
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        if method != "centered":
            assert values[0, 1] == memory.global_prior
        if method == "uncentered":
            assert values[0, 0] == memory.prior[1]
        if method == "prior":
            np.testing.assert_array_equal(values[0], values[1])


def test_forbidden_memory_and_self_retrieval_rejected():
    train, features = fixture()
    for bad in (train.assign(split="val"), pd.concat([train, train.iloc[:1]]), train.assign(n=4), train.assign(k=99)):
        with pytest.raises(ValueError):RetrievalMemory(bad, features)
    memory = RetrievalMemory(train, features)
    with pytest.raises(ValueError, match="overlap"):
        predict_pairs(memory, train, features, "centered", Parameters(1, .1, 5.))


def test_fusion_and_parameter_validation():
    large = np.ones((3, 768)); siglip = np.arange(3 * 768).reshape(3, 768) + 1.
    result = visual_features(large, siglip)
    assert result.shape == (3, 1536)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.)
    for args in ((0, .1, 5.), (1, 0., 5.), (1, .1, np.nan)):
        with pytest.raises(ValueError):Parameters(*args)
    with pytest.raises(ValueError):visual_features(np.zeros((1, 768)), siglip[:1])


def test_training_only_inner_selection_and_query_label_independence():
    path = Path(__file__).resolve().parents[1] / "scripts/run_exp024.py"
    spec = importlib.util.spec_from_file_location("exp024_test_runner", path)
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    features, rows = {}, []
    rng = np.random.default_rng(4)
    for i in range(60):
        name = f"p{i}"
        features[name + ".jpg"] = rng.normal(size=4)
        for language, k in zip(("A", "B", "C"), (1 + i % 2, 3, 4)):
            rows.append(dict(painting=name, image=name + ".jpg", image_sha256=name, language=language, n=5, k=k, split="train"))
    frame = pd.DataFrame(rows)
    config = dict(early_stop_seed=17029, early_stop_fraction=.5, neighbors=[4], temperatures=[.1], shrinkage_votes=[5.], language_prior_votes=20.)
    selected, history, partitions = runner.select_parameters(frame, features, config, 1)
    assert len(history) == 2 and set(selected) == {"uncentered", "centered"}
    assert not set(partitions["inner_paintings"]) & set(partitions["selection_paintings"])
    held = pd.DataFrame([dict(painting="new", image="q.jpg", language="A", n=5, k=5)])
    features["q.jpg"] = rng.normal(size=4)
    memory = RetrievalMemory(frame, features)
    first = predict_pairs(memory, held, features, "centered", selected["centered"])
    second = predict_pairs(memory, held.assign(n=999, k=0), features, "centered", selected["centered"])
    pd.testing.assert_frame_equal(first, second)
