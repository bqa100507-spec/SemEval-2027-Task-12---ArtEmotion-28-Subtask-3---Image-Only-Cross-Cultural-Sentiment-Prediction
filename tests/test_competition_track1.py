"""Synthetic regressions for the isolated competition challenger."""

from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.optimize import check_grad
from scipy.special import expit

from st3 import competition_challenger as challenger, data, submission
from st3.baseline import file_sha256
from st3.competition_ridge import Head, fit_head, make_panel, objective
from st3.competition_stack import ENCODERS, aligned_values, blend, fit_stack, stack_objective
from st3.e50_heads import read_json

CONFIG = read_json(challenger.CONFIG)


def fixture():
    gold = pd.DataFrame([(f"p{i:02d}", name, k, 5) for i in range(24)
                         for name, k in zip(("Korean", "Tamil", "Thai", "English"), (1, 4, 2, 5))],
                        columns=data.KEYS + ["k", "n"])
    good = gold.k.to_numpy() / gold.n.to_numpy()
    fair = .30 + .45 * good
    wrong = 1 - fair
    components = gold[data.KEYS].assign(**dict(zip(ENCODERS, (good, wrong, fair))))
    return gold, components


def test_stack_gradient_filter_simplex_and_key_alignment():
    gold, parts = fixture()
    objective_fn = stack_objective(gold, parts, CONFIG)
    x = np.array([.25, .25, .5])
    assert check_grad(lambda w: objective_fn(w)[0], lambda w: objective_fn(w)[1], x) < 1e-7
    actual = fit_stack(gold, parts, CONFIG)
    assert np.isclose(sum(actual["weights"]), 1)
    assert min(actual["weights"]) >= 0
    assert actual["weights"][0] > actual["weights"][1]
    assert actual["objective"] <= actual["anchor_objective"]
    modified = gold.assign(k=gold.k.where(gold.language.ne("English"), 0))
    mixed_order = parts.iloc[::-1]
    assert fit_stack(modified, mixed_order, CONFIG) == actual
    with pytest.raises(ValueError, match="Pair mismatch"):
        aligned_values(gold, parts.iloc[:-1], ENCODERS)


def test_ridge_gradient_and_unseen_language_fallback():
    gold, _ = fixture()
    gold["k"] = gold.k.where(gold.painting.lt("p12"), 5 - gold.k)
    panel = make_panel(gold)
    x = np.repeat([[-1., .5], [1., .5]], 12, axis=0)
    ridge = 1 / (CONFIG["ridge"]["reference_c"] * panel.n.sum())
    loss_grad, _, size = objective(panel, x, ridge, .5 * ridge * np.eye(len(panel.languages)))
    theta = np.linspace(-.1, .1, size)
    assert check_grad(lambda v: loss_grad(v)[0], lambda v: loss_grad(v)[1], theta) < 1e-7
    head, receipt = fit_head(panel, x, CONFIG["ridge"])
    assert receipt["converged"]
    assert np.linalg.norm(head.deviations) > 0
    np.testing.assert_allclose(head.logits(x, ["Kazakh"]).ravel(), head.intercept + x @ head.shared)


def test_crossfit_never_predicts_its_own_training_paintings(monkeypatch):
    gold, parts = fixture()
    visited = set()

    def train(fit, features, config):
        assert len(set(fit.painting)) == 16
        return set(fit.painting), {}

    def predict(fit, held, features):
        groups = set(held.painting)
        assert not (fit & groups or visited & groups)
        visited.update(groups)
        return held.merge(parts, on=data.KEYS, validate="one_to_one")

    monkeypatch.setattr(challenger, "fit_heads", train)
    monkeypatch.setattr(challenger, "component_predictions", predict)
    receipt = challenger.crossfit(gold, {}, CONFIG, 10)
    assert visited == set(gold.painting)
    assert [split["held_paintings"] for split in receipt["splits"]] == [8, 8, 8]


def test_bundle_prediction_uses_bytes_and_language_only(tmp_path, monkeypatch):
    bundle = tmp_path / "heads"
    heads = {}
    for j, (name, width) in enumerate(CONFIG["feature_widths"].items()):
        v = np.eye(1, width).ravel()
        heads[name] = Head(.05 * j, np.array([.2]), v, v[None] * .1, ("Korean",))
    challenger.write_bundle(bundle, heads, mode="fixed", weights=CONFIG["fixed_weights"], provenance={})
    images = tmp_path / "images"
    images.mkdir()
    for name, color in [("blue", "blue"), ("red", "red"), ("alias", "blue")]:
        Image.new("RGB", (8, 8), color).save(images / f"{name}.png")
    input_path = tmp_path / "pairs.csv"
    raw = [("a", "Korean", "blue.png"), ("a", "Kazakh", "blue.png"),
           ("b", "Korean", "red.png"), ("c", "Korean", "alias.png")]
    pairs = pd.DataFrame(raw, columns=data.INPUT_COLUMNS)
    pairs.to_csv(input_path, index=False)
    calls = []

    def worker(name, images_by_hash, args, profile, scratch):
        calls.append(name)
        assert args.device == "cpu"
        assert len(images_by_hash) == 2
        result = {}
        for digest, path in images_by_hash.items():
            with Image.open(path) as image:
                sign = 1 if image.getpixel((0, 0))[0] > 0 else -1
            result[digest] = np.eye(1, profile["width"], dtype=np.float32).ravel() * sign
        return result, {}

    monkeypatch.setattr(challenger, "run_feature_worker", worker)
    args = Namespace(input=input_path, data_dir=tmp_path, bundle=bundle, output=tmp_path / "first", device="cpu")
    challenger.predict_images(args)
    pred = submission.validate_archive(args.output / "submission.zip", pairs)
    assert list(pred.columns) == [*data.KEYS, "p_positive"]
    assert pred.p_positive.iloc[0] == pred.p_positive.iloc[3]
    assert pred.p_positive.iloc[0] != pred.p_positive.iloc[2]
    fallback = expit(np.array([[-1., -.95, -.90]])).round(6)
    assert pred.p_positive.iloc[1] == blend(fallback, CONFIG["fixed_weights"], fixed=True)[0]
    with pytest.raises(ValueError, match="already exists"):
        challenger.predict_images(args)
    args.output = tmp_path / "second"
    challenger.predict_images(args)
    assert (tmp_path / "first" / "predictions.csv").read_bytes() == (args.output / "predictions.csv").read_bytes()
    assert calls == list(ENCODERS) * 2
    with (bundle / "heads.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="incorrectly hashed"):
        challenger.load_bundle(bundle)


def test_official_refit_reads_training_votes_only(tmp_path, monkeypatch):
    raw = tmp_path / data.ANNOTATIONS
    raw.parent.mkdir()
    columns = ["painting", "language", "image_name", "split", "emotion"]
    records = [("a", "Korean", "a.png", "train", emotion) for emotion in ["awe"] * 3 + ["anger"] * 2]
    records.append(("b", "Korean", "b.png", "val", "forbidden-validation-label"))
    pd.DataFrame(records, columns=columns).to_csv(raw, index=False)
    dev = tmp_path / data.DEV_INPUT
    dev.parent.mkdir(parents=True)
    pd.DataFrame([("b", "Korean", "b.png")], columns=data.INPUT_COLUMNS).to_csv(dev, index=False)
    fixture_config = {"official_training": {"revision": "fixture", "annotations_sha256": file_sha256(raw),
                                             "paintings": 1, "pairs": 1}}
    protocol = pd.DataFrame([("a", "a.png")], columns=["painting", "image"])
    monkeypatch.setattr(challenger, "git_head", lambda _: "fixture")
    final = challenger.official_pairs(tmp_path, fixture_config, protocol)
    assert final[["painting", "k", "n"]].to_records(index=False).tolist() == [("a", 3, 5)]
