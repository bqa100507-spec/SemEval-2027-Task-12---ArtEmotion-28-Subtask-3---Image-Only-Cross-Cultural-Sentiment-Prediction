import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from st3 import data, residual_crossfit
from st3.ablation import painting_folds
from st3.baseline import file_sha256
from st3.residual import (COMPONENTS, MODELS, PairwiseRows, Residual, calibration_inputs, design, fit_residual,
                          objective)
from st3.residual_crossfit import disjoint, inner_oof, outer_reference_bytes, replay_outer, signals
from st3.residual_study import pair_ordering, run_residual_dev, select_residual


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "exp012.json").read_text())


def training_fixture():
    rows = [(f"p{i}", f"p{i}.jpg", lang, (i + j) % 5, 5, "train")
            for i in range(10) for j, lang in enumerate(("Tamil", "Yoruba", "English"))]
    return pd.DataFrame(rows, columns=["painting", "image", "language", "k", "n", "split"])


def base_frame(n):
    frame = training_fixture().iloc[:n].copy()
    frame["clip_b"] = np.linspace(.25, .45, n)
    frame["siglip2"] = np.linspace(.6, .5, n)
    frame["clip_l"] = np.linspace(.3, .8, n)
    frame["q0"] = np.linspace(.35, .65, n)
    return frame


def test_nested_folds_exclude_outer_held_out_paintings_and_cache_predictions(tmp_path, monkeypatch):
    train = training_fixture()
    fit_idx, held_idx = next(painting_folds(train, 5, 3))
    fit, held = train.iloc[fit_idx], train.iloc[held_idx]
    calls = []

    def fit_head(inner_fit, inner_held, bank, seed):
        disjoint(inner_fit, inner_held, held)
        assert set(inner_fit.painting) | set(inner_held.painting) == set(fit.painting)
        assert seed == 19
        calls.append(set(inner_held.painting))
        return inner_held[data.KEYS].assign(p_positive=bank), {"converged": True}

    monkeypatch.setattr(residual_crossfit, "fit_head", fit_head)
    features = dict(zip(COMPONENTS, (.1, .100001, .100001)))
    provenance = {"config": CONFIG, "exp002_config": {"fit_seed": 19}}
    rows, _ = inner_oof(tmp_path, fit, held, 1, features, provenance)
    assert len(calls) == 12 and set(rows.painting) == set(fit.painting)
    assert set(rows.q0) == {.1}
    assert len(rows) == len(fit) and rows.groupby("painting").inner_fold.nunique().eq(1).all()
    replay, _ = inner_oof(tmp_path, fit, held, 1, features, provenance)
    pd.testing.assert_frame_equal(rows, replay)
    assert len(calls) == 12
    with pytest.raises(ValueError, match="overlapping image"):
        disjoint(fit, held.assign(image=fit.iloc[0].image))
    with (tmp_path / "outer-1" / "predictions.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        inner_oof(tmp_path, fit, held, 1, features, provenance)


def test_outer_replay_preserves_exact_rounded_reference_bytes(tmp_path):
    train = training_fixture()
    folds = list(painting_folds(train, 5, 11))
    exp002, exp006, exp007, exp010 = [], [], [], []
    for fold, (_, held) in enumerate(folds, 1):
        frame = train.iloc[held][data.KEYS]
        clip, siglip, large = (frame.assign(p_positive=q) for q in (.1, .100001, .100001))
        exp002.append(frame.assign(p_C=.1, fold=fold))
        exp006.append(siglip.assign(model="C-SigLIP2", fold=fold))
        exp007.extend([clip.assign(model="C-CLIP", fold=fold), siglip.assign(model="C-SigLIP2", fold=fold),
                       frame.assign(p_positive=.1, model="E50", fold=fold)])
        exp010.extend([large.assign(model="CLIP-L", fold=fold, seed=19),
                       frame.assign(p_positive=.1, model="E50+L", fold=fold, seed=19)])
    saved, prior6, prior7, prior10 = (pd.concat(frames, ignore_index=True)
                                     for frames in (exp002, exp006, exp007, exp010))
    source = tmp_path / "predictions.csv"
    prior10.to_csv(source, index=False, lineterminator="\n")
    outer = replay_outer(train, folds, saved, prior6, prior7, source, 19)
    expected = prior10.loc[prior10.model.eq("E50+L")].to_csv(index=False, lineterminator="\n").encode()
    assert outer_reference_bytes(outer, 19) == expected
    source.write_bytes(source.read_bytes().replace(b",0.1,E50+L,", b",0.100000,E50+L,"))
    with pytest.raises(ValueError, match="byte-identical"):
        replay_outer(train, folds, saved, prior6, prior7, source, 19)


def test_vote_weighted_bce_and_hierarchical_penalty_scale():
    baseline = logit([.25, .8])
    x = np.array([[1., 0., 1.], [1., 1., 0.]])
    k, n = np.array([2, 8]), np.array([5, 9])
    params, penalty = np.array([.2, -.3, .1]), np.array([.02, .02, .20])
    value, gradient, bce, reg, _ = objective(params, baseline, x, k, n, penalty)
    q = 1. / (1. + np.exp(-(baseline + x @ params)))
    expected_bce = -(k @ np.log(q) + (n - k) @ np.log1p(-q)) / n.sum()
    assert bce == pytest.approx(expected_bce)
    assert reg == pytest.approx(.5 * (.02 * (.2 ** 2 + .3 ** 2) + .20 * .1 ** 2))
    assert value == pytest.approx(bce + reg)
    np.testing.assert_allclose(gradient, x.T @ ((n * q - k) / n.sum()) + penalty * params)


def test_pairwise_huber_gradient_in_row_order_and_full_objective():
    frame = base_frame(9).iloc[::-1]
    huber = PairwiseRows(frame, .20)
    baseline, d = calibration_inputs(frame, .0001)
    x = design(frame["language"], d, ("Tamil",))
    args = (baseline, x, frame["k"].to_numpy(), frame["n"].to_numpy(), np.array([.02, .02, .2, .2]), huber, .1)
    params = np.array([-.13, .15, .17, -.12])
    _, analytic, _, _, pairwise = objective(params, *args)
    epsilon = 1e-6
    directions = np.eye(len(params)) * epsilon
    numeric = [(objective(params + h, *args)[0] - objective(params - h, *args)[0]) / (2 * epsilon)
               for h in directions]
    assert pairwise > 0.
    np.testing.assert_allclose(analytic, numeric, atol=1e-9, rtol=0.)


@pytest.mark.parametrize("model", MODELS[1:])
def test_zero_identity_sparse_and_untrained_languages_and_determinism(model):
    frame = base_frame(30)
    frame.loc[frame.index[-1], "language"] = "Emakhuwa"
    config = {**CONFIG, "min_language_paintings": 3}
    a, info = fit_residual(frame, model, config)
    b, _ = fit_residual(frame, model, config)
    np.testing.assert_array_equal(a.parameters, b.parameters)
    assert info["converged"] and np.isfinite(a.parameters).all()
    assert "Emakhuwa" in a.seen and "Emakhuwa" not in a.support
    query = frame.iloc[:3][["language", *COMPONENTS, "q0"]].copy()
    query["q0"] = [0., 1., .123456]
    zero = Residual(model, np.zeros_like(a.parameters), a.seen, a.support, a.center, a.scale, a.epsilon)
    np.testing.assert_array_equal(zero.predict(query), query["q0"])
    query.loc[query.index[-1], "language"] = "Kazakh"
    prediction = a.predict(query)
    if model != "M1":
        assert prediction[-1] == .123456
    assert np.isfinite(prediction).all() and np.all((prediction >= 0) & (prediction <= 1))
    np.testing.assert_array_equal(prediction, Residual.from_dict(a.to_dict()).predict(query))
    assert prediction[1] == a.predict(query.iloc[[1]])[0]
    with pytest.raises(ValueError, match="original training pairs"):
        fit_residual(frame.assign(split="val"), model, config)


def test_pair_ordering_excludes_tied_gold_and_halves_predicted_ties():
    keys = pd.DataFrame({"painting": ["p"] * 3, "language": ["A", "B", "C"]})
    assert pair_ordering(keys.assign(p=[.1, .2, .2]), keys.assign(p_positive=[.4, .4, .1])) == {"p": .25}


def test_selection_uses_frozen_gain_loss_and_convergence_gates():
    summary = pd.DataFrame({"pooled_contrast": [.44, .4409, .442, .443], "log_loss_mean": [.72, .72, .725, .726],
                            "folds": 5, "scored": 5, "loss_scored": 5, "converged": 5}, index=MODELS)
    assert select_residual(summary, CONFIG, 5)["model"] == "M2"
    summary.loc["M2", "converged"] = 4
    assert select_residual(summary, CONFIG, 5)["model"] == "M0"
    summary.loc["M1", "pooled_contrast"] = .441
    assert select_residual(summary, CONFIG, 5)["model"] == "M1"


def test_no_go_does_not_open_development_inputs(tmp_path):
    cv = tmp_path / "cv"
    cv.mkdir()
    summary = pd.DataFrame({"pooled_contrast": .44, "log_loss_mean": .72, "folds": 5, "scored": 5,
                            "loss_scored": 5, "converged": 5}, index=pd.Index(MODELS, name="model"))
    decision = select_residual(summary, CONFIG, 5)
    summary.to_csv(cv / "summary.csv")
    (cv / "run.json").write_text(json.dumps({"config": CONFIG, "exp002_config": {"cv_folds": 5}}))
    (cv / "inner_audit.json").write_text("[]")
    (cv / "outer_components.csv").write_text("fixture")
    decision["sha256"] = {name: file_sha256(cv / name) for name in
                          ("run.json", "summary.csv", "outer_components.csv", "inner_audit.json")}
    (cv / "selection.json").write_text(json.dumps(decision))
    run_residual_dev(SimpleNamespace(run_dir=tmp_path), None)
    assert not (tmp_path / "dev").exists()
