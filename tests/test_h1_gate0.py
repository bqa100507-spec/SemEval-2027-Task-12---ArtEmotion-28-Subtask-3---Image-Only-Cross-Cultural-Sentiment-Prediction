"""Synthetic tests for the H1 Gate 0 analysis (no dataset, no downloads)."""

import numpy as np
import pandas as pd
import pytest

from st3 import h1_gate0 as gate


def panel(seed=0, paintings=400, languages=("English", "Arabic", "Chinese", "L1", "L2", "L3", "L4", "L5"),
          coupling=0.0, noise_votes=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(paintings):
        base = rng.normal()
        shared = rng.normal()
        for language in languages:
            deviation = rng.normal() * .6
            if language in ("English", "L1"):
                deviation = deviation * np.sqrt(1 - coupling) + shared * np.sqrt(coupling) * .6
            n = noise_votes or 2000
            k = rng.binomial(n, 1 / (1 + np.exp(-(base + deviation))))
            rows.append((f"p{i}", language, k, n))
    return pd.DataFrame(rows, columns=["painting", "language", "k", "n"])


def test_centering_excludes_source_and_target_and_requires_three_others():
    y = pd.DataFrame({"English": [1., 1.], "L1": [5., 5.], "L2": [0., 0.], "L3": [2., np.nan], "L4": [4., 2.]},
                     index=["a", "b"])
    v = y * 0 + .1
    frame = gate.centered_pair(y, v, "English", "L1")
    assert list(frame.painting) == ["a"]  # b has only two other languages
    ref = np.mean([0., 2., 4.])
    assert frame.d_source.iloc[0] == pytest.approx(1 - ref) and frame.d_target.iloc[0] == pytest.approx(5 - ref)
    split = gate.centered_pair(y, v, "English", "L1", mode="split")
    assert split.d_source.iloc[0] == pytest.approx(1 - np.mean([0., 4.]))
    assert split.d_target.iloc[0] == pytest.approx(5 - 2.)


def test_shared_reference_has_positive_null_bias_and_split_does_not():
    data = panel(paintings=1500, noise_votes=10)
    y, v = gate.wide(data)
    shared = gate.centered_pair(y, v, "English", "L2")
    split = gate.centered_pair(y, v, "English", "L2", mode="split")
    r_shared = np.corrcoef(shared.d_source, shared.d_target)[0, 1]
    r_split = np.corrcoef(split.d_source, split.d_target)[0, 1]
    assert r_shared > .08 and abs(r_split) < .06
    assert gate.shared_reference_null(5) == pytest.approx(1 / 6)


def test_disattenuation_recovers_a_known_correlation():
    rng = np.random.default_rng(1)
    n, true_r = 20000, .5
    latent = rng.multivariate_normal([0, 0], [[1, true_r], [true_r, 1]], size=n)
    vx, vy = np.full(n, 1.), np.full(n, .5)
    x = latent[:, 0] + rng.normal(scale=1., size=n)
    y = latent[:, 1] + rng.normal(scale=np.sqrt(.5), size=n)
    r, rx, ry = gate.weighted_stats(x, y, vx, vy, np.ones(n))
    assert r < .4 and rx == pytest.approx(.5, abs=.03) and ry == pytest.approx(2 / 3, abs=.03)
    assert gate.disattenuate(r, rx, ry) == pytest.approx(true_r, abs=.03)
    assert np.isnan(gate.disattenuate(.3, -.1, .5))


def test_pooling_weights():
    assert gate.pool([.2, .2], [100, 200]) == pytest.approx(.2, abs=1e-9)
    assert gate.pool([.1, .3], [103, 103], [1, 1], [1, 1]) == pytest.approx(.2)
    assert gate.pool([.1, .3], [103, 103], [1, .0], [1, 1]) == pytest.approx(.1)


def test_analysis_a_is_seed_reproducible_and_detects_coupling():
    data = panel(seed=2, paintings=300, coupling=.8)
    y, v = gate.wide(data)
    first, cells = gate.analysis_a(y, v, ["L1", "L2"], "split", minimum_n=50, draws=200, seed=7)
    second, _ = gate.analysis_a(y, v, ["L1", "L2"], "split", minimum_n=50, draws=200, seed=7)
    assert first == second
    l1 = cells[(cells.source == "English") & (cells.target == "L1")].r_disattenuated.iloc[0]
    l2 = cells[(cells.source == "English") & (cells.target == "L2")].r_disattenuated.iloc[0]
    assert l1 > .5 and abs(l2) < .2


def test_oracle_maps_never_use_held_out_paintings():
    rng = np.random.default_rng(3)
    paintings = [f"p{i}" for i in range(60)]
    features = pd.DataFrame({"d_English": rng.normal(size=60), "has_English": 1., "d_Arabic": 0., "has_Arabic": 0.,
                             "d_Chinese": 0., "has_Chinese": 0.}, index=paintings)
    rows = pd.DataFrame([(p, lang, rng.normal(), 0.) for p in paintings for lang in ("L1", "L2")],
                        columns=["painting", "language", "residual", "c2_logit"])
    train, held = rows[rows.painting.isin(paintings[:40])], rows[rows.painting.isin(paintings[40:])]
    models = gate.fit_oracle(train, features, (1., 10.), 3, 0)
    assert all(not (m[3] & set(held.painting)) for m in models.values())
    out = gate.apply_oracle(models, held, features)
    assert out.oracle.between(0, 1).all() and len(out) == len(held)
    with pytest.raises(ValueError, match="held-out painting"):
        gate.apply_oracle(models, train, features)
    control = gate.fit_oracle(train, features, (1.,), 3, 0, use_values=False)
    assert all(m[1] == ["has_English", "has_Arabic", "has_Chinese"] for m in control.values())


def test_source_features_use_no_target_gold():
    gold = gate.logits(pd.DataFrame({"painting": ["a", "a", "a"], "language": ["English", "L1", "L2"],
                                     "k": [5, 1, 4], "n": [5, 5, 5]}))
    c2 = pd.DataFrame({"painting": ["a", "a", "a"], "language": ["English", "L1", "L2"], "c2_logit": [0., 1., 2.]})
    features = gate.source_features(gold, c2)
    assert features.loc["a", "d_English"] == pytest.approx(gold.y.iloc[0] - 1.)
    assert features.loc["a", "has_Arabic"] == 0 and features.loc["a", "d_Arabic"] == 0


def test_learning_curve_is_seeded_and_fold_disjoint():
    rng = np.random.default_rng(4)
    paintings = [f"p{i}" for i in range(100)]
    x = pd.DataFrame(rng.normal(size=(100, 5)), index=paintings)
    frame = pd.DataFrame({"painting": paintings, "d": x[0] * 2 + rng.normal(scale=.5, size=100)})
    folds = pd.Series([1 + i % 5 for i in range(100)], index=paintings)
    a = gate.learning_curve(x, frame, folds, (.5, 1.), (0, 1), (1.,))
    b = gate.learning_curve(x, frame, folds, (.5, 1.), (0, 1), (1.,))
    pd.testing.assert_frame_equal(a, b)
    assert (a.r > .8).all() and gate.slope(a) == gate.slope(b)


def test_exact_logit_variance_matches_simulation_and_stays_bounded_at_extreme_counts():
    rng = np.random.default_rng(5)
    k, n = np.array([0, 2, 5]), np.array([5, 5, 5])
    exact = gate.exact_logit_variance(k, n)
    for value, kk in zip(exact, k):
        draws = rng.binomial(5, (kk + .5) / 6, size=200000)
        assert value == pytest.approx(np.log((draws + .5) / (5 - draws + .5)).var(), rel=.02)
    delta = gate.logits(pd.DataFrame({"k": k, "n": n})).v.to_numpy()
    assert exact[0] < delta[0] / 4 and exact[2] < delta[2] / 4  # the delta method blows up at k = 0 or n
    assert (gate.logits(pd.DataFrame({"k": k, "n": n}), "exact").v.to_numpy() == exact).all()


def test_stop_clauses_mark_undefined_disattenuation():
    assert gate.stop_clauses(float("nan"), [float("nan")] * 2, -.01) == {"A_null_correlation": None,
                                                                          "B_oracle_below_0_005": True}
    assert gate.stop_clauses(.05, [-.02, .1], .02) == {"A_null_correlation": True, "B_oracle_below_0_005": False}
    assert gate.decide(float("nan"), [float("nan")] * 2, .02, [.01, .03], 3) == "INCONCLUSIVE"


def test_decision_rule():
    assert gate.decide(.05, [-.02, .12], .02, [.01, .03], 3) == "STOP_H1"
    assert gate.decide(.2, [.05, .3], .004, [.001, .01], 3) == "STOP_H1"
    assert gate.decide(.2, [.05, .3], .012, [.002, .02], 2) == "PROMISING"
    assert gate.decide(.2, [.05, .3], .012, [.002, .02], 1) == "INCONCLUSIVE"
    assert gate.decide(.12, [.01, .2], .007, [-.001, .01], 3) == "INCONCLUSIVE"
