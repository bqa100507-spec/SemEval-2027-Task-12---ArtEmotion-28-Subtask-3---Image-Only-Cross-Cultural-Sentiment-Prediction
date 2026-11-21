import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.special import logit

from st3.exp032 import (Panel, crossfit_residuals, deviation_precision, estimate_graph, fit_head,
                       graph_stability, laplacian, objective, painting_splits)
from st3.exp032_evaluation import CANDIDATE, CONTROL, REFERENCE, choose_inner, decision, evaluate

OPT = {"reference_c": 0.3, "gtol": 1e-10, "ftol": 1e-14, "maxiter": 500, "max_final_gradient": 1e-7}
GRAPH = {"nuisance_splits": 3, "minimum_overlap": 8, "vote_weight_cap": 10, "overlap_shrinkage": 50,
         "minimum_common_edges": 6, "minimum_common_languages": 4, "minimum_similarity_rank_correlation": 0.25,
         "minimum_replicated_association": 0.03, "minimum_overlap_only_distance": 0.08,
         "minimum_isotropic_distance": 0.10}


def test_complete_uniform_graph_is_ridge_with_isolates():
    adjacency = np.ones((4, 4)) - np.eye(4)
    adjacency[3] = 0
    adjacency[:, 3] = 0
    graph, projector = laplacian(adjacency)
    np.testing.assert_allclose(graph, projector, atol=1e-15)
    np.testing.assert_allclose(deviation_precision(0.7, 0.5, graph, projector), 0.7 * np.eye(4), atol=1e-15)


def test_graph_penalty_holds_trace_constant_and_never_weakens_isolates():
    adjacency = np.array([[0., 1., 0.2, 0.], [1., 0., 0.5, 0.], [0.2, 0.5, 0., 0.], [0., 0., 0., 0.]])
    l_mat, proj = laplacian(adjacency)
    precision = deviation_precision(0.7, 0.5, l_mat, proj)
    np.testing.assert_allclose(precision[3], [0, 0, 0, 0.7])
    assert np.trace(precision) == pytest.approx(4 * 0.7)
    assert np.linalg.eigvalsh(precision).min() >= 0.35
    assert not np.allclose(precision, 0.7 * np.eye(4))


@pytest.mark.parametrize("with_graph", [False, True])
def test_gradient_constraints_and_fallback(with_graph):
    rng = np.random.default_rng(2032)
    x = rng.normal(size=(12, 3))
    n = rng.integers(0, 10, size=(12, 4)).astype(float)
    k = np.minimum(rng.integers(0, 10, size=n.shape), n)
    panel = Panel(tuple(map(str, range(12))), tuple("abcd"), k, n)
    l_mat, proj = laplacian(np.array([[0, 1, 0, 0], [1, 0, 0.7, 0], [0, 0.7, 0, 0], [0, 0, 0, 0]], dtype=float))
    precision = deviation_precision(0.07, 0.5, l_mat, proj) if with_graph else 0.07 * np.eye(4)
    func, unpack, size = objective(panel, x, 0.06, precision)
    theta = rng.normal(0, 0.2, size)
    assert check_grad(lambda t: func(t)[0], lambda t: func(t)[1], theta) < 1e-6
    head = unpack(theta)
    np.testing.assert_allclose(head.deviations.sum(axis=0), 0, atol=1e-14)
    assert head.language_intercepts.sum() == pytest.approx(0, abs=1e-14)
    np.testing.assert_allclose(head.logits(x, ["missing"]).ravel(), head.intercept + x @ head.shared)
    assert np.linalg.norm(head.deviations) > 0


def test_crossfitting_excludes_target_painting_votes():
    rng = np.random.default_rng(2032)
    names = tuple(f"p{i:02d}" for i in range(24))
    n = np.full((24, 3), 5.)
    k = rng.binomial(5, 0.6, size=n.shape).astype(float)
    x = rng.normal(size=(24, 3))
    first, _, receipts = crossfit_residuals(Panel(names, tuple("abc"), k, n), x, GRAPH, OPT, 2032)
    modified = k.copy()
    modified[0] = 5 - k[0]
    second, _, _ = crossfit_residuals(Panel(names, tuple("abc"), modified, n), x, GRAPH, OPT, 2032)
    np.testing.assert_allclose(
        logit((k[0] + .5) / 6) - first[0], logit((modified[0] + .5) / 6) - second[0], atol=1e-12,
    )
    assert [r["predicted_paintings"] for r in receipts] == [8, 8, 8]
    reversed_splits = dict(zip(names[::-1], painting_splits(names[::-1], 2032, 3)))
    assert reversed_splits == dict(zip(names, painting_splits(names, 2032, 3)))


def test_synthetic_graph_recovery_and_unsupported_node():
    rng = np.random.default_rng(2032)
    latent_a, latent_b, shared = rng.normal(size=(3, 240))
    residuals = np.stack([latent_a, latent_a * .65, latent_b, latent_b * .85, -latent_b], axis=1)
    residuals += rng.normal(0, 0.13, size=residuals.shape)
    n = np.full(residuals.shape, 5.)
    n[6:, 4] = 0
    n[::7, 1] = 0  # Well-supported edges may still have missing paintings.
    panel = Panel(tuple(map(str, range(len(n)))), tuple("abcde"), n / 2, n)
    full = estimate_graph(panel, residuals, shared, GRAPH)
    left = estimate_graph(panel.subset(np.arange(len(n)) % 2 == 0), residuals[::2], shared[::2], GRAPH)
    right = estimate_graph(panel.subset(np.arange(len(n)) % 2 == 1), residuals[1::2], shared[1::2], GRAPH)
    assert full["edges"]["a", "b"]["weight"] > full["edges"]["a", "c"]["weight"]
    assert full["edges"]["c", "d"]["weight"] > full["edges"]["b", "d"]["weight"]
    assert full["unsupported"] == ["e"]
    assert graph_stability(full, left, right, GRAPH)["pass"]
    no_signal = estimate_graph(panel, np.zeros_like(residuals), shared, GRAPH)
    assert not graph_stability(no_signal, no_signal, no_signal, GRAPH)["pass"]


def test_fit_zero_graph_and_uniform_graph_give_identical_predictions():
    rng = np.random.default_rng(2032)
    x = rng.normal(size=(30, 3))
    n = np.full((30, 4), 5.)
    k = rng.binomial(5, .65, size=n.shape)
    panel = Panel(tuple(map(str, range(len(n)))), tuple("abcd"), k, n)
    l_mat, proj = laplacian(np.ones((4, 4)) - np.eye(4))
    control, _ = fit_head(panel, x, OPT, deviation_ratio=1)
    candidate, _ = fit_head(panel, x, OPT, deviation_ratio=1, fraction=0.5,
                            graph={"laplacian": l_mat, "projection": proj})
    np.testing.assert_allclose(control.logits(x, panel.languages), candidate.logits(x, panel.languages), atol=1e-10)


def test_inner_guardrail_pairing_and_decision():
    spec = {
        "minimum_pooled_contrast_gain_vs_both": .001, "maximum_pooled_log_loss_regression_vs_both": .005,
        "minimum_winning_folds_vs_both": 3, "require_positive_paired_ci_lower_bound": True,
        "bootstrap_replicates": 500, "bootstrap_seed": 203204, "support_bin_edges": [1, 50, 200], "calibration_bins": 10,
    }
    trials = [{"contrast": .2, "log_loss": .7, "value": .5}, {"contrast": .3, "log_loss": .71, "value": 1}]
    assert choose_inner(trials, .005)[0]["value"] == .5
    gold = pd.DataFrame([
        (f"p{p}", language, k, 5, p % 5 + 1, support)
        for p in range(100) for language, k, support in zip("abc", (1, 2, 3), (15, 150, 500))
    ], columns=["painting", "language", "k", "n", "fold", "fitting_paintings"])
    winner = gold[["painting", "language"]].assign(p_positive=gold["k"] / gold["n"])
    weaker = winner.assign(p_positive=winner["language"].map(dict(a=.3, b=.2, c=.6)))
    report = evaluate(gold, {REFERENCE: weaker, CONTROL: weaker, CANDIDATE: winner}, spec)
    assert report["decision"]["status"] == "EXPLORATORY_GO"
    assert report["comparisons"][REFERENCE]["pooled"]["wins"] == 100
    assert report["systems"][CANDIDATE]["pooled"]["spearman"] == pytest.approx(1)
    assert report["systems"][CANDIDATE]["calibration"]["macro"]["ece"] == pytest.approx(0)
    report["comparisons"][CONTROL]["pooled"]["ci95"][0] = -0.001
    assert decision(report["comparisons"], spec)["status"] == "NO_GO_OOF"


def test_gate_stops_before_outer_fits_or_scoring(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_exp032.py"
    spec = importlib.util.spec_from_file_location("exp032_cli_for_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    frame = pd.DataFrame({"fold": [1, 2, 3, 4, 5]})
    tested = []

    def graphs(train, features, indices, config, fold):
        assert fold not in train["fold"].to_numpy()
        tested.append(fold)
        return {"fold": fold, "stability": {"pass": fold != 3}}

    def forbidden(*args, **kwargs):
        pytest.fail("Graph gate failure must stop outer fits and scoring")

    monkeypatch.setattr(cli, "prepare_fold_graphs", graphs)
    monkeypatch.setattr(cli, "select_parameters", forbidden)
    monkeypatch.setattr(cli, "evaluate", forbidden)
    result = cli.run((frame, None, {"dummy": np.ones((1, 1))}, {}), tmp_path, cli.read_json(cli.CONFIG))
    assert tested == [1, 2, 3, 4, 5]
    assert result["status"] == "NO_GO_GRAPH_STABILITY"
    assert not result["outer_scored"]
    assert not (tmp_path / "metrics.json").exists()
