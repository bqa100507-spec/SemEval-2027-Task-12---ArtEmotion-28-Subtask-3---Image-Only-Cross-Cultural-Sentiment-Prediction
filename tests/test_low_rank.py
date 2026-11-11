from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from st3.ablation import painting_folds
from st3.low_rank import (binomial_nll, factor_logits, factor_objective, fit_factors, main_logits,
                          spectrum, split_lr, truncate_svd, unpack_factors)
from st3.low_rank_study import grid_predictions, image_index, select_candidate, verify_folds, write_dev_analysis
from st3.submission import validate_predictions


def test_factorized_forward_equals_reconstructed_full_coefficients():
    x = np.array([[.2, -.4, .6], [1., 0., -.3]])
    theta = np.linspace(-.5, .5, 16)
    main, u, v = unpack_factors(theta, width=3, languages=2, rank=2)
    assert (u.shape, v.shape) == ((2, 2), (3, 2))
    weights = u @ v.T
    fitted = SimpleNamespace(coef_=np.r_[main[1:], weights.ravel()][None, :], intercept_=main[:1])
    recovered_main, recovered_weights = split_lr(fitted, 3, 2)
    np.testing.assert_array_equal(recovered_weights, weights)
    dense = main_logits(x, recovered_main)
    dense[:, :2] += x @ recovered_weights.T
    np.testing.assert_allclose(factor_logits(x, main, u, v), dense, atol=1e-14)
    np.testing.assert_allclose(dense[:, 2], main[0] + x @ main[1:4])


def test_binomial_loss_matches_explicit_votes_and_is_stable():
    logits = np.array([-1., 2.])
    expected = 2 * np.logaddexp(0, 1) + 3 * np.logaddexp(0, -1) + 5 * np.logaddexp(0, -2)
    assert binomial_nll(logits, np.array([2, 5]), np.array([5, 5])) == pytest.approx(expected)
    assert binomial_nll(np.array([-1000., 1000.]), np.array([5, 0]), np.array([5, 5])) == 10000.


def test_objective_gradient_matches_finite_differences():
    x = np.array([[.4, .9], [-.7, .5], [.4, -.2]])
    k = np.array([[0., 3.], [2., 4.], [5., 0.]])
    n = np.array([[0., 5.], [5., 6.], [5., 5.]])
    theta = np.linspace(-.3, .3, 9)
    args = (x, k, n, 1, 2.5)
    assert check_grad(lambda v: factor_objective(v, *args)[0],
                      lambda v: factor_objective(v, *args)[1], theta) < 1e-6


def test_fit_is_seeded_and_output_passes_official_prediction_validation():
    x = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    k = np.array([[5., 0.], [0., 5.], [0., 5.], [5., 0.]])
    config = {"init_scale": .05, "factor_penalty_multiplier": 1., "factor_threads": 1,
              "optimizer": {"maxiter": 400, "gtol": 2e-7, "ftol": 1e-10}}
    args = (x, k, np.full_like(k, 5), np.zeros(5), 1, 42, config)
    first, metadata = fit_factors(*args)
    second, other = fit_factors(*args)
    assert metadata == other
    assert metadata["converged"] and metadata["objective"] <= metadata["initial_objective"]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    dev = pd.DataFrame({"painting": ["p", "q"], "language": ["a", "unknown"], "image": ["a.jpg", "b.jpg"]})
    rows = image_index(dev, {"a.jpg": x[0], "b.jpg": x[1]}, ["a", "b"])
    result = grid_predictions(dev, factor_logits(rows[0], *first), rows)
    assert np.isfinite(validate_predictions(result, dev)).all()
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(result.assign(p_positive=np.nan), dev)


def test_svd_energy_and_reconstruction_error():
    w = np.array([[4., 0., 0.], [0., 3., 0.]])
    decomposition = np.linalg.svd(w, full_matrices=False)
    values, effective = spectrum(decomposition[1])
    np.testing.assert_allclose(values["energy"], [.64, .36])
    np.testing.assert_allclose(values["cumulative_energy"], [.64, 1.])
    np.testing.assert_allclose(values["relative_error"], [.6, 0.])
    assert 1 < effective <= 2
    assert np.linalg.norm(w - truncate_svd(decomposition, 1)) == pytest.approx(values["frobenius_error"][0])
    np.testing.assert_allclose(truncate_svd(decomposition, 2), w)


def test_exact_grouped_fold_membership_is_required():
    train = pd.DataFrame([(painting, language) for painting in "abcdef" for language in "XY"],
                         columns=["painting", "language"])
    config = {"cv_folds": 3, "cv_seed": 32027}
    saved = train.assign(fold=0)
    for fold, (_, held) in enumerate(painting_folds(train, 3, 32027), 1):
        saved.loc[held, "fold"] = fold
    for fit, held in verify_folds(train, saved.iloc[::-1], config):
        assert set(train.iloc[fit].painting).isdisjoint(train.iloc[held].painting)
    saved.loc[0, "fold"] = 99
    with pytest.raises(ValueError, match="split painting"):
        verify_folds(train, saved, config)


def test_selection_obeys_guardrail_convergence_and_small_rank_tie_break():
    summary = pd.DataFrame({"contrast_mean": [.44, .49, .45, .451, .50], "log_loss_mean": [.74, .751, .748, .748, .74],
                            "all_converged": [True, True, True, True, False]},
                           index=["C", "LR-2", "LR-4", "LR-8", "LR-16"])
    config = {"ranks": [2, 4, 8, 16], "log_loss_guardrail_bits": .01, "contrast_tie_band": .002, "final_seed": 2027}
    result = select_candidate(summary, config)
    assert result["model"] == "LR-4" and result["eligible_candidates"] == ["LR-4", "LR-8"]
    summary.loc["LR-4":"LR-8", "log_loss_mean"] = .8
    assert select_candidate(summary, config)["model"] == "C"


def test_selected_development_analysis_keeps_pairing_and_writes_predictions(tmp_path):
    gold = pd.DataFrame({"painting": ["p"] * 3, "language": list("abc"), "k": [0, 2, 5], "n": 5})
    predictions = {name: gold[["painting", "language"]].assign(p_positive=q)
                   for name, q in {"A": [.5, .5, .5], "B": [.1, .42, .9], "C": [.9, .58, .1],
                                   "LR-2": [.1, .42, .9]}.items()}
    config = {"bootstrap_replicates": 10, "bootstrap_seed": 42}
    metrics = write_dev_analysis(tmp_path, gold, predictions, "LR-2", config)
    assert metrics["LR-2"]["cultural_contrast"] == pytest.approx(1.)
    assert metrics["C"]["cultural_contrast"] == pytest.approx(-1.)
    draws = pd.read_csv(tmp_path / "bootstrap_distribution.csv")
    np.testing.assert_allclose(draws["contrast_LR_minus_C"], np.full(10, 2.))
    assert (tmp_path / "predictions/LR-2/predictions.csv").is_file()
