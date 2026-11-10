import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from st3.ablation import painting_folds
from st3.contrast import PairwiseHuber, PaintingPearson, fit_contrast, objective
from st3.contrast_study import choose, fit_c, write_development, zero_control
from st3.data import check_dev_isolation, check_split_integrity, training_pairs
from st3.low_rank_study import dense_logits, grid_predictions, verify_folds
from st3.submission import validate_predictions


def test_huber_balances_paintings_with_different_language_counts():
    n = np.array([[5, 5, 0], [5, 5, 5]])
    q = np.array([[.1, .5, .99], [.2, .2, .2]])
    aux = PairwiseHuber(np.zeros_like(n), n, .2)
    loss, grad = aux(q)
    assert loss == pytest.approx(.03)
    np.testing.assert_allclose(grad, [[-.1, .1, 0], [0, 0, 0]])


def test_pearson_eligibility_and_constant_predictions():
    n = np.array([[5, 5, 0], [5, 10, 5], [5, 5, 5]])
    k = np.array([[0, 5, 0], [2, 4, 2], [0, 2, 5]])
    aux = PaintingPearson(k, n, .01)
    assert aux.eligible.tolist() == [False, False, True]
    loss, gradient = aux(np.full((3, 3), .5))
    assert loss == pytest.approx(1.)
    assert np.isfinite(gradient).all() and np.abs(gradient[-1]).sum() > 0
    assert np.all(gradient[:2] == 0)
    true_order = np.array([[.5] * 3, [.5] * 3, [0, .4, 1.]])
    assert aux(true_order)[0] < .001
    assert aux(1 - true_order)[0] > 1.999


@pytest.mark.parametrize("loss_type", ["bce", "pairwise", "pearson"])
def test_full_analytic_gradient_and_bce_scaling(loss_type):
    images = np.array([[.4, .6], [-.5, .2]])
    n = np.array([[5, 5, 5], [6, 5, 0]])
    k = np.array([[0, 2, 5], [1, 3, 0]])
    aux = {"bce": None, "pairwise": PairwiseHuber(k, n, .2), "pearson": PaintingPearson(k, n, .01)}[loss_type]
    args = (images, k, n, aux, 0.05 if aux else 0.)
    theta = np.linspace(-.3, .4, 12)
    assert check_grad(lambda x: objective(x, *args)[0], lambda x: objective(x, *args)[1], theta) < 1e-6
    assert objective(np.zeros_like(theta), *args)[2] == pytest.approx(np.log(2))


def test_model_c_lambda_zero_and_seeded_fit():
    train = pd.DataFrame([{"painting": f"p{i}", "image": str(i), "language": language, "k": (i+j) % 6,
                           "n": 5, "p": ((i+j) % 6)/5}
                          for i in range(5) for j, language in enumerate("ABC")])
    embeddings = {str(i): np.array([i/5, (i % 2) / 2]) for i in range(5)}
    held = train.iloc[-3:][["painting", "image", "language"]]
    fit = train.iloc[:-3]
    config = {"optimizer": {"maxiter": 300, "ftol": 1e-11, "gtol": 1e-6}, "blas_threads": 1,
              "zero_optimizer": {"maxiter": 300, "ftol": 1.4210854715202004e-14, "gtol": 1e-4},
              "zero_probability_atol": 1e-10}
    grid, indices, initial, q, _ = fit_c(fit, held, embeddings, 2027)
    assert zero_control(grid, indices, initial, q, config)["max_probability_error"] < 1e-10
    aux = PaintingPearson(grid[1], grid[2], .01)
    a, first = fit_contrast(grid, initial, aux, .01, config)
    b, second = fit_contrast(grid, initial, aux, .01, config)
    assert first == second and first["converged"]
    np.testing.assert_array_equal(a[1], b[1])
    pred = grid_predictions(held, dense_logits(indices[0], *a), indices)
    assert np.isfinite(validate_predictions(pred, held)).all()


def test_exact_folds_train_only():
    annotations = pd.DataFrame([(p, p + ".jpg", lang, split, "awe") for p, split in
                                [(p, "train") for p in "abcdef"] + [("v", "val")]
                                for lang in "ABC" for _ in range(5)],
                               columns=["painting", "image_name", "language", "split", "emotion"])
    check_split_integrity(annotations)
    train = training_pairs(annotations)
    dev = pd.DataFrame({"painting": ["v"], "image": ["v.jpg"], "language": ["A"]})
    check_dev_isolation(annotations, dev)
    saved = train.assign(fold=0)
    for fold, (_, held) in enumerate(painting_folds(train, 3, 32027), 1):
        saved.iloc[held, saved.columns.get_loc("fold")] = fold
    for fit, held in verify_folds(train, saved, {"cv_folds": 3, "cv_seed": 32027}):
        assert set(train.iloc[fit].painting).isdisjoint(train.iloc[held].painting)
        assert "v" not in train.iloc[fit].painting.values
    with pytest.raises(ValueError, match="overlaps training"):
        check_dev_isolation(annotations, dev.assign(image="a.jpg"))


def test_selection_and_development_output(tmp_path):
    summary = pd.DataFrame({"contrast_mean": [.44, .448, .449, .46], "log_loss_mean": [.74, .744, .748, .751],
                            "contrast_gain": [0, .008, .009, .02], "log_loss_delta": [0, .004, .008, .011],
                            "fold_wins": [0, 4, 5, 5], "converged": 5, "scored": 5, "loss_scored": 5, "folds": 5},
                           index=["C", "P-small", "R-small", "R-large"])
    config = {"minimum_contrast_gain": .005, "minimum_fold_wins": 4, "log_loss_guardrail_bits": .01,
              "contrast_tie_band": .002, "preference_order": ["P-small", "R-small", "R-large"],
              "bootstrap_replicates": 10, "bootstrap_seed": 42}
    assert choose(summary, config, 5)["model"] == "P-small"
    summary.loc[["P-small", "R-small"], "converged"] = 4
    assert choose(summary, config, 5)["model"] == "C"
    gold = pd.DataFrame({"painting": ["x"] * 3, "language": list("ABC"), "k": [0, 2, 5], "n": 5, "p": [0, .4, 1.]})
    keys = gold[["painting", "language"]]
    write_development(tmp_path, gold, {"C": keys.assign(p_positive=[.9, .58, .1]),
                                       "P-small": keys.assign(p_positive=[.1, .42, .9])}, "P-small", config)
    assert (tmp_path / "predictions/P-small/predictions.csv").is_file()
    np.testing.assert_allclose(pd.read_csv(tmp_path / "bootstrap_distribution.csv").contrast_candidate_minus_C, 2.)
