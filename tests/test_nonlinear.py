import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from st3 import data, submission
from st3.baseline import file_sha256
from st3.nonlinear import (INPUT_WIDTH, NonlinearHead, concatenate_features, configure_device, early_split, fit_seed,
                           predict_seed, vote_loss)
from st3.nonlinear_study import MODELS, choose_head, compare_predictions, run_nonlinear_dev
from st3.residual_crossfit import disjoint

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "exp013.json").read_text())


def fixture():
    rows = [(f"p{i}", f"p{i}.jpg", lang, (i + j) % 6, 5, "train")
            for i in range(12) for j, lang in enumerate(("Tamil", "Yoruba"))]
    frame = pd.DataFrame(rows, columns=["painting", "image", "language", "k", "n", "split"])
    features = {f"p{i}.jpg": np.eye(1, INPUT_WIDTH, k=i, dtype=np.float32)[0] for i in range(12)}
    return frame, features


def test_concat_preserves_width_order_scaling_and_cache():
    clip, siglip, large = ({"a": np.eye(1, width, k=1, dtype=np.float32)[0]} for width in (512, 768, 768))
    values = concatenate_features(clip, siglip, large)["a"]
    assert values.shape == (2048,) and values.dtype == np.float32
    assert np.flatnonzero(values).tolist() == [1, 513, 1281]
    np.testing.assert_allclose(values[[1, 513, 1281]], 1 / np.sqrt(3), rtol=1e-7)
    assert np.linalg.norm(clip["a"]) == 1.
    with pytest.raises(ValueError, match="same nonempty image set"):
        concatenate_features(clip, {"b": siglip["a"]}, large)
    with pytest.raises(ValueError, match="unit-normalized"):
        concatenate_features({"a": clip["a"] * .5}, siglip, large)


def test_exact_architecture_parameter_count_and_untrained_language():
    head = NonlinearHead([str(i) for i in range(25)], CONFIG["hidden_width"], CONFIG["dropout"]).eval()
    assert head.visual[0].in_features == 2048 and head.visual[0].out_features == 64
    assert head.visual[3].p == .20
    assert sum(p.numel() for p in head.parameters()) == 132954
    image = torch.zeros((2, 2048))
    with torch.no_grad():
        head.language_bias.weight[0] = .2
        head.language_visual.weight[0, 0] = .7
        h = head.visual(image)
        shared = head.shared(h).squeeze(-1)
        actual = head(image, torch.tensor([-1, 0]))
        expected = shared + torch.tensor([0., .2 + .7 * float(h[1, 0]) / 8])
    torch.testing.assert_close(actual, expected, rtol=0., atol=1e-7)


def test_vote_loss_equals_expanded_annotator_bce_and_gradient():
    logits = torch.tensor([-.4, .6], dtype=torch.float64, requires_grad=True)
    k, n = torch.tensor([2., 1.], dtype=torch.float64), torch.tensor([5., 9.], dtype=torch.float64)
    loss = vote_loss(logits, k / n, n) / n.sum()
    expected = (k * torch.nn.functional.softplus(-logits) + (n - k) * torch.nn.functional.softplus(logits)).sum() / n.sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(logits.grad, (n * torch.sigmoid(logits) - k) / n.sum())


def test_early_stopping_and_outer_group_isolation():
    frame, _ = fixture()
    train, held = frame.iloc[:20], frame.iloc[20:]
    a, b = early_split(train, held, CONFIG, 2)
    disjoint(a, b, held)
    assert len(b["painting"].unique()) == 1
    assert set(pd.concat([a, b]).index) == set(train.index)
    _, replay = early_split(train, held, CONFIG, 2)
    pd.testing.assert_frame_equal(b, replay)
    with pytest.raises(ValueError, match="overlapping image"):
        early_split(train, held.assign(image=train.iloc[0]["image"]), CONFIG, 2)
    with pytest.raises(ValueError, match="original training"):
        early_split(train.assign(split="val"), held, CONFIG, 2)


def test_seeded_cpu_refit_restarts_for_selected_epoch_and_repeats():
    frame, features = fixture()
    partition, held = frame.iloc[:20], frame.iloc[20:]
    config = {**CONFIG, "max_epochs": 4, "patience": 2, "batch_size": 8}
    device = configure_device("cpu")
    fit, validation = early_split(partition, held, config, 1)
    model, trace, run = fit_seed(partition, fit, validation, features, 2027, config, device)
    replay, same_trace, _ = fit_seed(partition, fit, validation, features, 2027, config, device)
    assert trace == same_trace
    internal = [row for row in trace if row["phase"] == "selection"]
    full = [row for row in trace if row["phase"] == "refit"]
    assert len(internal) == run["selection_epochs"]
    assert len(full) == run["selected_epoch"] == min(internal, key=lambda row: row["validation_bce"])["epoch"]
    assert all(row["validation_bce"] is None for row in full)
    inputs = held[[*data.KEYS, "image"]]
    one, two = (predict_seed(m, inputs, features, device, 8) for m in (model, replay))
    np.testing.assert_array_equal(one, two)
    assert np.isfinite(one).all() and ((one >= 0.) & (one <= 1.)).all()


def test_averaging_blending_fallback_reference_and_zip(tmp_path):
    held = pd.DataFrame({"painting": ["b", "a"], "language": ["Tamil", "Kazakh"]})
    reference = held.iloc[::-1].assign(p_positive=[.6, .100001])
    seeds = np.array([[.1000001, .2], [.1000004, .4], [.10000097, .9]])
    preds = compare_predictions(held, reference, seeds, ("Tamil",), 3)
    pd.testing.assert_frame_equal(preds["M0"], reference)
    assert preds["M1"]["p_positive"].tolist() == [.1, .5]
    assert preds["M2"]["p_positive"].tolist() == [.6, .100001]
    assert np.round((.100001 + .1) * .5, 6) == .1
    with pytest.raises(ValueError, match="every requested seed"):
        compare_predictions(held, reference, seeds[:2], ("Tamil",), 3)
    csv = tmp_path / submission.FILENAME
    archive = tmp_path / "submission.zip"
    preds["M2"].to_csv(csv, index=False)
    submission.pack(csv, archive, held)
    pd.testing.assert_frame_equal(submission.validate_archive(archive, held), preds["M2"].reset_index(drop=True))


def test_selection_uses_gain_fold_consistency_and_mean_loss():
    summary = pd.DataFrame({"pooled_contrast": [.44, .443, .444], "log_loss_mean": [.72, .724, .725],
                            "fold_wins": [0, 3, 2], "folds": 5, "converged": 5, "scored": 5, "loss_scored": 5},
                           index=pd.Index(MODELS, name="model"))
    assert choose_head(summary, CONFIG, 5)["model"] == "M1"
    summary.loc["M2", "fold_wins"] = 3
    assert choose_head(summary, CONFIG, 5)["model"] == "M2"
    summary.loc["M2", "converged"] = 4
    summary.loc["M1", "log_loss_mean"] = .726
    assert choose_head(summary, CONFIG, 5)["model"] == "M0"
    summary.loc["M1", ["log_loss_mean", "pooled_contrast"]] = [.72, .4409]
    assert choose_head(summary, CONFIG, 5)["model"] == "M0"


def test_no_go_preserves_history_and_does_not_score_development(tmp_path):
    cv = tmp_path / "cv"
    cv.mkdir()
    history = tmp_path / "old_submission.zip"
    history.write_bytes(b"original artifact")
    original = file_sha256(history)
    summary = pd.DataFrame({"pooled_contrast": .44, "log_loss_mean": .72, "fold_wins": 3, "folds": 5,
                            "converged": 5, "scored": 5, "loss_scored": 5}, index=pd.Index(MODELS, name="model"))
    decision = choose_head(summary, CONFIG, 5)
    summary.to_csv(cv / "summary.csv")
    (cv / "run.json").write_text(json.dumps({"config": CONFIG, "exp002_config": {"cv_folds": 5}}))
    (cv / "seed_runs.csv").write_text("fixture")
    (cv / "split_audit.json").write_text("[]")
    decision["sha256"] = {name: file_sha256(cv / name) for name in
                          ("run.json", "summary.csv", "seed_runs.csv", "split_audit.json")}
    (cv / "selection.json").write_text(json.dumps(decision))
    run_nonlinear_dev(SimpleNamespace(run_dir=tmp_path), None)
    assert not (tmp_path / "dev").exists() and file_sha256(history) == original
