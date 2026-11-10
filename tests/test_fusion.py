import numpy as np
import pandas as pd
import pytest

from st3 import submission
from st3.ablation import painting_folds
from st3.contrast_study import new_output, write_development
from st3.data import check_dev_isolation, check_split_integrity, training_pairs
from st3.fusion import EARLY, blend, choose_fusion, combine_features, verify_reference
from st3.low_rank_study import verify_folds
from st3.visual_study import REFERENCE, SIGLIP


def test_blend_aligns_keys_and_is_deterministic():
    clip = pd.DataFrame({"painting": ["a", "b"], "language": ["X", "Y"], "p_positive": [.2, .8]})
    siglip = clip.assign(p_positive=[.6, .4]).iloc[::-1]
    for alpha, expected in [(.25, [.5, .5]), (.50, [.4, .6]), (.75, [.3, .7])]:
        actual = blend(clip, siglip, alpha)
        assert actual["p_positive"].tolist() == expected
        pd.testing.assert_frame_equal(actual, blend(clip, siglip, alpha))
    with pytest.raises(ValueError):
        blend(clip, siglip.iloc[:1], .5)
    with pytest.raises(ValueError, match="finite"):
        blend(clip, siglip.assign(p_positive=np.nan), .5)


def test_concat_preserves_image_alignment_and_unit_norm():
    clip = {"a": np.eye(1, 512, dtype=np.float32)[0], "b": np.eye(1, 512, k=1, dtype=np.float32)[0]}
    siglip = {"b": np.eye(1, 768, k=2, dtype=np.float32)[0], "a": np.eye(1, 768, dtype=np.float32)[0]}
    combined = combine_features(clip, siglip)
    assert list(combined) == ["a", "b"]
    for name in combined:
        assert combined[name].shape == (1280,) and combined[name].dtype == np.float32
        np.testing.assert_allclose(combined[name], np.r_[clip[name], siglip[name]] / np.sqrt(2.))
        assert np.linalg.norm(combined[name]) == pytest.approx(1.)
    with pytest.raises(ValueError, match="coverage"):
        combine_features(clip, {"a": siglip["a"]})
    with pytest.raises(ValueError, match="unit-normalized"):
        combine_features(clip, {**siglip, "a": 2 * siglip["a"]})


def test_reference_regression_and_train_only_exact_groups():
    annotations = pd.DataFrame([(p, p + ".jpg", lang, "train" if p != "v" else "val", "awe")
                                for p in "abcdefv" for lang in "XY" for _ in range(5)],
                               columns=["painting", "image_name", "language", "split", "emotion"])
    check_split_integrity(annotations)
    train = training_pairs(annotations)
    dev = pd.DataFrame({"painting": ["v"], "image": ["v.jpg"]})
    check_dev_isolation(annotations, dev)
    assert set(train.painting).isdisjoint(dev.painting)
    saved = train.assign(fold=0)
    for fold, (_, held) in enumerate(painting_folds(train, 3, 2027), 1):
        saved.iloc[held, saved.columns.get_loc("fold")] = fold
    for fit, held in verify_folds(train, saved, {"cv_folds": 3, "cv_seed": 2027}):
        assert set(train.iloc[fit].image).isdisjoint(train.iloc[held].image)
    pred = train[["painting", "language"]].assign(p_positive=.5)
    verify_reference(pred, pred.iloc[::-1])
    with pytest.raises(ValueError, match="differ"):
        verify_reference(pred, pred.assign(p_positive=.500001))


def test_selection_prefers_pooled_contrast_then_simplicity_within_tie():
    summary = pd.DataFrame({"pooled_contrast": [.43, .4302, .4326, .433], "contrast_mean": [.43, .45, .431, .429],
                            "log_loss_mean": [.74, .741, .738, .7385], "fold_wins": [0, 1, 2, 1],
                            "converged": 5, "folds": 5, "scored": 5, "loss_scored": 5},
                           index=[REFERENCE, SIGLIP, EARLY, "E50"])
    cfg = {"log_loss_guardrail_bits": .01, "contrast_tie_band": .001, "negligible_log_loss_bits": .001,
           "complexity": {SIGLIP: 1, EARLY: 2, "E50": 3}}
    result = choose_fusion(summary, cfg, 5)
    assert result["pooled_leader"] == "E50" and result["model"] == EARLY
    summary.loc[EARLY, "log_loss_mean"] = .751
    assert choose_fusion(summary, cfg, 5)["model"] == "E50"
    summary.loc["E50", "converged"] = 4
    assert choose_fusion(summary, cfg, 5)["model"] == SIGLIP
    summary.loc[SIGLIP, "pooled_contrast"] = .43
    assert choose_fusion(summary, cfg, 5)["model"] == REFERENCE


def test_fused_dev_zip_and_existing_artifact_protection(tmp_path):
    gold = pd.DataFrame({"painting": ["p"] * 3, "language": list("ABC"), "k": [0, 2, 5], "n": 5, "p": [0, .4, 1.]})
    clip = gold[["painting", "language"]].assign(p_positive=[.3, .2, .9])
    siglip = clip.assign(p_positive=[.1, .3, .95])
    combined = blend(clip, siglip, .5)
    original = tmp_path / "original"
    original.mkdir()
    protected = original / "predictions.csv"
    protected.write_bytes(b"prior bytes")
    with pytest.raises(ValueError, match="already exists"):
        with new_output(original):
            pytest.fail("Existing artifacts must not be written")
    output = tmp_path / "new"
    with new_output(output) as stage:
        write_development(stage, gold, {REFERENCE: clip, "E50": combined}, "E50",
                          {"bootstrap_replicates": 10, "bootstrap_seed": 7}, REFERENCE)
        submission.pack(stage / "predictions/E50/predictions.csv", stage / "submission.zip", gold)
    pd.testing.assert_frame_equal(combined, submission.validate_archive(output / "submission.zip", gold))
    assert protected.read_bytes() == b"prior bytes"
