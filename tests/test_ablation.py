from collections import Counter

import numpy as np
import pandas as pd
import pytest

from st3.ablation import paired_analysis, paired_bootstrap, painting_folds, shuffle_images
from st3.baseline import design_matrix
from st3.data import check_dev_isolation, check_split_integrity, training_pairs
from st3.metrics import score


def test_shared_design_is_image_and_language_prefix_of_official_design():
    args = (["im1", "im1", "im2"], ["a", "b", "new"], {"im1": [.25, .5], "im2": [-.1, .8]}, ["a", "b"])
    shared = design_matrix(*args, interactions=False).toarray()
    np.testing.assert_array_equal(shared, [[.25, .5, 1, 0], [.25, .5, 0, 1], [-.1, .8, 0, 0]])
    np.testing.assert_array_equal(shared, design_matrix(*args).toarray()[:, :4])


def test_shuffle_is_a_label_free_painting_derangement():
    dev = pd.DataFrame([(p, language, p + ".jpg") for p in "abcd" for language in ("X", "Y")],
                       columns=["painting", "language", "image"])
    shuffled, assignments = shuffle_images(dev, 12000)
    pd.testing.assert_frame_equal(shuffled[["painting", "language"]], dev[["painting", "language"]])
    assert shuffled.groupby("painting")["image"].nunique().eq(1).all()
    assert shuffled["image"].ne(dev["image"]).all()
    assert set(assignments["source_painting"]) == set(assignments["painting"]) == set("abcd")
    assert assignments["source_painting"].is_unique
    pd.testing.assert_frame_equal(assignments, shuffle_images(dev.iloc[::-1], 12000)[1])
    with pytest.raises(ValueError, match="at least two"):
        shuffle_images(dev.iloc[:2], 12000)


def test_paired_bootstrap_uses_the_same_painting_indices_and_fixed_seed():
    frame = pd.DataFrame({"painting": ["a", "b", "excluded"], "status": ["scored", "scored", "constant_gold"],
                          "contrast_C_minus_A": [0, 1, 99], "contrast_C_minus_B": [1, 0, -99]})
    sampled, intervals = paired_bootstrap(frame, 100, 42)
    assert set(sampled["contrast_C_minus_A"]) == {0, .5, 1}
    np.testing.assert_array_equal(sampled["contrast_C_minus_A"] + sampled["contrast_C_minus_B"], np.ones(100))
    assert intervals["contrast_C_minus_A"] == {"difference": .5, "ci95": [0, 1]}
    repeated, _ = paired_bootstrap(frame.iloc[::-1], 100, 42)
    pd.testing.assert_frame_equal(repeated, sampled)


def test_analysis_preserves_official_eligibility_and_constant_prediction_score():
    gold = pd.DataFrame({"painting": ["a"] * 3 + ["b"] * 3 + ["c"] * 2 + ["d"] * 3,
                         "language": list("XYZXYZXYXYZ"), "k": [0, 2, 5, 2, 2, 2, 1, 3, 1, 4, 5], "n": 5})
    predictions = {}
    for name, q in {"A": [.5] * 11, "B": [.1, .4, .9] * 3 + [.6, .8],
                    "C": [.9, .6, .1] * 3 + [.8, .6]}.items():
        predictions[name] = gold[["painting", "language"]].assign(p_positive=q)
    scores = {name: score(gold, frame) for name, frame in predictions.items()}
    _, paintings, languages = paired_analysis(gold, predictions, scores)
    assert paintings["status"].tolist() == ["scored", "constant_gold", "too_few_languages", "scored"]
    assert paintings["constant_prediction_A"].tolist() == [True, False, False, True]
    assert paintings["contrast_A"].dropna().tolist() == [0, 0]
    for name in "ABC":
        assert paintings[f"contrast_{name}"].mean() == pytest.approx(scores[name]["cultural_contrast"])
        for row in languages.itertuples():
            assert getattr(row, f"log_loss_{name}") == scores[name]["per_language"][row.language]["log_loss"]
    assert not languages["macro_eligible"].any()


def test_grouped_folds_and_fits_exclude_validation_paintings():
    rows = [(p, language, p + ".jpg", "train" if p != "z" else "val", emotion)
            for p in "abcdefz" for language in ("X", "Y") for emotion in ["awe"] * 3 + ["anger"] * 2]
    annotations = pd.DataFrame(rows, columns=["painting", "language", "image_name", "split", "emotion"])
    check_split_integrity(annotations)
    train = training_pairs(annotations)
    dev = pd.DataFrame({"painting": ["z"], "language": ["X"], "image": ["z.jpg"]})
    check_dev_isolation(annotations, dev)
    assert "z" not in train["painting"].values
    seen = Counter()
    for fit, holdout in painting_folds(train, 3, 32027):
        for column in ("painting", "image"):
            assert set(train.iloc[fit][column]).isdisjoint(train.iloc[holdout][column])
        seen.update(set(train.iloc[holdout]["painting"]))
    assert seen == Counter(dict.fromkeys("abcdef", 1))
    with pytest.raises(ValueError, match="overlaps training"):
        check_dev_isolation(annotations, dev.assign(image="a.jpg"))
