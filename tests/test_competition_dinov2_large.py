import numpy as np
import pandas as pd
import pytest

from st3.competition_dinov2_large import CONFIG, TRACK1, evaluate, predict
from st3.e50_heads import read_json


def test_frozen_head_excludes_held_paintings_and_requires_keys_only():
    x = pd.DataFrame([[0., 1.], [1., 0.], [.6, .8]], index=list("abc"))
    train = pd.DataFrame([(p, l, k, 7) for p in "ab" for l, k in (("Thai", 3), ("Hindi", 4), ("Tamil", 5))],
                         columns=["painting", "language", "k", "n"])
    keys = pd.DataFrame([("c", "Thai"), ("c", "Hindi")], columns=["painting", "language"])
    rule = read_json(TRACK1)["ridge"]
    result, details = predict(train, keys, x, rule)
    assert details["converged"] and result.dinov2_large.between(0, 1).all()
    with pytest.raises(ValueError, match="disjoint"):
        predict(train, keys.assign(painting="a"), x, rule)
    with pytest.raises(ValueError, match="keys"):
        predict(train, keys.assign(k=3), x, rule)


def test_paired_fusion_cannot_promote_the_same_predictions():
    gold = pd.DataFrame([(f"p{i}", l, k, 5, 1 + i // 20) for i in range(100)
                         for l, k in (("Thai", 4), ("Hindi", 2), ("Tamil", 3), ("English", 1))],
                        columns=["painting", "language", "k", "n", "fold"])
    q = gold.k / gold.n
    predictions = gold[["painting", "language"]].assign(corrected_ridge=q, dinov2_large=q, blend=q)
    stats, records = evaluate(gold, predictions, read_json(TRACK1), read_json(CONFIG))
    assert stats["selected"] == "corrected_ridge"
    assert stats["test_like"]["paired"]["blend"]["fold_wins"] == 0
    assert len(records) == 200
    np.testing.assert_array_equal(stats["test_like"]["paired"]["blend"]["bootstrap_95"], [0, 0])
