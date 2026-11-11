import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from st3.baseline import C, MAX_ITER, design_matrix, fit_lr, language_prior


def test_sparse_features_and_weighted_fit_match_dense_upstream():
    embeddings = {"a": [.2, .8], "b": [.7, -.1]}
    names = ["a", "a", "b", "b"]
    langs = ["A", "B", "A", "B"]
    counts = pd.DataFrame({"k": [5, 1, 3, 0], "n": [5, 6, 5, 7]})
    images = np.array([embeddings[name] for name in names])
    onehot = np.tile(np.eye(2), (2, 1))
    dense = np.hstack([images, onehot, (onehot[:, :, None] * images[:, None, :]).reshape(4, -1)])
    matrix = design_matrix(names, langs, embeddings, ["A", "B"])
    np.testing.assert_array_equal(matrix.toarray(), dense)
    sparse_model = fit_lr(matrix, counts, 2027)
    dense_model = LogisticRegression(C=C, max_iter=MAX_ITER).fit(
        np.vstack([dense, dense]), np.repeat([1, 0], len(counts)),
        sample_weight=np.r_[counts["k"], counts["n"] - counts["k"]])
    np.testing.assert_allclose(sparse_model.predict_proba(matrix), dense_model.predict_proba(dense), atol=1e-10)
    unseen = design_matrix(["a"], ["unseen"], embeddings, ["A", "B"])
    np.testing.assert_array_equal(unseen.toarray(), [[.2, .8, 0, 0, 0, 0, 0, 0]])


def test_language_prior_uses_pair_mean_and_fallback():
    train = pd.DataFrame({"language": ["A", "A"], "n": [5, 100], "p": [.2, .8]})
    dev = pd.DataFrame({"painting": ["a", "b"], "language": ["A", "B"]})
    assert language_prior(train, dev)["p_positive"].tolist() == [.5, .7]
