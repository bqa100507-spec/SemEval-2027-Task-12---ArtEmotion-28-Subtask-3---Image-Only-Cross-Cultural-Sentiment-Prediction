import json

import numpy as np
import pandas as pd
import pytest

from st3 import submission
from st3.baseline import file_sha256
from st3.contrast_study import choose, write_development
from st3.data import check_dev_isolation, check_split_integrity, training_pairs
from st3.dino import cache_location, cls_features, load_dino_cache, normalize_rows
from st3.visual_study import DINO, PCA_DINO, REFERENCE, fit_head, pca_embeddings


def test_extracts_and_normalizes_only_final_cls():
    tokens = np.zeros((2, 3, 768), dtype=np.float32)
    tokens[:, 1, 5] = 100
    tokens[0, 0, :2] = [3, 4]
    tokens[1, 0, 1] = -5
    extracted = normalize_rows(cls_features(tokens, 768))
    assert extracted.shape == (2, 768) and extracted.dtype == np.float32
    np.testing.assert_allclose(extracted[:, :2], [[.6, .8], [0, -1]])
    assert not extracted[:, 2:].any()
    with pytest.raises(ValueError, match="token sequence"):
        cls_features(tokens, 512)
    with pytest.raises(ValueError, match="nonzero"):
        normalize_rows(np.zeros((2, 768)))


def test_cache_identity_uses_image_bytes_and_encoder(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"first version")
    names = ["one.jpg"]
    identity = {"revision": "original", "width": 768}
    location = cache_location(tmp_path, names, tmp_path, identity)
    assert location != cache_location(tmp_path, names, tmp_path, {**identity, "revision": "other"})
    (images / "one.jpg").write_bytes(b"second version")
    assert location != cache_location(tmp_path, names, tmp_path, identity)
    location.mkdir()
    payload = location / "features.npz"
    np.savez(payload, names=np.array(names), features=np.eye(1, 768, dtype=np.float32))
    manifest = {"identity": identity, "features_sha256": file_sha256(payload)}
    (location / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    features, _ = load_dino_cache(location, names, identity)
    assert features["one.jpg"].shape == (768,)
    with payload.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dino_cache(location, names, identity)


def test_pca_uses_unique_fit_images_and_ignores_held_values_during_fit():
    vectors = {"a": [1, 2, 0], "b": [3, 4, 0], "c": [0, 2, 3], "d": [8, 2, 1]}
    fit = ["a", "b", "b", "c"]
    projected, pca = pca_embeddings(vectors, fit, ["d"], 2)
    different, other = pca_embeddings({**vectors, "d": [-100, 0, 99]}, fit, ["d"], 2)
    np.testing.assert_allclose(pca.mean_, np.mean([vectors[key] for key in "abc"], axis=0))
    np.testing.assert_array_equal(pca.components_, other.components_)
    assert pca.n_samples_ == 3
    for key in "abc":
        np.testing.assert_array_equal(projected[key], different[key])
    np.testing.assert_allclose(np.linalg.norm(list(projected.values()), axis=1), np.ones(4))
    with pytest.raises(ValueError, match="overlap"):
        pca_embeddings(vectors, fit, ["b"], 2)


def test_visual_head_and_package_on_isolated_dev_images(tmp_path):
    rows = []
    for p in ("p", "q", "v"):
        split = "val" if p == "v" else "train"
        for index, language in enumerate("ABC", 1):
            k = index if p == "p" else 5 - index
            rows.extend((p, p + ".jpg", split, language, emotion) for emotion in ["awe"] * k + ["fear"] * (5-k))
    annotations = pd.DataFrame(rows, columns=["painting", "image_name", "split", "language", "emotion"])
    check_split_integrity(annotations)
    train = training_pairs(annotations)
    dev = pd.DataFrame({"painting": ["v"] * 3, "image": ["v.jpg"] * 3, "language": list("ABC")})
    check_dev_isolation(annotations, dev)
    assert set(train.image).isdisjoint(dev.image)
    pred, info = fit_head(train, dev, {"p.jpg": [1, 0], "q.jpg": [0, 1], "v.jpg": [.7, .7]}, 2027)
    assert info["converged"] and info["parameters"] == 12
    gold = dev[["painting", "language"]].assign(k=[0, 2, 5], n=5, p=[0., .4, 1.])
    write_development(tmp_path, gold, {REFERENCE: pred, DINO: pred}, DINO,
                      {"bootstrap_replicates": 10, "bootstrap_seed": 42}, REFERENCE)
    archive = tmp_path / "submission.zip"
    submission.pack(tmp_path / "predictions" / DINO / "predictions.csv", archive, dev)
    pd.testing.assert_frame_equal(pred, submission.validate_archive(archive, dev))


def test_selection_requires_all_guardrails_and_prefers_512_if_tied():
    summary = pd.DataFrame({"contrast_mean": [.43, .439, .440], "log_loss_mean": [.74, .741, .747],
                            "contrast_gain": [0, .009, .010], "log_loss_delta": [0, .001, .007],
                            "fold_wins": [0, 4, 5], "converged": 5, "scored": 5, "loss_scored": 5, "folds": 5},
                           index=[REFERENCE, PCA_DINO, DINO])
    cfg = {"minimum_contrast_gain": .005, "minimum_fold_wins": 4, "log_loss_guardrail_bits": .01,
           "contrast_tie_band": .002, "preference_order": [PCA_DINO, DINO]}
    assert choose(summary, cfg, 5, REFERENCE)["model"] == PCA_DINO
    summary.loc[PCA_DINO, "converged"] = 4
    summary.loc[DINO, "log_loss_delta"] = .02
    assert choose(summary, cfg, 5, REFERENCE)["model"] == REFERENCE
