import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from st3.baseline import file_sha256
from st3.contrast_study import choose
from st3.frozen_features import cache_location, normalize_rows
from st3.siglip2 import FEATURE, PREFIX, load_siglip2_features, pooled_features
from st3.visual_study import REFERENCE, SIGLIP


def test_siglip_features_use_only_pixels_and_the_learned_pooler():
    class VisionTower:
        def __call__(self, pixel_values):
            assert pixel_values.shape == (2, 3, 224, 224)
            pool = np.zeros((2, 768), dtype=np.float32)
            pool[:, :2] = [3, 4]
            return SimpleNamespace(pooler_output=pool, last_hidden_state=np.ones((2, 196, 768)))

    inputs = {"pixel_values": np.zeros((2, 3, 224, 224), dtype=np.float32)}
    features = normalize_rows(pooled_features(VisionTower(), inputs, 768))
    assert features.shape == (2, 768)
    np.testing.assert_allclose(features[:, :2], [[.6, .8], [.6, .8]])
    assert not features[:, 2:].any()


def test_siglip_cache_bridge_binds_pinned_assets_and_image_identity(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"a")
    (images / "b.jpg").write_bytes(b"b")
    pointer = tmp_path / "extraction" / "features.json"
    pointer.parent.mkdir()
    config = {"model": "pinned-model", "revision": "pinned-revision", "width": 768, "batch_size": 16,
              "transformers_version": "4.51.3", "weights_sha256": "expected-weights"}
    identity = {key: config[key] for key in ("model", "revision", "width", "batch_size")}
    identity.update({"feature": FEATURE, "runtime": {"transformers": "4.51.3"}})
    names = ["a.jpg", "b.jpg"]
    location = cache_location(tmp_path, names, tmp_path, identity, PREFIX)
    location.mkdir()
    features_path = location / "features.npz"
    np.savez(features_path, names=names, features=np.eye(2, 768, dtype=np.float32))
    manifest = {"identity": identity, "features_sha256": file_sha256(features_path),
                "assets_sha256": {"model.safetensors": "expected-weights"}}
    (location / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pointer.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(data_dir=tmp_path, run_dir=tmp_path, cache_dir=tmp_path)
    train, dev = pd.DataFrame({"image": names[:1]}), pd.DataFrame({"image": names[1:]})
    cached, _ = load_siglip2_features(args, dev, train, config)
    assert set(cached) == set(names)
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_siglip2_features(args, dev, train, {**config, "weights_sha256": "wrong"})
    (images / "b.jpg").write_bytes(b"changed")
    with pytest.raises(FileNotFoundError):
        load_siglip2_features(args, dev, train, config)


def test_siglip_selection_uses_exact_absolute_guardrail():
    summary = pd.DataFrame({"contrast_mean": [.435705, .441], "contrast_gain": [0, .005295],
                            "log_loss_mean": [.7380560299, .748056], "fold_wins": [0, 4],
                            "converged": 5, "folds": 5, "scored": 5, "loss_scored": 5},
                           index=[REFERENCE, SIGLIP])
    cfg = {"maximum_log_loss": .748056, "minimum_contrast_gain": .005, "minimum_fold_wins": 4,
           "contrast_tie_band": 0., "preference_order": [SIGLIP]}
    assert choose(summary, cfg, 5, REFERENCE)["model"] == SIGLIP
    summary.loc[SIGLIP, "log_loss_mean"] = .74805601
    assert choose(summary, cfg, 5, REFERENCE)["model"] == REFERENCE
