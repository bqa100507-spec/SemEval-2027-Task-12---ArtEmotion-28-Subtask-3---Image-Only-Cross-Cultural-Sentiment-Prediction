import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from st3 import siglip_large, submission
from st3.baseline import file_sha256
from st3.frozen_features import cache_location
from st3.siglip_large_study import MODELS, compare_systems, run_large_dev, select_large

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "exp015.json").read_text())


def test_large_vision_verifies_width_parameter_count_and_fp32(monkeypatch):
    model = torch.nn.Linear(4, 2)
    model.config = SimpleNamespace(hidden_size=4, image_size=256, patch_size=16)
    config = {**CONFIG["encoder"], "width": 4, "vision_parameters": 10}
    monkeypatch.setattr(siglip_large, "load_vision", lambda _: model)
    assert siglip_large.load_large_vision(None, config) is model
    with pytest.raises(ValueError, match="pinned FP32"):
        siglip_large.load_large_vision(None, {**config, "width": 5})
    model.half()
    with pytest.raises(ValueError, match="pinned FP32"):
        siglip_large.load_large_vision(None, config)


def test_preflight_uses_repeated_trained_pool_output(monkeypatch, tmp_path):
    seen = []

    def fake_encoder(root, names, directory, config, device, *, prefix, load_model, encode):
        assert names == ["a"] and device == "cuda" and prefix == "siglip2-large"

        def model(**inputs):
            seen.append(inputs)
            return SimpleNamespace(pooler_output=torch.ones((1, 1024)))

        return encode(model, {"pixel_values": torch.zeros((1, 3, 256, 256))}, 1024)

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(siglip_large, "extract_frozen", fake_encoder)
    probe = {}
    result = siglip_large.extract_large(tmp_path, ["a"], tmp_path, CONFIG["encoder"], probe=probe)
    assert result.shape == (1, 1024) and len(seen) == 2 and probe["repeat_exact"]


def test_large_cache_checks_asset_identity_and_checksum(tmp_path):
    names = ["one.jpg"]
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / names[0]).write_bytes(b"fixture")
    config = CONFIG["encoder"]
    identity = {key: config[key] for key in ("model", "revision", "width", "batch_size", "feature", "assets_sha256")}
    identity.update({"runtime": {"transformers": "4.51.3"}, "initial_device": "cuda",
                     "attention": "eager", "processor": "slow"})
    folder = cache_location(tmp_path, names, tmp_path, identity, siglip_large.PREFIX)
    folder.mkdir()
    np.savez(folder / "features.npz", names=names, features=np.eye(1, 1024, dtype=np.float32))
    saved = {"identity": identity, "features_sha256": file_sha256(folder / "features.npz"),
             "assets_sha256": config["assets_sha256"], "final_device": "cuda"}
    (folder / "manifest.json").write_text(json.dumps(saved))
    vectors, _ = siglip_large.load_large_cache(tmp_path, names, tmp_path, config, saved)
    assert vectors[names[0]].shape == (1024,)
    with pytest.raises(ValueError, match="checksum"):
        siglip_large.load_large_cache(tmp_path, names, tmp_path, config, {**saved, "features_sha256": "wrong"})
    with pytest.raises(ValueError, match="pinned encoder"):
        siglip_large.load_large_cache(tmp_path, names, tmp_path, config, {**saved, "identity": {**identity, "width": 768}})


def test_four_fixed_systems_keep_original_reference_and_rounding(tmp_path):
    frame = pd.DataFrame({"painting": ["p"], "language": ["Tamil"], "clip_b": [.1], "clip_l": [.100001], "q0": [.4]})
    large = frame[["painting", "language"]].assign(p_positive=.100001)
    result = compare_systems(frame, large)
    assert tuple(result) == MODELS
    assert [pred["p_positive"].iloc[0] for pred in result.values()] == [.4, .100001, .1, .100001]
    assert np.round(.25 * .1 + .25 * .100001 + .5 * .100001, 6) == .100001
    csv, archive = tmp_path / submission.FILENAME, tmp_path / "submission.zip"
    result["M3"].to_csv(csv, index=False)
    submission.pack(csv, archive, frame)
    assert submission.validate_archive(archive, frame)["p_positive"].tolist() == [.100001]


def test_candidate_specific_promotion_and_diagnostic_exclusion():
    table = pd.DataFrame({"pooled_contrast": [.44, .47, .444, .443], "log_loss_mean": [.72, .72, .72, .725],
                          "fold_wins": 3, "folds": 5, "scored": 5, "loss_scored": 5, "converged": 5}, index=MODELS)
    assert select_large(table, CONFIG, 5)["model"] == "M3"
    table.loc["M2", "pooled_contrast"] = .445
    assert select_large(table, CONFIG, 5)["model"] == "M2"
    table.loc["M2", "converged"] = 4
    table.loc["M3", "log_loss_mean"] = .7251
    assert select_large(table, CONFIG, 5)["model"] == "M0"


def test_no_go_does_not_prepare_development(tmp_path):
    cv = tmp_path / "cv"
    cv.mkdir()
    table = pd.DataFrame({"pooled_contrast": .44, "log_loss_mean": .72, "fold_wins": 3, "folds": 5, "scored": 5,
                          "loss_scored": 5, "converged": 5}, index=pd.Index(MODELS, name="model"))
    decision = select_large(table, CONFIG, 5)
    table.to_csv(cv / "summary.csv")
    (cv / "run.json").write_text(json.dumps({"config": CONFIG, "exp002_config": {"cv_folds": 5}}))
    (cv / "fits.json").write_text("[]")
    decision["sha256"] = {name: file_sha256(cv / name) for name in ("run.json", "summary.csv", "fits.json")}
    (cv / "selection.json").write_text(json.dumps(decision))
    run_large_dev(SimpleNamespace(run_dir=tmp_path), None)
    assert not (tmp_path / "dev").exists()
