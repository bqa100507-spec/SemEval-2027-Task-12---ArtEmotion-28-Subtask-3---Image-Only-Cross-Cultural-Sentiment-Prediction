import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from st3 import clip_large
from st3.baseline import file_sha256
from st3.clip_large_study import E50, E50_LARGE, LARGE, LARGE_SIGLIP, MODELS, compose, run_clip_large_dev, select_large
from st3.frozen_features import verify_asset_hashes


CONFIG = {"log_loss_guardrail_bits": .01, "e50_tie_band": .001}


def summary_fixture():
    return pd.DataFrame({"pooled_contrast": [.44, .441, .442, .443], "log_loss_mean": [.72, .721, .72, .73],
                         "folds": 5, "scored": 5, "loss_scored": 5, "converged": 5},
                        index=pd.Index(MODELS, name="model"))


@pytest.mark.parametrize(("column", "value"), [
    ("log_loss_mean", .7301), ("converged", 4), ("folds", 4), ("scored", 4), ("loss_scored", 4),
    ("pooled_contrast", np.nan), ("pooled_contrast", np.inf),
])
def test_selection_keeps_only_valid_folds_within_the_loss_guardrail(column, value):
    summary = summary_fixture()
    assert select_large(summary, CONFIG, 5)["model"] == E50_LARGE
    summary.loc[E50_LARGE, column] = value
    decision = select_large(summary, CONFIG, 5)
    assert decision["model"] == LARGE_SIGLIP and E50_LARGE not in decision["eligible"]


def test_e50_wins_within_tie_band_including_boundary():
    summary = summary_fixture()
    summary.loc[[LARGE_SIGLIP, E50_LARGE], "pooled_contrast"] = .4405
    decision = select_large(summary, CONFIG, 5)
    assert decision["model"] == E50 and decision["pooled_leader"] == LARGE
    summary.loc[LARGE, "pooled_contrast"] = .4410001
    assert select_large(summary, CONFIG, 5)["model"] == LARGE


def test_candidate_blends_use_fixed_weights_and_persisted_e50():
    keys = pd.DataFrame({"painting": ["a", "b"], "language": ["Tamil", "Yoruba"]})
    large = keys.assign(p_positive=[.765433, .444441])
    siglip = keys.assign(p_positive=[.111113, .555559]).iloc[::-1]
    e50 = keys.assign(p_positive=[.333333, .888889])
    assert compose(LARGE, large) is large
    assert compose(LARGE_SIGLIP, large, siglip=siglip)["p_positive"].tolist() == [.438273, .5]
    assert compose(E50_LARGE, large, e50=e50)["p_positive"].tolist() == [.549383, .666665]
    with pytest.raises(ValueError, match="Pair mismatch"):
        compose(LARGE_SIGLIP, large, siglip=siglip.iloc[:1])


def test_pinned_config_processor_and_weights_are_all_checked():
    assets = {"config.json": "config", "preprocessor_config.json": "processor", "model.safetensors": "weights"}
    verify_asset_hashes(assets, {"assets_sha256": assets})
    for name in assets:
        with pytest.raises(ValueError, match=name):
            verify_asset_hashes({**assets, name: "corrupt"}, {"assets_sha256": assets})
    verify_asset_hashes(assets, {"weights_sha256": "weights"})
    with pytest.raises(ValueError, match="model.safetensors"):
        verify_asset_hashes({}, {"weights_sha256": "weights"})


def test_clip_large_reuses_frozen_image_extraction(monkeypatch, tmp_path):
    def extract(root, names, cache_dir, config, device, *, prefix, load_model, encode):
        assert (root, names, cache_dir, config, device, prefix) == (
            tmp_path, ["a"], tmp_path, {"width": 768}, "cuda", "clip-large")
        assert load_model is clip_large.load_vision
        return encode(lambda **inputs: SimpleNamespace(image_embeds=inputs["pixel_values"]),
                      {"pixel_values": [1., 2.]}, 768)

    monkeypatch.setattr(clip_large, "extract_frozen", extract)
    assert clip_large.extract_clip_large(tmp_path, ["a"], tmp_path, {"width": 768}) == [1., 2.]


def test_dev_no_go_does_not_load_annotations_or_publish_files(tmp_path):
    cv = tmp_path / "cv"
    cv.mkdir()
    summary = summary_fixture()
    summary["pooled_contrast"] = .44
    decision = select_large(summary, CONFIG, 5)
    summary.to_csv(cv / "summary.csv")
    (cv / "run.json").write_text(json.dumps({"config": CONFIG, "exp002_config": {"cv_folds": 5}}))
    decision["sha256"] = {name: file_sha256(cv / name) for name in ("summary.csv", "run.json")}
    (cv / "selection.json").write_text(json.dumps(decision))
    args = SimpleNamespace(run_dir=tmp_path)
    run_clip_large_dev(args, None)
    assert not (tmp_path / "dev").exists()
    with (cv / "summary.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="artifact changed"):
        run_clip_large_dev(args, None)
