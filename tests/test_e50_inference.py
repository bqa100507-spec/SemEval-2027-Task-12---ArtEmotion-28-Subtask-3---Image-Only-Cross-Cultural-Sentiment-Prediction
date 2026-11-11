import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from st3 import data, e50_inference, submission
from st3.ablation import predict_lr
from st3.baseline import file_sha256
from st3.e50_heads import HEADS, INDEX, MANIFEST, fit_frozen_head, head_from_arrays, inference_config, load_bundle
from st3.e50_inference import read_input, run_prediction, safe_image_path
from st3.e50l_heads import clip_large_config, compose_e50l, load_clip_large_bundle
from st3.fusion import blend
from st3.incremental import collect_features


def test_training_view_checks_split_without_using_validation_emotions(tmp_path):
    annotations = tmp_path / data.ANNOTATIONS
    annotations.parent.mkdir()
    pd.DataFrame([("p", "Tamil", "a.png", "train", "amusement")] * 5
                 + [("v", "Tamil", "b.png", "val", "ignored")], columns=data.ANNOTATION_COLUMNS).to_csv(
        annotations, index=False)
    train = data.training_pairs(data.read_annotations(tmp_path, training_only=True))
    assert train["painting"].tolist() == ["p"]
    text = annotations.read_text().replace("v,Tamil,b.png", "v,Tamil,a.png")
    annotations.write_text(text)
    with pytest.raises(ValueError, match="one-to-one"):
        data.read_annotations(tmp_path, training_only=True)


def test_head_round_trip_and_frozen_untrained_language_behavior():
    train = pd.DataFrame({"painting": ["a", "a", "b", "b"], "image": ["a", "a", "b", "b"],
                          "language": ["Tamil", "Yoruba"] * 2, "k": [0, 4, 2, 5], "n": 5, "split": "train"})
    languages = ["Tamil", "Yoruba"]
    embeddings = {"a": np.array([.2, .4]), "b": np.array([-.1, .7])}
    original = fit_frozen_head(train, embeddings, languages, 2027)
    reloaded = head_from_arrays(original.coef_, original.intercept_, 2, languages)
    query = train.iloc[:2].assign(language=["IsiNdebele", "Kazakh"])
    expected = predict_lr(original, query, embeddings, languages)
    pd.testing.assert_frame_equal(expected, predict_lr(reloaded, query, embeddings, languages))
    assert expected["p_positive"].nunique() == 1
    with pytest.raises(ValueError, match="only original training"):
        fit_frozen_head(train.assign(split="val"), embeddings, languages, 2027)
    with pytest.raises(ValueError, match="coefficients"):
        head_from_arrays(original.coef_, np.array([np.nan]), 2, languages)


def test_incremental_commit_restart_reuse_content_change_and_integrity(tmp_path):
    identity = {"encoder": "fixture", "width": 2, "revision": "one"}
    paths = {str(i): tmp_path / f"{i}.png" for i in range(3)}
    observed = []
    completed = []

    def encoder(items):
        observed.extend(digest for digest, _ in items)
        if items[0][0] == "2":
            raise InterruptedError("interrupted before commit")
        return np.tile(np.array([1., 0.], dtype=np.float32), (len(items), 1))

    with pytest.raises(InterruptedError):
        collect_features(paths, tmp_path, identity, 2, lambda: encoder, progress=completed.append)
    assert completed == [2]
    assert observed == ["0", "1", "2"]

    def resumed(items):
        observed.extend(digest for digest, _ in items)
        return np.tile(np.array([0., 1.], dtype=np.float32), (len(items), 1))

    result, meta = collect_features(paths, tmp_path, identity, 1, lambda: resumed)
    assert observed == ["0", "1", "2", "2"]
    assert meta["encoded"] == 1 and meta["incremental_hits"] == 2
    np.testing.assert_array_equal(result["1"], [1., 0.])

    def unavailable():
        pytest.fail("Cached subset should not load the encoder")

    collect_features({"1": paths["1"]}, tmp_path, identity, 3, unavailable)
    _, changed = collect_features({"new-digest": paths["1"]}, tmp_path, identity, 3, lambda: resumed)
    assert changed["encoded"] == 1 and observed[-1] == "new-digest"
    with sqlite3.connect(meta["database"]) as connection:
        connection.execute("UPDATE features SET checksum = 'tampered' WHERE image = '1'")
    with pytest.raises(ValueError, match="checksum mismatch"):
        collect_features(paths, tmp_path, identity, 3, unavailable)


@pytest.mark.parametrize(("rows", "error"), [
    ("p,Tamil,a.png\np,Tamil,a.png\n", "Duplicate"),
    ("p,Tamil,a.png\np,Yoruba,b.png\n", "one image path"),
    ("p,Unrecognized,a.png\n", "Unsupported target language"),
    ("p,Tamil,\n", "Invalid CSV"),
])
def test_input_rejects_invalid_rows(tmp_path, rows, error):
    path = tmp_path / "input.csv"
    path.write_text("painting,language,image\n" + rows)
    with pytest.raises(ValueError, match=error):
        read_input(path, ["Tamil", "Yoruba"])


@pytest.mark.parametrize("name", ["../image.png", "style/../../a.png", "C:/a.png", "a\\b.png", "/a.png", "a//b.png",
                                  "style/NUL.png", "style/a.png:stream", "style /a.png"])
def test_unsafe_image_paths_are_rejected(tmp_path, name):
    (tmp_path / "images").mkdir()
    with pytest.raises(ValueError, match="Unsafe image path"):
        safe_image_path(tmp_path, name)


def test_path_cannot_resolve_outside_images(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"fixture")
    try:
        (images / "link.png").symlink_to(outside)
    except OSError:
        pytest.skip("This machine does not permit test symlinks")
    with pytest.raises(ValueError, match="inside the images directory"):
        safe_image_path(tmp_path, "link.png")


def test_e50l_keeps_the_intermediate_rounding():
    keys = pd.DataFrame({"painting": ["p"], "language": ["Tamil"]})
    clip = keys.assign(p_positive=.1)
    siglip = keys.assign(p_positive=.100001)
    large = keys.assign(p_positive=.100001)
    result = compose_e50l(blend(clip, siglip, .5), large)
    assert result["p_positive"].tolist() == [.1]
    assert np.round(.25 * .1 + .25 * .100001 + .5 * .100001, 6) == .100001


@pytest.mark.parametrize(("mode", "probability", "active"), [
    ("clip", .562177, ("clip",)),
    ("e50", .562177, ("clip", "siglip2")),
    ("e50l", .592318, ("clip", "siglip2", "clip_large")),
])
def test_label_free_prediction_uses_legacy_by_content_and_preserves_order(tmp_path, monkeypatch, mode, probability, active):
    config = inference_config()
    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dataset = tmp_path / "input"
    images = dataset / "images"
    images.mkdir(parents=True)
    payload = b"fixture-content-injected-as-a-legacy-match"
    (images / "new-name.png").write_bytes(payload)
    (images / "same-content.png").write_bytes(payload)
    input_path = dataset / "st3.csv"
    input_path.write_text("painting,language,image\nz,Yoruba,new-name.png\na,Kazakh,same-content.png\n")

    languages = sorted(set(config["official_languages"]) - set(config["untrained_official_languages"]))
    archive_values, legacy = {}, {}
    for name, encoder in config["encoders"].items():
        width = encoder["width"]
        archive_values[f"{name}_coef"] = np.zeros((1, width + len(languages) * (width + 1)))
        archive_values[f"{name}_intercept"] = np.array([.25])
        path = cache_dir / f"{name}.npz"
        np.savez(path, names=["historical.png"], features=np.eye(1, width, dtype=np.float32))
        legacy[name] = {"file": path.name, "sha256": file_sha256(path), "source": {}}
    np.savez(model_dir / HEADS, **archive_values)
    (model_dir / INDEX).write_text(json.dumps({hashlib.sha256(payload).hexdigest(): "historical.png"}))
    manifest = {"version": config["bundle_version"], "system": "fixture", "encoders": config["encoders"],
                "blend_clip_weight": config["blend_clip_weight"], "reference_csv_sha256": config["reference_csv_sha256"],
                "languages": languages, "legacy": legacy, "upstream_revision": "fixture", "annotations_sha256": "fixture",
                "training_pairs": 4, "training_paintings": 2, "lr": {},
                "files_sha256": {name: file_sha256(model_dir / name) for name in (HEADS, INDEX)}}
    (model_dir / MANIFEST).write_text(json.dumps(manifest))

    large_dir = tmp_path / "clip-large-bundle"
    if mode == "e50l":
        large_dir.mkdir()
        encoder = clip_large_config()
        width = encoder["width"]
        path = cache_dir / "large.npz"
        np.savez(path, names=["historical.png"], features=np.eye(1, width, dtype=np.float32))
        split = {"file": path.name, "sha256": file_sha256(path), "source": {}}
        np.savez(large_dir / HEADS, coef=np.zeros((1, width + len(languages) * (width + 1))), intercept=np.array([.5]))
        (large_dir / INDEX).write_text(json.dumps({"train": {}, "dev": {hashlib.sha256(payload).hexdigest():
                                                                         "historical.png"}}))
        large_pins = json.loads((Path(__file__).resolve().parents[1] / "configs" / "e50l_inference.json").read_text())
        large_manifest = {**large_pins, "version": large_pins["bundle_version"], "system": "fixture with CLIP-L",
                          "encoder": encoder, "legacy": {"train": split, "dev": split}, "languages": languages,
                          "annotations_sha256": "fixture", "e50_heads_sha256": manifest["files_sha256"][HEADS],
                          "lr": {}, "files_sha256": {name: file_sha256(large_dir / name) for name in (HEADS, INDEX)}}
        (large_dir / MANIFEST).write_text(json.dumps(large_manifest))

    output = tmp_path / "result"
    args = SimpleNamespace(mode=mode, input=input_path, output_dir=output, data_dir=dataset, cache_dir=cache_dir,
                           model_dir=model_dir, clip_large_model_dir=large_dir, clip_batch_size=None,
                           siglip2_batch_size=None, clip_large_batch_size=None, device="auto")
    run_prediction(args)
    inputs = read_input(input_path, config["official_languages"])
    result = submission.validate_archive(output / "submission.zip", inputs)
    assert result["painting"].tolist() == ["z", "a"]
    assert result["p_positive"].tolist() == [probability, probability]
    report = json.loads((output / "run.json").read_text())
    assert report["unique_image_contents"] == 1
    assert report["untrained_language_rows"] == {"Kazakh": 1}
    assert tuple(report["features"]) == active
    assert all(info["historical_hits"] == 1 and info["encoded"] == 0 for info in report["features"].values())
    with pytest.raises(ValueError, match="Output already exists"):
        run_prediction(args)
    if mode == "e50l":
        def fail_worker(name, *args):
            assert name == "clip_large"
            raise RuntimeError("CLIP-L worker failed")

        monkeypatch.setattr(e50_inference, "load_clip_large_legacy",
                            lambda *args: ({}, {"status": "no_matching_content"}))
        monkeypatch.setattr(e50_inference, "run_feature_worker", fail_worker)
        args.output_dir = tmp_path / "failed"
        with pytest.raises(RuntimeError, match="CLIP-L worker failed"):
            run_prediction(args)
        assert not args.output_dir.exists()
        with pytest.raises(ValueError, match="different E50 heads"):
            load_clip_large_bundle(large_dir, {**manifest, "files_sha256": {HEADS: "changed"}})
        (large_dir / INDEX).write_text("{}")
        with pytest.raises(ValueError, match="CLIP-L bundle failed its checksum"):
            load_clip_large_bundle(large_dir, manifest)
    (model_dir / INDEX).write_text("{}")
    with pytest.raises(ValueError, match="failed its checksum"):
        load_bundle(model_dir, config, ("clip",))
