import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from st3.vlm_semantics import SemanticCache, token_chunks, encode_description, smoke_sample, training_images
from st3.vlm_probe import derangement, blend, choose_weight, head_predictions, screening_signal


def configuration():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/exp025.json").read_text(encoding="utf-8"))


def test_cache_keys_integrity_and_budget_free_reuse(tmp_path):
    config = configuration()
    cache = SemanticCache(tmp_path, config)
    vector = np.ones(768, dtype=np.float32) / np.sqrt(np.float32(768))
    cache.save("a" * 64, {"description": "A quiet landscape.", "embedding": vector.tolist(), "clip_tokens": 4, "tokens_encoded": 4})
    assert cache.read("b" * 64) is None
    np.testing.assert_array_equal(cache.read("a" * 64)["embedding"], vector.tolist())
    assert SemanticCache(tmp_path, config).directory == cache.directory
    head_change = copy.deepcopy(config);head_change["pca_dimensions"] = 32
    assert SemanticCache(tmp_path, head_change).directory == cache.directory
    for field in ("prompt", "vlm", "decoding", "text_encoder"):
        changed = copy.deepcopy(config)
        if field == "prompt":changed[field] += " Extra."
        else:changed[field]["test_change"] = True
        assert SemanticCache(tmp_path, changed).directory != cache.directory
    path = cache.directory / ("a" * 64 + ".json")
    payload = json.loads(path.read_text());payload["embedding"][0] *= -1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="embedding"):cache.read("a" * 64)


def test_all_clip_tokens_encoded_without_truncation():
    import torch
    ids = list(range(160))
    assert sum(token_chunks(ids), []) == ids
    assert list(map(len, token_chunks(ids))) == [75, 75, 10]
    class Tokenizer:
        bos_token_id, eos_token_id, pad_token_id = 200, 201, 0
        def __call__(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return {"input_ids": ids}
    class Model:
        def __call__(self, input_ids, attention_mask):
            assert input_ids.shape == (3, 77)
            assert attention_mask.sum().item() == 166
            return type("Output", (), {"text_embeds": torch.ones(3, 768)})()
    vector, info = encode_description("fixture", Tokenizer(), Model())
    assert vector.shape == (768,) and info == {"clip_tokens":160,"tokens_encoded":160,"chunks":3}
    assert np.isclose(np.linalg.norm(vector), 1.)


def test_label_free_representative_sample_and_protocol_guard(tmp_path):
    images = pd.DataFrame({"painting":[f"p{i}" for i in range(60)], "aspect_ratio":np.linspace(.3, 3., 60),
                           "image_sha256":[hashlib.sha256(str(i).encode()).hexdigest() for i in range(60)]})
    a = smoke_sample(images);b = smoke_sample(images.sample(frac=1., random_state=2))
    assert len(a) == a.painting.nunique() == 30
    assert a.painting.tolist() == b.painting.tolist()
    (tmp_path / "training").mkdir();(tmp_path / "training/protocol.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):training_images(tmp_path, "not_the_digest")


def test_partition_local_shuffle_preserves_distribution():
    first, second = ["a", "b", "c", "d"], ["x", "y", "z"]
    for names in (first, second):
        mapping = derangement(names, 9)
        assert set(mapping) == set(mapping.values()) == set(names)
        assert all(k != v for k,v in mapping.items())
        assert mapping == derangement(list(reversed(names)), 9)
    assert not set(derangement(first, 9).values()) & set(second)


def fixture():
    rng = np.random.default_rng(2)
    rows, visual, semantics = [], {}, {}
    for i in range(60):
        image = f"p{i}.jpg"
        v, s = rng.normal(size=8), rng.normal(size=6)
        visual[image], semantics[image] = v / np.linalg.norm(v), s / np.linalg.norm(s)
        for language, k in zip(("A", "B", "C"), (1 + i % 2, 3, 4)):
            rows.append(dict(painting=f"p{i}", image=image, image_sha256=f"hash{i}", language=language,
                             n=5,k=k,fold=1 if i < 30 else 3,split="train"))
    return pd.DataFrame(rows), visual, semantics


def test_matched_capacity_and_query_labels_cannot_change_heads():
    frame, visual, semantics = fixture()
    config = configuration();config["pca_dimensions"] = 4
    fitting, held = frame[frame.fold.eq(1)], frame[frame.fold.eq(3)]
    a, metadata = head_predictions(fitting, held, visual, semantics, config, 1, 0)
    b, _ = head_predictions(fitting, held.assign(n=999,k=0,caption="DO NOT READ"), visual, semantics, config, 1, 0)
    assert len({r["parameters"] for r in metadata["heads"].values()}) == 1
    assert set(metadata["fitting_shuffle"].values()) <= set(fitting.painting)
    assert set(metadata["query_shuffle"].values()) <= set(held.painting)
    for name in a:pd.testing.assert_frame_equal(a[name], b[name])
    with pytest.raises(ValueError, match="overlap"):
        head_predictions(fitting, fitting, visual, semantics, config, 1, 0)


def test_late_weight_selection_uses_inner_metrics():
    frame, _, _ = fixture()
    positive = frame[["painting","language"]].assign(p_positive=frame.k/frame.n)
    negative = positive.assign(p_positive=1-positive.p_positive)
    weight, _ = choose_weight(frame, negative, positive, configuration())
    assert weight == 1.
    pd.testing.assert_frame_equal(blend(negative, positive, 0.), negative.assign(p_positive=negative.p_positive.round(6)))
    with pytest.raises(ValueError):blend(negative, positive.iloc[::-1], .5)


def test_fusion_and_shuffle_both_required_for_continuation():
    frame, _, _ = fixture()
    positive = frame[["painting","language","fold"]].assign(p_positive=frame.k/frame.n)
    negative = positive.assign(p_positive=1-positive.p_positive)
    config = configuration();config["bootstrap_replicates"] = 100
    preds = pd.concat([negative.assign(model="A"), negative.assign(model="C_shuffle"), negative.assign(model="D_shuffle"),
                       positive.assign(model="C"), positive.assign(model="D")],ignore_index=True)
    assert screening_signal(frame,preds,config)["promising_fusions"] == ["C","D"]
    preds.loc[preds.model.eq("C_shuffle"),"p_positive"] = preds.loc[preds.model.eq("C"),"p_positive"].to_numpy()
    preds.loc[preds.model.eq("D_shuffle"),"p_positive"] = preds.loc[preds.model.eq("D"),"p_positive"].to_numpy()
    assert not screening_signal(frame,preds,config)["promising_fusions"]


def test_fp16_and_nf4_never_share_semantic_records(tmp_path):
    config = configuration();quantized = copy.deepcopy(config);quantized["vlm"]["quantization"]="nf4"
    assert SemanticCache(tmp_path,config).directory != SemanticCache(tmp_path,quantized).directory


def test_full_extraction_cannot_bypass_incomplete_smoke(tmp_path, monkeypatch):
    import sys
    import torch
    from st3 import vlm_semantics
    config = configuration()
    config["vlm"]["assets_sha256"] = {};config["vlm"]["assets_git_sha1"] = {}
    config["text_encoder"]["assets_sha256"] = {}
    images = pd.DataFrame({"painting":[f"p{i}" for i in range(60)], "aspect_ratio":np.linspace(.3, 3., 60),
                           "image_sha256":[hashlib.sha256(str(i).encode()).hexdigest() for i in range(60)]})
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setitem(sys.modules,"transformers",None)
    monkeypatch.setattr(vlm_semantics,"training_images",lambda *args:images)
    monkeypatch.setattr(vlm_semantics,"load_vlm",lambda *args:pytest.fail("Incomplete smoke must block VLM loading"))
    with pytest.raises(ValueError,match="Complete the 30-image smoke"):
        vlm_semantics.extract(tmp_path,tmp_path/"cache",config,tmp_path,tmp_path,full=True)
