import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from st3.baseline import fit_lr
from st3.exp028_cache import audit_cache, expected_identity, partition, state_key
from st3.exp028_stage1 import (
    ARMS, features_for_fit, fit_model, fit_pca, matched_reference, matrix,
    paired, run_screen, select_c, shuffle_indices, weighted_bce,
)
from st3.metrics import score
from st3.vlm_semantics import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def make_data():
    rows = pd.DataFrame([
        dict(painting=f"p{i}", image=f"p{i}.jpg", image_sha256=f"{i:064x}",
             fold=3 if i >= 50 else 1, language=language, k=k, n=8)
        for i in range(75) for language, k in zip("abc", [2 + i % 2, 4, 6])
    ])
    rng = np.random.default_rng(7)
    visual = {p: rng.standard_normal(6) for p in rows.painting.unique()}
    qwen = {(p, l): rng.standard_normal(8) for p, l in rows[["painting", "language"]].itertuples(index=False, name=None)}
    blocks = {m: {f"{p}.jpg": v for p, v in visual.items()} for m in ("CLIP-B", "CLIP-L", "SigLIP2-Base")}
    spec = json.loads((ROOT / "docs/exp028_design_spec.json").read_text())
    spec["features"].update(per_block_PCA=2, capacity_control_PCA=4)
    return rows, visual, qwen, blocks, spec


def seal(folder, digest, kind, identity, **arrays):
    path = folder / f"{digest}_{kind}.npz"
    receipts = {k: {"sha256": hashlib.sha256(v.tobytes()).hexdigest(), "shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in arrays.items()}
    np.savez(path, __metadata=json.dumps({"identity": fingerprint(identity), "filename": path.name, "arrays": receipts}), **arrays)
    return path


def test_cache_audits_prompt_key_pixel_hash_seal_and_weights_identity(tmp_path):
    rows, *_ = make_data()
    rows = rows.iloc[:3]
    identity = expected_identity(ROOT)
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    sha = rows.iloc[0].image_sha256
    vectors = {state_key(k): np.ones(2048, dtype=np.float32) for k in "ab"}
    seal(tmp_path, sha, "vision", identity, grid=np.array([1, 32, 16]), tokens=np.ones((128, 2048), dtype=np.float16))
    payload = seal(tmp_path, sha, "states", identity, **vectors)
    receipt, states, missing = audit_cache(rows, [tmp_path], identity, keep_vectors=True)
    assert receipt["verified_vision_images"] == 1 and receipt["verified_language_states"] == 2
    assert missing == [("p0", "c")] and not receipt["complete"]
    assert receipt["remaining_forwards_relative_to_supplied_cache"] == {
        "vision": 0, "decoder_sequences": 1, "decoder_batches_max8_per_image": 1,
    }
    assert set(states) == {("p0", "a"), ("p0", "b")}
    second = tmp_path / "second"
    second.mkdir()
    (second / "identity.json").write_text(json.dumps(identity))
    seal(second, sha, "states", identity, **vectors)
    assert audit_cache(rows, [tmp_path, second], identity)[0]["invalid_entries"] == 1
    bad = copy.deepcopy(identity)
    bad["vlm"]["revision"] = "wrong"
    (second / "identity.json").write_text(json.dumps(bad))
    assert audit_cache(rows, [second], identity)[0]["invalid_entries"] == 1
    payload.write_bytes(b"broken")
    assert audit_cache(rows, [tmp_path], identity)[0]["invalid_entries"] == 1
    seal(tmp_path, sha, "states", identity, **{state_key("a"): np.ones(2048, dtype=np.float16)})
    assert audit_cache(rows, [tmp_path], identity)[0]["invalid_entries"] == 1


def test_weighted_pca_signs_and_partition_isolation():
    rows, v, q, _, spec = make_data()
    fit, stop, outer_train, held = partition(rows, spec)
    output = features_for_fit(fit, (stop,), v, q, spec)
    second = features_for_fit(fit, (stop,), v, q, spec)
    for arm in ARMS:
        np.testing.assert_array_equal(output[arm][0], second[arm][0])
        if not arm.endswith("_raw"):
            np.testing.assert_allclose(np.linalg.norm(output[arm][0], axis=1), 1)
    assert output["C512"][0].shape == (len(fit), 4)
    with pytest.raises(ValueError, match="overlap"):
        features_for_fit(fit, (fit,), v, q, spec)
    changed = dict(q)
    for p, l in stop[["painting", "language"]].itertuples(index=False, name=None):
        changed[p, l] = np.ones(8)
    result = features_for_fit(fit, (stop,), v, changed, spec)
    for arm in ARMS:
        np.testing.assert_array_equal(output[arm][0], result[arm][0])
    x = np.stack(list(q.values()))
    m, basis = fit_pca(x, np.ones(len(x)), 4)
    np.testing.assert_allclose(m, x.mean(axis=0))
    assert (basis[np.argmax(abs(basis), axis=0), np.arange(4)] > 0).all()
    with pytest.raises(ValueError, match="rank"):
        fit_pca(np.ones((10, 8)), np.ones(10), 4)


def test_shuffle_keeps_language_and_partition_marginals():
    rows, _, _, _, spec = make_data()
    for frame in partition(rows, spec):
        i = shuffle_indices(frame)
        assert set(i) == set(range(len(frame))) and np.all(i != np.arange(len(frame)))
        np.testing.assert_array_equal(frame.language.to_numpy(), frame.language.to_numpy()[i])
    with pytest.raises(ValueError, match="Duplicate"):
        rows.loc[len(rows)] = rows.iloc[0]
        v, q = make_data()[1:3]
        features_for_fit(rows, (), v, q, spec)


def test_objective_matches_official_lr_and_unseen_fallback():
    rows, v, _, _, _ = make_data()
    x = matrix(np.stack([v[p] for p in rows.painting]), rows.language, list("abc"))
    model, meta = fit_model(x, rows, .3)
    other = fit_lr(x, rows, 2027)
    np.testing.assert_array_equal(model.coef_, other.coef_)
    assert meta["parameters"] == 28 and meta["objective_parity_abs"] <= 1e-12
    with pytest.raises(MemoryError):
        fit_model(x, rows, .3, memory_limit=1)
    test = matrix(np.array([v["p0"]]), ["z"], list("abc"))
    np.testing.assert_array_equal(test.toarray()[0, 6:], 0)
    assert select_c([{"C": 1., "inner_BCE_nats": .5}, {"C": .1, "inner_BCE_nats": .5}]) == .1
    assert abs(weighted_bce(np.zeros(len(rows)), rows) - np.log(2)) < 1e-12
    assert paired([1., 1.], [0., 0.], draws=10)["ci95"] == [1., 1.]


def test_complete_synthetic_fold_all_arms_reference_refit_and_metrics(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("st3.exp028_stage1.psutil.virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3))
    rows, _, qwen, blocks, spec = make_data()
    _, _, fit, held = partition(rows, spec)
    with threadpool_limits(limits=4):
        ref, _ = matched_reference(fit, held, blocks)
    ref["fold"] = 3
    predictions, evaluation, paintings, fits = run_screen(rows, blocks, qwen, ref, spec)
    assert set(predictions) == set(ARMS) | {"E50+L"}
    for arm in ARMS:
        assert len(fits[arm]["selection"]) == 3 and fits[arm]["refit"]["converged"]
        assert evaluation["metrics"][arm] == score(held, predictions[arm])
        assert len(paintings[arm]) == 25
        p = predictions[arm].p_positive.to_numpy()
        np.testing.assert_array_equal(p, p.round(6))
    assert "C512 vs C512_shuffled_Q" in evaluation["paired"]
    with pytest.raises(ValueError, match="before any real fit"):
        run_screen(rows, blocks, {}, ref, spec)
    ref.loc[ref.index[0], "p_positive"] = 0.123456
    with pytest.raises(ValueError, match="does not reproduce"):
        run_screen(rows, blocks, qwen, ref, spec)


def test_frozen_fold_manifest_and_source_pins():
    location = ROOT / "scripts/run_exp028_stage1.py"
    spec = importlib.util.spec_from_file_location("exp028_cli", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_pins()
    manifest = json.loads((ROOT / "configs/exp028_fold3_manifest.json").read_text())
    spec = json.loads((ROOT / "docs/exp028_design_spec.json").read_text())
    assert manifest["fold"] == 3 and manifest["spec_logical_sha256"] == fingerprint(spec)
    sets = {name: set(v["paintings"]) for name, v in manifest["populations"].items()}
    assert len(sets["refit"]) == 1068 and len(sets["held"]) == 267
    assert not (sets["refit"] & sets["held"] or sets["inner"] & sets["stop"])
    assert sets["inner"] | sets["stop"] == sets["refit"]
