"""CPU/synthetic tests for EXP-038: heads, mixes, banks, gate, caches, budget, packaging and the full worker."""

import ast
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp038_fixture as fx  # noqa: E402
from st3 import data, submission  # noqa: E402
from st3 import exp038 as science  # noqa: E402
from st3 import exp038_gpu as gpu  # noqa: E402
from st3 import exp038_io as io38  # noqa: E402
from st3.competition_neighborhood import fitting_predictions, make_bank, predict_correction  # noqa: E402
from st3.competition_ridge import fit_head, make_panel  # noqa: E402
from st3.competition_stack import ENCODERS, blend, compare  # noqa: E402
from st3.exp036_io import read, sha  # noqa: E402

RULE = json.loads((ROOT / "configs/competition_track1.json").read_text())["ridge"]


def toy(seed=0, paintings=30, languages=("A", "B", "C", "D"), width=5):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(paintings):
        for language in rng.choice(languages, size=rng.integers(2, len(languages) + 1), replace=False):
            n = int(rng.integers(5, 9))
            rows.append((f"q{i:02d}", language, int(rng.integers(0, n + 1)), n))
    gold = pd.DataFrame(rows, columns=["painting", "language", "k", "n"]).sort_values(data.KEYS, ignore_index=True)
    x = rng.standard_normal((paintings, width))
    features = pd.DataFrame(x / np.linalg.norm(x, axis=1, keepdims=True), index=[f"q{i:02d}" for i in range(paintings)])
    return gold, features, rng


# Heads.

def test_e4_head_with_zero_delta_equals_track1_head_and_gradient_is_exact():
    gold, x, rng = toy()
    zero = science.PairDelta.build(gold[data.KEYS], np.zeros((len(gold), 3), np.float32),
                                   pd.DataFrame(np.zeros((len(x), 3)), index=x.index))
    pair, receipt = science.fit_pair_head(gold, x, zero, RULE)
    panel = make_panel(gold)
    track, _ = fit_head(panel, x.loc[list(panel.paintings)].to_numpy(), RULE)
    np.testing.assert_allclose(pair.head.shared, track.shared, atol=1e-7)
    np.testing.assert_allclose(pair.head.deviations, track.deviations, atol=1e-7)
    assert np.all(pair.gamma == 0) and receipt["converged"]
    delta = science.PairDelta.build(gold[data.KEYS], rng.standard_normal((len(gold), 3)).astype(np.float32) * .1,
                                    pd.DataFrame(np.zeros((len(x), 3)), index=x.index))
    rows = gold.painting.map({p: i for i, p in enumerate(panel.paintings)}).to_numpy()
    cols = gold.language.map({p: i for i, p in enumerate(panel.languages)}).to_numpy()
    ridge = 1 / (RULE["reference_c"] * panel.n.sum())
    loss_grad, _, size = science.pair_objective(panel, x.loc[list(panel.paintings)].to_numpy(), rows, cols,
                                                delta.rows(gold), ridge, ridge * .5 * np.eye(len(panel.languages)))
    theta = rng.standard_normal(size) * .1
    numeric = approx_fprime(theta, lambda t: loss_grad(t)[0], 1e-7)
    np.testing.assert_allclose(loss_grad(theta)[1], numeric, atol=2e-5)


def test_e4_delta_is_pair_specific_and_missing_pairs_fail():
    keys = pd.DataFrame({"painting": ["a", "a"], "language": ["English", "Hausa"]})
    neutral = pd.DataFrame([[1., 0.]], index=["a"])
    delta = science.PairDelta.build(keys, np.array([[0., 1.], [1., 0.]], np.float32), neutral)
    np.testing.assert_array_equal(delta.rows(keys), [[-1, 1], [0, 0]])
    with pytest.raises(ValueError, match="missing"):
        delta.rows(pd.DataFrame({"painting": ["a"], "language": ["Thai"]}))
    spec = json.loads((ROOT / "configs/exp038.json").read_text())["arms"]["E4"]
    items = [("x|Hausa", "p", "Hausa"), ("x|Thai", "p", "Thai")]
    assert gpu.instructions_for("E4", spec, items) == [
        "Represent the emotion a Hausa-speaking viewer would feel looking at this artwork.",
        "Represent the emotion a Thai-speaking viewer would feel looking at this artwork."]


def test_qwen_formatting_matches_the_official_script():
    spec = json.loads((ROOT / "configs/exp038.json").read_text())["arms"]["E3"]
    conversation = gpu.qwen_conversation("IMG", "No final punctuation", spec)
    assert conversation[0] == {"role": "system", "content": [{"type": "text", "text": "No final punctuation."}]}
    assert conversation[1]["content"][0] == {"type": "image", "image": "IMG", "min_pixels": 4096, "max_pixels": 262144}
    import torch

    hidden = torch.arange(24.).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    assert gpu.last_token(hidden, mask).tolist() == [hidden[0, 2].tolist(), hidden[1, 3].tolist()]


def test_e1_pins_equal_the_exp036_processor_pins():
    e1 = json.loads((ROOT / "configs/exp038.json").read_text())["arms"]["E1"]
    visual = json.loads((ROOT / "configs/exp036.json").read_text())
    assert (e1["model"], e1["revision"]) == (visual["model"], visual["revision"])
    assert e1["weights_sha256"]["model.safetensors"] == visual["weights"]["model.safetensors"]
    assert all(visual["metadata_git_blobs"][k] == v for k, v in e1["metadata_git_blobs"].items())


def test_mix_and_similarity_spaces():
    frame = pd.DataFrame({"painting": ["a"], "language": ["x"], "clip": [.2], "siglip2": [.4], "clip_large": [.9],
                          "E1": [.1], "E2": [.3]})
    out = science.mix_columns(frame, {"C1": ("E1", "E2")})
    fixed = blend(frame[list(ENCODERS)].to_numpy(), None, fixed=True)
    assert out.fixed.tolist() == fixed.tolist() and out.C1.tolist() == [round(.5 * fixed[0] + .5 * .2, 6)]
    rng = np.random.default_rng(1)
    features = {n: pd.DataFrame(rng.standard_normal((4, 3)) * (i + 1), index=list("abcd"))
                for i, n in enumerate([*ENCODERS, "E1", "E3"])}
    space = science.candidate_space(features, ("E1", "E4"))
    assert space.shape == (4, 15)
    np.testing.assert_allclose(np.linalg.norm(space.to_numpy(), axis=1), 1)
    assert science.candidate_encoders(("E1", "E4")) == (*ENCODERS, "E1", "E3")


def test_r0_path_equals_the_original_neighborhood_code():
    gold, x, _ = toy(seed=3, paintings=40)
    gold["fold"] = gold.painting.map({p: 1 + i % 5 for i, p in enumerate(sorted(gold.painting.unique()))})
    features = {name: x.copy() * (i + 1) for i, name in enumerate(ENCODERS)}
    features = {n: f / np.linalg.norm(f.to_numpy(), axis=1, keepdims=True) for n, f in features.items()}
    track1 = json.loads((ROOT / "configs/competition_track1.json").read_text())
    nb = json.loads((ROOT / "configs/competition_neighborhood_residual.json").read_text())
    train, held = gold.loc[gold.fold.ne(2)], gold.loc[gold.fold.eq(2)]
    original, _ = fitting_predictions(train, features, track1, 2)
    mine, details = science.crossfit(train, features, None, (), track1, 2, {})
    np.testing.assert_array_equal(original.p_positive.to_numpy(), mine.fixed.to_numpy())
    visual = science.r0_space(features)
    bank_a = make_bank(train, original, visual, 2)
    bank_b = science.banks(train, mine, {"R0": visual}, {}, 2)["R0"]
    assert bank_a.numerator.equals(bank_b.numerator) and bank_a.votes.equals(bank_b.votes)
    base = held[data.KEYS].assign(p_positive=.5)
    a = predict_correction(bank_a, held[data.KEYS], visual, base, nb)
    b = predict_correction(bank_b, held[data.KEYS], visual, base, nb)
    pd.testing.assert_frame_equal(a, b)


def test_leakage_held_out_paintings_never_reach_outer_inner_fits_or_banks(monkeypatch, tmp_path):
    fixture = fx.build(tmp_path)
    seen = []
    original = science.fit_all

    def spy(gold, *args, **kwargs):
        seen.append(set(gold.painting))
        return original(gold, *args, **kwargs)

    monkeypatch.setattr(science, "fit_all", spy)
    from st3.exp038_io import base_features, painting_images, p2_population, training_population

    train, dev = training_population(fixture["base"], fixture["config"]["references"]["training_pair_sha256"])
    gold, _ = p2_population(fixture["overlay"], train, fixture["config"]["references"])
    paintings = painting_images(fixture["base"], read(fixture["base"] / "exp037_manifest.json"), train, dev)
    features = base_features(fixture["overlay"], paintings, fx.WIDTHS)
    science.evaluate_p2(gold, features, None, (), fixture["config"], fixture["track1"], fixture["neighborhood"],
                        fixture["config"]["references"]["track1_inner_oof_sha256"], log=lambda *_: None)
    assert len(seen) == 20
    for fold in range(1, 6):
        held = set(gold.painting[gold.fold.eq(fold)])
        assert all(not (held & fitted) for fitted in seen[4 * (fold - 1):4 * fold])
    assert not set(dev.painting) & set().union(*seen)


# Gate and selection.

def gate_report(filtered_delta, full_delta, folds, trimmed, lower, ll=0.):
    comparison = {"pooled_contrast_delta": filtered_delta, "bootstrap_95": [lower, .1], "fold_wins": folds,
                  "trimmed_contrast_delta": trimmed, "log_loss_delta": ll}
    report = {"full": {"comparisons": {"C1_minus_R0_corrected": comparison | {"pooled_contrast_delta": full_delta}},
                       "systems": {}},
              "filtered": {"comparisons": {"C1_minus_R0_corrected": comparison},
                           "systems": {"C1": {"pooled": {"cultural_contrast": .47}}}}}
    return science.gate(report, "C1", "R0_corrected", json.loads((ROOT / "configs/exp038.json").read_text())["gate"])


@pytest.mark.parametrize("args,passed", [((.004, 0., 4, .001, .0001), True), ((.0039, 0., 5, .001, .001), False),
                                         ((.01, -1e-9, 5, .001, .001), False), ((.01, 0., 3, .001, .001), False),
                                         ((.01, 0., 4, 0., .001), False), ((.01, 0., 4, .001, 0.), False)])
def test_gate_arithmetic(args, passed):
    assert gate_report(*args)["passed"] is passed
    assert gate_report(.01, .01, 5, .01, .01, ll=.0051)["passed"] is False


def test_bootstrap_is_reproducible_and_seeded():
    rule = {"bootstrap_seed": 203804, "bootstrap_draws": 20000, "trim_fraction": .05}
    delta = np.random.default_rng(0).normal(.003, .02, 400)
    a, b = compare(delta, rule), compare(delta, rule)
    assert a == b and len(set(map(tuple, [a["bootstrap_95"], compare(delta, rule | {"bootstrap_seed": 1})["bootstrap_95"]]))) == 2


def test_selection_prefers_simpler_within_tolerance():
    def g(value, ok=True):
        return {"passed": ok, "numbers": {"filtered_contrast": value}}

    order = ["C1", "C2", "C3"]
    assert science.select({"C1": g(.470), "C2": g(.4709), "C3": g(.4718)}, order, .001) == "C2"
    assert science.select({"C1": g(.470), "C2": g(.472, False), "C3": g(.4705)}, order, .001) == "C1"
    assert science.select({"C1": g(.47, False), "C2": g(.48, False)}, order, .001) is None
    assert science.select({"C2": g(.47), "C3": {"status": "BUDGET_SKIPPED", "passed": False}}, order, .001) == "C2"


# Budget.

def test_budget_skip_order_and_e4_cap():
    budget = json.loads((ROOT / "configs/exp038.json").read_text())["budget"]
    pieces = {"E1": 600, "E2": 600, "E3": 600, "E4": 3600, "cscva_seed_2": 1200, "cscva_seed_3": 1200,
              "cpu_p2_and_development": 1200, "export": 600}
    assert io38.plan_budget(pieces, 1e6, budget)["skipped"] == {}
    total = budget["forecast_margin"] * sum(pieces.values())
    plan = io38.plan_budget(pieces, total - 1, budget)
    assert list(plan["skipped"]) == ["cscva_seed_3"]
    plan = io38.plan_budget(pieces, total - 1300 * budget["forecast_margin"], budget)
    assert list(plan["skipped"]) == ["cscva_seed_3", "cscva_seed_2"]
    plan = io38.plan_budget(pieces, budget["forecast_margin"] * 4900, budget)
    assert list(plan["skipped"]) == ["cscva_seed_3", "cscva_seed_2", "E4"] and not plan["mandatory_exceeds_budget"]
    assert io38.plan_budget(pieces | {"E4": 11000}, 1e6, budget)["skipped"] == {"E4": "E4 forecast 11000s exceeds its 10800s cap"}
    assert io38.plan_budget(pieces, 10, budget)["mandatory_exceeds_budget"]


# Caches.

def test_feature_cache_reuse_and_rejection(tmp_path):
    keys = ["a", "b"]
    io38.save_features(tmp_path / "f", keys, np.eye(2), {"id": 1})
    np.testing.assert_array_equal(io38.load_features(tmp_path / "f", {"id": 1}, keys), np.eye(2))
    with pytest.raises(io38.BlockedCache, match="Stale"):
        io38.load_features(tmp_path / "f", {"id": 2}, keys)
    with pytest.raises(io38.BlockedCache, match="coverage"):
        io38.load_features(tmp_path / "f", {"id": 1}, ["a", "c"])
    (tmp_path / "f/features.npz").write_bytes(b"corrupt")
    with pytest.raises(io38.BlockedCache, match="corrupt"):
        io38.load_features(tmp_path / "f", {"id": 1}, keys)
    (tmp_path / "g").mkdir()
    (tmp_path / "g/features.npz").write_bytes(b"partial")
    with pytest.raises(io38.BlockedCache, match="Incomplete"):
        io38.completed(tmp_path / "g", {"id": 1})


def make_items(tmp_path, count):
    from PIL import Image

    items = []
    for i in range(count):
        path = tmp_path / f"i{i}.png"
        Image.new("RGB", (2, 2), (i, 2 * i, 3)).save(path)
        items.append((f"k{i:02d}", str(path)))
    return items


def test_extract_resumes_sealed_chunks_and_enforces_the_cap(tmp_path):
    spec = {"width": 4, "batch": 2}
    items = make_items(tmp_path, 9)
    calls = []

    def embed(images, instructions):
        calls.append(len(images))
        time.sleep(.05)
        return fx.MockBackend.embed_for("E2", 4)(images, instructions)

    folder = tmp_path / "E2"
    with pytest.raises(io38.BudgetStop):
        gpu.extract("E2", spec, embed, items, folder, {"x": 1}, 3, time.time() + 100, cap_deadline=time.time() + .12)
    sealed = sorted(p.name for p in (folder / "chunks").iterdir())
    assert 1 <= len(sealed) < 3
    before = sum(calls)
    values = gpu.extract("E2", spec, embed, items, folder, {"x": 1}, 3, time.time() + 100)
    assert sum(calls) - before == 9 - 3 * len(sealed) and values.shape == (9, 4)
    assert not (folder / "chunks").exists()
    again = gpu.extract("E2", spec, embed, items, folder, {"x": 1}, 3, time.time() + 100)
    np.testing.assert_array_equal(values, again)
    with pytest.raises(io38.BlockedCache):
        gpu.extract("E2", spec, embed, items, folder, {"x": 2}, 3, time.time() + 100)
    with pytest.raises(TimeoutError):
        gpu.extract("E2", spec, embed, items, tmp_path / "late", {"x": 1}, 3, time.time() - 1)


def test_snapshot_hash_verification(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b"{}")
    spec = {"model": "m", "revision": "r", "weights_sha256": {"w.bin": sha(tmp_path / "w.bin")},
            "metadata_git_blobs": {"config.json": gpu.git_blob_sha1(tmp_path / "config.json")}}
    assert gpu.verify_snapshot(tmp_path, spec)["model"] == "m"
    (tmp_path / "w.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        gpu.verify_snapshot(tmp_path, spec)


# Submissions and export.

def test_submission_keys_order_rounding_and_zip_structure(tmp_path):
    dev = pd.DataFrame({"painting": ["b", "a"], "language": ["x", "y"], "image": ["1", "2"]})
    frame = dev[data.KEYS].assign(p_positive=[.123456, .5])
    archive = io38.write_submission(frame, dev, tmp_path, "s")
    with ZipFile(archive) as z:
        assert z.namelist() == ["predictions.csv"]
    assert submission.validate_archive(archive, dev).p_positive.tolist() == [.123456, .5]
    with pytest.raises(ValueError, match="six-decimal"):
        io38.write_submission(frame.assign(p_positive=[.1234567, .5]), dev, tmp_path, "t")
    with pytest.raises(ValueError, match="order"):
        io38.write_submission(frame.iloc[::-1], dev, tmp_path, "u")
    with pytest.raises(ValueError):
        io38.write_submission(frame.assign(p_positive=[np.nan, .5]), dev, tmp_path, "v")


def test_export_read_back_detects_corruption(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "a.txt").write_text("payload")
    digest = io38.publish(tmp_path / "out.zip", root, [root / "a.txt"], "id")
    assert digest == sha(tmp_path / "out.zip")
    (tmp_path / "out.zip.sha256").write_text("0" * 64 + "  out.zip\n")
    with pytest.raises(ValueError, match="receipt"):
        io38.readback(tmp_path / "out.zip", "id")
    io38.publish(tmp_path / "out.zip", root, [root / "a.txt"], "id")
    with pytest.raises(ValueError, match="coverage"):
        io38.readback(tmp_path / "out.zip", "other")


# Full worker with mocked GPU children.

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("fixture"))


def run(built, tmp_path, backend_kwargs=None, **context):
    import run_exp038

    ctx = fx.context(built, tmp_path, **context)
    backend = fx.MockBackend(ctx, **(backend_kwargs or {}))
    return run_exp038, ctx, backend


def test_happy_path_order_outputs_and_no_go_best_zip(built, tmp_path, monkeypatch):
    run_exp038, ctx, backend = run(built, tmp_path)
    reads = []
    original = pd.read_csv

    def watch(path, *args, **kwargs):
        if str(path).endswith("dev_gold.csv"):
            reads.append(read(ctx.root / "status.json")["stage"])
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", watch)
    decision = run_exp038.run_pipeline(ctx, backend)
    status = read(ctx.root / "status.json")
    assert status["state"] == "COMPLETE" and status["completed_stages"] == list(run_exp038.STAGES)
    assert reads and set(reads) == {"Full refit and Development scoring"}
    assert [c[0] for c in backend.calls[:1]] == ["fetch"]
    assert [c[1] for c in backend.calls if c[0] == "extract"] == ["E1", "E2", "E3", "E4"]
    out = ctx.output
    for name in ("exp038_results.zip", "exp038_best_submission.zip", "exp038_SUCCESS.json",
                 "exp038_C1_submission.zip", "exp038_B2_C3_cscva3_submission.zip", "exp038_E4_submission.zip",
                 "exp038_CSCVA_3_submission.zip"):
        assert (out / name).is_file(), name
    gates = read(ctx.root / "p2/gates.json")
    assert set(gates["gates"]) == {"C1", "C2", "C3"}
    selection = read(ctx.root / "selection.json")
    if decision["decision"] == "NO_GO":
        assert sha(out / "exp038_best_submission.zip") == ctx.refs["exp037_original_blend_zip_sha256"]
        assert selection["decision"] == "NO_GO" and "No EXP-038 candidate" in selection["note"]
    scores = read(ctx.root / "development/scores.json")
    assert scores["archives"]["C1"]["eligibility"] == "VLM_FREE"
    assert scores["archives"]["B1_C1_cscva3"]["eligibility"] == "VLM_FREE"
    assert scores["archives"]["C2"]["eligibility"] == "UNCONFIRMED_QWEN_ST3"
    assert scores["archives"]["B2_C1_original"]["eligibility"] == "UNCONFIRMED_QWEN_ST3"
    diagnostics = read(ctx.root / "p2/diagnostics.json")
    assert "E4_minus_E3" in diagnostics["filtered"] and "C1" in diagnostics["full"]["diversity_vs_fixed"]
    with ZipFile(out / "exp038_results.zip") as z:
        names = z.namelist()
    assert "decision.json" in names and "development/bundle/heads.npz" in names
    assert all(not n.startswith(("tmp", "features/E1/chunks")) for n in names)
    # Resume on the same inputs reuses every sealed stage without any GPU extraction.
    second = fx.MockBackend(ctx)
    (out / "exp038_SUCCESS.json").unlink()
    run_exp038.run_pipeline(ctx, second)
    assert not [c for c in second.calls if c[0] in {"extract", "benchmark"}]


def test_go_path_uses_b2_with_cscva3(built, tmp_path, monkeypatch):
    run_exp038, ctx, backend = run(built, tmp_path)
    monkeypatch.setattr(science, "select", lambda gates, order, tolerance: "C1")
    run_exp038.run_pipeline(ctx, backend)
    selection = read(ctx.root / "selection.json")
    assert selection["decision"] == "GO" and selection["submission"] == "B2_C1_cscva3"
    scores = read(ctx.root / "development/scores.json")
    assert sha(ctx.output / "exp038_best_submission.zip") == scores["archives"]["B2_C1_cscva3"]["sha256"]


def test_budget_skips_seeds_then_e4_and_marks_c3(built, tmp_path):
    run_exp038, ctx, backend = run(built, tmp_path, {"rates": {"E1": 1e4, "E2": 1e4, "E3": 1e4, "E4": .01}}, hours=4.)
    run_exp038.run_pipeline(ctx, backend)
    plan = read(ctx.root / "budget_plan.json")
    assert list(plan["skipped"])[:1] == ["E4"]
    pieces = read(ctx.root / "status.json")["pieces"]
    assert pieces["E4"]["state"] == "BUDGET_SKIPPED"
    assert read(ctx.root / "p2/gates.json")["gates"]["C3"]["status"] == "BUDGET_SKIPPED"
    assert not [c for c in backend.calls if c[:2] == ("extract", "E4")]


def test_failed_optional_seed_falls_back_to_original_cscva(built, tmp_path, monkeypatch):
    seed = built["config"]["cscva"]["new_seeds"][1]
    run_exp038, ctx, backend = run(built, tmp_path, {"fail": {seed: RuntimeError("seed crash")}})
    monkeypatch.setattr(science, "select", lambda gates, order, tolerance: "C1")
    run_exp038.run_pipeline(ctx, backend)
    assert read(ctx.root / "status.json")["pieces"]["cscva_seed_3"]["state"] == "FAILED_OPTIONAL"
    selection = read(ctx.root / "selection.json")
    assert selection["submission"] == "B2_C1_original" and "original single" in selection["cscva_note"]


def test_failed_r0_replay_stops_before_gpu_and_never_claims_a_submission(built, tmp_path):
    import copy
    import hashlib
    import shutil

    import run_exp038

    overlay = tmp_path / "overlay"
    shutil.copytree(built["overlay"], overlay)
    manifest = read(overlay / "exp038_manifest.json")
    manifest["references"]["corrected_oof_sha256"] = "0" * 64
    content = (json.dumps(manifest, indent=2) + chr(10)).encode()
    (overlay / "exp038_manifest.json").write_bytes(content)
    config = copy.deepcopy(built["config"])
    config["references"] = manifest["references"]
    ctx = fx.context(built, tmp_path, overlay=overlay, overlay_sha=hashlib.sha256(content).hexdigest(), config=config)
    backend = fx.MockBackend(ctx)
    with pytest.raises(science.ReplayError, match="Corrected Ridge OOF"):
        run_exp038.run_pipeline(ctx, backend)
    status = read(ctx.root / "status.json")
    assert status["state"] == "FAILED" and status["stage"] == "Validate"
    assert not [c for c in backend.calls if c[0] in {"benchmark", "extract", "cscva"}]
    out = ctx.output
    assert (out / "exp038_diagnostic.zip").is_file() and not (out / "exp038_SUCCESS.json").exists()
    assert not (out / "exp038_best_submission.zip").exists()
    with ZipFile(out / "exp038_diagnostic.zip") as z:
        info = json.loads(z.read("exp038_diagnostic.json"))
    assert info["failed_stage"] == "Validate" and info["submission_claimed"] is False


def test_missing_inputs_are_rejected(built, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(built["base"], copy / "base")
    shutil.copytree(built["overlay"], copy / "overlay")
    first = next((copy / "base/artwork").iterdir())
    first.unlink()
    with pytest.raises(ValueError, match="Missing or corrupt"):
        io38.verify_both(copy / "base", copy / "overlay", built["base_sha"], built["overlay_sha"])
    with pytest.raises(ValueError, match="different EXP-037"):
        io38.verify_overlay(built["overlay"], built["overlay_sha"], "f" * 64)


def test_mandatory_timeout_is_resumable_and_exports_progress(built, tmp_path):
    run_exp038, ctx, backend = run(built, tmp_path, {"fail": {"E2": TimeoutError("allowance")}})
    with pytest.raises(TimeoutError):
        run_exp038.run_pipeline(ctx, backend)
    status = read(ctx.root / "status.json")
    assert status["state"] == "RESUMABLE_STOP" and status["stage"] == "Feature extraction"
    assert (ctx.output / "exp038_progress.zip").is_file() and (ctx.output / "exp038_diagnostic.zip").is_file()
    assert (ctx.root / "features/E1/seal.json").is_file()


def test_e4_budget_stop_is_recorded_not_a_crash(built, tmp_path):
    run_exp038, ctx, backend = run(built, tmp_path, {"fail": {"E4": io38.BudgetStop("cap")}})
    run_exp038.run_pipeline(ctx, backend)
    assert read(ctx.root / "status.json")["pieces"]["E4"]["state"] == "BUDGET_STOP"
    assert read(ctx.root / "p2/gates.json")["gates"]["C3"]["status"] == "BUDGET_SKIPPED"


# Notebook, launcher and deployment.

def test_launcher_and_worker_never_touch_colab_and_notebook_defaults():
    for name in ("exp038_kaggle.py", "run_exp038.py", "deploy_exp038.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
        imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any("colab" in module for module in imported)
    import build_exp038_notebook

    handoff = {"source": "a" * 40, "manifest_sha256": "b" * 64, "archive_sha256": "c" * 64,
               "base_zip_sha256": "d" * 64, "base_manifest_sha256": "e" * 64}
    notebook = build_exp038_notebook.build(handoff, "d" * 64, "e" * 64)
    config = "".join(notebook["cells"][1]["source"])
    assert "AUTHORIZE_RUN = False" in config and 'RESUME_INPUT = ""' in config
    code = "".join(notebook["cells"][2]["source"])
    assert all(value in code for value in ("a" * 40, "b" * 64, "c" * 64, "d" * 64, "e" * 64))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_deploy_prepare_and_infer_from_precomputed_features(built, tmp_path):
    import deploy_exp038
    import run_exp038

    ctx = fx.context(built, tmp_path)
    run_exp038.run_pipeline(ctx, fx.MockBackend(ctx))
    bundle = tmp_path / "deploy"
    deploy_exp038.prepare(type("A", (), {"results": ctx.output / "exp038_results.zip", "output": bundle,
                                         "identity": built["overlay_sha"]})())
    dev = data.read_dev_input(built["base"])
    pop = run_exp038.population(ctx)
    features, delta = run_exp038.features_with_arms(ctx, science.ARMS)
    names = sorted(dev.painting.unique())
    keys = dev[data.KEYS].reset_index(drop=True)
    arrays = {"paintings": np.asarray(names)} | {n: features[n].loc[names].to_numpy() for n in
                                                 [*ENCODERS, "E1", "E2", "E3"]}
    items = run_exp038.pair_items(ctx)
    cache = io38.load_features(ctx.root / "features/E4", run_exp038.arm_identity(ctx, "E4"), [i[0] for i in items])
    digest = pop["paintings"].set_index("painting").image_sha256
    row = {item[0]: i for i, item in enumerate(items)}
    arrays |= {"pair_keys": np.asarray([f"{p}|{l}" for p, l in keys.itertuples(index=False)]),
               "E4": cache[[row[f"{digest[p]}|{l}"] for p, l in keys.itertuples(index=False)]]}
    np.savez(tmp_path / "features.npz", **arrays)
    original = ctx.root / "development/submissions"
    seeds = [ctx.root / f"development/submissions/exp038_CSCVA_seed_{s}_submission.zip"
             for s in built["config"]["cscva"]["new_seeds"]]
    args = type("A", (), {"bundle": bundle, "input": built["base"] / data.DEV_INPUT, "images": built["base"],
                          "output": tmp_path / "inferred", "features": tmp_path / "features.npz",
                          "cscva": [built["overlay"] / "reference/exp036_standalone_submission.zip", *seeds],
                          "c6": original / "exp038_R0_fixed_submission.zip"})()
    # C6 for these keys is the saved EXP-037 C6 standalone; package it as a ZIP for the same keys.
    c6 = submission.read_predictions(built["overlay"] / "reference/exp037_c6_standalone/predictions.csv", dev)
    args.c6 = io38.write_submission(c6, dev, tmp_path / "c6", "c6")
    deploy_exp038.compose(args)
    receipt = read(tmp_path / "inferred/inference_receipt.json")
    run_archives = read(ctx.root / "development/scores.json")["archives"]
    for name in ("R0_corrected", "C1", "C3", "B2_C3_cscva3", "B1_R0_original", "E4"):
        deployed = tmp_path / f"inferred/submissions/exp038_{name}_submission.zip"
        assert io38.zip_member(deployed) == io38.zip_member(ctx.root / run_archives[name]["zip"]), name
    assert receipt["inputs_used"] == "artwork pixels and target language only"
    assert pop["dev"].shape[0] == receipt["pairs"]
