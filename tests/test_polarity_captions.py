import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from st3 import data, polarity_confirmation as confirmation, polarity_experiment as experiment
from st3 import polarity_workflow as workflow
from st3.baseline import file_sha256
from st3.caption_data import training_captions
from st3.caption_student import predict_student
from st3.polarity_student import (PolarityStudent, cosine_loss, gradient_diagnostic, polarity_epoch,
                                  polarity_rows, seed_polarity_model)
from st3.polarity_targets import condition_targets, fit_shared_geometry, polarity_means
from st3.polarity_workflow import BudgetStop, ExecutionBudget, StageBlocked, read_sealed, seal

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/exp018.json").read_text())


def test_polarity_mapping_keeps_repeated_votes_and_excludes_other():
    captions = pd.DataFrame({"painting": "p", "language": "Tamil", "split": "train", "valid_caption": True,
                             "emotion": ["awe", "awe", "sadness", "other"],
                             "caption_id": ["positive", "positive", "negative", "excluded"],
                             "binary": [True, True, True, False]})
    pairs = pd.DataFrame({"painting": ["p"], "language": ["Tamil"], "k": [2], "n": [5]})
    vectors = {"positive": np.array([1., 0., 0.], dtype=np.float32),
               "negative": np.array([0., 1., 0.], dtype=np.float32)}
    records, values, masks = polarity_means(captions, pairs, vectors)
    assert records[["n_all", "n_positive", "n_negative"]].values.tolist() == [[3, 2, 1]]
    np.testing.assert_allclose(values[0], [[2/3, 1/3, 0], [1, 0, 0], [0, 1, 0]])
    assert masks.tolist() == [[True, True, True]]
    with pytest.raises(ValueError, match="exceed"):
        polarity_means(captions, pairs.assign(k=5), vectors)


def test_caption_reader_does_not_retain_locked_or_official_validation_text(tmp_path):
    (tmp_path / "annotations").mkdir()
    frame = pd.DataFrame({"painting": ["fit", "locked", "val"], "language": "Tamil",
                          "image_name": ["a.png", "b.png", "c.png"], "split": ["train", "train", "val"],
                          "emotion": "awe", "caption": ["allowed", "LOCKED", "VALIDATION"], "image_id": 1})
    frame.to_csv(tmp_path / data.ANNOTATIONS, index=False)
    result = training_captions(tmp_path, allowed_paintings={"fit"})
    assert result.caption.tolist() == ["allowed"] and result.annotation_id.tolist() == [0]


def geometry_fixture():
    rng = np.random.default_rng(18)
    means = rng.normal(size=(8, 3, 5)).astype(np.float32)
    records = pd.DataFrame({"painting": ["a", "a", "b", "b", "c", "c", "d", "d"],
                            "language": ["Tamil", "Korean"] * 4, "n_all": 5, "n_positive": 3, "n_negative": 2})
    mask = np.ones((8, 3), dtype=bool)
    mask[0, 2] = False
    means[0, 2] = 0
    return records, means, mask


def test_one_train_local_pca_projects_all_three_means_with_missing_channels_masked():
    records, means, mask = geometry_fixture()
    cfg = {**CONFIG, "target_dimensions": 2}
    first = fit_shared_geometry(records.iloc[:6], means[:6], mask[:6], cfg)
    altered = means[:6].copy()
    altered[:, 1:] += 100
    second = fit_shared_geometry(records.iloc[:6], altered, mask[:6], cfg)
    np.testing.assert_array_equal(first["pca"].components_, second["pca"].components_)
    np.testing.assert_allclose(first["pca"].mean_, means[:6, 0].mean(axis=0))
    assert first["metadata"]["fit_paintings"] == ["a", "b", "c"] and not first["metadata"]["whiten"]
    np.testing.assert_allclose(np.linalg.norm(first["targets"][first["valid"]], axis=-1), 1, atol=2e-6)
    assert not first["valid"][0, 2] and not first["targets"][0, 2].any()
    pooled, _ = condition_targets(first, "M2")
    aware, weights = condition_targets(first, "M3")
    np.testing.assert_array_equal(pooled[("a", "Tamil")][0][0], pooled[("a", "Tamil")][0][1])
    assert aware[("a", "Tamil")][1].tolist() == [True, False]
    assert weights[("a", "Tamil")] == weights[("a", "Korean")] == .5


def test_geometry_subset_cannot_use_held_painting_means():
    records, means, mask = geometry_fixture()
    raw = records, means, mask
    pairs = records.iloc[:4][data.KEYS]
    first = workflow.fit_partition_geometry(raw, pairs, {**CONFIG, "target_dimensions": 2})
    altered = means.copy()
    altered[4:] = 10000
    second = workflow.fit_partition_geometry((records, altered, mask), pairs, {**CONFIG, "target_dimensions": 2})
    np.testing.assert_array_equal(first["pca"].components_, second["pca"].components_)
    assert first["metadata"]["fit_paintings"] == ["a", "b"]


def test_cosine_loss_masks_channels_averages_per_pair_and_balances_paintings():
    predicted = torch.tensor([[[1., 0.], [1., 0.]]] * 3, requires_grad=True)
    target = torch.tensor([[[-1., 0.], [-1., 0.]], [[1., 0.], [0., 0.]], [[-1., 0.], [-1., 0.]]])
    mask = torch.tensor([[True, True], [True, False], [True, True]])
    value = cosine_loss(predicted, target, mask, torch.tensor([.5, .5, 1.]))
    assert value.item() == pytest.approx(1.5)
    value.backward()
    assert not predicted.grad[1, 1].any()
    single = cosine_loss(predicted.detach()[:1], target[:1], torch.tensor([[True, False]]), torch.ones(1))
    both = cosine_loss(predicted.detach()[:1], target[:1], torch.tensor([[True, True]]), torch.ones(1))
    assert single.item() == both.item() == 2.


def test_unit_vector_cosine_is_32_times_coordinate_mse():
    rng = torch.Generator().manual_seed(18)
    first = F.normalize(torch.randn(5, 2, 64, generator=rng), dim=-1)
    second = F.normalize(torch.randn(5, 2, 64, generator=rng), dim=-1)
    torch.testing.assert_close(1 - (first * second).sum(dim=-1), 32 * (first - second).square().mean(dim=-1))


def model_rows():
    rng = np.random.default_rng(18)
    frame = pd.DataFrame({"painting": ["a", "b", "c", "d"], "image": list("abcd"),
                          "language": "Tamil", "k": [1, 2, 3, 4], "n": 5})
    features = {name: rng.normal(size=2048).astype(np.float32) for name in "abcd"}
    values = rng.normal(size=(4, 2, 64)).astype(np.float32)
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    targets = {(name, "Tamil"): (vector, np.ones(2, dtype=bool)) for name, vector in zip("abcd", values)}
    weights = {key: 1. for key in targets}
    return frame, features, targets, weights


def test_all_conditions_have_identical_initialization_and_zero_auxiliary_parity():
    frame, features, targets, weights = model_rows()
    initial, trained = [], []
    for use_targets in (False, True):
        model, rng, optimizer = seed_polarity_model(["Tamil"], 2027, CONFIG, "cpu")
        initial.append({key: value.clone() for key, value in model.state_dict().items()})
        rows = polarity_rows(frame, features, ["Tamil"], "cpu", targets if use_targets else None, weights)
        polarity_epoch(model, rows, rng, optimizer, CONFIG, 0.)
        trained.append(model.state_dict())
    assert all(torch.equal(initial[0][key], initial[1][key]) for key in initial[0])
    assert all(torch.equal(trained[0][key], trained[1][key]) for key in trained[0])
    assert torch.equal(initial[0]["polarity.weight"], trained[0]["polarity.weight"])


@pytest.mark.parametrize("condition", ["M2", "M3"])
def test_gradient_pilot_has_finite_shared_and_both_query_gradients(condition):
    frame, features, targets, weights = model_rows()
    if condition == "M2":
        targets = {key: (np.repeat(value[:1], 2, axis=0), mask) for key, (value, mask) in targets.items()}
    rows = polarity_rows(frame, features, ["Tamil"], "cpu", targets, weights)
    model, _, _ = seed_polarity_model(["Tamil"], 2027, CONFIG, "cpu")
    result = gradient_diagnostic(model, rows, torch.arange(4), .1)
    assert result["finite"] and min(result["polarity_gradient_norms"]) > 0
    assert result["scaled_auxiliary"] == pytest.approx(.1 * result["cosine_auxiliary"])
    assert result["cosine_to_mse_ratio"] == pytest.approx(32., abs=1e-4)
    assert -1 <= result["gradient_cosine"] <= 1


def test_invalid_gradients_stop_instead_of_adjusting_the_loss_coefficient():
    frame, features, targets, weights = model_rows()
    features["a"][:] = np.nan
    rows = polarity_rows(frame, features, ["Tamil"], "cpu", targets, weights)
    model, _, _ = seed_polarity_model(["Tamil"], 2027, CONFIG, "cpu")
    with pytest.raises(ValueError, match="Numerically invalid"):
        gradient_diagnostic(model, rows, torch.arange(4), .1)
    assert CONFIG["text_weight"] == .1


def test_inference_has_no_caption_or_ground_truth_polarity_input():
    frame, features, _, _ = model_rows()
    model = PolarityStudent(["Tamil"], CONFIG)
    baseline = frame[data.KEYS].assign(p_positive=.7)
    original = predict_student(model, frame, features, baseline, "cpu", 4)
    model.polarity.weight.data.fill_(1000.)
    changed = predict_student(model, frame.assign(caption=object(), polarity="secret"), features, baseline, "cpu", 4)
    np.testing.assert_array_equal(original, changed)
    unknown = frame.iloc[:1].assign(language="Kazakh")
    assert predict_student(model, unknown, features, unknown[data.KEYS].assign(p_positive=.123456), "cpu", 4)[0] == .123456


def test_sealed_checkpoint_rejects_corruption_partial_files_and_changed_sources(tmp_path):
    identity = {"source": "original"}
    (tmp_path / "student.pt").write_bytes(b"checkpoint")
    seal(tmp_path, identity, {"finite": True})
    assert read_sealed(tmp_path, identity)["finite"]
    with pytest.raises(ValueError, match="Incompatible"):
        read_sealed(tmp_path, {"source": "changed"})
    (tmp_path / "student.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_sealed(tmp_path, identity)
    (tmp_path / "run.json").unlink()
    with pytest.raises(StageBlocked, match="unpublished"):
        read_sealed(tmp_path, identity)


def test_changed_exp017_text_cache_is_rejected_before_execution(tmp_path, monkeypatch):
    path = tmp_path / "text/clip/features.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    cfg = {"exp017_config_sha256": file_sha256(Path(__file__).resolve().parents[1] / "configs/exp017.json"),
           "exp017_sha256": {"text/clip/features.npz": file_sha256(path)}}
    monkeypatch.setattr(workflow, "config018", lambda: cfg)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Pinned EXP-017 artifact changed"):
        workflow.fingerprint(SimpleNamespace(exp017_dir=tmp_path))


def test_incomplete_evaluation_is_never_classified_pass_or_no_go():
    assert experiment.choose_candidate(None, CONFIG, False)["status"] == "INCOMPLETE"


@pytest.mark.parametrize(("error", "code"), [(BudgetStop("budget"), 75), (StageBlocked("prior stage"), 3),
                                            (ValueError("invalid input"), 1)])
def test_stage_exit_codes_are_meaningful(monkeypatch, error, code):
    from st3 import polarity_cli
    def fail(args):
        raise error
    monkeypatch.setattr(polarity_cli, "run_cv", fail)
    with pytest.raises(SystemExit) as stopped:
        polarity_cli.run_polarity(SimpleNamespace(command="polarity-cv"))
    assert stopped.value.code == code


def test_no_go_guards_do_not_open_confirmation_or_development(monkeypatch):
    monkeypatch.setattr(confirmation, "verified_decision", lambda *args: {"status": "NO_GO"})
    monkeypatch.setattr(confirmation, "context", lambda *args: pytest.fail("Opened confirmation/public development after NO-GO"))
    confirmation.run_confirm(None)
    confirmation.run_dev(None)


@pytest.fixture
def tiny_workflow(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = {**CONFIG, "cv_folds": 2, "seeds": [2027], "max_epochs": 1, "patience": 1, "target_dimensions": 2,
           "pilot_warmup_epochs": 0, "pilot_timed_epochs": 1, "bootstrap_replicates": 50}
    rng = np.random.default_rng(18)
    rows = [{"painting": f"p{i:02}", "image": f"{i}.png", "image_sha256": f"{i:064x}", "split": "train",
             "language": lang, "k": k, "n": 5, "p": k/5, "fold": i % 2 + 1}
            for i in range(40) for lang, k in (("Tamil", 1), ("Korean", 2), ("Thai", 4))]
    train = pd.DataFrame(rows)
    reference = train[[*data.KEYS, "fold"]].assign(p_positive=train.p.to_numpy())
    banks = []
    for width in (512, 768, 768):
        values = rng.normal(size=(40, width)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        banks.append(dict(zip([f"{i}.png" for i in range(40)], values)))
    identity = {"device": "cpu", "fixture": True}
    ctx = SimpleNamespace(config=cfg, identity=identity, train=train, reference=reference, banks=banks)
    args = SimpleNamespace(run_dir=tmp_path / "run", exp017_dir=tmp_path / "old", max_seconds=120.,
                           deep_verify=True, device="cpu")
    (args.exp017_dir / "cv").mkdir(parents=True)
    (args.exp017_dir / "cv/fits.json").write_text(json.dumps([{"selected_epoch": 1}]))
    for module in (experiment, workflow):
        monkeypatch.setattr(module, "fingerprint", lambda *args: identity)
        monkeypatch.setattr(module, "config018", lambda: cfg)
    monkeypatch.setattr(experiment, "context", lambda *args, **kwargs: ctx)
    (args.run_dir / "preflight").mkdir(parents=True)
    (args.run_dir / "preflight/historical_sha256.json").write_text("{}")
    seal(args.run_dir / "preflight", identity, {"seconds": .01})
    audit = args.run_dir / "audit"
    audit.mkdir()
    records = train[data.KEYS].assign(n_all=5, n_positive=train.k.to_numpy(), n_negative=5-train.k.to_numpy())
    means = rng.normal(size=(len(records), 3, 5)).astype(np.float32)
    records.to_csv(audit / "pairs.csv", index=False)
    np.savez(audit / "raw_means.npz", means=means, available=np.ones((len(records), 3), dtype=bool))
    seal(audit, identity, {"seconds": .01})
    return args, ctx, (records, means, np.ones((len(records), 3), dtype=bool))


def test_tiny_pipeline_requires_pilot_resumes_and_replays_then_retains_baseline(tiny_workflow, monkeypatch):
    args, ctx, _ = tiny_workflow
    with pytest.raises(StageBlocked, match="unpublished"):
        experiment.run_cv(args)
    experiment.run_pilot(args)
    experiment.run_cv(args)
    assert read_sealed(args.run_dir / "cv", ctx.identity)["status"] == "AWAITING_REPLAY_VERIFICATION"
    with pytest.raises(StageBlocked):
        experiment.verified_decision(args)
    timestamps = {path: path.stat().st_mtime_ns for path in (args.run_dir / "work").rglob("student.pt")}
    monkeypatch.setattr(experiment, "fit_student", lambda *args, **kwargs: pytest.fail("Retrained a completed valid seed"))
    experiment.run_cv(args)
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in timestamps.items())
    experiment.run_verify(args)
    decision = experiment.verified_decision(args)
    assert decision["status"] == "NO_GO" and decision["model"] == "M0"
    assert read_sealed(args.run_dir / "verification", ctx.identity)["checkpoints_replayed"] == 6
    experiment.run_verify(args)
    path = next(iter(timestamps))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        experiment.run_cv(args)


def test_budget_stop_retains_completed_seed_and_restarts_only_unpublished_seed(tiny_workflow):
    args, ctx, raw = tiny_workflow
    fit, held = ctx.train[ctx.train.fold.eq(1)], ctx.train[ctx.train.fold.eq(2)]
    baseline = held[data.KEYS].assign(p_positive=held.p.to_numpy())
    class StopDuringSecondCondition:
        def check(self, phase="stage", epoch=0):
            if "M2" in phase and epoch == 1:
                raise BudgetStop("test interruption")
    directory = args.run_dir / "budget_work"
    with pytest.raises(BudgetStop):
        experiment.fit_conditions(args, ctx, raw, fit, held, baseline, directory, 1, StopDuringSecondCondition())
    checkpoint = directory / "seeds/M1_2027/student.pt"
    original = file_sha256(checkpoint)
    assert not (directory / "seeds/M2_2027").exists()
    predictions, fits = experiment.fit_conditions(args, ctx, raw, fit, held, baseline, directory, 1, ExecutionBudget(120))
    assert file_sha256(checkpoint) == original and len(fits) == 3 and set(predictions) == {"M0", "M1", "M2", "M3", "B1", "B2", "B3"}


def test_failed_later_verification_blocks_previous_success_until_replayed(tiny_workflow, monkeypatch):
    args, _, _ = tiny_workflow
    experiment.run_pilot(args)
    experiment.run_cv(args)
    experiment.run_verify(args)
    assert experiment.verified_decision(args)["status"] == "NO_GO"
    original_budget = experiment.stage_budget
    def stop(*args):
        raise BudgetStop("interrupted later replay")
    monkeypatch.setattr(experiment, "stage_budget", stop)
    with pytest.raises(BudgetStop):
        experiment.run_verify(args)
    with pytest.raises(StageBlocked, match="latest verification"):
        experiment.verified_decision(args)
    monkeypatch.setattr(experiment, "stage_budget", original_budget)
    experiment.run_verify(args)
    assert experiment.verified_decision(args)["status"] == "NO_GO"
