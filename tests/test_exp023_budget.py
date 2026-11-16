"""Operational budget tests: synthetic CPU data only; no scientific GPU jobs."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_exp023 import config, fixture
from st3.caption_data import inner_partition
from st3.caption_student import fit_student, predict_student
from st3.exp023_budget import condition_cost, forecast_work
from st3.exp023_runner import Experiment, ROOT
from st3.experiment_workflow import Budget, StopRun, sha256, verify_model, finalize


def operations():
    return json.loads((ROOT / "configs/exp023_runtime.json").read_text())


def samples():
    return [{"condition": c, "pairs": 900, "full_pairs": 1000,
             "full_inner_epoch_seconds": .1, "inner_training_epoch_seconds": .1,
             "validation_epoch_seconds": .02, "full_refit_epoch_seconds": .12,
             "checkpoint_seconds": .01, "replay_seconds": .02} for c in ("C0", "C1", "C2")]


def authorization(hours=4):
    return {"EXECUTION_SCOPE": "RESEARCH_ONLY_CV", "AUTHORIZE_EXPERIMENT": True,
            "AUTHORIZE_GUARDED_CONFIRMATION": False, "REQUIRE_SUBMISSION_ZIP": False,
            "MAX_GPU_HOURS": hours, "MAX_TOTAL_WALL_HOURS": hours + 1}


@pytest.mark.parametrize("hours", [1, 2, 4, 8])
def test_owner_budgets_not_rejected_by_experiment_or_generic_budget(tmp_path, config, hours):
    # Constructing a device descriptor allocates no CUDA tensors/context.
    engine = Experiment(tmp_path / "inputs", tmp_path / str(hours), config, authorization(hours), device="cuda")
    assert engine.budget.seconds == hours * 3600
    assert engine.config == config  # Owner budget never enters scientific configuration.


@pytest.mark.parametrize("hours", [1, 2, 4, 8])
def test_cloud_accepts_owner_budget_without_universal_ceiling(tmp_path, monkeypatch, hours):
    from st3 import exp023_cloud
    reached = []
    def inputs(*args):
        reached.append(True)
        raise StopRun("CPU_TEST_STOP", "No real setup/training in this test")
    monkeypatch.setattr(exp023_cloud, "restore_handoff", inputs)
    settings = {**authorization(hours), "LOCAL_ROOT": str(tmp_path / "local"), "PERSISTENT_ROOT": str(tmp_path / "outputs"),
                "AUTO_BROWSER_DOWNLOAD": True, "AUTO_RELEASE_COLAB": True, "INPUT_ZIP": "unused", "ZIP_SHA256": "x", "MANIFEST_SHA256": "x"}
    events = []
    report = exp023_cloud.run_cloud(settings, platform="colab", mount=lambda:None,
                                    download=lambda path:events.append("download"), release=lambda:events.append("release"))
    assert reached and report["status"] == "CPU_TEST_STOP"
    assert events == ["download", "release"]


def test_theoretical_bound_is_diagnostic_and_no_measured_expectation_before_fits():
    cfg = json.loads((ROOT / "configs/exp023.json").read_text())
    remaining = [(f, c, s) for f in range(1, 6) for c in cfg["conditions"] for s in cfg["seeds"]]
    result = forecast_work(samples(), remaining, [], cfg, operations(), fit_pairs=1000, target_seconds=1, overhead=60)
    legacy = 1.25 * (45 * 150 * 2 * (.1 * 1000 / 900) + 12 + 300 + 120)
    assert result["original_max_epoch_formula_seconds"] == pytest.approx(legacy)
    assert result["theoretical_worst_case_seconds"] >= legacy
    assert result["measured_expected_remaining_seconds"] is None
    assert "not confidence intervals" in result["limitations"]


def test_measured_fast_slow_stopping_changes_expected_not_science():
    cfg = json.loads((ROOT / "configs/exp023.json").read_text())
    remaining = [(1, "C0", 2028)]
    fast = [{"condition":"C0", "selection_epochs":20, "selected_epoch":5, "fit_seconds":5., "end_to_end_seconds":6.}]
    slow = [{"condition":"C0", "selection_epochs":150, "selected_epoch":140, "fit_seconds":70., "end_to_end_seconds":74.}]
    a = forecast_work(samples(), remaining, fast, cfg, operations(), fit_pairs=1000, target_seconds=1, overhead=60)
    b = forecast_work(samples(), remaining, slow, cfg, operations(), fit_pairs=1000, target_seconds=1, overhead=60)
    assert b["measured_expected_remaining_seconds"] > a["measured_expected_remaining_seconds"]
    assert b["per_condition"]["C0"]["admission_seconds"] >= a["per_condition"]["C0"]["admission_seconds"]
    assert b["per_condition"]["C0"]["selection_epochs"] == [150]
    assert b["per_condition"]["C0"]["refit_epochs"] == [140]
    assert cfg["max_epochs"] == 150 and cfg["patience"] == 15


def make_engine(tmp_path, fixture, config):
    pairs, _, _, visual, _ = fixture
    fitting, held = pairs[pairs.painting.ne("p7")], pairs[pairs.painting.eq("p7")]
    inner, stopping = inner_partition(fitting, config, 1)
    config = {**config, "conditions":["C0"], "seeds":[2027]}
    engine = Experiment(tmp_path / "inputs", tmp_path / "run", config, authorization(), device="cpu")
    engine.partition = lambda fold:(fitting, held, inner, stopping)
    engine.identity = {"source_revision":"CPU_SYNTHETIC", "source":{"model":"pinned"}, "config":"fixed", "inputs":"fixed", "runtime":"CPU_FIXTURE"}
    engine.train, engine.visual = fitting, visual
    engine.timing_samples = [samples()[0]]
    engine.target_seconds = .001
    baseline = held[["painting","language"]].assign(p_positive=.5)
    return engine, fitting, held, inner, stopping, visual, baseline


def test_global_worst_case_cannot_veto_safe_partial_progress(tmp_path, fixture, config):
    engine, *rest = make_engine(tmp_path, fixture, config)
    # A huge global overhead is diagnostic, never a whole-job admission gate.
    engine.scoring_seconds = 100000.
    result = engine.forecast(engine.timing_samples, .001)
    assert result["theoretical_worst_case_seconds"] > engine.budget.remaining()
    engine.admit_fit(1, "C0", 2027, len(engine.train))


@pytest.mark.parametrize("condition", ["C0", "C1", "C2"])
def test_timed_fit_parity_atomic_verified_publication_and_budget_only_resume(tmp_path, fixture, config, monkeypatch, condition):
    engine, fitting, held, inner, stopping, visual, baseline = make_engine(tmp_path, fixture, config)
    from st3.exp023_targets import prepare_partitions
    prepared = prepare_partitions(fixture[1], fitting, inner, fixture[2], engine.config)
    engine.config["conditions"] = [condition]
    engine.timing_samples = [r for r in samples() if r["condition"] == condition]
    original, _, info = fit_student(fitting, inner, stopping, visual, prepared.get(condition), 2027, engine.config, torch.device("cpu"))
    expected = predict_student(original, held, visual, baseline, torch.device("cpu"), engine.config["batch_size"])
    path = engine.fit_one(1, condition, 2027, prepared, baseline, held)
    payload = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(original.state_dict()[k], payload["state_dict"][k]) for k in payload["state_dict"])
    np.testing.assert_array_equal(expected, np.load(path / "raw_predictions.npy"))
    observed = json.loads((path / "fit.json").read_text())
    assert observed["selected_epoch"] == info["selected_epoch"]
    assert observed["selection_epochs"] == info["selection_epochs"]
    assert json.loads((path / "initial_replay.json").read_text())["status"] == "PASS"
    before = sha256(path / "seal.json")
    from st3 import exp023_runner
    monkeypatch.setattr(exp023_runner, "fit_student", lambda *args,**kwargs:pytest.fail("Verified model retrained"))
    engine.authorization.update(MAX_GPU_HOURS=8., MAX_TOTAL_WALL_HOURS=9.)
    engine.budget = Budget(8,9,cuda=False)
    engine.fit_one(1, condition, 2027, prepared, baseline, held)
    assert sha256(path / "seal.json") == before
    changed = copy.deepcopy(engine.identity);changed["source_revision"] = "different"
    with pytest.raises(StopRun):verify_model(path, changed)
    (path / "checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(StopRun):engine.fit_one(1, condition, 2027, prepared, baseline, held)


def test_insufficient_budget_stops_before_fit(tmp_path, fixture, config):
    engine, *rest = make_engine(tmp_path, fixture, config)
    now = [0.]
    engine.budget = Budget(1,2,cuda=False,clock=lambda:now[0])
    now[0] = 7200 - 181
    with pytest.raises(StopRun) as error:engine.admit_fit(1,"C0",2027,len(engine.train))
    assert error.value.status == "STOPPED_BUDGET"
    assert not (engine.run_dir / "models").exists()


def test_mid_fit_exhaustion_preserves_verified_previous_model(tmp_path, fixture, config):
    engine, _, held, _, _, _, baseline = make_engine(tmp_path, fixture, config)
    first = engine.fit_one(1,"C0",2027,{},baseline,held)
    before = sha256(first / "seal.json")
    now = [0.]
    engine.budget = Budget(1,2,cuda=True,clock=lambda:now[0])
    ordinary = engine.progress
    def progress(phase, **kwargs):
        if phase in ("selection", "refit"):now[0] += 1800
        ordinary(phase, **kwargs)
    engine.progress = progress
    with pytest.raises(StopRun) as error:engine.fit_one(2,"C0",2027,{},baseline,held)
    assert error.value.status == "STOPPED_BUDGET"
    assert not (engine.run_dir / "models/fold_2_C0_2027").exists()
    assert sha256(first / "seal.json") == before
    calls=[]
    report=finalize(engine.run_dir,tmp_path/"drive",{"status":"STOPPED_BUDGET"},platform="colab",
                    download=lambda p:calls.append("download"),release=lambda:calls.append("release"))
    assert report["persistent_readback"]=="PASS" and calls==["download","release"]


@pytest.mark.parametrize("count", [0, 1, 44])
def test_incomplete_cv_never_becomes_scientific_no_go(count):
    engine=object.__new__(Experiment);engine.verified=count
    with pytest.raises(StopRun) as error:engine.decision()
    assert error.value.status=="INCOMPLETE_CV"


def test_cv_still_requires_45_verified_fits():
    engine=object.__new__(Experiment)
    engine.completed_timings=lambda:([{"fold":1}]*44,[(5,"C2",2029)])
    with pytest.raises(StopRun) as error:engine.cv_placeholder()
    assert error.value.status=="INCOMPLETE_CV"
    engine.completed_timings=lambda:([{"fold":1}]*45,[])
    assert engine.cv_placeholder()["status"]=="FITS_COMPLETE_PENDING_INDEPENDENT_REPLAY"
