"""CPU/synthetic tests for EXP-040 (the GPU child is mocked; no downloads)."""

import json
import sys
import time
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp039_fixture as fx39  # noqa: E402
import exp040_fixture as fx  # noqa: E402
from st3 import data  # noqa: E402
from st3 import exp038 as science  # noqa: E402
from st3 import exp038_io as io38  # noqa: E402
from st3 import exp040 as plan  # noqa: E402
from st3 import exp040_gpu as gpu  # noqa: E402
from st3.competition_ridge import Head, fit_head, make_panel  # noqa: E402
from st3.exp036_io import read, sha  # noqa: E402


@pytest.fixture(autouse=True)
def class_keys(monkeypatch):
    from st3 import key_audit

    monkeypatch.setattr(key_audit, "qwen_class_keys", fx39.fake_class_keys)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("exp040"))


def synthetic(seed=0, paintings=40, languages=("English", "Hausa", "Korean", "Thai"), width=6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(paintings, width))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    names = [f"p{i:03d}" for i in range(paintings)]
    rows = []
    for i, p in enumerate(names):
        for j, language in enumerate(languages):
            n = 5 + (i + j) % 3
            prob = 1 / (1 + np.exp(-(x[i, 0] * 2 + x[i, 1] * (j - 1.5))))
            rows.append((p, language, int(rng.binomial(n, prob)), n))
    frame = pd.DataFrame(rows, columns=["painting", "language", "k", "n"])
    return frame, pd.DataFrame(x, index=names)


RULE = {"reference_c": 0.3, "gtol": 1e-8, "ftol": 1e-13, "maxiter": 1600, "max_final_gradient": 5e-7,
        "deviation_ratio": 0.5}


# Head, loss, anchor and step 0.

def test_head_initialized_from_ridge_maps_exactly():
    frame, x = synthetic()
    head, _ = fit_head(make_panel(frame), x.loc[sorted(frame.painting.unique())].to_numpy(), RULE)
    languages = ["Burmese", *head.languages]  # one language unseen by the ridge keeps the shared row
    weight, bias = plan.head_tensors(head, languages)
    logits = head.logits(x.to_numpy(), languages)
    assert np.allclose(x.to_numpy() @ weight.T + bias, logits, atol=1e-12, rtol=0)
    assert np.allclose(weight[0], head.shared) and bias[0] == pytest.approx(head.intercept)
    inverse = plan.head_from_tensors(weight, bias, languages)
    assert np.allclose(inverse.logits(x.to_numpy(), languages), logits, atol=1e-12)


def test_step0_equality_on_synthetic_data():
    frame, x = synthetic(1)
    head, _ = fit_head(make_panel(frame), x.loc[sorted(frame.painting.unique())].to_numpy(), RULE)
    languages = sorted(frame.language.unique())
    module = plan.interaction_head(head, languages, 1e-3)
    keys = frame[data.KEYS]
    expected = plan.ridge_probabilities(head, x, keys)
    index = torch.as_tensor(keys.language.map({l: i for i, l in enumerate(languages)}).to_numpy())
    with torch.no_grad():
        got = torch.sigmoid(module(torch.as_tensor(x.loc[keys.painting].to_numpy(dtype=np.float32))).double()[
            torch.arange(len(keys)), index]).numpy()
    assert np.abs(got - expected).max() <= 1e-5
    assert float(module.anchor()) == 0.
    # LoRA at init is the identity on the backbone path (zero right factor, dropout off in eval).
    layers = torch.nn.ModuleList()
    for _ in range(2):
        attention = torch.nn.Module()
        for name in gpu.TARGETS:
            setattr(attention, name, torch.nn.Linear(8, 8))
        layer = torch.nn.Module()
        layer.self_attn = attention
        layers.append(layer)
    lm = torch.nn.Module()
    lm.layers = layers
    states = torch.randn(3, 8)
    before = [getattr(l.self_attn, t)(states) for l in layers for t in gpu.TARGETS]
    names = gpu.attach_lora(lm, {"decoder_layers": 2, "targets": list(gpu.TARGETS), "rank": 4, "alpha": 8,
                                 "dropout": .5}, 0)
    lm.eval()
    after = [getattr(l.self_attn, t)(states) for l in layers for t in gpu.TARGETS]
    assert len(names) == 8 and all(torch.equal(a, b) for a, b in zip(before, after))
    params = gpu.lora_parameters(lm)
    assert len(params) == 16 and all(float(p.abs().max()) == 0 for p in params[1::2])
    with pytest.raises(ValueError, match="targets"):
        gpu.attach_lora(lm, {"decoder_layers": 2, "targets": ["q_proj", "v_proj"], "rank": 4, "alpha": 8,
                             "dropout": 0}, 0)


def test_binomial_loss_on_counts():
    z = torch.tensor([0.3, -1.2, 2.0])
    k, n = torch.tensor([2., 0., 5.]), torch.tensor([5., 6., 5.])
    manual = sum(nn * np.log1p(np.exp(zz)) - kk * zz for zz, kk, nn in zip(z.tolist(), k.tolist(), n.tolist())) / float(n.sum())
    assert float(plan.binomial_nll(z, k, n)) == pytest.approx(manual, rel=1e-6)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z, k / n, reduction="none")
    assert float(plan.binomial_nll(z, k, n)) == pytest.approx(float((bce * n).sum() / n.sum()), rel=1e-6)


def test_l2_to_init_term():
    frame, x = synthetic(2)
    head, _ = fit_head(make_panel(frame), x.loc[sorted(frame.painting.unique())].to_numpy(), RULE)
    module = plan.interaction_head(head, sorted(frame.language.unique()), 1e-3)
    with torch.no_grad():
        module.linear.weight[0, 0] += .5
        module.linear.bias[1] -= .25
    assert float(module.anchor()) == pytest.approx(.5 ** 2 + .25 ** 2, rel=1e-6)
    logits = module(torch.zeros(1, x.shape[1]))
    total, nll = plan.objective(logits[0], torch.zeros(4), torch.ones(4), module)
    assert float(total) - float(nll) == pytest.approx(1e-3 * (.5 ** 2 + .25 ** 2), rel=1e-3)
    module.zero_grad()
    (module.lam * module.anchor()).backward()
    assert float(module.linear.weight.grad[0, 0]) == pytest.approx(2e-3 * .5, rel=1e-5)
    assert not module.weight0.requires_grad and not module.bias0.requires_grad


# Epochs.

def test_inner_split_and_epoch_selection_including_epoch_zero():
    names = [f"p{i}" for i in range(50)]
    fit, held = plan.inner_split(names, .1, 7)
    assert len(held) == 5 and not set(fit) & set(held) and sorted(fit + held) == sorted(names)
    assert plan.inner_split(names, .1, 7) == (fit, held) and plan.inner_split(names, .1, 8) != (fit, held)
    history = [{"epoch": e, "inner_test_like_contrast": c} for e, c in enumerate([.42, .41, .40, .39, .38])]
    assert plan.choose_epoch(history) == 0
    history[2]["inner_test_like_contrast"] = .43
    history[4]["inner_test_like_contrast"] = .43
    assert plan.choose_epoch(history) == 2  # exact tie goes to the smaller epoch
    with pytest.raises(ValueError, match="epoch 0"):
        plan.choose_epoch(history[1:])


def test_median_epoch_rule():
    assert plan.median_epoch([0, 2, 4, 3, 1]) == 2
    assert plan.median_epoch([0, 0, 4, 4, 4]) == 4
    assert plan.median_epoch([1, 3]) == 1 and plan.median_epoch([2, 3, 3, 2]) == 2
    with pytest.raises(ValueError):
        plan.median_epoch([])


def test_schedule_matches_exp036():
    factor = plan.schedule(100, .05)
    assert factor(0) == pytest.approx(.2) and factor(4) == pytest.approx(1.)
    assert factor(5) == pytest.approx(1.) and factor(99) == pytest.approx(.1, abs=1e-3)


# Gate and selection.

def gate_result(passed, contrast):
    return {"passed": passed, "status": "PASS" if passed else "FAIL", "numbers": {"filtered_contrast": contrast}}


def test_gate_arithmetic_and_tie_rules(built):
    assert plan.select({"F1": gate_result(True, .470), "F2": gate_result(True, .4709)}, .001) == "F1"
    assert plan.select({"F1": gate_result(True, .470), "F2": gate_result(True, .4712)}, .001) == "F2"
    assert plan.select({"F1": gate_result(False, .48), "F2": gate_result(True, .46)}, .001) == "F2"
    assert plan.select({"F1": gate_result(False, .48), "F2": gate_result(False, .49)}, .001) is None
    config = built["config"]
    assert config["gate"]["minimum_filtered_gain"] == .003 and config["gate"]["minimum_fold_wins"] == 4
    assert config["gate"]["bootstrap_draws"] == 20000 and config["gate"]["maximum_log_loss_delta"] == .005
    report = {"full": {"comparisons": {"F1_minus_C2": {"pooled_contrast_delta": .001, "log_loss_delta": .004}}},
              "filtered": {"systems": {"F1": {"pooled": {"cultural_contrast": .47}}},
                           "comparisons": {"F1_minus_C2": {"pooled_contrast_delta": .003, "bootstrap_95": [.0001, .006],
                                                           "fold_wins": 4, "trimmed_contrast_delta": .002,
                                                           "log_loss_delta": .005}}}}
    assert science.gate(report, "F1", "C2", config["gate"])["passed"]
    report["filtered"]["comparisons"]["F1_minus_C2"]["fold_wins"] = 3
    assert not science.gate(report, "F1", "C2", config["gate"])["passed"]


def test_config_pins_and_protocol(built):
    config = json.loads((ROOT / "configs/exp040.json").read_text(encoding="utf-8"))
    io38.check_config_pins(config)
    assert config["candidates"] == {"C2": ["E1", "E2", "E3"], "F1": ["E1", "E2", "E3L"], "F2": ["E1", "E2", "E3", "E3L"]}
    assert config["lora"] | {} == config["lora"] and config["lora"]["rank"] == 16 and config["lora"]["alpha"] == 32
    assert config["lora"]["targets"] == ["q_proj", "k_proj", "v_proj", "o_proj"] and config["lora"]["dropout"] == .05
    t = config["train"]
    assert (t["lora_lr"], t["head_lr"], t["epochs"], t["batch_paintings"], t["warmup_fraction"]) == (5e-5, 5e-4, 4, 16, .05)
    assert t["lambda"] == 1e-3 and "rationale" in " ".join(config["train"]) and config["head"]["step0_tolerance"] == 1e-5
    assert config["arms"]["E3"]["revision"] == "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
    assert set(config["arms"]) == {"E1", "E2", "E3"}


# Worker paths on the synthetic overlay.

def run(built, tmp_path, **backend):
    import run_exp040

    ctx = fx.context(built, tmp_path)
    return run_exp040, ctx, fx.Backend(ctx, **backend)


def test_full_run_no_go_returns_incumbent_bytes_and_never_leaks(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path)
    with mock.patch.object(plan, "select", lambda gates, tolerance: None):
        decision = run_exp040.run_pipeline(ctx, backend)
    status = read(ctx.root / "status.json")
    assert status["state"] == "COMPLETE" and status["completed_stages"] == list(run_exp040.STAGES)
    assert decision["decision"] == "NO_GO"
    assert sha(ctx.output / "exp040_best_submission.zip") == ctx.refs["exp038_carried_sha256"]["incumbent_zip"]
    validate = read(ctx.root / "validate/validate.json")
    assert all(validate[k]["status"] == "PASS" for k in ("pin_formats", "inputs", "snapshot_2B", "sealed_exp038",
                                                          "key_audit", "head_cpu", "step0_gpu", "r0_replay", "c2_replay"))
    assert validate["step0_gpu"]["result"]["max_abs_probability_difference"] <= 1e-5
    metrics = read(ctx.root / "p2/metrics.json")
    assert all(c["exact"] for c in metrics["c2_replay"])
    gates = read(ctx.root / "p2/gates.json")
    assert set(gates["gates"]) == {"F1", "F2"} and gates["selected_epochs"] == {str(f): 2 for f in range(1, 6)} \
        or gates["selected_epochs"] == {f: 2 for f in range(1, 6)}
    scores = read(ctx.root / "development/scores.json")
    assert {"E3L", "F1", "F2", "B2_F1_cscva3", "B2_F2_cscva3", "B2_C2_cscva3"} <= set(scores["archives"])
    assert all(c["exact"] or c["max_abs_difference"] <= c["threshold"] for c in scores["c2_replay"])
    assert scores["full_refit"]["epoch"] == 2 and (ctx.root / "development/bundle/e3l/e3l.json").is_file()
    # Leakage: training, inner and held-out paintings are disjoint; Development is never a training painting.
    gold = run_exp040.r38.population(ctx)["gold"]
    dev = set(run_exp040.r38.population(ctx)["dev"].painting)
    for body in backend.requests:
        held = set(body["held"])
        assert not held & set(body["train_paintings"]) and not held & set(body["inner"])
        assert not set(body["inner"]) & set(body["train_paintings"])
        if body["mode"] == "fold":
            fold = body["identity"]["fold"]
            assert held == set(gold[gold.fold.eq(fold)].painting)
            assert not dev & (set(body["train_paintings"]) | set(body["inner"]) | held)
        if body["mode"] == "full":
            assert held == dev and not dev & set(body["train_paintings"])
            pairs = pd.read_csv(body["pairs"])
            assert (pairs[pairs.painting.isin(dev)][["k", "n"]] == 0).all().all()
    out = ctx.output
    assert (out / "exp040_results.zip").is_file() and (out / "exp040_SUCCESS.json").is_file()


def test_go_path_uses_b2_of_the_selected_candidate(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path)
    with mock.patch.object(plan, "select", lambda gates, tolerance: "F1"):
        run_exp040.run_pipeline(ctx, backend)
    selection = read(ctx.root / "selection.json")
    assert selection["decision"] == "GO" and selection["submission"] == "B2_F1_cscva3"
    assert sha(ctx.output / "exp040_best_submission.zip") == read(ctx.root / "development/scores.json")["archives"][
        "B2_F1_cscva3"]["sha256"]


def test_epoch_zero_everywhere_makes_f1_equal_c2(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path, epochs={f: 0 for f in range(1, 6)})
    run_exp040.run_pipeline(ctx, backend)
    oof = pd.read_csv(ctx.root / "p2/oof.csv")
    assert np.array_equal(oof.F1.to_numpy(), oof.C2.to_numpy()) and np.array_equal(oof.E3L, oof.E3)
    assert not [c for c in backend.calls if c[:2] == ("e3l", "full")]
    assert read(ctx.root / "decision.json")["decision"] == "NO_GO"


def test_technical_failure_is_a_reported_no_go(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path, fail={("fold", 3): RuntimeError("CUDA out of memory")})
    decision = run_exp040.run_pipeline(ctx, backend)
    assert decision["decision"] == "NO_GO" and "TECHNICAL_FAILURE fold 3" in decision["e3l"]["reason"]
    assert {g["status"] for g in decision["gates"].values()} == {"NOT_AVAILABLE"}
    assert sha(ctx.output / "exp040_best_submission.zip") == ctx.refs["exp038_carried_sha256"]["incumbent_zip"]
    assert "E3L" not in read(ctx.root / "development/scores.json")["archives"]


def test_forecast_over_session_stops_before_training(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path, step_seconds=1e5)
    decision = run_exp040.run_pipeline(ctx, backend)
    assert "FORECAST_EXCEEDS_ALLOWANCE" in decision["e3l"]["reason"]
    assert not [c for c in backend.calls if c[:2] == ("e3l", "fold")]
    assert read(ctx.root / "budget_plan.json")["fits"] is False


def test_smoke_mode_writes_no_submission(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path)
    result = run_exp040.run_pipeline(ctx, backend, smoke=True)
    out = ctx.output
    assert (out / "exp040_smoke.json").is_file() and result["submission_produced"] is False
    assert not list(out.glob("*.zip")) and not (out / "exp040_SUCCESS.json").exists()
    assert read(ctx.root / "status.json")["state"] == "SMOKE_COMPLETE"
    smoke = read(out / "exp040_smoke.json")
    assert smoke["validate"]["r0_replay"] == "PASS" and smoke["validate"]["step0_gpu"] == "PASS"
    assert read(ctx.root / "validate_smoke.json")["r0_replay"]["result"] == {"status": "SKIPPED_SMOKE"}
    assert smoke["forecast"]["fits_session"] and set(smoke["forecast"]["pieces"]) >= {"fold_1", "full_refit"}
    # Step 0 on GPU 0, then one smoke child per GPU at once (fold 1 on GPU 0, fold 2 on GPU 1).
    assert backend.placements[0] == ("step0", None, 0)
    assert sorted(backend.placements[1:]) == [("smoke", 1, 0), ("smoke", 2, 1)]
    assert smoke["concurrent"] and set(smoke["per_gpu"]) == {"0", "1"}
    assert smoke["per_gpu"]["1"]["gpu"] == {"visible_devices": "1"} and smoke["per_gpu"]["1"]["fold"] == 2
    forecast = smoke["forecast"]
    assert {str(g) for g in forecast["queues"]} == {"0", "1"}
    assert forecast["total"] < forecast["serial_total"] and smoke["vision_cache"]["enabled"]


def test_validate_collects_failures_and_step0_mismatch(built, tmp_path):
    import run_exp040

    class Off(fx.Backend):
        def e3l(self, request_path, deadline, gpu=0):
            super().e3l(request_path, deadline, gpu)
            body = json.loads(Path(request_path).read_text(encoding="utf-8"))
            if body["mode"] == "step0":
                path = Path(body["folder"]) / "step0_probabilities.csv"
                frame = pd.read_csv(path)
                frame.loc[0, "p_positive"] += 1e-4
                frame.to_csv(path, index=False, float_format="%.17g")

    ctx = fx.context(built, tmp_path)
    ctx.config["references"] = ctx.config["references"] | {"training_pair_sha256": "c" * 63}
    with pytest.raises(run_exp040.ValidationFailed) as error:
        run_exp040.run_pipeline(ctx, Off(ctx))
    assert any(f.startswith("pin_formats") for f in error.value.failures)
    ctx = fx.context(built, tmp_path / "second")
    with pytest.raises(run_exp040.ValidationFailed, match="step0_gpu: ValueError: Step-0 equality failed"):
        run_exp040.run_pipeline(ctx, Off(ctx))


def test_single_gpu_placement_is_asserted_and_child_sees_one_gpu(built, tmp_path):
    import run_exp040

    module = torch.nn.Linear(2, 2)
    assert gpu.single_gpu(module, "cpu") == ["cpu"]
    with pytest.raises(RuntimeError, match="one GPU"):
        gpu.single_gpu(module, "cuda:0")
    ctx = fx.context(built, tmp_path)
    seen = {}
    with mock.patch.object(run_exp040, "run_child", lambda args, deadline, ctx, env=None, label=None,
                           registry=None: seen.update(env | {"label": label})):
        run_exp040.Backend(ctx).e3l(Path("request.json"), 0., gpu=1)
    assert seen["CUDA_VISIBLE_DEVICES"] == "1" and seen["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and seen["label"] == "gpu1"
    source = (ROOT / "src/st3/exp040_gpu.py").read_text(encoding="utf-8")
    assert 'device_map={"": 0}' in source and "torch.cuda.device_count() != 1" in source


# Two-GPU fold parallelism.

def test_queue_plan_and_parallel_forecast(built, tmp_path):
    import run_exp040

    queues = run_exp040.queue_plan({1: 10., 2: 10., 3: 10., 4: 10., 5: 10.}, [0, 1])
    assert queues == {0: {"folds": [1, 3, 5], "seconds": 30.}, 1: {"folds": [2, 4], "seconds": 20.}}
    queues = run_exp040.queue_plan({1: 30., 2: 10., 3: 10., 4: 10., 5: 10.}, [0, 1])
    assert queues[1]["folds"] == [2, 3, 4] and queues[0]["folds"] == [1, 5]
    assert run_exp040.queue_plan({1: 5., 2: 5.}, [0]) == {0: {"folds": [1, 2], "seconds": 10.}}
    ctx = fx.context(built, tmp_path)
    measured = {0: {"warm_seconds_per_optimizer_step": 2., "inner_paintings_per_second": 4., "load_seconds": 10.},
                1: {"warm_seconds_per_optimizer_step": 3., "inner_paintings_per_second": 2., "load_seconds": 10.}}
    one = run_exp040.forecast(ctx, measured, 100., [0])
    two = run_exp040.forecast(ctx, measured, 100., [0, 1])
    assert two["inputs"]["rates"]["step"] == 3. and two["inputs"]["rates"]["forward"] == .5  # slowest GPU
    folds = [two["pieces"][f"fold_{f}"] for f in range(1, 6)]
    assert one["fold_phase_seconds"] == pytest.approx(sum(folds))
    assert two["fold_phase_seconds"] == pytest.approx(max(q["seconds"] for q in two["queues"].values()))
    after = two["pieces"]["full_refit"] + two["pieces"]["cpu_p2_and_development"] + two["pieces"]["export"]
    assert two["total"] == pytest.approx(two["fold_phase_seconds"] + after)
    assert two["serial_total"] == pytest.approx(one["total"])
    # Revisit rates count only with the vision cache on.
    measured[0] |= {"revisit_seconds_per_optimizer_step": .5, "revisit_inner_paintings_per_second": 40.}
    measured[1] |= {"revisit_seconds_per_optimizer_step": 1., "revisit_inner_paintings_per_second": 20.}
    assert run_exp040.forecast(ctx, measured, 100., [0, 1])["total"] == pytest.approx(two["total"])
    ctx.vision_cache = {"enabled": True}
    cached = run_exp040.forecast(ctx, measured, 100., [0, 1])
    assert cached["inputs"]["rates"]["step_revisit"] == 1. and cached["total"] < two["total"]


def test_folds_run_on_both_gpus_and_full_refit_on_one(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path)
    run_exp040.run_pipeline(ctx, backend)
    folds = [p for p in backend.placements if p[0] == "fold"]
    assert sorted(f for _, f, _ in folds) == [1, 2, 3, 4, 5] and {g for *_, g in folds} == {0, 1}
    assert [p for p in backend.placements if p[0] == "full"] == [("full", None, 0)]
    budget = read(ctx.root / "budget_plan.json")
    assert budget["gpus"] == [0, 1] and budget["fits"] and budget["planned_seconds"] < budget["serial_planned_seconds"]
    queues = {int(g): q["folds"] for g, q in budget["queues"].items()}
    assert sorted(queues[0] + queues[1]) == [1, 2, 3, 4, 5] and abs(len(queues[0]) - len(queues[1])) <= 1
    for fold in range(1, 6):
        result = read(ctx.root / f"p2_folds/fold_{fold}/result.json")
        assert (ctx.root / f"p2_folds/fold_{fold}/seal.json").is_file() and result["gpu_index"] in (0, 1)
    fold_bodies = [b for b in backend.requests if b["mode"] == "fold"]
    assert all(b["vision_cache"]["enabled"] and b["identity"]["vision_cache"] for b in fold_bodies)


def test_sealed_folds_are_reused_whatever_gpu_trained_them(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path)
    run_exp040.run_pipeline(ctx, backend)
    ctx2 = fx.context(built, tmp_path)
    again = fx.Backend(ctx2, gpu_count=1)
    with mock.patch.object(run_exp040, "stage_validate", lambda c, b, s, smoke=False: run_exp040.read(
            ctx.root / "validate/validate.json")):
        run_exp040.run_pipeline(ctx2, again)
    assert not [p for p in again.placements if p[0] in {"fold", "full"}]


def test_one_gpu_falls_back_to_a_single_queue(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path, gpu_count=1)
    run_exp040.run_pipeline(ctx, backend)
    assert {g for *_, g in backend.placements} == {0}
    budget = read(ctx.root / "budget_plan.json")
    assert budget["gpus"] == [0] and budget["planned_seconds"] == pytest.approx(budget["serial_planned_seconds"])


def test_failure_on_one_gpu_cancels_the_other_and_is_a_no_go(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path, fail={("fold", 2): RuntimeError("CUDA error: illegal address")})
    decision = run_exp040.run_pipeline(ctx, backend)
    assert backend.cancels >= 1 and "TECHNICAL_FAILURE fold 2" in decision["e3l"]["reason"]
    assert decision["decision"] == "NO_GO"
    assert sha(ctx.output / "exp040_best_submission.zip") == ctx.refs["exp038_carried_sha256"]["incumbent_zip"]


def test_timeout_on_a_gpu_is_resumable_after_every_child_stops(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path, fail={("fold", 1): TimeoutError("session allowance")})
    with pytest.raises(TimeoutError):
        run_exp040.run_pipeline(ctx, backend)
    assert read(ctx.root / "status.json")["state"] == "RESUMABLE_STOP" and backend.cancels == 0
    # The other GPU finished its fold before the stop; it is sealed and reused on resume.
    sealed = [f for f in range(1, 6) if (ctx.root / f"p2_folds/fold_{f}/seal.json").is_file()]
    assert sealed and 1 not in sealed


def test_run_child_prefixes_each_line_once_and_cancel_kills(built, tmp_path):
    import threading

    import run_exp040

    ctx = fx.context(built, tmp_path)
    ctx.root.mkdir(parents=True)
    run_exp040.run_child(["x"], time.time() + 60, ctx, label="gpu1",
                         executable=[sys.executable, "-c", "print('alpha'); print('beta')"])
    log = (ctx.root / "gpu_children.log").read_text(encoding="utf-8").splitlines()
    assert log == ["[gpu1] alpha", "[gpu1] beta"]

    class Registry:
        def __init__(self):
            self.lock, self.children = threading.Lock(), {}

        def cancel(self):
            with self.lock:
                for child in self.children.values():
                    run_exp040.kill(child)

    registry = Registry()
    timer = threading.Timer(1.5, registry.cancel)
    timer.start()
    tick = time.time()
    with pytest.raises(RuntimeError, match="exited with status"):
        run_exp040.run_child(["x"], time.time() + 60, ctx, label="gpu0", registry=registry,
                             executable=[sys.executable, "-c", "import time; print('up', flush=True); time.sleep(60)"])
    assert time.time() - tick < 30 and not registry.children
    with pytest.raises(TimeoutError):
        run_exp040.run_child(["x"], time.time() - 1, ctx, executable=[sys.executable, "-c", "pass"])


def test_progress_snapshot_leaves_out_in_flight_folds(built, tmp_path):
    import run_exp040

    ctx = fx.context(built, tmp_path)
    for fold, sealed in ((1, True), (2, False)):
        folder = ctx.root / f"p2_folds/fold_{fold}"
        (folder / "epochs").mkdir(parents=True)
        (folder / "epochs/epoch_0.pt").write_bytes(b"w")
        if sealed:
            (folder / "seal.json").write_text("{}", encoding="utf-8")
    (ctx.root / "status.json").write_text("{}", encoding="utf-8")
    run_exp040.progress_sealed(ctx)
    with ZipFile(ctx.output / "exp040_progress.zip") as z:
        names = set(z.namelist())
    assert any("fold_1/epochs/epoch_0.pt" in n for n in names) and not any("fold_2" in n for n in names)


# Vision cache.

def test_vision_cache_decision(built, tmp_path):
    import run_exp040

    ctx = fx.context(built, tmp_path)
    check = {"passes": {"fill": {"bitwise": True}, "hit": {"bitwise": True}}, "max_abs_probability_difference": 0.,
             "tolerance": 1e-5, "passed": True, "cache": {"bytes_per_entry": 3e6, "max_entry_bytes": 4e6}}
    on = run_exp040.vision_cache_decision(ctx, {"vision_cache": check}, [0, 1])
    assert on["enabled"] and on["check"]["bitwise"]
    off = run_exp040.vision_cache_decision(
        ctx, {"vision_cache": check | {"passed": False, "max_abs_probability_difference": 2e-5}}, [0])
    assert not off["enabled"] and "skipped" in off["reason"]
    big = run_exp040.vision_cache_decision(ctx, {"vision_cache": check | {"cache": {"max_entry_bytes": 1e12}}}, [0])
    assert not big["enabled"] and "max_bytes_per_child" in big["reason"]
    with mock.patch.object(run_exp040, "available_memory", lambda: 10_000):
        assert not run_exp040.vision_cache_decision(ctx, {"vision_cache": check}, [0, 1])["enabled"]
    # The peak is two fold children at once or the full refit alone, never the full refit times the GPU count.
    n = run_exp040.counts(ctx)
    fold, full = max(c["embed"] for c in n["folds"].values()), n["train"] + n["dev"]
    assert on["check"]["projected_peak_bytes"] == int(4e6 * max(2 * fold, full))
    assert on["check"]["projected_bytes_per_child"] == int(4e6 * max(fold, full))
    assert not run_exp040.vision_cache_decision(ctx, {}, [0])["enabled"]


def test_cache_mismatch_runs_uncached(built, tmp_path):
    run_exp040, ctx, backend = run(built, tmp_path, cache_difference=2e-5)
    run_exp040.run_pipeline(ctx, backend, smoke=True)
    smoke = read(ctx.output / "exp040_smoke.json")
    assert not smoke["vision_cache"]["enabled"] and smoke["validate"]["step0_gpu"] == "PASS"
    assert all(not b["vision_cache"]["enabled"] for b in backend.requests if b["mode"] == "smoke")
    rates = smoke["forecast"]["inputs"]["rates"]
    assert rates["step_revisit"] == rates["step"] and rates["forward_revisit"] == rates["forward"]


class FakeBackbone(torch.nn.Module):
    """The Qwen3VLModel contract the cache relies on: ``get_image_features`` is called from ``forward`` when
    ``pixel_values`` is not None and returns (per-image embeds, deepstack list); image tokens are replaced."""

    IMAGE = 7

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(16, 8)
        self.visual = torch.nn.Linear(4, 8)
        self.deep = torch.nn.Linear(4, 8)
        self.mix = torch.nn.Linear(8, 8)
        self.calls = 0
        self.eval()

    def get_image_features(self, pixel_values, image_grid_thw=None):
        self.calls += 1
        return (self.visual(pixel_values),), [self.deep(pixel_values)]

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, use_cache=False):
        states = self.embed(input_ids)
        if pixel_values is not None:
            embeds, deep = self.get_image_features(pixel_values, image_grid_thw)
            mask = (input_ids == self.IMAGE).unsqueeze(-1).expand_as(states)
            image = torch.cat(embeds).to(states.dtype) + deep[0]
            states = states.masked_scatter(mask, image)
        return type("Out", (), {"last_hidden_state": self.mix(states)})()


def test_vision_cache_injection_matches_the_uncached_forward():
    backbone = FakeBackbone()
    inputs = {"input_ids": torch.tensor([[1, 7, 7, 7, 2, 3]]), "attention_mask": torch.ones(1, 6, dtype=torch.long),
              "pixel_values": torch.randn(3, 4), "image_grid_thw": torch.tensor([[1, 2, 6]])}
    with torch.no_grad():
        expected = gpu.pooled(backbone, inputs)
    original, slot = gpu.attach_vision_cache(backbone)
    entry = gpu.cache_entry(backbone, original, inputs)
    assert "pixel_values" not in entry["inputs"] and backbone.calls == 2
    with torch.no_grad():
        got = gpu.cached_pooled(backbone, slot, entry, "cpu")
        assert backbone.calls == 2  # injected: the vision tower did not run
        uncached_again = gpu.pooled(backbone, inputs)  # an empty slot falls through to the vision tower
    assert torch.equal(got, expected) and torch.equal(uncached_again, expected) and backbone.calls == 3
    assert slot["value"] is None
    cache = gpu.VisionCache(max_bytes=gpu.VisionCache.size(entry))
    cache.put("a", entry)
    cache.put("b", entry)
    assert cache.get("a") is entry and cache.get("b") is None
    receipt = cache.receipt()
    assert {k: receipt[k] for k in ("entries", "overflow", "hits", "misses")} == \
        {"entries": 1, "overflow": 1, "hits": 1, "misses": 1}
    assert receipt["max_entry_bytes"] == gpu.VisionCache.size(entry)


def test_vision_cache_on_a_tiny_qwen3_vl():
    """The real Qwen3-VL forward (random tiny config, CPU): the cached injection equals the uncached path."""
    qwen = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
    from transformers import Qwen3VLConfig

    config = Qwen3VLConfig(
        text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True}},
        vision_config={"depth": 3, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "patch_size": 4,
                       "spatial_merge_size": 2, "temporal_patch_size": 2, "out_hidden_size": 32,
                       "num_position_embeddings": 64, "deepstack_visual_indexes": [0, 1]},
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    torch.manual_seed(0)
    model = qwen.Qwen3VLForConditionalGeneration(config).eval()
    backbone = model.model
    grid = torch.tensor([[1, 4, 4]])
    tokens = int(grid.prod()) // 4
    ids = torch.tensor([[1, 2, 62, *([60] * tokens), 63, 3, 4]])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "image_grid_thw": grid,
              "pixel_values": torch.randn(int(grid.prod()), 3 * 2 * 4 * 4)}
    with torch.no_grad():
        expected = gpu.pooled(backbone, inputs)
        original, slot = gpu.attach_vision_cache(backbone)
        entry = gpu.cache_entry(backbone, original, inputs)
        got = gpu.cached_pooled(backbone, slot, entry, "cpu")
    assert len(entry["deep"]) == 2 and entry["embeds"].shape == (tokens, 32)
    assert torch.equal(got, expected)


def test_fit_and_predict_refuses_overlap(built, tmp_path):
    import run_exp040

    ctx = fx.context(built, tmp_path)
    ctx.identity = {}
    gold = run_exp040.r38.population(ctx)["gold"]
    features = run_exp040.base_features(ctx)
    with pytest.raises(ValueError, match="overlap"):
        plan.fit_and_predict(gold, gold[gold.fold.eq(1)][data.KEYS], plan.with_e3l(features, features["E3"]), None,
                             ctx.track1, ctx.neighborhood, 1)


# Packaging.

def test_packager_refuses_archives_and_survives_recursive_expansion(built, tmp_path):
    import build_exp040_notebook

    with pytest.raises(ValueError, match="must not be archives"):
        io38.refuse_archives({"exp038/features/E3/features.npz": b"PK\x03\x04"})
    stored = [p for p in built["mounted"].rglob("*") if p.is_file()]
    assert stored and not [p for p in stored if io38.archive_like(p.name, p.read_bytes())]
    assert not [n for n in built["members"] if n.startswith(("keys/E5", "runtime/scripts/run_exp039"))]
    archive = fx.zip_dir(built["mounted"], tmp_path / "upload" / "exp040_inputs.zip", "exp040_inputs/")
    mount = tmp_path / "input" / "datasets" / "o" / "exp040-inputs"
    with ZipFile(archive) as z:
        z.extractall(mount)
    scope = {}
    exec(build_exp040_notebook.DEFS, scope)
    for where in (tmp_path / "input", tmp_path / "upload"):
        root, _ = scope["locate"](where, tmp_path / f"w_{where.name}", "exp040_manifest.json", built["overlay_sha"],
                                  "exp040_inputs.zip", sha(archive), "exp040_inputs")
        restored = io38.restore_overlay(root, built["overlay_sha"], built["base_sha"], tmp_path / f"r_{where.name}",
                                        manifest_name="exp040_manifest.json")
        for name in ("exp038/incumbent_submission.zip", "exp038/features/E3/features.npz"):
            assert (restored / name).read_bytes() == built["members"][name]
    source = (ROOT / "scripts/package_exp040.py").read_text(encoding="utf-8")
    assert "check_config_pins(config)" in source and "archive_like(name, z.read(name))" in source


def test_notebook_has_smoke_only_and_single_relay():
    import build_exp040_notebook

    notebook = build_exp040_notebook.build({"source": "s", "base_manifest_sha256": "a" * 64, "base_zip_sha256": "b" * 64,
                                            "manifest_sha256": "c" * 64, "archive_sha256": "d" * 64})
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "SMOKE_ONLY = True" in text and 'command.append("--smoke")' in text and "AUTHORIZE_RUN = False" in text
    assert "exp039" not in text
    launcher = (ROOT / "scripts/exp040_kaggle.py").read_text(encoding="utf-8")
    assert "exp039" not in launcher and "google.colab" not in launcher and 'worker.append("--smoke")' in launcher
