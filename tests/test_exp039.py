"""CPU/synthetic tests for EXP-039 (GPU children mocked; no downloads)."""

import ast
import hashlib
import io
import json
import sys
from pathlib import Path
from unittest import mock
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp039_fixture as fx  # noqa: E402
from st3 import data, submission  # noqa: E402
from st3 import exp038 as science  # noqa: E402
from st3 import exp038_io as io38  # noqa: E402
from st3 import exp039 as plan39  # noqa: E402
from st3.exp036_io import read, sha  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("exp039"))


def run(built, tmp_path, **backend):
    import run_exp039

    ctx = fx.context(built, tmp_path)
    return run_exp039, ctx, fx.Backend(ctx, **backend)


# Config, pins and candidates.

def test_config_is_generated_with_well_formed_pins_and_one_change_per_arm():
    config = json.loads((ROOT / "configs/exp039.json").read_text())
    assert io38.check_config_pins(config)
    e3, e5, e6, e7 = (config["arms"][a] for a in ("E3", "E5", "E6", "E7"))
    assert e5["model"] == "Qwen/Qwen3-VL-Embedding-8B" and len(e5["revision"]) == 40 and e5["license"] == "apache-2.0"
    assert e5["max_pixels"] == e3["max_pixels"] and e5["instruction"] == e3["instruction"]
    assert e6["instructions"][0] == e3["instruction"] and len(e6["instructions"]) == 4
    assert e7["max_pixels"] == 1048576 and e7["min_pixels"] == e3["min_pixels"] and e7["instruction"] == e3["instruction"]
    assert config["gate"]["reference"] == "C2" and config["gate"]["minimum_filtered_gain"] == 0.003
    assert config["budget"]["skip_order"] == ["E5", "E7"] and config["budget"]["mandatory"] == ["R0", "C2_replay"]
    assert config["references"]["exp038_carried_sha256"]["incumbent_zip"].startswith("62093a31")


def test_make_config_refuses_drifted_2b_pins(built):
    import make_exp039_config

    exp038 = json.loads((ROOT / "configs/exp038.json").read_text())
    e3 = exp038["arms"]["E3"]
    wrong = {"revision": e3["revision"], "weights_sha256": e3["weights_sha256"],
             "metadata_git_blobs": e3["metadata_git_blobs"] | {"config.json": "0" * 40}}
    with pytest.raises(ValueError, match="2B pins differ"):
        make_exp039_config.build(exp038, built["results38"], fx.fake_e5_pins(), wrong)


def test_pin_generator_uses_the_tree_api_and_rejects_bad_hashes():
    import pin_hf_snapshot

    responses = {
        "/revision/main": {"sha": "d" * 40, "cardData": {"license": "apache-2.0"}},
        "/tree/" + "d" * 40: [{"type": "file", "path": "model.safetensors", "size": 3, "oid": "e" * 40,
                               "lfs": {"oid": "f" * 64}},
                              {"type": "file", "path": "config.json", "size": 2, "oid": "a" * 40}]}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def opener(url, timeout):
        key = next(k for k in responses if k in url)
        return Response(json.dumps(responses[key]).encode())

    result = pin_hf_snapshot.pins("m/x", "main", ["model.safetensors"], ["config.json"], opener=opener)
    assert result["revision"] == "d" * 40 and result["weights_sha256"]["model.safetensors"] == "f" * 64
    assert result["metadata_git_blobs"]["config.json"] == "a" * 40 and result["license"] == "apache-2.0"
    responses["/tree/" + "d" * 40][0]["lfs"]["oid"] = "f" * 63
    with pytest.raises(ValueError, match="Malformed"):
        pin_hf_snapshot.pins("m/x", "main", ["model.safetensors"], ["config.json"], opener=opener)


def test_d4_vlm_third_formula_and_partial_availability():
    frame = pd.DataFrame({"painting": ["a"], "language": ["x"], "clip": [.2], "siglip2": [.4], "clip_large": [.9],
                          "E1": [.1], "E2": [.3], "E3": [.6], "E6": [.7]})
    out = science.mix_columns(frame, {"D4": science.VlmThird(("E1", "E2"), ("E3", "E6"))})
    fixed = out.fixed.iloc[0]
    assert out.D4.iloc[0] == round((fixed + round(.2, 6) + round(.65, 6)) / 3, 6)
    config = json.loads((ROOT / "configs/exp039.json").read_text())
    got = plan39.candidate_set(config, ("E1", "E2", "E3", "E6"))
    assert set(got) == {"C2", "D2", "D4"} and tuple(got["D4"]) == ("E1", "E2", "E3", "E6")
    assert "D4" not in plan39.candidate_set(config, ("E1", "E2", "E3"))
    nf4 = plan39.candidate_set(config, ("E1", "E2", "E3", "E5_nf4"), "E5_nf4")
    assert nf4["D1"] == ("E1", "E2", "E5_nf4") and "E5_nf4" in tuple(nf4["D4"])


def test_e6_pooling_and_items():
    e3 = np.array([[1., 0.], [0., 1.]])
    raw = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    pooled = plan39.pooled_instructions(e3, raw, 4)
    np.testing.assert_allclose(np.linalg.norm(pooled, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(pooled[1], [0, 1], atol=1e-6)
    assert plan39.pool_items([("s", "p")], 4) == [("s|1", "p", 1), ("s|2", "p", 2), ("s|3", "p", 3)]


def test_selection_tie_rules():
    config = json.loads((ROOT / "configs/exp039.json").read_text())

    def g(value, ok=True):
        return {"passed": ok, "numbers": {"filtered_contrast": value}}

    assert plan39.select({"D1": g(.4700), "D2": g(.4695)}, config, {}) == "D2"
    assert plan39.select({"D1": g(.4720), "D2": g(.4700)}, config, {}) == "D1"
    assert plan39.select({"D2": g(.470), "D3": g(.4705)}, config, {"E6": 900, "E7": 400}) == "D3"
    assert plan39.select({"D2": g(.470), "D3": g(.4705)}, config, {"E6": 300, "E7": 400}) == "D2"
    assert plan39.select({"D4": g(.4709), "D1": g(.4700)}, config, {}) == "D1"
    assert plan39.select({"D1": g(.48, False)}, config, {}) is None


def test_gate_uses_the_exp039_threshold_and_seed():
    config = json.loads((ROOT / "configs/exp039.json").read_text())
    report = {"full": {"comparisons": {"D2_minus_C2": {"pooled_contrast_delta": 0., "log_loss_delta": 0.}}},
              "filtered": {"comparisons": {"D2_minus_C2": {"pooled_contrast_delta": .0030, "bootstrap_95": [1e-4, .01],
                                                            "fold_wins": 4, "trimmed_contrast_delta": 1e-4,
                                                            "log_loss_delta": 0.}},
                           "systems": {"D2": {"pooled": {"cultural_contrast": .47}}}}}
    assert science.gate(report, "D2", "C2", config["gate"])["passed"]
    report["filtered"]["comparisons"]["D2_minus_C2"]["pooled_contrast_delta"] = .0029
    assert not science.gate(report, "D2", "C2", config["gate"])["passed"]
    delta = np.random.default_rng(0).normal(.003, .02, 300)
    from st3.competition_stack import compare

    assert compare(delta, config["gate"]) == compare(delta, config["gate"]) and config["gate"]["bootstrap_draws"] == 20000


# Full worker.

def test_happy_path_no_go_returns_incumbent_bytes(built, tmp_path):
    run_exp039, ctx, backend = run(built, tmp_path)
    decision = run_exp039.run_pipeline(ctx, backend)
    status = read(ctx.root / "status.json")
    assert status["state"] == "COMPLETE" and status["completed_stages"] == list(run_exp039.STAGES)
    assert [c[1] for c in backend.calls if c[0] == "extract"] == ["E5", "E6", "E7"]
    assert {"fetch"} <= {c[0] for c in backend.calls}
    validate = read(ctx.root / "validate/validate.json")
    assert validate["c2_replay"]["status"] == "PASS" and validate["c2_replay"]["result"]["exact"]
    assert all(validate[k]["status"] == "PASS" for k in ("pin_formats", "inputs", "snapshot_2B", "snapshot_8B",
                                                          "sealed_exp038", "r0_replay"))
    gates = read(ctx.root / "p2/gates.json")["gates"]
    assert set(gates) == {"D1", "D2", "D3", "D4"}
    metrics = read(ctx.root / "p2/metrics.json")
    assert metrics["gate_reference"].startswith("on-platform C2") and all(c["exact"] for c in metrics["c2_replay"])
    scores = read(ctx.root / "development/scores.json")
    assert {"B2_D1_cscva3", "B2_D4_cscva3", "B2_C2_cscva3", "E5", "E6", "E7", "D4"} <= set(scores["archives"])
    assert {"D2_plus_C6", "C2_plus_C6"} <= set(scores["diagnostic_not_selectable"])
    assert scores["archives"]["E5"]["eligibility"] == "UNCONFIRMED_QWEN_ST3"
    diagnostics = read(ctx.root / "p2/diagnostics.json")
    assert {"E5", "E6", "E7"} <= set(diagnostics["vs_E3"]["filtered"])
    out = ctx.output
    assert (out / "exp039_results.zip").is_file() and (out / "exp039_SUCCESS.json").is_file()
    if decision["decision"] == "NO_GO":
        assert sha(out / "exp039_best_submission.zip") == ctx.refs["exp038_carried_sha256"]["incumbent_zip"]
    second = fx.Backend(ctx)
    (out / "exp039_SUCCESS.json").unlink()
    run_exp039.run_pipeline(ctx, second)
    assert not [c for c in second.calls if c[0] in {"extract", "benchmark"}]


def test_go_path_uses_b2_of_the_selected_candidate(built, tmp_path):
    run_exp039, ctx, backend = run(built, tmp_path)
    with mock.patch.object(plan39, "select", lambda gates, config, seconds: "D2"):
        run_exp039.run_pipeline(ctx, backend)
    selection = read(ctx.root / "selection.json")
    assert selection["decision"] == "GO" and selection["submission"] == "B2_D2_cscva3"
    assert sha(ctx.output / "exp039_best_submission.zip") == read(ctx.root / "development/scores.json")["archives"][
        "B2_D2_cscva3"]["sha256"]


def test_single_gpu_runs_e5_as_nf4_and_labels_every_output(built, tmp_path):
    run_exp039, ctx, backend = run(built, tmp_path, gpu_count=1)
    run_exp039.run_pipeline(ctx, backend)
    assert ctx.e5 == "E5_nf4" and run_exp039.spec(ctx, "E5_nf4")["placement"] == "nf4"
    assert (ctx.root / "features/E5_nf4/seal.json").is_file() and not (ctx.root / "features/E5").exists()
    scores = read(ctx.root / "development/scores.json")
    assert "E5_nf4" in scores["archives"] and "E5" not in scores["archives"]
    assert read(ctx.root / "p2/metrics.json")["candidates"]["D1"] == ["E1", "E2", "E5_nf4"]
    assert read(ctx.root / "selection.json")["e5_label"] == "E5_nf4"


def test_two_gpu_placement_and_child_environment(built, tmp_path):
    import run_exp039

    ctx = fx.context(built, tmp_path)
    ctx.identity = {}
    backend = run_exp039.Backend(ctx)
    assert run_exp039.spec(ctx, "E5")["placement"] == "two_gpu"
    assert backend._env("E5")["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert backend._env("E6")["CUDA_VISIBLE_DEVICES"] == "0" and backend._env("E7")["CUDA_VISIBLE_DEVICES"] == "0"


def test_failure_isolation_and_partial_d4(built, tmp_path):
    run_exp039, ctx, backend = run(built, tmp_path, fail={"E6": RuntimeError("E6 crash"),
                                                         ("benchmark", "E7"): RuntimeError("E7 OOM")})
    run_exp039.run_pipeline(ctx, backend)
    pieces = read(ctx.root / "status.json")["pieces"]
    assert pieces["E6"]["state"] == "FAILED_OPTIONAL" and pieces["E7"]["state"] == "FAILED_OPTIONAL"
    assert "benchmark" in pieces["E7"]["reason"] and pieces["E5"]["state"] == "COMPLETE"
    gates = read(ctx.root / "p2/gates.json")["gates"]
    assert gates["D2"]["status"] == "NOT_AVAILABLE" and gates["D3"]["status"] == "NOT_AVAILABLE"
    assert gates["D2"]["reason"] == "NOT_AVAILABLE (E6 FAILED_OPTIONAL)"
    assert gates["D4"]["status"] in {"PASS", "FAIL"}
    assert read(ctx.root / "p2/metrics.json")["candidates"]["D4"] == ["E1", "E2", "E3", "E5"]
    assert set(read(ctx.root / "decision.json")["unavailable_candidates"]) == {"D2", "D3"}


def test_all_new_arms_failing_leaves_c2_and_no_go(built, tmp_path):
    run_exp039, ctx, backend = run(built, tmp_path, fail={a: RuntimeError("x") for a in ("E5", "E6", "E7")},
                                   nan_arms=())
    decision = run_exp039.run_pipeline(ctx, backend)
    gates = read(ctx.root / "p2/gates.json")["gates"]
    assert all(gates[d]["status"] == "NOT_AVAILABLE" for d in ("D1", "D2", "D3", "D4"))
    assert decision["decision"] == "NO_GO"
    assert sha(ctx.output / "exp039_best_submission.zip") == ctx.refs["exp038_carried_sha256"]["incumbent_zip"]


def test_nan_embedding_is_an_optional_failure(built, tmp_path):
    run_exp039, ctx, backend = run(built, tmp_path, nan_arms=("E7",))
    run_exp039.run_pipeline(ctx, backend)
    assert "FloatingPointError" in read(ctx.root / "status.json")["pieces"]["E7"]["reason"]


def test_budget_skips_e5_first_then_e7(built, tmp_path):
    run_exp039, ctx, backend = run(built, tmp_path, rates={"E5": 1e-3, "E6": 1e4, "E7": 1e4})
    ctx.clock.deadline = ctx.clock.started + 3 * 3600
    run_exp039.run_pipeline(ctx, backend)
    plan = read(ctx.root / "budget_plan.json")
    assert list(plan["skipped"]) == ["E5"]
    assert read(ctx.root / "status.json")["pieces"]["E5"]["state"] == "BUDGET_SKIPPED"
    assert not [c for c in backend.calls if c[:2] == ("extract", "E5")]


def test_validate_collects_every_independent_failure(built, tmp_path):
    def change(members, config):
        config["arms"]["E5"]["metadata_git_blobs"]["config.json"] = "b" * 39
        members["exp038/features/E2/seal.json"] = members["exp038/features/E2/seal.json"] + b" "

    changed = fx.rebuild(built, tmp_path, change)
    import run_exp039

    ctx = fx.context(changed, tmp_path)
    backend = fx.Backend(ctx, fail={("fetch", "E5"): ValueError("Pinned metadata differs for 8B")})
    with pytest.raises(run_exp039.ValidationFailed) as error:
        run_exp039.run_pipeline(ctx, backend)
    text = str(error.value)
    assert "pin_formats" in text and "snapshot_8B" in text and "sealed_exp038" in text
    assert len(error.value.failures) == 3
    receipts = read(ctx.root / "validate_failures.json")
    assert receipts["r0_replay"]["status"] == "PASS" and receipts["c2_replay"]["status"] == "SKIPPED_DEPENDENCY"
    assert not [c for c in backend.calls if c[0] in {"benchmark", "extract"}]
    status = read(ctx.root / "status.json")
    assert status["state"] == "FAILED" and status["stage"] == "Validate" and len(status["failures"]) == 3
    assert (ctx.output / "exp039_diagnostic.zip").is_file() and not (ctx.output / "exp039_best_submission.zip").exists()


def shift(column, delta, rows=None):
    def change(members, config):
        frame = pd.read_csv(io.BytesIO(members["exp038/p2_oof.csv"]))
        index = frame.index if rows is None else frame.index[:rows]
        frame.loc[index, column] = (frame.loc[index, column] + delta).clip(0, 1).round(6)
        content = frame.to_csv(index=False, lineterminator="\n", float_format="%.6f").encode()
        members["exp038/p2_oof.csv"] = content
        config["references"]["exp038_carried_sha256"]["p2_oof"] = hashlib.sha256(content).hexdigest()
    return change


def test_c2_replay_tolerance_pass_and_fail(built, tmp_path):
    import run_exp039

    ok = fx.rebuild(built, tmp_path / "ok", shift("C2", 5e-6))
    ctx = fx.context(ok, tmp_path / "ok")
    ctx.identity = {}
    frame, *_ = run_exp039.c2_p2(ctx)
    check = next(c for c in run_exp039.c2_p2_checks(ctx, frame) if c["check"] == "C2 replay P2 C2")
    assert not check["exact"] and check["max_abs_difference"] == pytest.approx(5e-6, abs=1e-9)
    bad = fx.rebuild(built, tmp_path / "bad", shift("C2", 5e-5, rows=1))
    ctx = fx.context(bad, tmp_path / "bad")
    with pytest.raises(run_exp039.ValidationFailed, match=r"c2_replay: ReplayError: C2 replay P2 C2"):
        run_exp039.run_pipeline(ctx, fx.Backend(ctx))


# Packaging, layouts, logging, deployment.

def test_packager_refuses_archives_and_overlay_has_none(built):
    with pytest.raises(ValueError, match="must not be archives"):
        io38.refuse_archives({"exp038/features/E1/features.npz": b"PK\x03\x04"})
    with pytest.raises(ValueError, match="must not be archives"):
        io38.refuse_archives({"exp038/x.zip": b"text"})
    stored = [p for p in built["mounted"].rglob("*") if p.is_file()]
    assert stored and not [p for p in stored if io38.archive_like(p.name, p.read_bytes())]
    encoded = read(built["mounted"] / "exp039_manifest.json")["encoded"]
    assert {"exp038/features/E3/features.npz", "exp038/incumbent_submission.zip",
            "exp038/CSCVA_3_submission.zip"} <= set(encoded)
    source = (ROOT / "scripts/package_exp039.py").read_text(encoding="utf-8")
    assert "check_config_pins(config)" in source and "archive_like(name, z.read(name))" in source


def test_overlay_survives_simulated_recursive_expansion(built, tmp_path):
    import build_exp039_notebook

    archive = tmp_path / "upload" / "exp039_inputs.zip"
    archive.parent.mkdir()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as z:
        for path in sorted(built["mounted"].rglob("*")):
            if path.is_file():
                z.write(path, "exp039_inputs/" + path.relative_to(built["mounted"]).as_posix())
    mount = tmp_path / "input" / "datasets" / "o" / "exp039-inputs"
    with ZipFile(archive) as z:
        z.extractall(mount)
    scope = {}
    exec(build_exp039_notebook.DEFS, scope)
    for where in (tmp_path / "input", tmp_path / "upload"):
        root, _ = scope["locate"](where, tmp_path / f"w_{where.name}", "exp039_manifest.json", built["overlay_sha"],
                                  "exp039_inputs.zip", sha(archive), "exp039_inputs")
        restored = io38.restore_overlay(root, built["overlay_sha"], built["base_sha"], tmp_path / f"r_{where.name}",
                                        manifest_name="exp039_manifest.json")
        for name in ("exp038/incumbent_submission.zip", "exp038/features/E3/features.npz"):
            assert (restored / name).read_bytes() == built["members"][name]


def test_notebook_stream_prints_each_line_once(capfd):
    import build_exp039_notebook

    scope = {}
    exec(build_exp039_notebook.DEFS, scope)
    lines = []
    scope["stream"]([sys.executable, "-c", "print('alpha'); print('beta')"], write=lines.append)
    out = capfd.readouterr().out
    assert lines == ["alpha\n", "beta\n"] and "alpha" not in out
    with pytest.raises(Exception):
        scope["stream"]([sys.executable, "-c", "import sys; sys.exit(3)"], write=lines.append)
    notebook = build_exp039_notebook.build({"source": "a" * 40, "manifest_sha256": "b" * 64, "archive_sha256": "c" * 64,
                                            "base_zip_sha256": "d" * 64, "base_manifest_sha256": "e" * 64})
    code = [("".join(c["source"])) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert "AUTHORIZE_RUN = False" in code[0] and "stream(command" in code[3]
    assert "subprocess.run(command" not in "".join(code)
    for cell in code:
        ast.parse(cell)


def test_launcher_relay_writes_each_line_once_and_never_touches_colab(tmp_path):
    import exp039_kaggle

    class Child:
        stdout = iter(["one\n", "two\n"])

    printed = []
    with (tmp_path / "log").open("w", encoding="utf-8") as stream:
        exp039_kaggle.relay(Child(), stream, printed.append)
    assert printed == ["one\n", "two\n"] and (tmp_path / "log").read_text() == "one\ntwo\n"
    env = exp039_kaggle.environment(tmp_path)
    assert "CUDA_VISIBLE_DEVICES" not in env and env["OPENBLAS_CORETYPE"] == "Haswell"
    for name in ("exp039_kaggle.py", "run_exp039.py", "deploy_exp039.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
        modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any("colab" in m for m in modules)


def test_deploy_selected_only_reproduces_the_run(built, tmp_path):
    import deploy_exp039
    import run_exp039

    run_exp039_, ctx, backend = run(built, tmp_path)
    with mock.patch.object(plan39, "select", lambda gates, config, seconds: "D2"):
        run_exp039.run_pipeline(ctx, backend)
    bundle = tmp_path / "deploy"
    deploy_exp039.prepare(type("A", (), {"results": ctx.output / "exp039_results.zip", "output": bundle,
                                         "identity": built["overlay_sha"]})())
    dev = data.read_dev_input(built["base"])
    features = run_exp039.features_for(ctx, ("E1", "E2", "E3", "E6"))
    names = sorted(dev.painting.unique())
    np.savez(tmp_path / "f.npz", paintings=np.asarray(names),
             **{n: features[n].loc[names].to_numpy() for n in ["clip", "siglip2", "clip_large", "E1", "E2", "E3", "E6"]})
    cs3 = run_exp039.carried(ctx, "CSCVA_3_submission.zip")
    c6 = submission.read_predictions(ctx.overlay / "reference/exp037_c6_standalone/predictions.csv", dev)
    args = type("A", (), {"bundle": bundle, "input": built["base"] / data.DEV_INPUT, "images": built["base"],
                          "output": tmp_path / "jan", "features": tmp_path / "f.npz", "cscva": [cs3],
                          "c6": io38.write_submission(c6, dev, tmp_path / "c6", "c6"), "selected_only": True})()
    deploy_exp039.compose(args)
    receipt = read(tmp_path / "jan/inference_receipt.json")
    assert receipt["arms_extracted"] == ["E1", "E2", "E6"]
    run_zip = ctx.root / read(ctx.root / "development/scores.json")["archives"]["B2_D2_cscva3"]["zip"]
    assert io38.zip_member(tmp_path / "jan/exp039_best_submission.zip") == io38.zip_member(run_zip)
