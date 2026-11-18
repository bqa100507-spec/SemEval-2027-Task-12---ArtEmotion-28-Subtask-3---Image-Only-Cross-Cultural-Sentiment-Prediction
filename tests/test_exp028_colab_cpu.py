"""CPU tests for the Drive wrapper. All model runs here use synthetic rows."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from st3 import exp028_cache as cache
from st3.exp028_stage1 import ARMS, matched_reference, run_screen
from st3.vlm_semantics import file_hash
from test_exp028_stage1 import make_data
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_wrapper", ROOT / "scripts/run_exp028_colab_cpu.py")
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


def test_source_pin_rejects_wrong_revision():
    import subprocess
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with pytest.raises(ValueError, match="Git revision"):
        cpu.pinned_source("0" * 40)
    # Frozen scientific files must remain byte-identical, even in a dirty test checkout.
    for name in cpu.SCIENCE:
        prior = subprocess.check_output(["git", "show", f"{cpu.BASE}:{name}"], cwd=ROOT)
        assert cpu.expected_science_bytes(name, prior) == (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    assert len(head) == 40


def test_missing_dependency_and_bad_input_zip_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu.platform, "python_version", lambda: "3.12.10")

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cpu.importlib.metadata, "version", missing)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        cpu.dependency_receipt()
    wrong = tmp_path / "wrong.zip"
    wrong.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Input ZIP SHA"):
        cpu.prepare_inputs(wrong, tmp_path / "inputs", tmp_path / "workspace")


def test_portable_npz_recreates_frozen_header(tmp_path):
    output = tmp_path / "features.npz"
    names = np.array(["a", "abc"], dtype="<U3")
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    cpu.write_original_npz(output, names, data)
    with zipfile.ZipFile(output) as archive:
        assert [(i.filename, i.create_system, i.create_version) for i in archive.infolist()] == [
            ("names.npy", 0, 45), ("features.npy", 0, 45),
        ]
    with np.load(output, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["names"], names)
        np.testing.assert_array_equal(arrays["features"], data)


def test_inputs_restore_only_explicit_members_and_verify_hashes(tmp_path):
    import hashlib
    z = tmp_path / "handoff.zip"
    with zipfile.ZipFile(z, "w") as out:
        out.writestr("training/pairs.csv", b"pairs")
        out.writestr("confirmation/unused.csv", b"unread")
        out.writestr("evaluation/unused.csv", b"unread")
    restore = tmp_path / "restore"
    with zipfile.ZipFile(z) as source:
        cpu.restore_member(source, {"training/pairs.csv": hashlib.sha256(b"pairs").hexdigest()},
                           "training/pairs.csv", restore)
        with pytest.raises(ValueError, match="SHA-256"):
            cpu.restore_member(source, {"training/pairs.csv": "bad"}, "training/pairs.csv", restore)
    assert not (restore / "confirmation").exists()
    assert not (restore / "evaluation").exists()
    assert (restore / "training/pairs.csv").read_bytes() == b"pairs"


def test_missing_and_invalid_qwen_cache_do_not_reach_worker(tmp_path):
    rows = pd.DataFrame({"image_sha256": ["a" * 64]})
    folder = tmp_path / "cache"
    folder.mkdir()
    path = folder / "identity.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        cpu.preflight_cache(folder, rows, tmp_path / "audit.json")
    path.write_text(json.dumps(cache.expected_identity(ROOT)))
    with pytest.raises(ValueError, match="lacks 2"):
        cpu.preflight_cache(folder, rows, tmp_path / "audit.json")
    for suffix in ("vision", "states"):
        (folder / f"{'a' * 64}_{suffix}.npz").touch()
    cpu.preflight_cache(folder, rows, tmp_path / "audit.json")
    assert cpu.json_read(tmp_path / "audit.json") == {"required_NPZs": 2, "missing_NPZs": []}


def test_ram_shortage_saves_receipt_and_blocks_before_fitting(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    args = SimpleNamespace(attempt=attempt, audit_only=False, revision="test", scratch=tmp_path / "scratch",
                           handoff=tmp_path / "input.zip", qwen_cache=tmp_path / "cache", seconds=99)
    monkeypatch.setattr(cpu, "pinned_source", lambda rev: {})
    monkeypatch.setattr(cpu, "dependency_receipt", lambda: {})
    monkeypatch.setattr(cpu, "prepare_inputs", lambda *args: {"ok": True})
    monkeypatch.setattr(cpu, "preflight_cache", lambda *a: None)
    monkeypatch.setattr(cache, "read_protocol", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(cpu, "memory_report", lambda n: {"pass": False, "effective_available_bytes": 1})
    monkeypatch.setattr(cpu, "guarded_worker", lambda *a: pytest.fail("Fit must never start"))
    assert cpu.run(args) == 2
    assert cpu.json_read(attempt / "workflow_status.json")["failed_stage"] == "RAM_PREFLIGHT"
    assert cpu.json_read(attempt / "ram_preflight.json")["effective_available_bytes"] == 1


@pytest.fixture
def completed_screen(tmp_path, monkeypatch):
    rows, _, qwen, blocks, example = make_data()
    _, _, fit, held = cache.partition(rows, example)
    with threadpool_limits(limits=int(os.environ.get("ST3_EXP028_CPU_THREADS", "4"))):
        historical, _ = matched_reference(fit, held, blocks)
    historical["fold"] = 3
    monkeypatch.setattr("st3.exp028_stage1.psutil.virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3))
    preds, evaluation, paintings, fits = run_screen(rows, blocks, qwen, historical, example)
    inputs, results = tmp_path / "inputs", tmp_path / "results"
    (inputs / "history").mkdir(parents=True)
    results.mkdir()
    historical.to_csv(inputs / "history/exp022_R0.csv", index=False)
    monkeypatch.setattr(cache, "read_protocol", lambda *a, **k: rows)
    original = cpu.json_read
    monkeypatch.setattr(cpu, "json_read", lambda p: example if p.name == "exp028_design_spec.json" else original(p))
    for name, frame in preds.items():
        frame.to_csv(results / f"{name}_predictions.csv", float_format="%.6f", index=False)
        paintings[name].to_csv(results / f"{name}_paintings.csv")
    for name in ("manifest.json", "cache_audit.json", "missing_pairs.json"):
        cpu.json_write(results / name, {})
    cpu.json_write(results / "fits.json", fits)
    cpu.json_write(results / "evaluation.json", evaluation)
    hashes = {p.name: file_hash(p) for p in results.iterdir()}
    cpu.json_write(results / "status.json", {"status": "FOLD3_SCREEN_COMPLETE / EXPLORATORY", "files_sha256": hashes})
    return results, inputs, evaluation


def test_verified_synthetic_fold3_recomputes_metrics_bootstrap_and_csv(completed_screen):
    output, inputs, evaluation = completed_screen
    checked = cpu.verify_screen(output, inputs)
    assert checked == evaluation
    summary = cpu.compact_table(checked)
    assert all(f"| {name} |" in summary for name in (*ARMS, "E50+L"))
    assert "C512 vs C512_shuffled_Q" in summary


@pytest.mark.parametrize("failure", ["predictions", "numerical", "convergence", "historical"])
def test_failed_science_receipt_is_rejected(completed_screen, failure):
    output, inputs, _ = completed_screen
    status = cpu.json_read(output / "status.json")
    if failure == "predictions":
        with (output / "C512_predictions.csv").open("a") as handle:
            handle.write("changed")
        match = "artifact SHA"
    elif failure == "historical":
        historical = inputs / "history/exp022_R0.csv"
        frame = pd.read_csv(historical)
        frame.loc[0, "p_positive"] = .999993
        frame.to_csv(historical, index=False)
        match = "six-decimal"
    else:
        fit = cpu.json_read(output / "fits.json")
        field, value = ("converged", False) if failure == "convergence" else ("objective_parity_abs", .001)
        fit["C512"]["refit"][field] = value
        cpu.json_write(output / "fits.json", fit)
        status["files_sha256"]["fits.json"] = file_hash(output / "fits.json")
        cpu.json_write(output / "status.json", status)
        match = "convergence or numerical"
    with pytest.raises(ValueError, match=match):
        cpu.verify_screen(output, inputs)
