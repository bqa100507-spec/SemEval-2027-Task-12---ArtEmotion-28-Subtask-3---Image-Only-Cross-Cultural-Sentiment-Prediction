"""Exercise the CPU notebook with simulated Colab IO and process outcomes."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import pytest

from st3.experiment_workflow import validate_notebook_contract

ROOT = Path(__file__).resolve().parents[1]


def load_notebook():
    notebook = json.loads((ROOT / "notebooks/EXP028_Stage1_CPU_Colab.ipynb").read_text(encoding="utf-8"))
    assert validate_notebook_contract(notebook)["status"] == "PASS"
    namespace = {}
    for cell in notebook["cells"][1:3]:
        exec("".join(cell["source"]), namespace)
    return namespace


@pytest.mark.parametrize("mode", ["success", "audit", "missing", "wrong_source", "dependencies", "gpu", "worker", "export", "native_crash"])
def test_colab_workflow_persistence_and_failure_gates(tmp_path, monkeypatch, mode):
    notebook = load_notebook()
    assert notebook["AUDIT_ONLY"] is False and notebook["MAX_GPU_HOURS"] == 0
    config = dict(notebook, AUDIT_ONLY=mode == "audit", LOCAL_ROOT=str(tmp_path / "local"),
                  PERSISTENT_ROOT=str(tmp_path / "drive"), INPUT_ZIP=str(tmp_path / "handoff.zip"),
                  QWEN_CACHE=str(tmp_path / "qwen"))
    if mode != "missing":
        Path(config["INPUT_ZIP"]).touch()
    Path(config["QWEN_CACHE"]).mkdir()
    events, calls = [], []

    def git(command, **kwargs):
        if "status" in command:
            return ""
        return "different\n" if mode == "wrong_source" else config["SOURCE_REVISION"] + "\n"

    def invoke(command, log, env, timeout):
        assert env["CUDA_VISIBLE_DEVICES"] == "-1"
        assert env["OMP_NUM_THREADS"] == "1"
        assert env["ST3_EXP028_CPU_THREADS"] == "1"
        assert 0 < timeout < 10800
        calls.append(command)
        log.touch()
        if mode == "dependencies" and log.name == "dependencies.log":
            return 1
        if log.name == "native_preflight.log":
            native = Path(command[command.index("--output") + 1])
            native.mkdir()
            checks = {name: {"status": "PASS", "exit_code": 0}
                      for name in ("scipy", "sklearn", "original_regression")}
            if mode == "native_crash":
                checks["original_regression"] = {"status": "NATIVE_SIGNAL", "exit_code": -11, "signal": "SIGSEGV"}
            (native / "native_preflight.json").write_text(json.dumps(
                {"status": "BLOCKED_VERIFICATION" if mode == "native_crash" else "PASS", "checks": checks}))
            return 2 if mode == "native_crash" else 0
        if log.name == "cpu_workflow.log":
            assert ("--audit-only" in command) == (mode == "audit")
            assert "--authorize-extraction" not in command
            attempt = Path(command[command.index("--attempt") + 1])
            status = "COMPLETE_CACHE_VERIFIED" if mode == "audit" else "FOLD3_SCREEN_COMPLETE / EXPLORATORY"
            result = {"status": status, "source_revision": config["SOURCE_REVISION"]}
            if mode == "worker":
                result.update(status="BLOCKED_VERIFICATION", failed_stage="RAM_PREFLIGHT", error="RAM unavailable")
            (attempt / "workflow_status.json").write_text(json.dumps(result))
            if mode == "success":
                (attempt / "summary.md").write_text("synthetic verified result")
            return int(mode == "worker")
        return 0

    def browser_download(path):
        events.append("download")
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest_sha256.json"))
            import hashlib
            assert set(archive.namelist()) == set(manifest) | {"manifest_sha256.json"}
            assert all(hashlib.sha256(archive.read(name)).hexdigest() == digest for name, digest in manifest.items())

    monkeypatch.setenv("COLAB_GPU", "1" if mode == "gpu" else "0")
    monkeypatch.setattr(notebook["subprocess"], "check_output", git)
    notebook["invoke"] = invoke
    if mode == "export":
        notebook["verified_export"] = lambda _: (_ for _ in ()).throw(OSError("corrupt backup"))
        config["AUDIT_ONLY"] = True
    with patch.dict("sys.modules", {"IPython.display": SimpleNamespace(display=lambda _: None, Markdown=str)}):
        result = notebook["run_pipeline"](config, SimpleNamespace(mount=lambda _: events.append("mount")),
                                           SimpleNamespace(download=browser_download),
                                           SimpleNamespace(unassign=lambda: events.append("release")))
    expected = {
        "success": "FOLD3_SCREEN_COMPLETE / EXPLORATORY",
        "audit": "COMPLETE_CACHE_VERIFIED",
    }.get(mode, "BLOCKED_VERIFICATION")
    assert result["status"] == expected
    assert events == (["mount", "release"] if mode == "export" else ["mount", "download", "release"])
    if mode in ("missing", "gpu"):
        assert not calls
    if mode in ("dependencies", "wrong_source", "native_crash"):
        assert not any("run_exp028_colab_cpu.py" in token for call in calls for token in call)
    if mode == "missing":
        assert config["INPUT_ZIP"] in result["error"]
    if mode == "worker":
        assert result["failed_stage"] == "RAM_PREFLIGHT"
    if mode == "native_crash":
        assert result["failed_stage"] == "NATIVE_PREFLIGHT"
        assert not any("pytest" in command for command in calls)
        assert next((tmp_path / "drive").rglob("native_preflight.json")).is_file()
    assert not any("extract_exp028_missing.py" in token or "torch" in token for call in calls for token in call)
    final = next((tmp_path / "drive").rglob("final_report.json"))
    assert json.loads(final.read_text()) == result
