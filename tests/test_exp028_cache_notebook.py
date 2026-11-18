"""CPU probes of the extraction-only notebook and its deadline/cleanup paths."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from st3.experiment_workflow import validate_notebook_contract

ROOT = Path(__file__).resolve().parents[1]


def notebook_code():
    notebook = json.loads((ROOT / "notebooks/EXP028_Cache_Colab.ipynb").read_text(encoding="utf-8"))
    assert validate_notebook_contract(notebook)["status"] == "PASS"
    module = {}
    for cell in notebook["cells"][1:3]:
        exec("".join(cell["source"]), module)
    return module


@pytest.mark.parametrize("authorize,fail", [(False, False), (True, False), (False, True)])
def test_colab_audit_default_authorized_extraction_and_persistent_export(tmp_path, monkeypatch, authorize, fail):
    code = notebook_code()
    assert code["AUTHORIZE_EXPERIMENT"] is False
    cfg = dict(code, AUTHORIZE_EXPERIMENT=authorize, PERSISTENT_ROOT=str(tmp_path / "outputs"),
               LOCAL_ROOT=str(tmp_path / "local"), INPUT_ZIP=str(tmp_path / "inputs.zip"),
               ORIGINAL_CACHE=str(tmp_path / "original"))
    Path(cfg["ORIGINAL_CACHE"]).mkdir()
    Path(cfg["INPUT_ZIP"]).touch()
    commands, events = [], []
    monkeypatch.setattr(code["subprocess"], "check_output", lambda *a, **k: cfg["SOURCE_REVISION"])

    def command(args, **kwargs):
        commands.append(args)
        assert 0 < kwargs["timeout"] <= 9000
        if any(Path(s).name == "extract_exp028_missing.py" for s in args):
            index = args.index("--stage")
            assert args[index+1] == ("extract" if authorize else "audit")
            assert ("--authorize-extraction" in args) is authorize
            assert kwargs["env"].get("CUDA_VISIBLE_DEVICES") == ("-1" if not authorize else None)
            if fail:
                return SimpleNamespace(returncode=2)
            dest = Path(args[args.index("--output")+1])
            dest.mkdir()
            (dest / "result.json").write_text(json.dumps({"status":"AUDIT_MOCK"}))
        return SimpleNamespace(returncode=0)

    def download(path):
        with zipfile.ZipFile(path) as archive:
            assert "manifest_sha256.json" in archive.namelist()
        events.append("download")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(code["subprocess"], "run", command)
    report = code["run_pipeline"](cfg, SimpleNamespace(mount=lambda p: events.append("mount")),
                                  SimpleNamespace(download=download),
                                  SimpleNamespace(unassign=lambda: events.append("release")))
    assert report["status"] == ("STOPPED" if fail else "AUDIT_MOCK")
    assert report["persistent_readback"] == "PASS"
    assert events == ["mount", "download", "release"]
    assert sum(any(Path(arg).name=="extract_exp028_missing.py" for arg in cmd) for cmd in commands) == 1
    assert not any("run_exp028_stage1.py" in arg for cmd in commands for arg in cmd)


def test_missing_drive_input_stops_before_any_process(tmp_path, monkeypatch):
    code = notebook_code()
    cfg = dict(code, PERSISTENT_ROOT=str(tmp_path / "outputs"), LOCAL_ROOT=str(tmp_path / "cache"),
               ORIGINAL_CACHE=str(tmp_path / "missing"), INPUT_ZIP=str(tmp_path / "missing.zip"))
    launched = []
    monkeypatch.setattr(code["subprocess"], "run", lambda *a, **k: launched.append(a))
    release = []
    report = code["run_pipeline"](cfg, SimpleNamespace(mount=lambda p: None),
                                  SimpleNamespace(download=lambda p: None),
                                  SimpleNamespace(unassign=lambda: release.append(True)))
    assert not launched and release == [True]
    assert "Original Drive cache directory is missing" in report["error"]
