"""Native preflight must stop and save distinct failure receipts."""
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "preflight", Path(__file__).resolve().parents[1] / "scripts/exp028_native_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


@pytest.mark.parametrize("code,status", [
    (-signal.SIGSEGV, "NATIVE_SIGNAL"), (1, "PYTHON_OR_TEST_FAILURE"), (0, "PASS"),
])
def test_native_preflight_receipt_and_gate(tmp_path, monkeypatch, code, status):
    calls = []

    def launch(command, **kwargs):
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        assert kwargs["env"]["ST3_EXP028_CPU_THREADS"] == "1"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        assert "faulthandler" in command
        calls.append(command)
        return SimpleNamespace(returncode=code if len(calls) == 2 else 0)

    monkeypatch.setattr(preflight.subprocess, "run", launch)
    assert preflight.run(tmp_path) == (2 if code else 0)
    report = json.loads((tmp_path / "native_preflight.json").read_text())
    assert report["checks"]["sklearn"]["status"] == status
    if code:
        assert len(calls) == 2 and report["status"] == "BLOCKED_VERIFICATION"
        if code < 0:
            assert report["checks"]["sklearn"]["signal"] == "SIGSEGV"
    else:
        assert len(calls) == 3 and calls[-1][-1] == preflight.REGRESSION
