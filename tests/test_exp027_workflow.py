"""No-GPU orchestration/cleanup behavior and structural workflow contract."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from st3.experiment_workflow import validate_notebook_contract

ROOT=Path(__file__).resolve().parents[1]


def code():
    nb=json.loads((ROOT/"notebooks/EXP027_Colab.ipynb").read_text(encoding="utf-8"))
    validate_notebook_contract(nb)
    for c in nb["cells"]:
        if c["cell_type"]=="code":ast.parse("".join(c["source"]))
    settings={};exec("".join(nb["cells"][1]["source"]),settings)
    scope={};exec("".join(nb["cells"][2]["source"]),scope)
    return settings,scope


def test_notebook_frozen_stage_permissions_paths_and_manifest_pin():
    s,c=code();assert not s["AUTHORIZE_EXPERIMENT"] and not s["AUTHORIZE_GUARDED_CONFIRMATION"]
    assert s["STAGE"]=="pilot" and not s["REQUIRE_SUBMISSION_ZIP"] and s["EVAL_INPUT_CSV"] is None
    assert s["MAX_GPU_HOURS"]==s["MAX_TOTAL_WALL_HOURS"]==2
    assert len(s["SOURCE_REVISION"])==40 and s["SOURCE_REVISION"]!="0"*40
    assert s["MANIFEST_SHA256"]=="b5315db3847c29c432e91510a3d8f99c4b1bf13aca8b17282a53059347a20aa9"
    command=c["stage_command"]("python",ROOT,ROOT,ROOT,ROOT,s,6000)
    assert command[2].endswith("run_exp027.py") and "--authorize-gpu" in command
    for stage in ("outer-fold","one-fold","full-cv","development"):
        with pytest.raises(ValueError):c["stage_command"]("python",ROOT,ROOT,ROOT,ROOT,{**s,"STAGE":stage},6000)


@pytest.mark.parametrize("failure",["authorization","mount","scope","confirmation","submission","eval","budget","wall_nan","input","download","export"])
def test_bootstrap_failures_preserve_export_and_release_without_gpu(tmp_path,monkeypatch,failure):
    s,c=code();events=[]
    s.update(LOCAL_ROOT=str(tmp_path/"local"),PERSISTENT_ROOT=str(tmp_path/"drive"),INPUT_ZIP=str(tmp_path/"missing.zip"))
    s["AUTHORIZE_EXPERIMENT"]=failure not in ("authorization","download","export")
    if failure=="scope":s["EXECUTION_SCOPE"]="COMPETITION_READY"
    if failure=="confirmation":s["AUTHORIZE_GUARDED_CONFIRMATION"]=True
    if failure=="submission":s["REQUIRE_SUBMISSION_ZIP"]=True
    if failure=="eval":s["EVAL_INPUT_CSV"]="forbidden.csv"
    if failure=="budget":s["MAX_GPU_HOURS"]=2.01
    if failure=="wall_nan":s["MAX_TOTAL_WALL_HOURS"]=float("nan")
    def mount(*args):
        if failure=="mount":raise IOError("MOCK mount failed")
    def download(path):
        with zipfile.ZipFile(path) as archive:assert archive.testzip() is None
        events.append("download")
        if failure=="download":raise IOError("MOCK browser blocked")
    monkeypatch.setattr(c["subprocess"],"run",lambda *a,**k:pytest.fail("No training/dependency command may launch"))
    if failure=="export":monkeypatch.setattr(c["zipfile"],"ZipFile",lambda *a,**k:(_ for _ in ()).throw(IOError("MOCK export error")))
    result=c["pilot_notebook"](s,SimpleNamespace(mount=mount),SimpleNamespace(download=download),SimpleNamespace(unassign=lambda:events.append("release")))
    assert events[-1]=="release" and result["status"]==("FAILED_EXPORT" if failure=="export" else "EXECUTION_STOP")
    assert result["browser_arrival"]=="UNVERIFIED" and result["physical_GPU_deallocation"]=="UNVERIFIED"
    if failure not in ("export","mount"):assert events==["download","release"] and result["persistent_readback"]=="PASS"
    if failure=="download":assert result["download_request"].startswith("FAILED")
    if failure=="export":assert "export_error" in result
    if failure in ("confirmation","submission","eval"):assert "must remain disabled" in result["error"]
    if failure in ("budget","wall_nan"):assert "at most2 hours" in result["error"]
    if failure=="scope":assert "research-only" in result["error"]


@pytest.mark.parametrize("failure",[None,"manifest","export_readback"])
def test_mocked_complete_pilot_and_manifest_failure_export_before_release(tmp_path,monkeypatch,failure):
    import hashlib,csv,io
    s,c=code();s.update(AUTHORIZE_EXPERIMENT=True,LOCAL_ROOT=str(tmp_path/"local"),PERSISTENT_ROOT=str(tmp_path/"drive"),INPUT_ZIP=str(tmp_path/"inputs.zip"))
    source=tmp_path/"local/source";source.mkdir(parents=True)
    protocol=io.StringIO();writer=csv.DictWriter(protocol,fieldnames=["painting","image","image_sha256","role"]);writer.writeheader()
    for i in range(1335):writer.writerow(dict(painting=f"p{i}",image=f"p{i}.jpg",image_sha256="mock",role="development"))
    pairs=b"painting,language\np0,L\n";raw_protocol=protocol.getvalue().encode();image=b"CPU fixture, not a real model input"
    manifest=b'{"cpu_fixture":true}'
    with zipfile.ZipFile(s["INPUT_ZIP"],"w") as z:
        z.writestr("input_manifest.json",manifest);z.writestr("training/protocol.csv",raw_protocol);z.writestr("training/pairs.csv",pairs);z.writestr("artwork/images/p0.jpg",image)
    s["ZIP_SHA256"]=hashlib.sha256(Path(s["INPUT_ZIP"]).read_bytes()).hexdigest();s["MANIFEST_SHA256"]=hashlib.sha256(manifest).hexdigest() if failure!="manifest" else "bad"
    names=["configs/exp026.json","configs/exp027.json","requirements-exp026.lock.txt","scripts/run_exp027.py","scripts/verify_exp027_validation.py","src/st3/vlm_supervised.py","src/st3/vlm_supervised_training.py","src/st3/contrast_qlora.py","src/st3/exp027_data.py","src/st3/exp027_pilot.py","src/st3/exp027_validation.py","src/st3/metrics.py","src/st3/submission.py"]
    for name in names:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("CPU fixture")
    (source/"configs/exp026.json").write_text(json.dumps({"input_sha256":{"training/pairs.csv":hashlib.sha256(pairs).hexdigest(),"training/protocol.csv":hashlib.sha256(raw_protocol).hexdigest()}}))
    events=[];commands=[]
    def output(command,**kwargs):return s["SOURCE_REVISION"]+"\n" if "rev-parse" in command else ""
    monkeypatch.setattr(c["subprocess"],"check_output",output)
    def run(command,**kwargs):
        commands.append(command)
        if "-c" in command and "pilot_images.json" in str(command[-1]):
            Path(command[-1]).write_text(json.dumps([{"image":"p0.jpg","image_sha256":hashlib.sha256(image).hexdigest()}]))
        if any(Path(str(x)).name=="run_exp027.py" for x in command):
            path=Path(command[command.index("--run-dir")+1]);path.mkdir(parents=True)
            (path/"pilot_report.json").write_text(json.dumps({"status":"GPU_PILOT_COMPLETE","outer_evaluation_performed":False,"CPU_MOCK_ONLY":True}))
            events.append("pilot")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c["subprocess"],"run",run)
    if failure=="export_readback":
        original=c["zipfile"].ZipFile
        def corrupt(path,*args,**kwargs):
            if Path(path).name=="outputs.zip" and not args:raise IOError("MOCK output read-back failure")
            return original(path,*args,**kwargs)
        monkeypatch.setattr(c["zipfile"],"ZipFile",corrupt)
    result=c["pilot_notebook"](s,SimpleNamespace(mount=lambda *a:None),SimpleNamespace(download=lambda path:events.append("download")),SimpleNamespace(unassign=lambda:events.append("release")))
    if failure=="manifest":
        assert result["status"]=="EXECUTION_STOP" and "manifest SHA" in result["error"]
        assert not commands and events==["download","release"]
    elif failure=="export_readback":
        assert result["status"]=="FAILED_EXPORT" and result["status_before_export"]=="GPU_PILOT_COMPLETE"
        assert events==["pilot","release"] and result["persistent_readback"]=="FAIL"
    else:
        assert result["status"]=="GPU_PILOT_COMPLETE" and result["CPU_MOCK_ONLY"] is True
        assert events==["pilot","download","release"]
        commands=[x for x in commands if any(Path(str(y)).name=="run_exp027.py" for y in x)]
        assert len(commands)==1 and commands[0][commands[0].index("--stage")+1]=="pilot"
    if failure!="export_readback":assert result["persistent_readback"]=="PASS"
