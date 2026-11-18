"""Real notebook control flow with mocked processes; never runs a GPU."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest
from st3.experiment_workflow import validate_notebook_contract

ROOT=Path(__file__).resolve().parents[1]


def code():
    nb=json.loads((ROOT/"notebooks/TASK031_Colab.ipynb").read_text(encoding="utf-8"))
    validate_notebook_contract(nb)
    for c in nb["cells"]:
        if c["cell_type"]=="code":ast.parse("".join(c["source"]))
    settings={};exec("".join(nb["cells"][1]["source"]),settings)
    scope={};exec("".join(nb["cells"][2]["source"]),scope)
    return settings,scope


def test_frozen_source_contract_no_training_and_explicit_head_paths():
    s,c=code();assert not s["AUTHORIZE_EXPERIMENT"] and not s["AUTHORIZE_GUARDED_CONFIRMATION"]
    assert s["STAGE"]=="probe" and s["MAX_GPU_HOURS"]==s["MAX_TOTAL_WALL_HOURS"]==.5
    pin=json.loads((ROOT/"docs/task031_source.json").read_text())
    assert s["SOURCE_REVISION"]==pin["source_revision"]
    for name,digest in pin["files_sha256"].items():
        blob=subprocess.check_output(["git","show",s["SOURCE_REVISION"]+":"+name],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==digest
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()==digest
    command=c["stage_command"]("python",ROOT,ROOT,ROOT,ROOT,s,300)
    assert command[2].endswith("run_task031.py") and "--checkpoints" in command and "--authorize-gpu" in command
    for stage in ("pilot","train","outer-fold","one-fold","full-cv","development"):
        with pytest.raises(ValueError):c["stage_command"]("python",ROOT,ROOT,ROOT,ROOT,{**s,"STAGE":stage},300)


@pytest.mark.parametrize("failure",["authorization","mount","scope","confirmation","submission","eval","budget","wall_nan","input","download","export"])
def test_failures_export_download_release_without_launching_gpu(tmp_path,monkeypatch,failure):
    s,c=code();events=[]
    s.update(LOCAL_ROOT=str(tmp_path/"local"),PERSISTENT_ROOT=str(tmp_path/"drive"),INPUT_ZIP=str(tmp_path/"missing.zip"))
    s["AUTHORIZE_EXPERIMENT"]=failure not in ("authorization","download","export")
    if failure=="scope":s["EXECUTION_SCOPE"]="COMPETITION_READY"
    if failure=="confirmation":s["AUTHORIZE_GUARDED_CONFIRMATION"]=True
    if failure=="submission":s["REQUIRE_SUBMISSION_ZIP"]=True
    if failure=="eval":s["EVAL_INPUT_CSV"]="forbidden.csv"
    if failure=="budget":s["MAX_GPU_HOURS"]=.51
    if failure=="wall_nan":s["MAX_TOTAL_WALL_HOURS"]=float("nan")
    def mount(*args):
        if failure=="mount":raise IOError("CPU MOCK mount failure")
    def download(path):
        with zipfile.ZipFile(path) as archive:assert archive.testzip() is None
        events.append("download")
        if failure=="download":raise IOError("CPU MOCK blocked browser")
    monkeypatch.setattr(c["subprocess"],"run",lambda *a,**k:pytest.fail("No process allowed in this failure fixture"))
    if failure=="export":monkeypatch.setattr(c["zipfile"],"ZipFile",lambda *a,**k:(_ for _ in ()).throw(IOError("CPU MOCK export failure")))
    result=c["probe_notebook"](s,SimpleNamespace(mount=mount),SimpleNamespace(download=download),SimpleNamespace(unassign=lambda:events.append("release")))
    assert events[-1]=="release" and result["status"]==("FAILED_EXPORT" if failure=="export" else "EXECUTION_STOP")
    assert result["browser_arrival"]==result["physical_GPU_deallocation"]=="UNVERIFIED"
    if failure not in ("mount","export"):assert events==["download","release"] and result["persistent_readback"]=="PASS"


@pytest.mark.parametrize("failure",[None,"manifest","checkpoint","stage","timeout","verify","export_readback"])
def test_complete_mocked_inference_isolated_inputs_and_cleanup(tmp_path,monkeypatch,failure):
    s,c=code();events=[];commands=[]
    s.update(AUTHORIZE_EXPERIMENT=True,LOCAL_ROOT=str(tmp_path/"local"),PERSISTENT_ROOT=str(tmp_path/"drive"),
             INPUT_ZIP=str(tmp_path/"inputs.zip"),CHECKPOINT_ROOT=str(tmp_path/"old_checkpoints"))
    source=tmp_path/"local/source";source.mkdir(parents=True)
    pairs=b"painting,language\np0,A\n";protocol=b"painting,role\n"+b"p,development\n"*1335
    image=b"CPU mock image only";manifest=b'{"CPU_MOCK_ONLY":true}'
    with zipfile.ZipFile(s["INPUT_ZIP"],"w") as z:
        for name,raw in {"input_manifest.json":manifest,"training/pairs.csv":pairs,"training/protocol.csv":protocol,"artwork/images/p0.jpg":image}.items():z.writestr(name,raw)
    s["ZIP_SHA256"]=hashlib.sha256(Path(s["INPUT_ZIP"]).read_bytes()).hexdigest()
    s["MANIFEST_SHA256"]=hashlib.sha256(manifest).hexdigest() if failure!="manifest" else "BAD"
    config={"input_sha256":{n:hashlib.sha256(v).hexdigest() for n,v in {"training/pairs.csv":pairs,"training/protocol.csv":protocol}.items()}}
    names=["configs/exp026.json","configs/task031.json","configs/task031_manifest.json","docs/task031_protocol.md","scripts/run_task031.py","scripts/verify_task031.py","src/st3/interaction_probe.py","src/st3/probe_inference.py","requirements-exp026.lock.txt","docs/exp027_closure_evidence.json"]
    for name in names:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("CPU fixture")
    (source/"configs/exp026.json").write_text(json.dumps(config))
    evidence={}
    for name in ["identity.json", *["models/"+m+suffix for m in ("BCE","P_low","P_high") for suffix in (".pt",".sha256.json")]]:
        p=Path(s["CHECKPOINT_ROOT"])/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"CPU checkpoint fixture")
        evidence[name.replace("/","\\")]=hashlib.sha256(p.read_bytes()).hexdigest()
    (source/"docs/exp027_closure_evidence.json").write_text(json.dumps({"files_sha256":evidence}))
    if failure=="checkpoint":(Path(s["CHECKPOINT_ROOT"])/"models/BCE.pt").write_bytes(b"corrupt")
    monkeypatch.setattr(c["subprocess"],"check_output",lambda command,**k:s["SOURCE_REVISION"]+"\n" if "rev-parse" in command else "")
    def run(command,**kwargs):
        commands.append(command)
        if any(Path(str(x)).name=="run_task031.py" for x in command):
            stage=command[command.index("--stage")+1];directory=Path(command[command.index("--run-dir")+1]);directory.mkdir(parents=True,exist_ok=True)
            events.append(stage)
            if stage=="prepare":
                (directory/"probe_manifest.json").write_text(json.dumps({"counts":{"CPU_MOCK_ONLY":True},"images":{"p0":{"image":"p0.jpg","image_sha256":hashlib.sha256(image).hexdigest()}}}))
            else:
                assert (tmp_path/"local/inputs/artwork/images/p0.jpg").read_bytes()==image
                assert not (tmp_path/"local/inputs/history").exists()
                if failure=="timeout":raise c["subprocess"].TimeoutExpired("CPU mock",30)
                if failure=="stage":return SimpleNamespace(returncode=2)
                (directory/"probe_report.json").write_text(json.dumps({"status":"INFERENCE_COMPLETE_OFFLINE_VERIFIED","CPU_MOCK_ONLY":True}))
                (directory/"verification.json").write_text(json.dumps({"status":"FAIL" if failure=="verify" else "PASS"}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c["subprocess"],"run",run)
    if failure=="export_readback":
        original=c["zipfile"].ZipFile
        def corrupt(path,*args,**kwargs):
            if Path(path).name=="outputs.zip" and not args:raise IOError("CPU MOCK read-back failure")
            return original(path,*args,**kwargs)
        monkeypatch.setattr(c["zipfile"],"ZipFile",corrupt)
    result=c["probe_notebook"](s,SimpleNamespace(mount=lambda *a:None),SimpleNamespace(download=lambda p:events.append("download")),SimpleNamespace(unassign=lambda:events.append("release")))
    assert events[-1]=="release"
    if failure in ("manifest","checkpoint"):assert "probe" not in events and "prepare" not in events
    if failure is None:
        assert result["status"]=="INFERENCE_COMPLETE_OFFLINE_VERIFIED" and result["CPU_MOCK_ONLY"]
        assert events==["prepare","probe","download","release"]
    elif failure=="export_readback":assert result["status"]=="FAILED_EXPORT" and "download" not in events
    else:assert result["status"]=="EXECUTION_STOP" and events[-2:]==["download","release"]


def test_real_tiny_image_pipeline_interrupt_resume_and_cached_pixel_guard(tmp_path,monkeypatch):
    import torch
    from st3.probe_inference import run_inference, read_batch, batch_identity
    from st3.vlm_supervised import attach_adapters, make_head
    from st3.vlm_supervised_training import model_state
    from test_vlm_supervised import fixture as images_fixture, config
    from test_interaction_probe import fixture as probe_fixture
    frame,backend,cache=images_fixture(tmp_path);c=config();attach_adapters(backend.model,c)
    backend.device="cpu";original_hidden=backend.hidden
    backend.hidden=lambda visions,languages,train=False:backend.model(original_hidden(visions,languages,train))
    backend.parity=lambda *a:{"status":"PASS","CPU_MOCK_ONLY":True}
    head=make_head(["A","B"],c);state=model_state(backend,head)
    models={m:{"state":state} for m in ("BCE","P_low","P_high")}
    m,_,_=probe_fixture()
    m["images"]={p:frame[frame.painting.eq(source)][["image","image_sha256"]].iloc[0].to_dict() for p,source in (("I","p0"),("J","p1"))}
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    calls=[]
    def stop():
        calls.append(1)
        if len(calls)==3:raise TimeoutError("CPU mock interrupted at image boundary")
    identity={"CPU_MOCK_ONLY":True};directory=tmp_path/"probe"
    with pytest.raises(TimeoutError):run_inference(backend,models,c,m,tmp_path,tmp_path/"cache",directory,identity,stop)
    path=directory/"logits/BCE/trained/I.json";key=batch_identity(identity,"BCE","trained","I")
    assert read_batch(path,key) is not None
    first_bytes=path.read_bytes()
    diagnostics=run_inference(backend,models,c,m,tmp_path,tmp_path/"cache",directory,identity,lambda:None)
    assert diagnostics["resumed_image_batches"]==1 and diagnostics["new_image_batches"]==11
    assert len(diagnostics["exact_determinism_checks"])==6 and path.read_bytes()==first_bytes
    with pytest.raises(ValueError,match="identity"):
        read_batch(path,{"different_source":True})
    (tmp_path/"artwork/images/p0.jpg").write_bytes(b"changed image despite valid old cache")
    with pytest.raises(ValueError,match="pixels differ"):
        run_inference(backend,models,c,m,tmp_path,tmp_path/"cache",directory,identity,lambda:None)
