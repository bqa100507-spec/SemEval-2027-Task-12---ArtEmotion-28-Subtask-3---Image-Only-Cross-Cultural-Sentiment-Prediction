"""Scope boundaries use CPU/synthetic callbacks only, never scientific GPU fits or sealed gold."""

import copy
import json
from pathlib import Path

import pytest

from st3.experiment_workflow import (Budget, StopRun, STAGES, execute_stages, execution_scope,
                                     scoped_cv_decision, require_run_identity, publish_model, verify_model,
                                     finalize, RESEARCH_ONLY_CV, COMPETITION_READY, eligibility_evidence, sha256)
from st3.exp023_runner import Experiment, ROOT, verify_eligibility
from st3.exp023_data import read_confirmation
from st3 import exp023_cloud


def auth(scope=RESEARCH_ONLY_CV):
    return {"EXECUTION_SCOPE":scope,"AUTHORIZE_EXPERIMENT":True,
            "AUTHORIZE_GUARDED_CONFIRMATION":scope==COMPETITION_READY,
            "REQUIRE_SUBMISSION_ZIP":scope==COMPETITION_READY,
            "CAPTION_ELIGIBILITY_STATUS":"UNRESOLVED"}


@pytest.mark.parametrize("scope",[None,"","research","UNKNOWN"])
def test_scope_must_be_explicit(scope):
    with pytest.raises(StopRun,match="explicitly"):execution_scope({**auth(),"EXECUTION_SCOPE":scope})


@pytest.mark.parametrize("flag",["AUTHORIZE_GUARDED_CONFIRMATION","REQUIRE_SUBMISSION_ZIP"])
def test_research_conflicting_authorizations_fail(flag):
    with pytest.raises(StopRun):execution_scope({**auth(),flag:True})


def callbacks(events,go=True):
    result={"Verify":"PASS","Decision":"CV_GO" if go else "CV_NO_GO","GuardedConfirmation":"CONFIRMATION_GO",
            "FinalModel":"PASS","SubmissionValidation":"PASS"}
    return {stage:(lambda stage=stage:events.append(stage) or {"status":result.get(stage,"PASS")}) for stage in STAGES}


@pytest.mark.parametrize("go",[True,False])
def test_research_stops_after_cv_decision_and_exports_releases(tmp_path,go):
    events=[]
    hooks=callbacks(events,go)
    for stage in STAGES[6:]:hooks[stage]=lambda:pytest.fail("Competition callback reached by research scope")
    if go:
        results=execute_stages(hooks,Budget(1,2,cuda=False),lambda stage:None,scope=RESEARCH_ONLY_CV)
        status=results["Decision"]["status"]
        assert status=="CV_GO_PENDING_ELIGIBILITY"
        assert results["Decision"]["competition_eligibility"]=="NOT_ESTABLISHED"
    else:
        with pytest.raises(StopRun) as error:execute_stages(hooks,Budget(1,2,cuda=False),lambda stage:None,scope=RESEARCH_ONLY_CV)
        status=error.value.status
        assert status=="CV_NO_GO"
    assert tuple(events)==STAGES[:6]
    run=tmp_path/"run";run.mkdir();(run/"cv_report.json").write_text(json.dumps({"status":status}))
    lifecycle=[]
    report=finalize(run,tmp_path/"drive",{"status":status,"execution_scope":RESEARCH_ONLY_CV},platform="colab",
                    download=lambda path:lifecycle.append("download:"+Path(path).name),release=lambda:lifecycle.append("release"))
    assert report["persistent_readback"]=="PASS"
    assert lifecycle==["download:outputs.zip","release"]
    assert not (tmp_path/"drive/submission.zip").exists()


def test_competition_full_sequence_unchanged():
    events=[]
    result=execute_stages(callbacks(events),Budget(1,2,cuda=False),lambda stage:None,scope=COMPETITION_READY)
    assert tuple(events)==STAGES
    assert result["Decision"]["status"]=="CV_GO"
    assert result["SubmissionValidation"]["status"]=="PASS"


@pytest.mark.parametrize("scope",[RESEARCH_ONLY_CV,COMPETITION_READY])
def test_failed_verification_never_reaches_decision(scope):
    events=[];hooks=callbacks(events)
    hooks["Verify"]=lambda:{"status":"FAIL"}
    with pytest.raises(StopRun) as error:execute_stages(hooks,Budget(1,2,cuda=False),lambda stage:None,scope=scope)
    assert error.value.status=="FAILED_VERIFICATION"
    assert "Decision" not in events


@pytest.mark.parametrize("method",["confirmation","final_model","inference","validate_submission"])
def test_direct_competition_methods_reject_research(method):
    engine=object.__new__(Experiment)
    engine.execution_scope=RESEARCH_ONLY_CV
    engine.competition_eligible=False
    with pytest.raises(StopRun) as error:getattr(engine,method)()
    assert error.value.status=="BLOCKED_SCOPE"


@pytest.mark.parametrize("method,args",[("final_baseline",(None,)),("final_predict",("C2",None,None)),
                                       ("fit_one",(0,"C2",2027,None,None,None))])
def test_direct_final_fit_helpers_reject_research(method,args):
    engine=object.__new__(Experiment);engine.execution_scope=RESEARCH_ONLY_CV;engine.competition_eligible=False
    with pytest.raises(StopRun):getattr(engine,method)(*args)


def test_direct_confirmation_read_rejects_research_even_with_all_flags(tmp_path):
    with pytest.raises(StopRun,match="sealed"):
        read_confirmation(tmp_path,None,cv_verified=True,authorized=True,final_models_verified=True,execution_scope=RESEARCH_ONLY_CV)


def test_competition_methods_require_real_eligibility_validation():
    engine=object.__new__(Experiment);engine.execution_scope=COMPETITION_READY;engine.competition_eligible=False
    with pytest.raises(StopRun):engine.confirmation()


def identity():
    return {"source_revision":"immutable","source":{"model.py":"sha"},"config":"frozen",
            "inputs":"same_handoff", "runtime":{"torch":"pinned","gpu":"same"}}


@pytest.mark.parametrize("key",["source_revision","source","config","inputs","runtime"])
def test_incompatible_scientific_identity_always_fails(key):
    before=identity();after=copy.deepcopy(before);after[key]="changed"
    with pytest.raises(StopRun,match=key):require_run_identity(before,after)


def test_scope_transition_resumes_exact_model_without_retraining(tmp_path):
    scientific=identity()
    assert execution_scope(auth(RESEARCH_ONLY_CV))==RESEARCH_ONLY_CV
    require_run_identity(scientific,identity())
    calls=[]
    def writer(stage):calls.append("fit");(stage/"checkpoint.pt").write_bytes(b"synthetic sealed checkpoint")
    model=tmp_path/"fold_1_C2_2027"
    publish_model(model,scientific,writer)
    assert execution_scope(auth(COMPETITION_READY))==COMPETITION_READY
    publish_model(model,identity(),lambda stage:pytest.fail("Compatible research fit retrained"))
    assert verify_model(model,identity())["complete"]
    assert calls==["fit"]
    (model/"checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(StopRun):publish_model(model,identity(),writer)
    assert calls==["fit"]


@pytest.mark.parametrize("scope,expected",[(RESEARCH_ONLY_CV,"CV_GO_PENDING_ELIGIBILITY"),(COMPETITION_READY,"BLOCKED_ELIGIBILITY")])
def test_cloud_scope_eligibility_isolation(tmp_path,monkeypatch,scope,expected):
    source=tmp_path/"source";(source/"configs").mkdir(parents=True)
    (source/"configs/exp023_eligibility.json").write_text(json.dumps({"status":"ELIGIBILITY_UNRESOLVED"}))
    monkeypatch.setattr(exp023_cloud,"restore_handoff",lambda *args:{"source_revision":"same_source"})
    monkeypatch.setattr(exp023_cloud,"restore_source",lambda *args:source)
    if scope==RESEARCH_ONLY_CV:
        monkeypatch.setattr(exp023_cloud,"eligibility_evidence",lambda *args:pytest.fail("Research fetched competition evidence"))
    events=[]
    monkeypatch.setattr(exp023_cloud,"bootstrap",lambda *args:events.append("bootstrap") or Path("synthetic_python"))
    run=tmp_path/"drive/run"
    def child(command,**kwargs):
        if "pytest" in command:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"]=="-1"
            events.append("cpu_compliance")
        else:
            events.append("synthetic_pipeline")
            (run/"child_report.json").write_text(json.dumps({"status":"CV_GO_PENDING_ELIGIBILITY","execution_scope":scope}))
    monkeypatch.setattr(exp023_cloud,"child",child)
    settings={**auth(scope),"LOCAL_ROOT":str(tmp_path/"local"),"PERSISTENT_ROOT":str(tmp_path/"drive"),
              "MAX_GPU_HOURS":1.,"MAX_TOTAL_WALL_HOURS":2.,"INPUT_ZIP":"unused", "ZIP_SHA256":"x","MANIFEST_SHA256":"x",
              "SOURCE_REVISION":"same_source","AUTO_BROWSER_DOWNLOAD":True,"AUTO_RELEASE_COLAB":True}
    report=exp023_cloud.run_cloud(settings,platform="colab",mount=lambda:None,download=lambda path:events.append("download:"+Path(path).name),release=lambda:events.append("release"))
    assert report["status"]==expected
    assert ("bootstrap" in events)==(scope==RESEARCH_ONLY_CV)
    assert events[-2:]==["download:outputs.zip","release"]


def test_eligibility_config_and_scientific_protocol_still_frozen():
    config=json.loads((ROOT/"configs/exp023_eligibility.json").read_text())
    assert config["status"]=="ELIGIBILITY_UNRESOLVED"


def test_separate_source_backed_evidence_does_not_rewrite_frozen_config(tmp_path):
    import hashlib
    frozen=ROOT/"configs/exp023_eligibility.json"
    before=sha256(frozen)
    raw=b"SYNTHETIC ORGANIZER RESPONSE FOR CPU TEST ONLY: permission for the specified procedure."
    evidence={"status":"VERIFIED_ORGANIZER_PERMISSION","covers":["ST3","original_training_captions_as_auxiliary_targets","pinned_pretrained_CLIP_L_text"],
              "url":"https://github.com/artemotion-28/artemotion-28/issues/999",
              "source_sha256":hashlib.sha256(raw).hexdigest(),"exact_permission_excerpt":raw.decode()}
    supplied=tmp_path/"synthetic_permission.json";supplied.write_text(json.dumps(evidence))
    loaded,receipt=eligibility_evidence({"COMPETITION_ELIGIBILITY_EVIDENCE":str(supplied)},frozen)
    assert receipt["sha256"]==sha256(supplied)
    assert verify_eligibility(loaded,"VERIFIED",fetch=lambda url:raw)["status"]=="PASS"
    assert sha256(frozen)==before
    assert json.loads(frozen.read_text())["status"]=="ELIGIBILITY_UNRESOLVED"
