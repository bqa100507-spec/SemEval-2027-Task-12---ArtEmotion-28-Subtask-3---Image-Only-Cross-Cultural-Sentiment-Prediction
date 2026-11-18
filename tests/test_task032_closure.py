"""CPU-only synthetic archive tests and proposed-design arithmetic, no model fits."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest
from st3.probe_inference import SOURCE_FILES
from st3.interaction_probe import analyze
from st3.vlm_semantics import atomic_json,file_hash,fingerprint
from test_interaction_probe import fixture

ROOT=Path(__file__).resolve().parents[1]


def module():
    spec=importlib.util.spec_from_file_location("closure",ROOT/"scripts/close_task031.py")
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def synthetic_run(tmp_path,monkeypatch):
    closure=module();m,targets,rows=fixture()
    blobs={n:(ROOT/n).read_bytes().replace(b"\r\n",b"\n") for n in SOURCE_FILES}
    blobs["configs/task031_manifest.json"]=json.dumps(m).encode()
    monkeypatch.setattr(closure.subprocess,"check_output",lambda command,**kwargs:blobs[command[-1].split(":",1)[1]])
    identity=dict(source_revision=closure.SOURCE,manifest_sha256=fingerprint(m),checkpoint_sha256=m["checkpoint_sha256"],
                  source_sha256={n:hashlib.sha256(v).hexdigest() for n,v in blobs.items()})
    run=tmp_path/"run"
    for name,value in (("identity.json",identity),("probe_manifest.json",m),("selected_training_targets.json",targets)):
        atomic_json(run/name,value)
    for model in ("BCE","P_low","P_high"):
        for state in m["config"]["adapter_states"]:
            for painting in m["images"]:
                p=run/"logits"/model/state/(painting+".json")
                part=[r for r in rows if (r["model"],r["adapter_state"],r["painting"])==(model,state,painting)]
                atomic_json(p,dict(identity=dict(run=fingerprint(identity),model=model,adapter_state=state,painting=painting),rows=part))
                atomic_json(p.with_suffix(".sha256.json"),{"sha256":file_hash(p)})
    frame,result=analyze(rows,m,targets)
    frame.to_csv(run/"rectangles.csv",index=False,lineterminator="\n");atomic_json(run/"analysis.json",result)
    diagnostics=dict(exact_determinism_checks={model+"/"+state:True for model in ("BCE","P_low","P_high") for state in m["config"]["adapter_states"]},
                     official_cache_parity={"status":"PASS"},fresh_vision_cache_exact=True,new_image_batches=12,resumed_image_batches=0)
    atomic_json(run/"inference_diagnostics.json",diagnostics)
    atomic_json(run/"probe_report.json",dict(inference=diagnostics,status="INFERENCE_COMPLETE_OFFLINE_VERIFIED",training_performed=False,outer_scoring=False,
        GPU="CPU_MOCK_ONLY",elapsed_seconds=1.,peak_allocated_bytes=0,peak_reserved_bytes=0))
    atomic_json(run/"verification.json",dict(status="PASS",raw_logits=96,files_sha256={n:file_hash(run/n) for n in ("analysis.json","rectangles.csv","inference_diagnostics.json")}))
    return closure,run


def test_complete_read_only_statistical_replay_and_bootstrap(tmp_path,monkeypatch):
    c,run=synthetic_run(tmp_path,monkeypatch)
    before={str(p):file_hash(p) for p in run.rglob("*") if p.is_file()}
    r=c.reconstruct(run,ROOT)
    assert r["raw_logits"]==96 and r["sealed_image_batches"]==12 and r["status"]=="PASS"
    assert r["analysis"]["summaries"][0]["pair_cluster_95_CI_mean_abs_D"]==[8.,8.]
    assert before=={str(p):file_hash(p) for p in run.rglob("*") if p.is_file()}


@pytest.mark.parametrize("failure",["source","artifact","keys","targets","replay","analysis"])
def test_failed_evidence_cannot_close_science(tmp_path,monkeypatch,failure):
    c,run=synthetic_run(tmp_path,monkeypatch)
    if failure=="source":
        x=json.loads((run/"identity.json").read_text());x["source_sha256"].pop(next(iter(x["source_sha256"])));atomic_json(run/"identity.json",x)
    elif failure in ("artifact","keys"):
        p=next((run/"logits").rglob("*.json"));x=json.loads(p.read_text());x["rows"][0]["head_language"]="wrong";atomic_json(p,x)
        if failure=="keys":atomic_json(p.with_suffix(".sha256.json"),{"sha256":file_hash(p)})
    elif failure=="targets":
        x=json.loads((run/"selected_training_targets.json").read_text());x[0]["k"]-=1;atomic_json(run/"selected_training_targets.json",x)
    else:
        name="inference_diagnostics.json" if failure=="replay" else "analysis.json"
        x=json.loads((run/name).read_text())
        if failure=="replay":x["exact_determinism_checks"].pop(next(iter(x["exact_determinism_checks"])))
        else:x["summaries"][0]["mean_abs_D"]+=.1
        atomic_json(run/name,x)
        v=json.loads((run/"verification.json").read_text());v["files_sha256"][name]=file_hash(run/name);atomic_json(run/"verification.json",v)
    with pytest.raises((ValueError,AssertionError)):c.reconstruct(run,ROOT)


def test_zip_checksum_coverage_and_unsafe_paths(tmp_path):
    c=module();archive=tmp_path/"bad.zip";destination=tmp_path/"out"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("../escape",b"bad");z.writestr("manifest_sha256.json",json.dumps({"../escape":hashlib.sha256(b"bad").hexdigest()}))
    with pytest.raises(ValueError,match="checksum"):c.unpack(archive,destination,"wrong")
    with pytest.raises(ValueError,match="Unsafe"):c.unpack(archive,destination,file_hash(archive))
    with zipfile.ZipFile(archive,"w") as z:z.writestr("extra",b"bad");z.writestr("manifest_sha256.json","{}")
    with pytest.raises(ValueError,match="coverage"):c.unpack(archive,destination,file_hash(archive))


def test_recorded_closure_and_design_only_dimensions_boundaries():
    r=json.loads((ROOT/"docs/task031_closure_receipt.json").read_text())
    assert r["raw_logits"]==3216 and r["export_files"]==406 and r["source_files"]==17
    assert r["original_checkpoint_target_verification"].startswith("PASS")
    assert len(r["analysis"]["summaries"])==24 and r["sealed_image_batches"]==192
    s=json.loads((ROOT/"docs/exp028_design_spec.json").read_text());assert not s["execution_authorized"]
    for arm in s["LR_arms"]:assert arm["parameters_at_25_languages"]==26*arm["dimension"]+26
    assert s["gated"]["parameters_at_25_languages"]==2*(256*64+64)+25*8+(128+8+1)+(64+1)+25+25*64
    assert not (ROOT/"scripts/run_exp028.py").exists()
    assert not (ROOT/"src/st3/exp028.py").exists()
    inventory=json.loads((ROOT/"docs/exp028_feature_inventory.json").read_text())
    assert all(x["P2_covered"]==1335 and x["P2_missing"]==0 for x in inventory["visual_caches"])
    assert inventory["exact_cold_extraction"]["language_decoder_sequences"]==13600
    assert inventory["Drive_Qwen_cache"]["full_payload_coverage"].startswith("UNVERIFIED")
