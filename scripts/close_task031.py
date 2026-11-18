"""Read-only CPU reconstruction of TASK-031 from its sealed export; no encoder."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from st3.vlm_semantics import file_hash, fingerprint, atomic_json

SOURCE="8d2d24b6b83b31fdf5164da6717b13c0f94c9bf7"
ZIP_SHA="86f06117ff1e9e09615b8488c0298ef6369d94faf76478f01b16bdc0cbf2d3b1"


def same(a,b):
    """Cross-platform arithmetic tolerance; no relaxation of file hashes/keys."""
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and np.isfinite(a) and np.isfinite(b) and abs(a-b)<=1e-12
    return a==b


def unpack(archive,destination,expected):
    if file_hash(archive)!=expected:raise ValueError("Export ZIP checksum differs")
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError("Duplicate ZIP names")
        manifest=json.loads(z.read("manifest_sha256.json"))
        if set(names)!=set(manifest)|{"manifest_sha256.json"}:raise ValueError("Export coverage differs")
        for name,digest in manifest.items():
            path=(destination/name).resolve()
            if not path.is_relative_to(destination.resolve()):raise ValueError("Unsafe archive path")
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError("Export file checksum differs: "+name)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return manifest


def reconstruct(run,root,*,inputs=None,checkpoints=None):
    """Never write into the archived run or import/construct the Qwen encoder."""
    identity=json.loads((run/"identity.json").read_text());read=lambda n:json.loads((run/n).read_text())
    if identity["source_revision"]!=SOURCE:raise ValueError("Wrong frozen inference source")
    source_hashes=identity["source_sha256"]
    expected_sources={
        "configs/task031.json","configs/task031_manifest.json","src/st3/interaction_probe.py",
        "src/st3/probe_inference.py","scripts/run_task031.py","scripts/verify_task031.py",
        "scripts/run_exp026.py","src/st3/vlm_supervised.py","src/st3/vlm_supervised_training.py",
        "src/st3/vlm_semantics.py","requirements-exp026.lock.txt","docs/exp027_closure_evidence.json",
        "docs/task031_protocol.md","configs/exp026.json","configs/exp027.json","src/st3/caption_data.py","src/st3/data.py"}
    if set(source_hashes)!=expected_sources:raise ValueError("Source manifest coverage differs")
    blobs={n:subprocess.check_output(["git","show",SOURCE+":"+n],cwd=root) for n in source_hashes}
    if any(hashlib.sha256(blobs[n]).hexdigest()!=h for n,h in source_hashes.items()):raise ValueError("Pinned source blob differs")
    # The pinned analysis imports this unchanged, hash-verified helper from this checkout.
    if hashlib.sha256((root/"src/st3/vlm_semantics.py").read_bytes().replace(b"\r\n",b"\n")).hexdigest()!=source_hashes["src/st3/vlm_semantics.py"]:
        raise ValueError("Analysis helper changed")
    manifest=read("probe_manifest.json");targets=read("selected_training_targets.json")
    if manifest!=json.loads(blobs["configs/task031_manifest.json"]) or fingerprint(manifest)!=identity["manifest_sha256"]:
        raise ValueError("Frozen manifest differs")
    if identity["checkpoint_sha256"]!=manifest["checkpoint_sha256"]:raise ValueError("Checkpoint identity differs")
    verification=read("verification.json")
    expected_logits=len(manifest["queries"])*3*len(manifest["config"]["adapter_states"])
    expected_batches=len(manifest["images"])*3*len(manifest["config"]["adapter_states"])
    if verification["status"]!="PASS" or verification["raw_logits"]!=expected_logits:raise ValueError("Original verification was not complete")
    if set(verification["files_sha256"])!={"rectangles.csv","analysis.json","inference_diagnostics.json"}:
        raise ValueError("Original verification coverage differs")
    for name,digest in verification["files_sha256"].items():
        if file_hash(run/name)!=digest:raise ValueError("Verified output checksum differs: "+name)
    diagnostics=read("inference_diagnostics.json");report=read("probe_report.json")
    expected_checks={m+"/"+s for m in ("BCE","P_low","P_high") for s in manifest["config"]["adapter_states"]}
    checks=diagnostics["exact_determinism_checks"]
    if set(checks)!=expected_checks or not all(v is True for v in checks.values()):raise ValueError("Deterministic checks incomplete")
    if diagnostics["official_cache_parity"]["status"]!="PASS" or not diagnostics["fresh_vision_cache_exact"]:raise ValueError("Saved parity failed")
    if report["status"]!="INFERENCE_COMPLETE_OFFLINE_VERIFIED" or report["training_performed"] or report["outer_scoring"]:
        raise ValueError("Unexpected execution scope")
    if diagnostics!=report["inference"] or diagnostics["new_image_batches"]+diagnostics["resumed_image_batches"]!=expected_batches:
        raise ValueError("Inference batch coverage differs")
    rows=[]
    for model in ("BCE","P_low","P_high"):
        for state in manifest["config"]["adapter_states"]:
            for painting in manifest["images"]:
                path=run/"logits"/model/state/(painting+".json")
                if read_seal(path)!=file_hash(path):raise ValueError("Raw logit checksum differs")
                payload=json.loads(path.read_text())
                expected=dict(run=fingerprint(identity),model=model,adapter_state=state,painting=painting)
                if payload["identity"]!=expected:raise ValueError("Raw logit identity differs")
                part=payload["rows"]
                if any((r["model"],r["adapter_state"],r["painting"])!=(model,state,painting) for r in part):raise ValueError("Misfiled logit batch")
                rows.extend(part)
    with tempfile.TemporaryDirectory() as scratch:
        module_path=Path(scratch)/"analysis.py";module_path.write_bytes(blobs["src/st3/interaction_probe.py"])
        spec=importlib.util.spec_from_file_location("st3._task031_pinned_analysis",module_path)
        pinned=importlib.util.module_from_spec(spec);spec.loader.exec_module(pinned)
        frame,analysis=pinned.analyze(rows,manifest,targets)
        checkpoint_validation="NOT_REQUESTED; selected targets still match the pre-inference committed hash"
        if inputs is not None:
            if checkpoints is None:raise ValueError("Both original inputs and checkpoints are required")
            repeated,truth=pinned.make_manifest(inputs,checkpoints,root,manifest["config"])
            if repeated!=manifest or truth!=targets:raise ValueError("Original training-only target/partition reconstruction differs")
            checkpoint_validation="PASS: all three original checkpoint SHA/source/config/selection and selected training targets"
    if not same(analysis,read("analysis.json")):raise ValueError("Saved analysis/intervals do not reproduce within1e-12")
    saved=pd.read_csv(run/"rectangles.csv",float_precision="round_trip")
    pd.testing.assert_frame_equal(saved,frame,check_exact=False,atol=1e-12,rtol=0)
    # A second, direct four-key reconstruction checks every raw mixed contrast.
    index={(r["model"],r["adapter_state"],r["condition"],r["painting"],r["target_language"]):r["logit"] for r in rows}
    for r in frame.to_dict("records"):
        base=(r["model"],r["adapter_state"],r["condition"])
        value=index[(*base,r["I"],r["a"]) ]-index[(*base,r["I"],r["b"]) ]-index[(*base,r["J"],r["a"]) ]+index[(*base,r["J"],r["b"]) ]
        if abs(value-r["D"])>1e-12:raise ValueError("Independent raw mixed contrast differs")
    return {"status":"PASS","frozen_source":SOURCE,"source_files":len(source_hashes),"sealed_image_batches":expected_batches,
            "raw_logits":len(rows),"rectangles_per_model_state_condition":len(manifest["rectangles"]),"analysis_rows":len(frame),
            "analysis_absolute_tolerance":1e-12,"original_checkpoint_target_verification":checkpoint_validation,
            "engineering_record":{k:report[k] for k in ("GPU","elapsed_seconds","peak_allocated_bytes","peak_reserved_bytes")},
            "parity_record":diagnostics["official_cache_parity"],"six_saved_replays_pass":True,
            "verification_boundary":"Independent read-only CPU artifact/source/key/statistical replay; no new encoder or hardware measurement",
            "original_files_sha256":{p.relative_to(run).as_posix():file_hash(p) for p in sorted(run.rglob("*")) if p.is_file()},
            "analysis":analysis}


def read_seal(path):return json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]


def verify_archive(archive,root,expected=ZIP_SHA,*,inputs=None,checkpoints=None):
    with tempfile.TemporaryDirectory() as scratch:
        directory=Path(scratch);manifest=unpack(archive,directory,expected)
        result=reconstruct(directory/"run",root,inputs=inputs,checkpoints=checkpoints)
    return dict(result,export_sha256=expected,export_files=len(manifest))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--export-zip",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--inputs",type=Path);p.add_argument("--checkpoints",type=Path)
    args=p.parse_args()
    # Failure never leaves a stale successful receipt at the requested NEW output path.
    if args.output.exists():p.error("Choose a new output filename; historical receipts are read-only")
    result=verify_archive(args.export_zip,ROOT,inputs=args.inputs,checkpoints=args.checkpoints)
    atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("analysis","original_files_sha256")},indent=2))


if __name__=="__main__":main()
