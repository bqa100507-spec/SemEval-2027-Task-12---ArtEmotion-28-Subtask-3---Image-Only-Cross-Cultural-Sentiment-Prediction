"""Training-only TASK-031 preparation or explicitly authorized bounded L4 inference."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from st3.interaction_probe import make_manifest,checkpoints
from st3.vlm_semantics import atomic_json,file_hash,fingerprint


def prepare(args):
    config=json.loads((ROOT/"configs/task031.json").read_text())
    manifest,targets=make_manifest(args.inputs,args.checkpoints,ROOT,config)
    if manifest!=json.loads((ROOT/"configs/task031_manifest.json").read_text()):
        raise ValueError("Probe sampling differs from frozen training-only manifest")
    args.run_dir.mkdir(parents=True,exist_ok=True)
    for name,value in (("probe_manifest.json",manifest),("selected_training_targets.json",targets)):
        path=args.run_dir/name
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError("Existing probe inputs changed; use a new run directory")
        atomic_json(path,value)
    print(json.dumps(manifest["counts"],indent=2),flush=True)
    return manifest,targets


def worker(args):
    import torch
    from st3.probe_inference import SOURCE_FILES,run_inference,verify_outputs
    started=time.monotonic();report={"task":"TASK-031","status":"RUNNING","training_performed":False,"outer_scoring":False}
    atomic_json(args.run_dir/"verification.json",{"status":"NOT_VERIFIED", "mechanism_decision":"INVALID_UNTIL_PASS"})
    def check():
        if time.monotonic()-started>args.max_seconds-30:raise TimeoutError("Probe budget exhausted; sealed image batches persist")
    try:
        if subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain","--untracked-files=no"],text=True).strip():
            raise ValueError("Tracked source is modified; run a clean pinned checkout")
        manifest,targets=prepare(args);original,models=checkpoints(args.checkpoints,ROOT)
        config=original["config"]
        # Reuse only the pinned asset loader, never an earlier experiment's stages.
        spec=importlib.util.spec_from_file_location("qwen_asset_loader",ROOT/"scripts/run_exp026.py")
        loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
        backend,names=loader.build_backend(args,config)
        torch.use_deterministic_algorithms(True)
        identity={"source_revision":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
                  "source_sha256":{name:file_hash(ROOT/name) for name in SOURCE_FILES},
                  "manifest_sha256":fingerprint(manifest),"checkpoint_sha256":manifest["checkpoint_sha256"],
                  "original_run_identity":fingerprint(original),"runtime":config["runtime"],"GPU":torch.cuda.get_device_name()}
        identity_path=args.run_dir/"identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            raise ValueError("Source/config/runtime/GPU identity differs; do not bypass provenance; use a new run directory")
        atomic_json(identity_path,identity)
        report["target_modules"]=names
        report["inference"]=run_inference(backend,models,config,manifest,args.inputs,args.cache,args.run_dir,identity,check)
        check();checkpoints(args.checkpoints,ROOT)  # Original checkpoint bytes remain unchanged.
        verify_outputs(args.run_dir,manifest,targets,ROOT)
        report.update(status="INFERENCE_COMPLETE_OFFLINE_VERIFIED",GPU=torch.cuda.get_device_name())
    except torch.cuda.OutOfMemoryError as error:report.update(status="CUDA_OOM",error=str(error),retry_attempted=False)
    except Exception as error:report.update(status="STOPPED_BUDGET" if isinstance(error,TimeoutError) else "FAILED",error=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        if torch.cuda.is_initialized():report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(args.run_dir/"probe_report.json",report);print(json.dumps(report,indent=2),flush=True)
    return 0 if report["status"]=="INFERENCE_COMPLETE_OFFLINE_VERIFIED" else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["prepare","probe"],default="prepare")
    for name in ("inputs","checkpoints","run-dir"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--cache",type=Path,default=Path("cache/task031_vision"));p.add_argument("--model-path",type=Path)
    p.add_argument("--model-cache",type=Path,default=Path("cache/task031_model"));p.add_argument("--max-seconds",type=float,default=1500)
    p.add_argument("--authorize-gpu",action="store_true");p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.stage=="prepare":prepare(args);return 0
    if not args.authorize_gpu:p.error("GPU inference is not authorized. CPU preparation requires no GPU switch.")
    if not math.isfinite(args.max_seconds) or not 30<args.max_seconds<=1800:p.error("Use a finite inference limit between30 and1800 seconds")
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:return worker(args)
    atomic_json(args.run_dir/"verification.json",{"status":"NOT_VERIFIED", "mechanism_decision":"INVALID_UNTIL_PASS"})
    log=args.run_dir/("probe_"+str(time.time_ns())+".log")
    try:
        with log.open("w") as handle:
            result=subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"],stdout=handle,stderr=subprocess.STDOUT,timeout=args.max_seconds)
        print(log.read_text(errors="replace")[-12000:]);return result.returncode
    except subprocess.TimeoutExpired:
        atomic_json(args.run_dir/"probe_report.json",{"status":"STOPPED_BUDGET","max_seconds":args.max_seconds,"training_performed":False})
        return 2


if __name__=="__main__":raise SystemExit(main())
