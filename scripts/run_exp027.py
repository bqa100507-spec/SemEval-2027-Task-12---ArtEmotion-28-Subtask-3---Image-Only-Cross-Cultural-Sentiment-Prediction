"""EXP-027 authorized future L4 pilot only. No outer-fold/CV/Development endpoint."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from st3.exp027_data import load_config,read_inputs
from st3.vlm_semantics import atomic_json,file_hash


def build_backend(args,config):
    spec=importlib.util.spec_from_file_location("exp026_backend_only",ROOT/"scripts/run_exp026.py")
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    return old.build_backend(args,config)  # Only pinned assets/encoder construction, never EXP-026 training.


def worker(args):
    import torch
    from st3.vlm_supervised_training import FeatureCache
    from st3.exp027_pilot import run_pilot
    started=time.monotonic();report={"experiment":"EXP-027","status":"RUNNING","outer_evaluation_performed":False,"GPU_training_authorized":True,"production":"E50+L unchanged"}
    def check():
        if time.monotonic()-started>args.max_seconds-30:raise TimeoutError("Pilot budget exhausted; completed epochs/models persist")
    try:
        config,protocol=load_config(ROOT);fit,val,parts=read_inputs(args.inputs,config,protocol)
        backend,names=build_backend(args,config)
        report.update(GPU=torch.cuda.get_device_name(),target_modules=names,model_load_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        # Allow validation image pixels, never outer paintings. No frozen decoder-state extraction needed.
        import pandas as pd
        cache=FeatureCache(args.cache,args.inputs,config,pd.concat([fit,val]))
        sources=["scripts/run_exp027.py","src/st3/contrast_qlora.py","src/st3/exp027_data.py","src/st3/exp027_pilot.py",
                 "scripts/run_exp026.py","src/st3/vlm_supervised.py","src/st3/vlm_supervised_training.py","src/st3/vlm_semantics.py",
                 "src/st3/caption_data.py","src/st3/vlm_exploratory.py","src/st3/data.py","requirements-exp026.lock.txt"]
        identity={"config":config,"source_sha256":{name:file_hash(ROOT/name) for name in sources},"GPU":torch.cuda.get_device_name(),
                  "source_revision":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
                  "inputs":config["input_sha256"],"fresh_initialization":"Pinned pretrained + seed2027 zero-output decoder LoRA; never EXP-026 adapters"}
        run_pilot(args,config,protocol,backend,cache,fit,val,parts,report,check,identity)
    except torch.cuda.OutOfMemoryError as error:report.update(status="CUDA_OOM",error=str(error),retry_attempted=False)
    except Exception as error:report.update(status="STOPPED_BUDGET" if isinstance(error,TimeoutError) else "FAILED",error=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        if torch.cuda.is_initialized():report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(args.run_dir/"pilot_report.json",report);print(json.dumps(report,indent=2),flush=True)
    return 0 if report["status"]=="GPU_PILOT_COMPLETE" else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=["pilot"],default="pilot")
    p.add_argument("--inputs",type=Path,required=True);p.add_argument("--cache",type=Path,required=True);p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--model-path",type=Path);p.add_argument("--model-cache",type=Path,default=Path("cache/exp027_model"))
    p.add_argument("--max-seconds",type=float,default=3600);p.add_argument("--authorize-gpu",action="store_true")
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS);args=p.parse_args()
    import math
    if not args.authorize_gpu:p.error("No GPU authorization. This implementation task authorizes CPU validation only.")
    if not math.isfinite(args.max_seconds) or args.max_seconds<=30:p.error("Finite budget greater than30s required")
    args.max_seconds=min(args.max_seconds,7200);args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:return worker(args)
    log=args.run_dir/("pilot_"+str(time.time_ns())+".log")
    try:
        with log.open("w") as handle:
            result=subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"],stdout=handle,stderr=subprocess.STDOUT,timeout=args.max_seconds)
        print(log.read_text(errors="replace")[-16000:]);return result.returncode
    except subprocess.TimeoutExpired:
        atomic_json(args.run_dir/"pilot_report.json",{"experiment":"EXP-027","status":"STOPPED_BUDGET","hard_seconds":args.max_seconds,"no_outer_evaluation":True})
        return 2


if __name__=="__main__":raise SystemExit(main())
