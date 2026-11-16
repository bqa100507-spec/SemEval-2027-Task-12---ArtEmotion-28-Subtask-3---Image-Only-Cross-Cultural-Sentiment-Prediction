"""Read-only CPU input/history/source validation; never opens the confirmation payload."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from st3.exp023_data import load_inputs
from st3.experiment_workflow import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((ROOT/"configs/exp023.json").read_text())
    manifest=json.loads((args.inputs/"input_manifest.json").read_text())
    expected=manifest["source_files_sha256"]
    actual={name:sha256(ROOT/name) for name in expected}
    if expected!=actual or manifest["config_sha256"]!=sha256(ROOT/"configs/exp023.json"):
        raise ValueError("Source/configuration differs from the private handoff")
    train,captions,_,_,_,_,historical=load_inputs(args.inputs,cfg)
    result={"status":"PASS","source_revision":manifest["source_revision"],"source_files":len(actual),
            "input_files":len(manifest["files"]),"training_pairs":len(train),"painting_count":train.painting.nunique(),
            "caption_rows":len(captions),"historical_parity":historical,"confirmation_payload_parsed":False,
            "development_gold_read":False,"training_or_gpu_execution":False}
    write_json(args.report,result)
    print(json.dumps(result,indent=2))
