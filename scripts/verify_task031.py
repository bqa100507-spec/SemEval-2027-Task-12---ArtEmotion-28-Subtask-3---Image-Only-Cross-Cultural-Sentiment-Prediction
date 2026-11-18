"""CPU-only source/cache/key/metric verification of a completed TASK-031 probe."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from st3.probe_inference import verify_outputs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-dir",type=Path,required=True)
    args=p.parse_args();manifest=json.loads((ROOT/"configs/task031_manifest.json").read_text())
    targets=json.loads((args.run_dir/"selected_training_targets.json").read_text())
    summary=verify_outputs(args.run_dir,manifest,targets,ROOT)
    print(json.dumps({"verification":"PASS","mechanism_outcome":summary["outcome"],"GPU_replay":False},indent=2))


if __name__=="__main__":main()
