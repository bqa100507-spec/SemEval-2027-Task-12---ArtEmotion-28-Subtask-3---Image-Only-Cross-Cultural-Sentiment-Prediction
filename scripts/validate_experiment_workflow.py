"""Reusable structural notebook contract validator; behavioral tests remain mandatory."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from st3.experiment_workflow import validate_notebook_contract


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks",type=Path,nargs="+")
    args=parser.parse_args()
    for path in args.notebooks:
        result=validate_notebook_contract(json.loads(path.read_text(encoding="utf-8")))
        print(path, json.dumps(result))
