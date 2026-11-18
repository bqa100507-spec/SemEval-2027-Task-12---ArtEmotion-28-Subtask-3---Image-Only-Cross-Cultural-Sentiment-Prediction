"""Independent CPU report reconstruction from inner-validation files; no encoder load."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from st3.exp027_validation import verify
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path);a=p.parse_args()
    print(json.dumps(verify(a.directory),indent=2,allow_nan=False))
