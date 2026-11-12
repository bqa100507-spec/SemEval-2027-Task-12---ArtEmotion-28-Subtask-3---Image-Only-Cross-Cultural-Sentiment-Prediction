"""Optional matched-population raw/PCA description; never changes training targets."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from st3.baseline import file_sha256
from st3.emotion_targets import EMOTIONS,geometry_statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"runs/exp020_emotion_targets")
    parser.add_argument("--exp018-dir",type=Path,default=ROOT/"runs/exp018_polarity_captions/prepared")
    parser.add_argument("--fold",type=int,choices=range(1,6),default=1)
    parser.add_argument("--phase",choices=["inner","full"],default="full")
    args=parser.parse_args();root=args.run_dir
    target=root/"targets"/f"fold_{args.fold}"/args.phase
    output=root/"descriptive"/f"fold_{args.fold}_{args.phase}"
    if output.exists():raise ValueError("Descriptive output already exists; preserve it")
    records=pd.read_csv(root/"audit/pairs.csv");partition=pd.read_csv(target/"pairs.csv")
    index=pd.MultiIndex.from_frame(records[["painting","language"]]).get_indexer(pd.MultiIndex.from_frame(partition[["painting","language"]]))
    if (index<0).any():raise ValueError("Missing audit pairs")
    means=np.load(root/"audit/emotion_means.npy",mmap_mode="r",allow_pickle=False)
    counts=np.load(root/"audit/counts.npy",allow_pickle=False)[index]
    class SelectedMeans:
        shape=(len(index),8,means.shape[-1])
        def __getitem__(self,key):
            rows,channel=key
            return means[index[rows],channel]
    parent_records=pd.read_csv(args.exp018_dir/"audit/pairs.csv")
    pi=pd.MultiIndex.from_frame(parent_records[["painting","language"]]).get_indexer(pd.MultiIndex.from_frame(partition[["painting","language"]]))
    with np.load(args.exp018_dir/"audit/raw_means.npz",allow_pickle=False) as archive:
        polarity=archive["means"][pi,1:];available=archive["available"][pi,1:]
    config=json.loads((ROOT/"configs/exp020.json").read_text())
    summary,matrix,coobserved,variation,_=geometry_statistics(partition,SelectedMeans(),counts>0,polarity,available,config["near_identical_cosine"])
    output.mkdir(parents=True)
    pd.DataFrame(matrix,index=EMOTIONS,columns=EMOTIONS).to_csv(output/"raw_cosine.csv")
    pd.DataFrame(coobserved,index=EMOTIONS,columns=EMOTIONS).to_csv(output/"raw_coobserved.csv")
    variation.to_csv(output/"raw_variation.csv",index=False)
    raw_dispersion=pd.read_csv(root/"audit/raw_caption_dispersion.csv").merge(partition,on=["painting","language"],validate="many_to_one")
    raw_dispersion.to_csv(output/"raw_caption_dispersion.csv",index=False)
    payload={"pairs":len(partition),"geometry":summary,"script_sha256":file_sha256(Path(__file__)),
        "config_sha256":file_sha256(ROOT/"configs/exp020.json"),
        "input_manifests":{str(path):file_sha256(path) for path in (root/"audit/run.json",target/"run.json",args.exp018_dir/"audit/run.json")},
        "outputs_sha256":{p.name:file_sha256(p) for p in output.iterdir()}}
    (output/"summary.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=="__main__":main()
