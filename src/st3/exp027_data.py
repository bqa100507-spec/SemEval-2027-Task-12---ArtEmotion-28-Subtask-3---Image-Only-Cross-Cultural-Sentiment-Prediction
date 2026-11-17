"""Released training-only inputs; exclude outer-fold labels before indexing votes."""
import csv
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

from .caption_data import inner_partition, require_isolation
from .vlm_semantics import file_hash, fingerprint


def load_config(root):
    root=Path(root)
    protocol=json.loads((root/"configs/exp027.json").read_text(encoding="utf-8"))
    raw=(root/"configs/exp026.json").read_bytes().replace(b"\r\n",b"\n")
    if hashlib.sha256(raw).hexdigest()!=protocol["base_config_sha256"]:raise ValueError("EXP-026 scientific base differs")
    config=copy.deepcopy(json.loads(raw));config["fold"]=protocol["outer_fold"];config["max_epochs"]=protocol["max_epochs"]
    config["pearson_epsilon"]=protocol["pearson_epsilon"]
    config["input_sha256"]={k:v for k,v in config["input_sha256"].items() if k in ("training/pairs.csv","training/protocol.csv")}
    return config,protocol


def partitions(metadata,config,protocol):
    if len(metadata)!=1335 or metadata.painting.duplicated().any():raise ValueError("Original P2 metadata coverage differs")
    outer=metadata[metadata.fold.ne(protocol["outer_fold"])]
    held=metadata[metadata.fold.eq(protocol["outer_fold"])]
    inner,validation=inner_partition(outer,config,protocol["outer_fold"])
    require_isolation(inner,validation,held)
    def choose(frame,count,seed):
        names=sorted(frame.painting,key=lambda name:fingerprint([seed,name]))[:count]
        if len(names)!=count:raise ValueError("Insufficient pilot paintings")
        return frame[frame.painting.isin(names)].copy()
    pilot_inner=choose(inner,protocol["pilot"]["training_paintings"],config["seed"])
    pilot_val=choose(validation,protocol["pilot"]["validation_paintings"],config["seed"]+1)
    require_isolation(pilot_inner,pilot_val,held)
    return inner,validation,pilot_inner,pilot_val,held


def filter_votes(rows,allowed):
    """Rows outside the permitted population never have target fields indexed."""
    selected=[]
    for row in rows:
        if row["painting"] not in allowed:continue
        selected.append({key:row[key] for key in ("painting","language","image","image_sha256","fold","split","k","n")})
    result=pd.DataFrame(selected)
    if result.empty:raise ValueError("No permitted training pairs")
    for key in ("k","n","fold"):result[key]=pd.to_numeric(result[key],errors="raise")
    if (not result.split.eq("train").all() or result[["painting","language"]].duplicated().any()
            or not np.isfinite(result[["k","n"]]).all().all() or (result.n<5).any() or (result.k<0).any() or (result.k>result.n).any()
            or (result.k%1!=0).any() or (result.n%1!=0).any()):
        raise ValueError("Invalid permitted training votes")
    return result


def read_inputs(inputs,config,protocol,*,pilot=True,include_validation=True,excluded_label_folds=()):
    inputs=Path(inputs)
    for name,digest in config["input_sha256"].items():
        if file_hash(inputs/name)!=digest:raise ValueError("Pinned EXP-027 training input differs: "+name)
    meta=pd.read_csv(inputs/"training/protocol.csv")
    allowed_meta=meta[meta.role.eq("development")]
    if len(meta[meta.role.eq("holdout")])!=236:raise ValueError("Sealed identifier count differs")
    inner,validation,pi,pv,held=partitions(allowed_meta,config,protocol)
    fitting,stopping=(pi,pv) if pilot else (inner,validation)
    if excluded_label_folds:fitting=fitting[~fitting.fold.isin(excluded_label_folds)]
    if not include_validation:stopping=stopping.iloc[:0]
    permitted=pd.concat([fitting,stopping])
    require_isolation(permitted,held,meta[meta.role.eq("holdout")])
    with (inputs/"training/pairs.csv").open(encoding="utf-8",newline="") as handle:
        pairs=filter_votes(csv.DictReader(handle),set(permitted.painting))
    expected=permitted[["painting","image","image_sha256","fold"]].sort_values("painting").reset_index(drop=True)
    actual=pairs[expected.columns].drop_duplicates().sort_values("painting").reset_index(drop=True)
    if not actual.equals(expected):raise ValueError("Permitted pair metadata differs from P2 protocol")
    fit=pairs[pairs.painting.isin(fitting.painting)].copy()
    val=pairs[pairs.painting.isin(stopping.painting)].copy()
    require_isolation(fit,val,held)
    return fit,val,{"inner":fitting,"validation":stopping,"outer_held_ids_only":held}
