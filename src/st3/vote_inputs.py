"""Annotation-only development-training inputs; no captions or locked votes retained."""

import csv
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import data
from .baseline import file_sha256
from .cli import ROOT,json_output
from .contrast_study import new_output
from .polarity_workflow import StageBlocked,read_sealed,seal

EMOTIONS=("amusement","awe","contentment","excitement","anger","disgust","fear","sadness")
COUNT_COLUMNS=["count_"+name for name in EMOTIONS]


def config021():
    cfg=json.loads((ROOT/"configs/exp021.json").read_text())
    if cfg["emotions"]!=list(EMOTIONS) or cfg["text_weight"]!=.1 or cfg["auxiliary_normalization"]!="ln(8)":
        raise ValueError("Frozen emotion order, coefficient or normalization changed")
    return cfg


def development_votes(root,protocol):
    allowed=protocol[protocol.role.eq("development")].copy()
    if allowed.painting.duplicated().any() or not allowed.fold.between(1,5).all():
        raise ValueError("Invalid original painting protocol")
    allowed_ids=set(allowed.painting)
    votes=[]
    with (root/data.ANNOTATIONS).open(encoding="utf-8",newline="") as handle:
        for annotation_id,row in enumerate(csv.DictReader(handle)):
            # Filter by split/identity BEFORE reading or retaining its emotion.
            painting=row["painting"].strip()
            if row["split"].strip()!="train" or painting not in allowed_ids:continue
            emotion=row["emotion"].strip().lower()
            if emotion in data.AMBIGUOUS:continue
            if emotion not in EMOTIONS:raise ValueError("Unknown original training emotion")
            language=row["language"].strip()
            language={"arabic":"Arabic","chinese":"Chinese","english":"English"}.get(language,language)
            votes.append({"annotation_id":annotation_id,"painting":painting,"language":language,
                          "emotion":emotion,"image":row["image_name"].strip(),"split":"train"})
    frame=pd.DataFrame(votes)
    if frame.empty:raise ValueError("No development-training votes")
    counts=frame.groupby(data.KEYS+["emotion"]).size().unstack(fill_value=0).reindex(columns=EMOTIONS,fill_value=0)
    counts.columns=COUNT_COLUMNS
    counts=counts.reset_index()
    counts["n"]=counts[COUNT_COLUMNS].sum(axis=1)
    counts["k"]=counts[COUNT_COLUMNS[:4]].sum(axis=1)
    counts=counts[counts.n.ge(5)].copy();counts["p"]=counts.k/counts.n
    original=data.aggregate(frame)
    check=counts.merge(original,on=data.KEYS,suffixes=("","_binary"),validate="one_to_one")
    data.require_same_pairs(counts,original)
    if not (check.n.eq(check.n_binary)&check.k.eq(check.k_binary)&check.p.eq(check.p_binary)).all():
        raise ValueError("Eight-way and original binary vote reconstruction differ")
    images=frame[data.KEYS+["image"]].drop_duplicates()
    if images.duplicated(data.KEYS).any():raise ValueError("Artwork identity differs within a pair")
    counts=counts.merge(images,on=data.KEYS,validate="one_to_one").merge(allowed.drop(columns="image"),on="painting",validate="many_to_one")
    expected=dict(allowed[["painting","image"]].itertuples(index=False,name=None))
    if not counts.image.eq(counts.painting.map(expected)).all():raise ValueError("Original artwork path differs")
    counts["split"]="train"
    counts=counts.sort_values(data.KEYS).reset_index(drop=True)
    empirical_distributions(counts)
    return counts,frame.merge(counts[data.KEYS],on=data.KEYS,validate="many_to_one")


def empirical_distributions(frame):
    c=frame[COUNT_COLUMNS].to_numpy(dtype=np.float64)
    n=c.sum(axis=1);k=c[:,:4].sum(axis=1)
    if (c.shape[1]!=8 or not np.isfinite(c).all() or (c<0).any() or (c!=np.floor(c)).any() or (n<5).any()
            or not np.array_equal(n,frame.n) or not np.array_equal(k,frame.k)):
        raise ValueError("Invalid empirical eight-vote counts")
    d=c/n[:,None];y=k/n
    if not np.allclose(d.sum(1),1.,atol=1e-12,rtol=0.) or not np.array_equal(y,frame.p):
        raise ValueError("Binary/distribution target inconsistency")
    if "split" in frame and not frame.split.eq("train").all():raise ValueError("Only original training targets are allowed")
    return d,y


def verify_parent(args,cfg):
    if file_sha256(ROOT/"configs/exp020.json")!=cfg["exp020_config_sha256"]:raise ValueError("EXP-020 configuration changed")
    path=args.exp020_dir/"decision/run.json"
    if not path.exists():raise StageBlocked("Completed EXP-020 missing; on its original branch run run_exp020.ps1 -Stage CV then -Stage Verify")
    profile=json.loads(path.read_text())["identity"]
    old=profile["runtime_and_source"]
    for name,digest in old["source_sha256"].items():
        if file_sha256(ROOT/name)!=digest:raise ValueError("Historical source changed: "+name)
    for name,digest in profile["runners"].items():
        if file_sha256(ROOT/"scripts"/name)!=digest:raise ValueError("Historical runner changed")
    for name,digest in cfg["parent_manifests"].items():
        path=args.exp020_dir/name
        if file_sha256(path)!=digest:raise ValueError("EXP-020 manifest changed: "+name)
        read_sealed(path.parent,profile)
    decision=read_sealed(args.exp020_dir/"decision",profile)
    replay=read_sealed(args.exp020_dir/"verification",profile)
    cv=read_sealed(args.exp020_dir/"cv",profile)
    attempts=sorted((args.exp020_dir/"verification_attempts").glob("*.json"))
    latest=json.loads(attempts[-1].read_text()) if attempts else {}
    if latest.get("status")!="PASS" or latest.get("cv_manifest_sha256")!=file_sha256(args.exp020_dir/"cv/run.json"):
        raise StageBlocked("Latest EXP-020 replay incomplete; verify on original branch")
    if decision["status"]!="COMPETITION_NO_GO" or not decision["all_seeds_finite_and_replayed"] or replay["checkpoints_replayed"]!=30:
        raise ValueError("Unexpected completed EXP-020 decision/replay")
    for phase in ("cv","verification"):
        if decision[phase+"_manifest_sha256"]!=file_sha256(args.exp020_dir/phase/"run.json"):raise ValueError("EXP-020 decision evidence changed")
    for key,digest in cv["seed_manifests"].items():
        fold,name=key.split("/");path=args.exp020_dir/"work"/f"fold_{fold}"/name
        if file_sha256(path/"run.json")!=digest:raise ValueError("EXP-020 checkpoint reference changed")
    if any((args.exp020_dir/phase).exists() for phase in ("confirm","dev")):raise ValueError("Expected locked/public scoring to remain sealed")
    return decision


def audit_identity(args):
    cfg=config021()
    files=[ROOT/"src/st3"/name for name in ("vote_inputs.py","vote_audit.py","data.py","baseline.py")]
    return {"config_sha256":file_sha256(ROOT/"configs/exp021.json"),"annotations_sha256":cfg["annotations_sha256"],
            "protocol_sha256":cfg["protocol_sha256"],"audit_source_sha256":{p.name:file_sha256(p) for p in files},
            "python":platform.python_version(),"numpy":np.__version__}


def input_context(args,*,preflight=True):
    cfg=config021()
    if file_sha256(args.data_dir/data.ANNOTATIONS)!=cfg["annotations_sha256"] or file_sha256(args.exp017_dir/"audit/protocol.csv")!=cfg["protocol_sha256"]:
        raise ValueError("Original data or painting protocol changed")
    identity=audit_identity(args)
    if preflight:read_sealed(args.run_dir/"preflight",identity)
    protocol=pd.read_csv(args.exp017_dir/"audit/protocol.csv")
    if int(protocol.role.eq("development").sum())!=1335 or int(protocol.role.eq("holdout").sum())!=236:
        raise ValueError("Original development/locked identities differ")
    train,votes=development_votes(args.data_dir,protocol)
    reference=pd.read_csv(args.exp020_dir/"cv/predictions.csv",float_precision="round_trip")
    reference=reference[reference.model.eq("R0")][data.KEYS+["p_positive","fold"]]
    data.require_same_pairs(train,reference)
    check=train[data.KEYS+["fold"]].merge(reference,on=data.KEYS,suffixes=("_expected",""),validate="one_to_one")
    if not check.fold.eq(check.fold_expected).all():raise ValueError("Original OOF painting-fold identity differs")
    return SimpleNamespace(config=cfg,identity=identity,train=train,votes=votes,reference=reference)


def run_preflight(args):
    started=time.perf_counter();cfg=config021();decision=verify_parent(args,cfg);ctx=input_context(args,preflight=False)
    if (args.run_dir/"preflight").exists():read_sealed(args.run_dir/"preflight",ctx.identity);return
    roots=[args.exp017_dir,args.exp018_dir,args.exp019_dir,args.exp020_dir]
    history={str(p.resolve()):file_sha256(p) for root in roots for p in root.rglob("*") if p.is_file()}
    with new_output(args.run_dir/"preflight") as stage:
        json_output(history,stage/"protected_sha256.json")
        json_output({"EXP020":decision,"locked_emotion_rows_retained":0,"caption_fields_retained":0},stage/"provenance.json")
        seal(stage,ctx.identity,{"status":"INPUTS_VERIFIED","protected_files":len(history),"seconds":time.perf_counter()-started})
    print(f"Verified completed EXP-020; protected {len(history)} files; annotations only",flush=True)


def preserve(args,identity):
    read_sealed(args.run_dir/"preflight",identity)
    history=json.loads((args.run_dir/"preflight/protected_sha256.json").read_text())
    actual={str(p.resolve()) for root in (args.exp017_dir,args.exp018_dir,args.exp019_dir,args.exp020_dir) for p in root.rglob("*") if p.is_file()}
    if actual!=set(history):raise ValueError("Historical artifact set changed")
    for name,digest in history.items():
        if file_sha256(Path(name))!=digest:raise ValueError("Historical artifact changed: "+name)
    return len(history)
