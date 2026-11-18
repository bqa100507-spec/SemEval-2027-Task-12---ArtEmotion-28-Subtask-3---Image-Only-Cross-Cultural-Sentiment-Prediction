"""Read-only inner-validation reporting; never selects a checkpoint or lambda."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import score,painting_scores
from .submission import validate_predictions
from .vlm_semantics import atomic_json,file_hash


def evaluate(gold,predictions,*,replicates=10000,seed=27029):
    results={};paintings={}
    for name,pred in predictions.items():
        validate_predictions(pred,gold)
        metrics=score(gold,pred)
        frame=gold.merge(pred,on=["painting","language"],validate="one_to_one")
        frame["p"]=frame.k/frame.n
        q=frame.p_positive.clip(1e-7,1-1e-7)
        descriptive=float((-(frame.k*np.log(q)+(frame.n-frame.k)*np.log1p(-q))).sum()/frame.n.sum())
        results[name]={"mean_contrast":metrics["cultural_contrast"],"eligible_paintings":metrics["contrast_paintings"],
                       "macro_log_loss_bits":metrics["log_loss"],"log_loss_eligible_languages":metrics["log_loss_languages"],
                       "log_loss_status":"AVAILABLE" if metrics["log_loss"] is not None else "UNAVAILABLE_NO_LANGUAGE_WITH_20_PAIRS",
                       "descriptive_vote_weighted_BCE_nats_from_rounded_predictions":descriptive}
        paintings[name]=painting_scores(frame).set_index("painting").contrast
    comparisons={}
    for name in ("P_low","P_high"):
        paired=pd.concat([paintings["BCE"].rename("control"),paintings[name].rename("candidate")],axis=1).dropna()
        delta=(paired.candidate-paired.control).to_numpy(dtype=float)
        interval=None
        if len(delta):
            rng=np.random.default_rng(seed)
            boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(replicates)])
            interval=np.quantile(boot,[.025,.975]).tolist()
        comparisons[name]={"mean_contrast_difference_vs_BCE":float(delta.mean()) if len(delta) else None,
                           "paired_eligible_paintings":len(delta),"wins":int((delta>1e-12).sum()),"losses":int((delta< -1e-12).sum()),
                           "ties":int((np.abs(delta)<=1e-12).sum()),"painting_paired_percentile_95_CI":interval,
                           "bootstrap_replicates":replicates,"bootstrap_seed":seed}
    return {"scope":"READ_ONLY_INNER_VALIDATION_NOT_FOR_SELECTION_OR_TUNING","metrics":results,"paired_comparisons":comparisons,
            "scorer":"Existing published-specification ST3 scorer; not an independently obtained hosted scorer",
            "uncertainty_caveat":"Descriptive inner-validation interval after BCE checkpoint selection; not independent generalization evidence"}


def verify(directory):
    directory=Path(directory);receipt=json.loads((directory/"receipt.json").read_text())
    for name,digest in receipt["files_sha256"].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.parent.resolve()) or file_hash(path)!=digest:raise ValueError("Validation artifact integrity failure: "+name)
    gold=pd.read_csv(directory/"targets.csv",float_precision="round_trip")
    predictions={name:pd.read_csv(directory/(name+".csv"),float_precision="round_trip") for name in receipt["variants"]}
    for name,pred in predictions.items():
        if pred[["painting","language"]].to_dict("records")!=receipt["keys"]:raise ValueError("Validation keys/order differ")
    return evaluate(gold,predictions)


def publish(run_dir,val,payloads):
    directory=Path(run_dir)/"inner_validation";directory.mkdir(parents=True,exist_ok=True)
    gold=val[["painting","language","k","n"]].reset_index(drop=True)
    gold.to_csv(directory/"targets.csv",index=False,lineterminator="\n")
    files={"targets.csv":file_hash(directory/"targets.csv")}
    for name,payload in payloads.items():
        pred=gold[["painting","language"]].assign(p_positive=payload["validation_prediction_six_decimals"])
        validate_predictions(pred,gold)
        pred.to_csv(directory/(name+".csv"),index=False,float_format="%.6f",lineterminator="\n")
        files[name+".csv"]=file_hash(directory/(name+".csv"))
        files["../models/"+name+".pt"]=file_hash(Path(run_dir)/"models"/(name+".pt"))
    atomic_json(directory/"receipt.json",{"files_sha256":files,"keys":gold[["painting","language"]].to_dict("records"),
                 "variants":list(payloads),"selected_epochs":{n:p["selected_epoch"] for n,p in payloads.items()},
                 "checkpoint_selection_unchanged":"Minimum original inner-validation BCE; all reporting is subsequent and read-only"})
    result=verify(directory);atomic_json(directory/"metrics.json",result)
    return result
