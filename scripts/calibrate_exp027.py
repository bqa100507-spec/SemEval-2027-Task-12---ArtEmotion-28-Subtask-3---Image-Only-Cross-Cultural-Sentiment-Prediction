"""CPU-only training-target/logit-gradient scale diagnostics; never a VLM fit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import torch
from st3.contrast_qlora import pearson_loss
from st3.exp027_data import load_config,read_inputs
from st3.vlm_supervised import vote_loss
from st3.vlm_semantics import fingerprint,atomic_json,file_hash


def calibrate(inputs):
    torch.set_num_threads(2)
    config,protocol=load_config(ROOT)
    fit,_,_=read_inputs(inputs,config,protocol,pilot=False,include_validation=False,excluded_label_folds=(1,))
    names=sorted(fit.painting.unique(),key=lambda name:fingerprint([2711,name]))[:64]
    rows=[];rng=np.random.default_rng(2711)
    for first in range(0,64,4):
        batch=fit[fit.painting.isin(names[first:first+4])].sort_values(["painting","language"])
        k=torch.tensor(batch.k.to_numpy());n=torch.tensor(batch.n.to_numpy())
        for sigma in (.001,.05,.2):
            # Label-independent synthetic logits; not cached/pseudo VLM predictions.
            logits=torch.tensor(rng.normal(.5,sigma,len(batch)),dtype=torch.float32,requires_grad=True)
            bce=vote_loss(logits,k,n);aux,eligible=pearson_loss(logits,k,n,batch.painting.tolist(),.01)
            gb=torch.autograd.grad(bce,logits,retain_graph=True)[0];ga=torch.autograd.grad(aux,logits)[0]
            ratio=float(ga.norm()/gb.norm())
            rows.append({"sigma":sigma,"bce":float(bce.detach()),"pearson":float(aux.detach()),"eligible":eligible,
                         "unscaled_logit_gradient_ratio":ratio})
    ratios=np.array([r["unscaled_logit_gradient_ratio"] for r in rows])
    median=float(np.median(ratios));p90=float(np.quantile(ratios,.9))
    # Frozen engineering targets:5%/15% median gradient, capped at50% p90.
    weights=[float(f"{min(target/median,.5/p90):.3g}") for target in (.05,.15)]
    return {"status":"TRAINING_ONLY_CPU_DIAGNOSTIC","seed":2711,"outer_fold_excluded":2,"examined_fold1_excluded":True,
            "paintings":len(names),"target_rows":int(fit.painting.isin(names).sum()),"batches":len(rows),
            "source_sha256":file_hash(Path(__file__)),"protocol_base_sha256":protocol["base_config_sha256"],
            "training_pairs_sha256":config["input_sha256"]["training/pairs.csv"],
            "median_unscaled_gradient_ratio":median,"p90_unscaled_gradient_ratio":p90,
            "recommended_grid":[0.,*weights],"historical_lambda_gradient_ratios":{str(w):{"median":w*median,"p90":w*p90} for w in (.01,.05)},
            "boundary":"Real permitted training votes + synthetic label-independent logits. Not Qwen feature/parameter gradients or optimal-lambda evidence.",
            "rows":rows}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--inputs",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();result=calibrate(args.inputs);atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
