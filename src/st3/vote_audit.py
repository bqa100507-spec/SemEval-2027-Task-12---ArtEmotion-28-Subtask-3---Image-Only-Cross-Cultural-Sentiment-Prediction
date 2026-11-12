"""Descriptive Jeffreys-prior ordering uncertainty, never a training target."""

import time

import numpy as np
import pandas as pd

from .cli import json_output
from .contrast_study import new_output
from .metrics import painting_scores
from .vote_inputs import COUNT_COLUMNS,empirical_distributions,input_context
from .polarity_workflow import read_sealed,seal


def quantiles(values):
    return {str(q):float(np.quantile(values,q)) for q in (0,.1,.25,.5,.75,.9,1)} if len(values) else {}


def ordering_audit(train,config):
    rng=np.random.default_rng(config["posterior_seed"]);rows=[];paintings=[]
    prior_a,prior_b=config["posterior_prior"];samples=config["posterior_samples"]
    for painting,group in train.sort_values(["painting","language"]).groupby("painting",sort=True):
        n=group.n.to_numpy();k=group.k.to_numpy();rates=k/n;languages=group.language.tolist()
        if len(group)<2:
            paintings.append({"painting":painting,"comparisons":0,"uncertain_fraction":None,"near_tie_fraction":None,"dominated_by_uncertainty":None});continue
        posterior=rng.beta(k+prior_a,n-k+prior_b,size=(samples,len(group)))
        first=len(rows)
        for left in range(len(group)):
            for right in range(left+1,len(group)):
                probability=float((posterior[:,left]>posterior[:,right]).mean())
                confidence=max(probability,1-probability)
                # Exact integer cross-products avoid 0.8-0.6 crossing the
                # descriptive <=0.2 boundary through floating-point rounding.
                difference=float((k[left]*n[right]-k[right]*n[left])/(n[left]*n[right]))
                rows.append({"painting":painting,"language_a":languages[left],"language_b":languages[right],
                    "n_a":int(n[left]),"n_b":int(n[right]),"min_votes":int(min(n[left],n[right])),
                    "total_votes":int(n[left]+n[right]),"difference":difference,"abs_difference":abs(difference),
                    "observed_tie":bool(k[left]*n[right]==k[right]*n[left]),"p_a_gt_b":probability,
                    "confidence":confidence,"uncertain":confidence<config["ordering_uncertain_confidence"],
                    "near_tie":abs(difference)<=config["near_tie_difference"]})
        current=rows[first:];uncertain=float(np.mean([r["uncertain"] for r in current]));near=float(np.mean([r["near_tie"] for r in current]))
        paintings.append({"painting":painting,"comparisons":len(current),"uncertain_fraction":uncertain,"near_tie_fraction":near,
                          "dominated_by_uncertainty":uncertain>=config["painting_dominated_fraction"]})
    return pd.DataFrame(rows),pd.DataFrame(paintings)


def reliability_summary(train,comparisons,paintings,config):
    distribution,_=empirical_distributions(train)
    entropy=-(distribution*np.log2(np.maximum(distribution,1e-300))).sum(1)
    eligibility=painting_scores(train.assign(p_positive=train.p))
    return {"pairs":len(train),"paintings":train.painting.nunique(),"binary_votes":int(train.n.sum()),
        "positive_votes":int(train.k.sum()),"n_quantiles":quantiles(train.n),"n5_pairs":int(train.n.eq(5).sum()),
        "positive_rate_quantiles":quantiles(train.p),"extreme_rate_fraction":float(train.p.isin([0.,1.]).mean()),
        "zero_rate_pairs":int(train.p.eq(0).sum()),"one_rate_pairs":int(train.p.eq(1).sum()),
        "emotion_entropy_bits":quantiles(entropy),"languages_per_painting":quantiles(train.groupby("painting").size()),
        "contrast_eligible_paintings":int(eligibility.status.eq("scored").sum()),
        "language_comparisons":len(comparisons),"observed_ties":int(comparisons.observed_tie.sum()),
        "absolute_rate_difference":quantiles(comparisons.abs_difference),
        "posterior_confidence_ge_095":float(comparisons.confidence.ge(.95).mean()),
        "posterior_confidence_ge_080":float(comparisons.confidence.ge(.8).mean()),
        "posterior_confidence_lt_060":float(comparisons.confidence.lt(.6).mean()),
        "posterior_uncertain_fraction":float(comparisons.uncertain.mean()),
        "near_tie_fraction":float(comparisons.near_tie.mean()),
        "paintings_dominated_by_uncertainty":int(paintings.dominated_by_uncertainty.fillna(False).sum()),
        "sampling":{k:config[k] for k in ("posterior_prior","posterior_samples","posterior_seed","ordering_uncertain_confidence","near_tie_difference","painting_dominated_fraction")},
        "assumptions":"Independent language-specific binomial votes, exchangeable annotators and independent Beta(.5,.5) priors; posterior confidence is descriptive, not ground truth or an evaluation noise ceiling.",
        "training_policy":"Empirical counts only. No posterior smoothing, filtering or reliability weighting."}


def run_audit(args):
    started=time.perf_counter();ctx=input_context(args)
    if (args.run_dir/"audit").exists():read_sealed(args.run_dir/"audit",ctx.identity);return
    comparisons,paintings=ordering_audit(ctx.train,ctx.config)
    summary=reliability_summary(ctx.train,comparisons,paintings,ctx.config)
    with new_output(args.run_dir/"audit") as stage:
        ctx.train.to_csv(stage/"targets.csv",index=False)
        comparisons.to_csv(stage/"language_comparisons.csv",index=False)
        paintings.to_csv(stage/"painting_reliability.csv",index=False)
        ctx.train.groupby("n").size().rename("pairs").to_csv(stage/"vote_count_distribution.csv")
        bins=pd.cut(comparisons.abs_difference,[-.001,0,.1,.2,.4,.6,1.],include_lowest=True)
        counts=pd.cut(comparisons.min_votes,[0,5,6,8,10,1000])
        comparisons.assign(difference_bin=bins,vote_bin=counts).groupby(["vote_bin","difference_bin"],observed=True).agg(
            comparisons=("confidence","size"),mean_abs_difference=("abs_difference","mean"),
            mean_confidence=("confidence","mean"),uncertain_fraction=("uncertain","mean")).to_csv(stage/"ordering_bins.csv")
        json_output(summary,stage/"summary.json")
        seal(stage,ctx.identity,{"status":"AUDITED","seconds":time.perf_counter()-started,"all_eight_counts_exact":True,
                                "locked_votes_retained":0,"caption_fields_retained":0})
    print(f"Vote audit: {summary['pairs']} pairs; {summary['posterior_uncertain_fraction']:.1%} uncertain orderings in {time.perf_counter()-started:.2f}s",flush=True)
