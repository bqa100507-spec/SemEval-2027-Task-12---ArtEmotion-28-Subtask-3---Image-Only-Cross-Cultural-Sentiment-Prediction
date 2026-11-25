"""Re-evaluate verified EXP-037 predictions and characterize within-painting errors."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from st3 import data, submission
from st3.exp036_io import read, sha, write
from st3.exp037 import diagnostics
from st3.exp037_closure import recipe_prediction
from st3.exp037_diagnostics import (
    centered, decompose, logit_ablations, oracle_variants, pair_diversity, paired_bootstrap, paired_log_loss,
)
from st3.metrics import painting_scores, score

AUDITED = ("ridge","cscva","incumbent","A4","B6","C6","selected")
EXCLUDED = ("English","Arabic","Chinese")
GRID = (0., .10, .20, .25, .35, .50, .75, 1.)


def average(*parts):
    return [(1/len(parts), p) for p in parts]


def share(part, alpha):
    return [(1-alpha, "incumbent"), (alpha,part)]


def bounded_recipes():
    averages = {"abc_mean":average("A4","B6","C6"), "bc_mean":average("B6","C6"),
                "a34_mean":average("A3","A4"), "b56_mean":average("B5","B6"),
                "c56_mean":average("C5","C6"), "c678_mean":average("C6","C7","C8")}
    return averages | {name+"_blend20":share(recipe,.20) for name,recipe in averages.items()} | {
        "flat_60_20_20":[(.60,"ridge"),(.20,"cscva"),(.20,"C6")],
        "flat_65_15_20":[(.65,"ridge"),(.15,"cscva"),(.20,"C6")],
        "flat_65_20_15":[(.65,"ridge"),(.20,"cscva"),(.15,"C6")],
    }


def key(row):
    return row["full_contrast"], -row["full_log_loss"]


def compact_metrics(gold, pred):
    result = diagnostics(gold, pred, EXCLUDED)
    return {f"{name}_{field}": result[group][metric]
            for name,group in (("full","full"),("like","evaluation_like"))
            for field,metric in (("contrast","cultural_contrast"),("log_loss","log_loss"),
                                 ("eligible","contrast_paintings"),("macro_languages","log_loss_languages"))}


def grid_and_candidates(closure, output, dev, gold, predictions, audit):
    history = {row["id"]:row["state_sha256"] for row in audit["history"]}
    rows = []
    recipes = {"selected":share("C6",.20)}
    for checkpoint in history:
        for alpha in GRID:
            label = f"{checkpoint}_a{round(100*alpha):03d}"
            recipe = share(checkpoint,alpha)
            frame = recipe_prediction(dev,predictions,recipe)
            rows.append({"id":label,"epoch":checkpoint,"alpha":alpha,**compact_metrics(gold,frame)})
            recipes[label] = recipe
    posthoc = []
    for label,recipe in bounded_recipes().items():
        frame = recipe_prediction(dev,predictions,recipe)
        posthoc.append({"id":label,**compact_metrics(gold,frame)})
        recipes[label] = recipe
    stage_best = {stage: max((r for r in rows if r["epoch"].startswith(stage)),
                             key=lambda r: (*key(r),-r["alpha"],-int(r["epoch"][1:]))) for stage in "ABC"}
    stage_best_like = {stage: max((r for r in rows if r["epoch"].startswith(stage)),
                             key=lambda r: (r["like_contrast"],-r["like_log_loss"],-r["alpha"],-int(r["epoch"][1:])))
                             for stage in "ABC"}
    original = {"id":"selected",**compact_metrics(gold,predictions["selected"])}
    best = max([original,*rows,*posthoc],key=key)
    checkpoints_sha256 = history | {"ridge":sha(closure/"predictions/ridge.csv"),"cscva":sha(closure/"predictions/cscva.csv"),
                                      "incumbent":sha(closure/"predictions/incumbent.csv")}
    pd.DataFrame(rows).to_csv(output/"checkpoint_alpha_grid.csv",index=False)
    pd.DataFrame(posthoc).to_csv(output/"additional_ensembles.csv",index=False)
    write(output/"candidate_recipes.json",{"stage_best":stage_best,"stage_best_like":stage_best_like,
          "best":best,"original":original,"checkpoint_or_input_sha256":checkpoints_sha256,"recipes":recipes})
    # Offer a distinct archive for each distinct improvement on the original full selection.
    new_artifacts = []
    for row in sorted([*rows,*posthoc],key=key,reverse=True):
        if row["full_contrast"] <= original["full_contrast"]+1e-12:
            break
        label = row["id"]
        candidate = recipe_prediction(dev,predictions,recipes[label])
        if any(candidate.p_positive.equals(other) for _,other in new_artifacts):
            continue
        folder = output/"posthoc_candidates"/label
        folder.mkdir(parents=True)
        csv = folder/"predictions.csv"
        candidate.to_csv(csv,index=False,float_format="%.6f")
        zip_path = folder/f"exp037_posthoc_{label}_submission.zip"
        submission.pack(csv,zip_path,dev)
        pd.testing.assert_frame_equal(candidate,submission.validate_archive(zip_path,dev))
        write(folder/"receipt.json",{"selection":"POST_HOC_PUBLIC_DEVELOPMENT", "recipe":recipes[label],
             "metrics":row,"zip_sha256":sha(zip_path)})
        new_artifacts.append((label,candidate.p_positive))
    return rows, posthoc, stage_best,stage_best_like,best,recipes


def evaluate_scope(frame, models, output, label, comparisons):
    decomposed = {}
    all_paintings = {}
    all_languages = []
    parts = []
    yc = centered(frame,"p")
    for name in models:
        stats, paintings, orders = decompose(frame,name)
        decomposed[name] = stats
        all_paintings[name] = paintings
        q = frame[name].clip(.01,.99)
        per_language = frame[["painting","language","n"]].assign(
            bias=frame[name]-frame.p, ll=-(frame.p*np.log2(q)+(1-frame.p)*np.log2(1-q)),
            center_sq=(centered(frame,name)-yc)**2,
        ).groupby("language").agg(paintings=("painting","size"),votes=("n","sum"),
                                    mean_bias=("bias","mean"),log_loss=("ll","mean"),centered_mse=("center_sq","mean"))
        per_language["centered_rmse"] = per_language.centered_mse.pow(.5)
        per_language["order_accuracy"] = [hit/total if total else None for hit,total in per_language.index.map(orders)]
        per_language["eligible_order_pairs"] = [total for _,total in per_language.index.map(orders)]
        all_languages.append(per_language.drop(columns="centered_mse").assign(model=name))
        parts.append(paintings.assign(model=name))
    joined = pd.concat(parts).reset_index()
    joined["delta_vs_incumbent"] = joined.contrast - joined.painting.map(all_paintings["incumbent"].contrast)
    joined.to_csv(output/f"per_painting_{label}.csv",index=False)
    table = pd.concat(all_languages).reset_index()
    table.to_csv(output/f"per_language_{label}.csv",index=False)
    eligibility = all_paintings["incumbent"].contrast.dropna().index
    measures, loss_measures, largest = {}, {}, {}
    for a,b in comparisons:
        x,y = all_paintings[a].loc[eligibility].contrast,all_paintings[b].loc[eligibility].contrast
        difference = (x-y).sort_values(kind="stable")
        measures[f"{a}-{b}"] = paired_bootstrap(x,y)
        loss_measures[f"{a}-{b}"] = paired_log_loss(frame,a,b)
        largest[f"{a}-{b}"] = {"wins":difference.tail(5).sort_values(ascending=False).to_dict(),
                                "losses":difference.head(5).to_dict()}
    main = ["ridge","cscva","incumbent","A4","B6","C6","selected"]
    diversity = pair_diversity(frame,main,eligibility)
    wins = all_paintings["selected"].loc[eligibility].contrast-all_paintings["incumbent"].loc[eligibility].contrast
    complement = {n:float(np.corrcoef(wins,all_paintings[n].loc[eligibility].contrast)[0,1]) for n in ("ridge","cscva","incumbent")}
    low_ridge = all_paintings["ridge"].loc[eligibility].contrast <= all_paintings["ridge"].loc[eligibility].contrast.median()
    slices = {"low_ridge":float(wins[low_ridge].mean()),"high_ridge":float(wins[~low_ridge].mean())}
    low_cscva = all_paintings["cscva"].loc[eligibility].contrast <= all_paintings["cscva"].loc[eligibility].contrast.median()
    slices |= {"low_cscva":float(wins[low_cscva].mean()),"high_cscva":float(wins[~low_cscva].mean())}
    return {"summary":decomposed,"paired_bootstrap":measures,"paired_log_loss":loss_measures,
            "largest":largest,"diversity":diversity,
            "selected_gain_by_baseline_quality":complement,"selected_gain_slices":slices}


def diagnose(args):
    closure,output = args.closure,args.output
    output.mkdir(parents=True,exist_ok=False)
    audit = read(closure/"audit.json")
    gold = pd.read_csv(closure/"development_counts.csv")
    keyset = gold[data.KEYS]
    dev = pd.read_csv(closure/"predictions/selected.csv")[data.KEYS]
    predictions = {p.stem:pd.read_csv(p,float_precision="round_trip") for p in (closure/"predictions").glob("*.csv")}
    rows,posthoc,stage_best,stage_best_like,best,recipes = grid_and_candidates(closure,output,dev,gold,predictions,audit)
    compare = [*AUDITED,"C7","C8"]
    if best["id"] not in compare:
        predictions["posthoc_best"] = recipe_prediction(dev,predictions,recipes[best["id"]])
        compare.append("posthoc_best")
    frame = gold.assign(p=gold.k/gold.n)
    for name in compare:
        aligned = keyset.merge(predictions[name],on=data.KEYS,validate="one_to_one")
        frame[name] = aligned.p_positive
    comparisons = [(name,"incumbent") for name in compare if name!="incumbent"]
    comparisons += [("B6","A4"),("B6","cscva"),("C6","B6"),("C7","C6"),("C8","C6")]
    frozen = pd.read_csv(closure/"training_counts.csv")
    fit = frozen.groupby("language")[["k","n"]].sum()
    priors = (fit.k+1)/(fit.n+2)
    train_prior = frame.language.map(priors).fillna((frozen.k.sum()+1)/(frozen.n.sum()+2))
    logit_summary = {"train_prior_beta_1_1":compact_metrics(gold,keyset.assign(p_positive=train_prior))}
    intercepts = read(closure/"intercepts.json")
    for name, values in intercepts.items():
        variants, components = logit_ablations(frame,name,dict(zip(fit.index,values,strict=True)))
        logit_summary[name] = {"components":components,"metrics":{
            label:compact_metrics(gold,keyset.assign(p_positive=p)) for label,p in variants.items()}}
    scores = {name:compact_metrics(gold,predictions[name]) for name in compare}
    by_scope, oracles = {},{}
    for label, local in (("full",frame),("like",frame.loc[~frame.language.isin(EXCLUDED)].copy())):
        by_scope[label] = evaluate_scope(local,compare,output,label,comparisons)
        oracles[label] = {}
        for name in ("ridge","incumbent","C6","selected"):
            oracles[label][name] = {}
            for variant,p in oracle_variants(local,name).items():
                oracle_frame = local[data.KEYS+["p"]].assign(p_positive=p)
                unclipped = painting_scores(oracle_frame).contrast.mean()
                clipped = score(local[data.KEYS+["k","n"]],local[data.KEYS].assign(p_positive=p.clip(0,1)))
                oracles[label][name][variant] = {"unclipped_pearson":float(unclipped),
                                                "bounded_pearson":clipped["cultural_contrast"],
                                                "bounded_log_loss":clipped["log_loss"],
                                                "out_of_range_rows":int(((p<0)|(p>1)).sum())}
    oracle_dir = output/"oracle_analysis_only"
    oracle_dir.mkdir()
    write(oracle_dir/"summary.json",oracles)
    flat = recipe_prediction(dev,predictions,recipes["flat_60_20_20"])
    result = {"standalone_and_original":scores,"checkpoint_grid":rows,"additional_ensembles":posthoc,
              "best_by_stage":stage_best,"best_by_stage_like_diagnostic":stage_best_like,
              "best_posthoc_or_original":best,"logit_variants":logit_summary,"scopes":by_scope,
              "oracles_analysis_only":oracles,
              "flat_differs_from_hierarchical_n":int((predictions["selected"].p_positive!=flat.p_positive).sum()),
              "flat_max_abs_difference":float((predictions["selected"].p_positive-flat.p_positive).abs().max())}
    write(output/"diagnostics.json",result)
    print(json.dumps({"best":best,"original":scores["selected"],"incumbent":scores["incumbent"]},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    diagnose(parser.parse_args())
