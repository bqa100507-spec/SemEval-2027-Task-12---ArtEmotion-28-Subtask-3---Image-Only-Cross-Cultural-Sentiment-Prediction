"""CPU-only closure audit. Reads only saved inner-validation targets/predictions and selected checkpoints."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from st3.exp027_validation import verify
from st3.metrics import painting_scores
from st3.vlm_semantics import file_hash,fingerprint,atomic_json


def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>=3 and np.ptp(a)>0 and np.ptp(b)>0 else None


def validate_selection(payload):
    best=min(payload["curves"],key=lambda row:(row["validation_BCE_nats"],row["epoch"]))
    if payload["selected_epoch"]!=best["epoch"]:raise ValueError("Selected checkpoint does not match earliest minimum BCE")


def verify_export(path,run):
    import zipfile
    path=Path(path);digest=file_hash(path)
    assert digest==path.with_name("outputs.sha256").read_text().split()[0]
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read("manifest_sha256.json"))
        assert set(z.namelist())==set(manifest)|{"manifest_sha256.json"}
        for name,expected in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==expected
        for p in Path(run).rglob("*"):
            if p.is_file():assert manifest["run/"+p.relative_to(run).as_posix()]==file_hash(p)
    return {"status":"FULL_EXPORT_ZIP_INTEGRITY_PASS","sha256":digest,"verified_manifest_files":len(manifest)}


def crossfit(frame,seed=30027):
    """Two-way additive prediction-only fit; test profiles are recentered on observed languages.

    Painting intercepts remove missing-language-set confounding in the training fit.
    No gold is used in fitting. OOS sums need not be orthogonal: retain the cross term.
    """
    names=np.array(sorted(frame.painting.unique()));np.random.default_rng(seed).shuffle(names)
    rows=[];unseen=0;profiles=[]
    for half,held in enumerate((set(names[:len(names)//2]),set(names[len(names)//2:]))):
        fit=frame[~frame.painting.isin(held)];test=frame[frame.painting.isin(held)]
        languages=sorted(fit.language.unique());paintings=sorted(fit.painting.unique())
        li={x:i for i,x in enumerate(languages)};pi={x:i for i,x in enumerate(paintings)}
        matrix=np.zeros((len(fit),len(li)+len(pi)))
        for j,r in enumerate(fit.itertuples()):matrix[j,li[r.language]]=1;matrix[j,len(li)+pi[r.painting]]=1
        coefficients=np.linalg.lstsq(matrix,fit.p_positive.to_numpy(),rcond=None)[0][:len(li)]
        coefficients-=coefficients.mean()  # Fix the gauge; unseen languages fall back to mean offset0.
        profiles.extend(dict(test_half=half,language=l,offset=float(coefficients[li[l]])) for l in languages)
        for name,g in test.groupby("painting",sort=True):
            profile=np.array([coefficients[li[l]] if l in li else 0. for l in g.language]);unseen+=sum(l not in li for l in g.language)
            profile-=profile.mean();q=g.p_positive.to_numpy();center=q-q.mean();residual=center-profile
            y=g.k.to_numpy()/g.n.to_numpy();yc=y-y.mean()
            rows.append(dict(painting=name,test_half=half,prediction_centered_SS=float(center@center),profile_SS=float(profile@profile),
                        residual_SS=float(residual@residual),profile_residual_cross_term=float(2*profile@residual),
                        profile_rank_agreement=corr(pd.Series(q).rank(),pd.Series(profile).rank()),
                        residual_gold_correlation=corr(residual,yc),profile_gold_correlation=corr(profile,yc)))
    result=pd.DataFrame(rows);ss=result.prediction_centered_SS.sum()
    result.attrs["language_profiles"]=profiles
    summary={"held_out_explained_variance":float(1-result.residual_SS.sum()/ss) if ss else None,
             "median_profile_rank_agreement":float(result.profile_rank_agreement.median()),
             "mean_residual_gold_correlation":float(result.residual_gold_correlation.mean()),
             "centered_SS":float(ss),"profile_SS":float(result.profile_SS.sum()),"residual_SS":float(result.residual_SS.sum()),
             "cross_term":float(result.profile_residual_cross_term.sum()),"unseen_language_rows":unseen,"seed":seed,
             "boundary":"Prediction-only cross-fit; residual includes interaction, estimation error/noise and misspecification, not an identified causal component"}
    return result,summary


def analyze(run,out,inputs=None,export_zip=None):
    run=Path(run);out=Path(out)
    if out.exists():raise ValueError("Use a new analysis output directory; historical audits are immutable")
    out.mkdir(parents=True)
    metrics=verify(run/"inner_validation")
    original=json.loads((run/"inner_validation/metrics.json").read_text());assert metrics==original
    report=json.loads((run/"pilot_report.json").read_text());identity=json.loads((run/"identity.json").read_text())
    complete=json.loads((run/"PILOT_COMPLETE.json").read_text());assert complete["identity"]==fingerprint(identity)
    assert report["status"]=="GPU_PILOT_COMPLETE" and not report["outer_evaluation_performed"]
    assert complete["status"]=="GPU_PILOT_COMPLETE" and report["inner_validation_reporting"]==metrics
    for name,expected in identity["source_sha256"].items():
        raw=subprocess.check_output(["git","show",identity["source_revision"]+":"+name],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==expected,name
    gold=pd.read_csv(run/"inner_validation/targets.csv",float_precision="round_trip")
    assert len(gold)==504 and gold.painting.nunique()==48
    from st3.exp027_data import load_config,filter_votes
    config,protocol=load_config(ROOT)
    assert identity["config"]==config and identity["protocol"]==protocol
    if inputs is not None:
        import csv
        for name,digest in identity["inputs"].items():assert file_hash(Path(inputs)/name)==digest
        with (Path(inputs)/"training/pairs.csv").open(encoding="utf-8",newline="") as f:
            original=filter_votes(csv.DictReader(f),set(gold.painting))
        keys=["painting","language"]
        assert gold.sort_values(keys).reset_index(drop=True).equals(original[keys+["k","n"]].sort_values(keys).reset_index(drop=True))
    models={};curves=[];states={};selected={}
    for name in ("BCE","P_low","P_high"):
        path=run/"models"/(name+".pt");seal=json.loads(path.with_suffix(".sha256.json").read_text());assert file_hash(path)==seal["sha256"]
        payload=torch.load(path,map_location="cpu",weights_only=True)
        assert payload["complete"] and payload["identity"]["run"]==identity
        assert payload["curves"]==report["variants"][name]["curves"]
        assert len(payload["identity"]["fit"])==192 and len(payload["curves"])==2
        assert all(r["pairs"]==1931 and r["paintings"]==192 for r in payload["curves"])
        assert set(payload["identity"]["validation"])==set(gold.painting)
        assert not set(payload["identity"]["fit"])&set(gold.painting)
        validate_selection(payload)
        assert payload["identity"]["variant"]==next(v for v in identity["protocol"]["variants"] if v["name"]==name)
        pred=pd.read_csv(run/"inner_validation"/(name+".csv"),float_precision="round_trip")
        assert pred.p_positive.tolist()==payload["validation_prediction_six_decimals"]
        models[name]=gold.merge(pred,on=["painting","language"],validate="one_to_one")
        curves.extend({"model":name,**row} for row in payload["curves"])
        states[name]=payload["state"];selected[name]={"epoch":payload["selected_epoch"],"LoRA_update_norm":payload["LoRA_update_norm"]}
    painting=[];language=[];offset={};offset_rows=[];profiles=[]
    for name,frame in models.items():
        frame["p"]=frame.k/frame.n
        scores=painting_scores(frame).set_index("painting")
        base=models["BCE"].set_index(["painting","language"]).p_positive
        for pid,g in frame.groupby("painting",sort=True):
            q=g.p_positive.to_numpy();y=g.p.to_numpy();b=base.loc[pd.MultiIndex.from_frame(g[["painting","language"]])].to_numpy()
            painting.append({"model":name,"painting":pid,"eligible":scores.loc[pid,"status"]=="scored",
                "contrast":scores.loc[pid,"contrast"],"languages":len(g),"votes":int(g.n.sum()),"min_pair_votes":int(g.n.min()),
                "target_variance":float(np.var(y)),"prediction_variance":float(np.var(q)),"mean_prediction":float(q.mean()),
                "mean_abs_change_vs_BCE":float(np.abs(q-b).mean()),"mean_relative_abs_change_vs_BCE":float((np.abs(q-b)/np.maximum(np.abs(b),1e-7)).mean()),
                "order_agreement_vs_BCE":corr(pd.Series(q).rank(),pd.Series(b).rank())})
        q=frame.p_positive.clip(.01,.99);frame["log_loss_bits"]=-(frame.p*np.log2(q)+(1-frame.p)*np.log2(1-q))
        frame["centered_prediction"]=frame.p_positive-frame.groupby("painting").p_positive.transform("mean")
        l=frame.groupby("language").agg(pairs=("painting","size"),votes=("n","sum"),observed_positive=("p","mean"),predicted_positive=("p_positive","mean"),
                    log_loss_bits=("log_loss_bits","mean"),mean_centered_prediction=("centered_prediction","mean"))
        l["included_in_macro"]=l.pairs.ge(20);l["model"]=name;language.append(l.reset_index())
        details,offset[name]=crossfit(frame);profiles.extend({"model":name,**row} for row in details.attrs["language_profiles"]);details["model"]=name;offset_rows.append(details)
    paintings=pd.DataFrame(painting);wide=paintings.pivot(index="painting",columns="model",values="contrast")
    paintings["delta_contrast_vs_BCE"]=paintings.apply(lambda r:r.contrast-wide.loc[r.painting,"BCE"] if pd.notna(r.contrast) else np.nan,axis=1)
    paintings.to_csv(out/"painting_diagnostics.csv",index=False,lineterminator="\n");pd.concat(language).to_csv(out/"language_diagnostics.csv",index=False,lineterminator="\n")
    pd.concat(offset_rows).to_csv(out/"language_offset_crossfit.csv",index=False,lineterminator="\n");pd.DataFrame(curves).to_csv(out/"training_curves.csv",index=False,lineterminator="\n")
    pd.DataFrame(profiles).to_csv(out/"language_profiles.csv",index=False,lineterminator="\n")
    effects={}
    for name in ("P_low","P_high"):
        g=paintings[paintings.model.eq(name)&paintings.eligible];delta=g.delta_contrast_vs_BCE
        states_delta={section:float(np.sqrt(sum(float((states[name][section][key]-v).float().square().sum()) for key,v in states["BCE"][section].items()))) for section in ("head","adapters")}
        effects[name]={"median_delta":float(delta.median()),"trimmed_mean_drop_2_each_tail":float(np.sort(delta)[2:-2].mean()),
             "negative_painting_fraction":float((delta<0).mean()),"mean_abs_probability_change":float(g.mean_abs_change_vs_BCE.mean()),
             "median_rank_agreement_vs_BCE":float(g.order_agreement_vs_BCE.median()),"parameter_difference_norms_vs_BCE":states_delta,
             "delta_vs_target_variance_rank_correlation":corr(g.delta_contrast_vs_BCE.rank(),g.target_variance.rank()),
             "low_variance_quartile_mean_abs_delta":float(g[g.target_variance<=g.target_variance.quantile(.25)].delta_contrast_vs_BCE.abs().mean()),
             "other_paintings_mean_abs_delta":float(g[g.target_variance>g.target_variance.quantile(.25)].delta_contrast_vs_BCE.abs().mean()),
             "lowest_order_agreement_examples":g.nsmallest(3,"order_agreement_vs_BCE")[["painting","order_agreement_vs_BCE","delta_contrast_vs_BCE"]].to_dict("records"),
             "best_paintings":g.nlargest(3,"delta_contrast_vs_BCE")[["painting","delta_contrast_vs_BCE"]].to_dict("records"),
             "worst_paintings":g.nsmallest(3,"delta_contrast_vs_BCE")[["painting","delta_contrast_vs_BCE"]].to_dict("records"),
             "high_target_variance_poor_correlation":g[(g.target_variance>=g.target_variance.median())&(g.contrast<0)].painting.tolist(),
             "low_target_variance_bottom_quartile":g[g.target_variance<=g.target_variance.quantile(.25)].painting.tolist()}
    result={"status":"GPU_PILOT_COMPLETED / NO_GO_FOR_EXPANSION","verification":"Saved metrics/bootstrap/keys/checkpoint SHA/prediction vectors/selection/source identity PASS; encoder inference not replayed",
       "metrics":metrics,"offset_crossfit":offset,"effects":effects,"selected_checkpoints":selected,
       "engineering_report":{k:report[k] for k in ("GPU","gradient_preflight","elapsed_seconds","peak_allocated_bytes","peak_reserved_bytes")},
       "files_sha256":{str(p.relative_to(run)):file_hash(p) for p in run.rglob("*") if p.is_file()},
       "source_revision":identity["source_revision"],"run_identity":fingerprint(identity),"outer_labels_read":False,"GPU_executed":False,
       "original_validation_votes_checked":inputs is not None,
       "export_integrity":verify_export(export_zip,run) if export_zip else {"status":"NOT_CHECKED"},
       "analysis_script_sha256":file_hash(Path(__file__)),
       "diagnostic_csv_sha256":{p.name:file_hash(p) for p in out.glob("*.csv")},
       "readback_boundary":"Saved files/optional export bytes verified; CUDA gradient arrays and encoder inference not independently replayed"}
    atomic_json(out/"analysis.json",result)
    print(json.dumps({k:result[k] for k in ("status","offset_crossfit","effects","selected_checkpoints")},indent=2))
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--inputs",type=Path);p.add_argument("--export-zip",type=Path)
    a=p.parse_args();analyze(a.run_dir,a.output,a.inputs,a.export_zip)
