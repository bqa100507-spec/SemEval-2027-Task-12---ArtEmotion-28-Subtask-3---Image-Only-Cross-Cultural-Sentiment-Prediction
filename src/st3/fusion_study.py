"""EXP-007 cached-feature comparison and locked development run."""

import hashlib
import json
import time

import numpy as np
import pandas as pd

from . import data, submission
from .ablation import paired_bootstrap
from .baseline import file_sha256
from .cli import ROOT, json_output
from .contrast_study import new_output, summarize, write_development
from .fusion import EARLY, blend, choose_fusion, combine_features, verify_reference
from .low_rank_study import FOLD_FILE, run_metadata, score_result, study_inputs, verify_folds
from .metrics import score
from .siglip2 import load_siglip2_features
from .visual_study import REFERENCE, SIGLIP, deltas, fit_head


def fusion_inputs(args, dev):
    config, annotations, train, clip, provenance = study_inputs(args, dev, "exp007")
    siglip_config = json.loads((ROOT / "configs" / "exp006.json").read_text(encoding="utf-8"))
    siglip, embedding = load_siglip2_features(args, dev, train, siglip_config,
                                              extraction_dir=args.exp006_dir / "extraction")
    prior_path = args.exp006_dir / "cv" / "run.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    keys = ("annotations_sha256", "dev_input_sha256", "exp002_folds_sha256", "embedding")
    if any(prior[key] != provenance[key] for key in keys) or \
            prior["siglip2_embedding"]["features_sha256"] != embedding["features_sha256"]:
        raise ValueError("Fusion inputs differ from the EXP-006 reference run")
    provenance.update({"siglip2_embedding": embedding, "exp006_run_sha256": file_sha256(prior_path),
                       "exp006_predictions_sha256": file_sha256(args.exp006_dir / "cv" / "predictions.csv")})
    return config, annotations, train, clip, siglip, provenance


def write_cv_analysis(stage, train, records, rows, config):
    predictions = pd.concat(records["predictions"], ignore_index=True)
    paintings = deltas(pd.concat(records["paintings"], ignore_index=True), ["fold", "painting"], "contrast")
    languages = deltas(pd.concat(records["languages"], ignore_index=True), ["fold", "language"], "log_loss")
    pooled = {name: score(train, frame[[*data.KEYS, "p_positive"]])
              for name, frame in predictions.groupby("model", sort=False)}
    summary = summarize(rows, REFERENCE)
    summary["pooled_contrast"] = pd.Series({name: result["cultural_contrast"] for name, result in pooled.items()})
    summary["pooled_log_loss"] = pd.Series({name: result["log_loss"] for name, result in pooled.items()})
    summary["pooled_contrast_delta"] = summary["pooled_contrast"] - summary.loc[REFERENCE, "pooled_contrast"]
    summary["pooled_log_loss_delta"] = summary["pooled_log_loss"] - summary.loc[REFERENCE, "pooled_log_loss"]
    summary["fit_seconds"] = rows.groupby("model")["fit_seconds"].sum()
    summary["predict_seconds"] = rows.groupby("model")["predict_seconds"].sum()
    scored = paintings[paintings["status"].eq("scored")]
    summary["painting_wins"] = scored.groupby("model")["delta_vs_clip"].apply(lambda d: int(d.gt(0).sum()))
    summary["painting_regressions"] = scored.groupby("model")["delta_vs_clip"].apply(lambda d: int(d.lt(0).sum()))
    wide = scored.pivot(index=["fold", "painting"], columns="model", values="delta_vs_clip").reset_index()
    compared = [name for name in summary.index if name != REFERENCE]
    draws, intervals = paired_bootstrap(wide.assign(status="scored"), config["bootstrap_replicates"],
                                       config["bootstrap_seed"], columns=compared)
    for name, frame in (("fold_metrics", rows), ("predictions", predictions), ("paintings", paintings),
                        ("languages", languages), ("bootstrap_distribution", draws)):
        frame.to_csv(stage / f"{name}.csv", index=False, lineterminator="\n")
    summary.to_csv(stage / "summary.csv", lineterminator="\n")
    json_output(pooled, stage / "pooled_metrics.json")
    json_output(intervals, stage / "paired_bootstrap.json")
    return summary, predictions


def run_fusion_cv(args, dev):
    start = time.perf_counter()
    with new_output(args.run_dir / "cv") as stage:
        config, _, train, clip, siglip, provenance = fusion_inputs(args, dev)
        tick = time.perf_counter()
        combined = combine_features(clip, siglip)
        feature_path = stage / "fused_features.npz"
        np.savez(feature_path, names=np.array(list(combined)), features=np.stack(list(combined.values())))
        provenance.update({"fused_features_sha256": file_sha256(feature_path),
                           "fusion_cache_seconds": time.perf_counter() - tick})
        saved = pd.read_csv(args.exp002_dir / FOLD_FILE, float_precision="round_trip")
        exp006 = pd.read_csv(args.exp006_dir / "cv" / "predictions.csv", float_precision="round_trip")
        folds = verify_folds(train, saved, provenance["exp002_config"])
        seed = provenance["exp002_config"]["fit_seed"]
        records = {name: [] for name in ("predictions", "paintings", "languages")}
        rows = []
        for fold, (fit_ids, held_ids) in enumerate(folds, 1):
            print(f"EXP-007 train-only fold {fold}/{len(folds)}", flush=True)
            fit, held = train.iloc[fit_ids], train.iloc[held_ids]
            predictions, info = {}, {}
            for name, features in ((REFERENCE, clip), (SIGLIP, siglip), (EARLY, combined)):
                predictions[name], info[name] = fit_head(fit, held, features, seed)
            for name in (REFERENCE, SIGLIP):
                expected = exp006.loc[exp006["model"].eq(name) & exp006["fold"].eq(fold),
                                      [*data.KEYS, "p_positive"]]
                verify_reference(predictions[name], expected)
                if not info[name]["converged"]:
                    raise ValueError(f"{name} failed convergence in fold {fold}")
            verify_reference(predictions[REFERENCE], saved.loc[saved["fold"].eq(fold), [*data.KEYS, "p_C"]]
                             .rename(columns={"p_C": "p_positive"}))
            for name, alpha in config["alphas"].items():
                tick = time.perf_counter()
                predictions[name] = blend(predictions[REFERENCE], predictions[SIGLIP], alpha)
                seconds = time.perf_counter() - tick
                info[name] = {"converged": True, "fit_seconds": 0., "predict_seconds": seconds, "seconds": seconds,
                              "parameters": info[REFERENCE]["parameters"] + info[SIGLIP]["parameters"]}
            for name, pred in predictions.items():
                context = {"fold": fold, "model": name, "seed": seed}
                contrast, loss = score_result(held, pred, context, records)
                rows.append({**context, **info[name], "contrast": contrast, "log_loss": loss})
                print(f"  {name}: {contrast:.6f} / {loss:.6f}, converged={info[name]['converged']}", flush=True)

        summary, predictions = write_cv_analysis(stage, train, records, pd.DataFrame(rows), config)
        reference_bytes = predictions[predictions["model"].isin([REFERENCE, SIGLIP])].to_csv(
            index=False, lineterminator="\n").encode("utf-8")
        if hashlib.sha256(reference_bytes).hexdigest() != provenance["exp006_predictions_sha256"]:
            raise ValueError("Reproduced EXP-006 out-of-fold CSV is not byte-identical")
        decision = choose_fusion(summary, config, len(folds))
        json_output(run_metadata(provenance, start), stage / "run.json")
        decision.update({"summary_sha256": file_sha256(stage / "summary.csv"), "run_sha256": file_sha256(stage / "run.json")})
        json_output(decision, stage / "selection.json")
    print(f"Frozen EXP-007 decision: {decision['model']}", flush=True)


def run_fusion_dev(args, dev):
    start = time.perf_counter()
    with new_output(args.run_dir / "dev") as stage:
        cv = args.run_dir / "cv"
        selection = cv / "selection.json"
        decision = json.loads(selection.read_text(encoding="utf-8"))
        metadata = json.loads((cv / "run.json").read_text(encoding="utf-8"))
        if (decision["summary_sha256"] != file_sha256(cv / "summary.csv")
                or decision["run_sha256"] != file_sha256(cv / "run.json")):
            raise ValueError("Frozen EXP-007 CV artifacts have changed")
        summary = pd.read_csv(cv / "summary.csv", index_col="model", float_precision="round_trip")
        expected = choose_fusion(summary, metadata["config"], metadata["exp002_config"]["cv_folds"])
        if any(decision[key] != value for key, value in expected.items()):
            raise ValueError("Decision disagrees with the frozen EXP-007 rule")
        winner = decision["model"]
        result = {"selection": decision, "selection_sha256": file_sha256(selection), "status": "fallback"}
        if winner != REFERENCE:
            config, annotations, train, clip, siglip, provenance = fusion_inputs(args, dev)
            if any(metadata[key] != value for key, value in provenance.items()):
                raise ValueError("Development inputs differ from frozen EXP-007 CV")
            seed = provenance["exp002_config"]["fit_seed"]
            baseline, clip_info = fit_head(train, dev, clip, seed)
            original_path = args.exp002_dir / "predictions" / "C" / submission.FILENAME
            verify_reference(baseline, submission.read_predictions(original_path, dev))
            if not clip_info["converged"]:
                raise ValueError("Reference full-train head failed convergence")
            if winner == EARLY:
                cache = cv / "fused_features.npz"
                if file_sha256(cache) != metadata["fused_features_sha256"]:
                    raise ValueError("Frozen fused-feature cache has changed")
                with np.load(cache, allow_pickle=False) as archive:
                    combined = dict(zip(archive["names"], archive["features"]))
                pred, info = fit_head(train, dev, combined, seed)
            else:
                pred, info = fit_head(train, dev, siglip, seed)
                if winner in config["alphas"]:
                    tick = time.perf_counter()
                    pred = blend(baseline, pred, config["alphas"][winner])
                    info["blend_seconds"] = time.perf_counter() - tick
            result.update({**run_metadata(provenance, start), "reference_fit": clip_info, "candidate_fit": info,
                           "reference_predictions_sha256": file_sha256(original_path),
                           "status": "scored" if info["converged"] else "fit_failed"})
            if info["converged"]:
                gold = data.validation_gold(annotations, dev)
                write_development(stage, gold, {REFERENCE: baseline, winner: pred}, winner, config, REFERENCE)
                submission.pack(stage / "predictions" / winner / submission.FILENAME, stage / "submission.zip", dev)
                submission.validate_archive(stage / "submission.zip", dev)
                result["submission_sha256"] = file_sha256(stage / "submission.zip")
        json_output(result, stage / "run.json")
    print(f"EXP-007 development stage: {result['status']}", flush=True)
