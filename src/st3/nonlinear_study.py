"""Run the fixed nonlinear-head comparison on verified EXP-002 folds."""

import time

import numpy as np
import pandas as pd

from . import data, submission
from .baseline import file_sha256
from .cli import json_output
from .clip_large_study import E50_LARGE
from .contrast_study import new_output, write_development
from .e50_heads import read_json
from .fusion import blend
from .fusion_study import write_cv_analysis
from .low_rank_study import run_metadata, score_result
from .metrics import score
from .nonlinear import concatenate_features, configure_device, torch, train_partition
from .residual_crossfit import outer_reference_bytes, residual_inputs
from .residual_study import development_components, pair_ordering

MODELS = ("M0", "M1", "M2")


def compare_predictions(held, reference, seed_probabilities, known_languages, expected_seeds):
    data.require_same_pairs(held, reference)
    if (seed_probabilities.shape != (expected_seeds, len(held)) or not np.isfinite(seed_probabilities).all()
            or ((seed_probabilities < 0) | (seed_probabilities > 1)).any()):
        raise ValueError("Expected finite probabilities for every requested seed and pair")
    mean = seed_probabilities.mean(axis=0)
    raw = held[data.KEYS].assign(p_positive=mean)
    ensemble = blend(reference, raw, .5)
    ensemble["p_positive"] = ensemble["p_positive"].where(ensemble["language"].isin(known_languages),
                                                           reference["p_positive"].to_numpy())
    return {"M0": reference, "M1": raw.assign(p_positive=mean.round(6)), "M2": ensemble}


def choose_head(summary, config, folds):
    baseline = summary.loc["M0"]
    candidates = summary.loc[list(MODELS)]
    complete = candidates[["converged", "folds", "scored", "loss_scored"]].eq(folds).all(axis=1)
    complete &= np.isfinite(candidates[["pooled_contrast", "log_loss_mean"]]).all(axis=1)
    if not complete.loc["M0"]:
        raise ValueError("Incomplete frozen reference evaluation")
    loss_limit = float(baseline["log_loss_mean"] + config["log_loss_guardrail_bits"])
    minimum = float(baseline["pooled_contrast"] + config["minimum_pooled_contrast_gain"])
    eligible = candidates[complete & candidates["log_loss_mean"].le(loss_limit)
                          & candidates["pooled_contrast"].ge(minimum)
                          & candidates["fold_wins"].ge(config["minimum_fold_wins"])]
    winner = eligible["pooled_contrast"].idxmax() if len(eligible) else "M0"
    return {"model": winner, "eligible_candidates": eligible.index.tolist(), "minimum_pooled_contrast": minimum,
            "maximum_mean_log_loss": loss_limit, "minimum_fold_wins": config["minimum_fold_wins"]}


def device_metadata(provenance, start, device):
    return {**run_metadata(provenance, start), "device": str(device), "torch": str(torch.__version__),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}


def append_seed_scores(held, probabilities, runs, predictions):
    for info, q in zip(runs, probabilities):
        pred = held[data.KEYS].assign(p_positive=q.round(6))
        metrics = score(held, pred)
        info["contrast"], info["log_loss"] = metrics["cultural_contrast"], metrics["log_loss"]
        predictions.append(pred.assign(fold=info["fold"], seed=info["seed"]))


def write_seed_artifacts(stage, gold, runs, curves, seed_predictions):
    pd.DataFrame(runs).to_csv(stage / "seed_runs.csv", index=False, lineterminator="\n")
    pd.DataFrame(curves).to_csv(stage / "curves.csv", index=False, lineterminator="\n")
    by_seed = []
    predictions = pd.concat(seed_predictions, ignore_index=True)
    for seed, pred in predictions.groupby("seed"):
        metrics = score(gold, pred[[*data.KEYS, "p_positive"]])
        by_seed.append({"seed": seed, "pooled_contrast": metrics["cultural_contrast"],
                        "pooled_log_loss": metrics["log_loss"]})
    predictions.to_csv(stage / "seed_predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame(by_seed).to_csv(stage / "seed_summary.csv", index=False, lineterminator="\n")


def run_nonlinear_cv(args, dev):
    start = time.perf_counter()
    with new_output(args.run_dir / "cv") as stage:
        train, banks, folds, outer, provenance = residual_inputs(args, dev, "exp013")
        config = provenance["config"]
        device = configure_device(args.device)
        features = concatenate_features(*(banks[name] for name in ("clip_b", "siglip2", "clip_l")))
        rows, runs, curves, audits, seed_predictions = [], [], [], [], []
        records = {name: [] for name in ("predictions", "paintings", "languages")}
        for fold, (fit_idx, held_idx) in enumerate(folds, 1):
            fit, held = train.iloc[fit_idx], train.iloc[held_idx]
            reference = outer.loc[outer["outer_fold"].eq(fold), [*data.KEYS, "q0"]].rename(columns={"q0": "p_positive"})
            probabilities, seeded, trace, audit = train_partition(fit, held, features, config, fold, device)
            runs.extend(seeded)
            curves.extend(trace)
            audits.append({"fold": fold, **audit})
            append_seed_scores(held, probabilities, seeded, seed_predictions)
            preds = compare_predictions(held, reference, probabilities, seeded[0]["known_languages"], len(config["seeds"]))
            for name, pred in preds.items():
                context = {"model": name, "fold": fold}
                contrast, loss = score_result(held, pred, context, records)
                paintings = records["paintings"][-1]
                paintings["ordering_agreement"] = paintings["painting"].map(pair_ordering(held, pred))
                fit_seconds = sum(item["fit_seconds"] for item in seeded) if name == "M1" else 0.
                predict_seconds = sum(item["predict_seconds"] for item in seeded) if name == "M1" else 0.
                rows.append({**context, "contrast": contrast, "log_loss": loss, "converged": True,
                             "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                             "seconds": fit_seconds + predict_seconds,
                             "ordering_agreement": paintings["ordering_agreement"].mean()})
                print(f"Outer {fold} {name}: {contrast:.6f} / {loss:.6f}", flush=True)

        table = pd.DataFrame(rows)
        summary, _ = write_cv_analysis(stage, train, records, table, config, "M0", "M0")
        summary["ordering_mean"] = table.groupby("model")["ordering_agreement"].mean()
        summary["pooled_ordering"] = pd.concat(records["paintings"]).groupby("model")["ordering_agreement"].mean()
        expected = pd.read_csv(args.exp010_dir / "cv" / "summary.csv", index_col="model", float_precision="round_trip")
        fields = ["contrast_mean", "log_loss_mean", "pooled_contrast", "pooled_log_loss"]
        np.testing.assert_allclose(summary.loc["M0", fields].astype(float),
                                   expected.loc[E50_LARGE, fields].astype(float), rtol=0., atol=1e-14)
        summary.to_csv(stage / "summary.csv", lineterminator="\n")
        (stage / "reference_e50l.csv").write_bytes(outer_reference_bytes(outer, provenance["exp002_config"]["fit_seed"]))
        json_output(audits, stage / "split_audit.json")
        write_seed_artifacts(stage, train, runs, curves, seed_predictions)
        json_output(device_metadata(provenance, start, device), stage / "run.json")
        decision = choose_head(summary, config, len(folds))
        decision["sha256"] = {name: file_sha256(stage / name) for name in
                              ("run.json", "summary.csv", "seed_runs.csv", "split_audit.json")}
        json_output(decision, stage / "selection.json")
    print(f"Frozen EXP-013 decision: {decision['model']}", flush=True)


def run_nonlinear_dev(args, dev):
    start = time.perf_counter()
    cv = args.run_dir / "cv"
    decision = read_json(cv / "selection.json")
    for name in ("run.json", "summary.csv", "seed_runs.csv", "split_audit.json"):
        if decision["sha256"][name] != file_sha256(cv / name):
            raise ValueError(f"Frozen EXP-013 artifact changed: {name}")
    metadata = read_json(cv / "run.json")
    config = metadata["config"]
    summary = pd.read_csv(cv / "summary.csv", index_col="model", float_precision="round_trip")
    expected = choose_head(summary, config, metadata["exp002_config"]["cv_folds"])
    if any(decision[key] != value for key, value in expected.items()):
        raise ValueError("EXP-013 decision disagrees with the frozen selection rule")
    winner = decision["model"]
    if winner == "M0":
        print("EXP-013 NO-GO. Retained E50+L without development scoring or packaging.", flush=True)
        return

    with new_output(args.run_dir / "dev") as stage:
        train, banks, _, _, provenance = residual_inputs(args, dev, "exp013")
        if any(metadata[key] != value for key, value in provenance.items()):
            raise ValueError("Development inputs differ from the frozen EXP-013 CV run")
        reference, bundles, dev_banks = development_components(args, dev, include_features=True)
        train_banks = (banks[name] for name in ("clip_b", "siglip2", "clip_l"))
        features = concatenate_features(*({**fit, **held} for fit, held in zip(train_banks, dev_banks)))
        device = configure_device(args.device)
        probabilities, seeded, curves, audit = train_partition(train, dev, features, config, 0, device)
        baseline = reference[[*data.KEYS, "q0"]].rename(columns={"q0": "p_positive"})
        preds = compare_predictions(dev, baseline, probabilities, seeded[0]["known_languages"], len(config["seeds"]))
        gold = dev[data.KEYS].merge(data.validation_gold(data.read_annotations(args.data_dir), dev),
                                    on=data.KEYS, validate="one_to_one")
        seed_predictions = []
        append_seed_scores(gold, probabilities, seeded, seed_predictions)
        write_seed_artifacts(stage, gold, seeded, curves, seed_predictions)
        json_output(audit, stage / "split_audit.json")
        write_development(stage, gold, {"M0": baseline, winner: preds[winner]}, winner, config, "M0")
        archive = stage / "submission.zip"
        submission.pack(stage / "predictions" / winner / submission.FILENAME, archive, dev)
        submission.validate_archive(archive, dev)
        json_output(device_metadata({**provenance, "selection": decision, "head_manifests_sha256": bundles,
                                     "submission_sha256": file_sha256(archive)}, start, device), stage / "run.json")
    print(f"EXP-013 scored {winner} and validated {len(dev)} development predictions", flush=True)
