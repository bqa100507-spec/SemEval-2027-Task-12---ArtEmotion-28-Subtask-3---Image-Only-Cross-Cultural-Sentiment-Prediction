"""EXP-010 fixed CLIP-L comparison and guarded development evaluation."""

import time

import numpy as np
import pandas as pd

from . import data, submission
from .baseline import file_sha256
from .cli import ROOT, json_output
from .clip_large import extract_clip_large
from .contrast_study import new_output, write_development
from .coverage_study import E50, fold_reference
from .e50_heads import read_json
from .fusion import blend, verify_reference
from .fusion_study import fusion_inputs, write_cv_analysis
from .low_rank_study import FOLD_FILE, run_metadata, score_result, verify_folds
from .visual_study import REFERENCE, SIGLIP, fit_head

LARGE = "CLIP-L"
LARGE_SIGLIP = "L+SigLIP"
E50_LARGE = "E50+L"
MODELS = (E50, LARGE, LARGE_SIGLIP, E50_LARGE)
ALPHA = 0.5


def select_large(summary, config, folds):
    baseline = summary.loc[E50]
    limit = float(baseline["log_loss_mean"] + config["log_loss_guardrail_bits"])
    candidates = summary.loc[list(MODELS)]
    complete = candidates[["converged", "folds", "scored", "loss_scored"]].eq(folds).all(axis=1)
    finite = np.isfinite(candidates[["pooled_contrast", "log_loss_mean"]]).all(axis=1)
    if not complete.loc[E50] or not finite.loc[E50]:
        raise ValueError("Original E50 has incomplete train-only results")
    eligible = candidates[complete & finite & candidates["log_loss_mean"].le(limit)]
    leader = eligible["pooled_contrast"].idxmax()
    cutoff = baseline["pooled_contrast"] + config["e50_tie_band"]
    winner = leader if eligible.loc[leader, "pooled_contrast"] > cutoff else E50
    return {"model": winner, "pooled_leader": leader, "eligible": eligible.index.tolist(),
            "maximum_mean_log_loss": limit, "e50_tie_band": config["e50_tie_band"]}


def compose(model, large, *, e50=None, siglip=None):
    if model == LARGE:
        return large
    if model == LARGE_SIGLIP:
        return blend(large, siglip, ALPHA)
    if model == E50_LARGE:
        return blend(e50, large, ALPHA)
    raise ValueError(f"Unknown CLIP-L candidate: {model}")


def experiment_inputs(args, dev):
    exp007_config, annotations, train, clip, siglip, provenance = fusion_inputs(args, dev)
    source = args.exp007_dir / "cv"
    prior = read_json(source / "run.json")
    selection = read_json(source / "selection.json")
    if (selection["model"] != E50 or selection["run_sha256"] != file_sha256(source / "run.json")
            or selection["summary_sha256"] != file_sha256(source / "summary.csv")
            or exp007_config["alphas"][E50] != ALPHA
            or any(prior[key] != value for key, value in provenance.items())):
        raise ValueError("EXP-010 inputs differ from the frozen EXP-007 E50 reference")
    provenance.update({"exp007_sha256": {name: file_sha256(source / name) for name in
                                        ("run.json", "selection.json", "summary.csv", "predictions.csv")},
                       "config": read_json(ROOT / "configs" / "exp010.json"),
                       "config_sha256": file_sha256(ROOT / "configs" / "exp010.json")})
    return annotations, train, clip, siglip, provenance


def run_clip_large_cv(args, dev):
    start = time.perf_counter()
    with new_output(args.run_dir / "cv") as stage:
        _, train, _, _, provenance = experiment_inputs(args, dev)
        config = provenance["config"]
        saved = pd.read_csv(args.exp002_dir / FOLD_FILE, float_precision="round_trip")
        folds = verify_folds(train, saved, provenance["exp002_config"])
        reference_path = args.exp006_dir / "cv" / "predictions.csv"
        if file_sha256(reference_path) != provenance["exp006_predictions_sha256"]:
            raise ValueError("EXP-006 fold prediction checksum differs from EXP-007 provenance")
        exp006 = pd.read_csv(reference_path, float_precision="round_trip")
        source = args.exp007_dir / "cv"
        exp007 = pd.read_csv(source / "predictions.csv", float_precision="round_trip")
        prior_fits = pd.read_csv(source / "fold_metrics.csv")
        reference_fits = prior_fits.loc[prior_fits["model"].isin([REFERENCE, SIGLIP, E50])]
        expected_fits = {(model, fold) for model in (REFERENCE, SIGLIP, E50) for fold in range(1, len(folds) + 1)}
        actual_fits = list(reference_fits[["model", "fold"]].itertuples(index=False, name=None))
        if (len(actual_fits) != len(expected_fits) or set(actual_fits) != expected_fits
                or not reference_fits["converged"].eq(True).all()):
            raise ValueError("Original E50 components do not have complete converged EXP-007 fits")
        large, features = extract_clip_large(args.data_dir, train["image"], args.cache_dir, config["encoder"], args.device)

        seed = provenance["exp002_config"]["fit_seed"]
        records = {name: [] for name in ("predictions", "paintings", "languages")}
        rows = []
        for fold, (fit_ids, held_ids) in enumerate(folds, 1):
            print(f"EXP-010 train-only fold {fold}/{len(folds)}", flush=True)
            fit, held = train.iloc[fit_ids], train.iloc[held_ids]
            e50, siglip = fold_reference(saved, exp006, exp007, held, fold, ALPHA)
            pred, fit_info = fit_head(fit, held, large, seed)
            for model in MODELS:
                tick = time.perf_counter()
                prediction = e50 if model == E50 else compose(model, pred, e50=e50, siglip=siglip)
                seconds = time.perf_counter() - tick
                info = fit_info if model == LARGE else {
                    "converged": model == E50 or fit_info["converged"], "fit_seconds": 0.,
                    "predict_seconds": seconds, "seconds": seconds}
                context = {"model": model, "fold": fold, "seed": seed}
                contrast, loss = score_result(held, prediction, context, records)
                rows.append({**context, **info, "contrast": contrast, "log_loss": loss})
                print(f"  {model}: {contrast:.6f} / {loss:.6f}", flush=True)
        summary, predictions = write_cv_analysis(stage, train, records, pd.DataFrame(rows), config, E50, E50)
        expected = pd.read_csv(source / "summary.csv", index_col="model", float_precision="round_trip")
        for field in ("contrast_mean", "log_loss_mean", "pooled_contrast", "pooled_log_loss"):
            if abs(summary.loc[E50, field] - expected.loc[E50, field]) > 1e-14:
                raise ValueError(f"Original E50 score differs from EXP-007: {field}")
        reference = predictions.loc[predictions["model"].eq(E50)]
        verify_reference(reference[[*data.KEYS, "p_positive"]],
                         exp007.loc[exp007["model"].eq(E50), [*data.KEYS, "p_positive"]])
        decision = select_large(summary, config, len(folds))
        metadata = run_metadata({**provenance, "clip_large_embedding": features,
                                 "reference_oof_pairs_verified": len(reference), "blend_weight": ALPHA}, start)
        json_output(metadata, stage / "run.json")
        decision["sha256"] = {name: file_sha256(stage / name) for name in ("run.json", "summary.csv")}
        json_output(decision, stage / "selection.json")
    print(f"Frozen EXP-010 decision: {decision['model']}", flush=True)


def run_clip_large_dev(args, dev):
    start = time.perf_counter()
    cv = args.run_dir / "cv"
    decision = read_json(cv / "selection.json")
    for name in ("run.json", "summary.csv"):
        if decision["sha256"][name] != file_sha256(cv / name):
            raise ValueError(f"Frozen EXP-010 artifact changed: {name}")
    metadata = read_json(cv / "run.json")
    config = metadata["config"]
    summary = pd.read_csv(cv / "summary.csv", index_col="model", float_precision="round_trip")
    expected = select_large(summary, config, metadata["exp002_config"]["cv_folds"])
    if any(decision[key] != value for key, value in expected.items()):
        raise ValueError("EXP-010 decision disagrees with the frozen selection rule")
    winner = decision["model"]
    if winner == E50:
        print("EXP-010 NO-GO. Retained E50 without development scoring or packaging.", flush=True)
        return

    with new_output(args.run_dir / "dev") as stage:
        annotations, train, clip, siglip, provenance = experiment_inputs(args, dev)
        if any(metadata[key] != value for key, value in provenance.items()):
            raise ValueError("Development inputs differ from the frozen EXP-010 CV run")
        train_vectors, train_features = extract_clip_large(args.data_dir, train["image"], args.cache_dir,
                                                          config["encoder"], args.device)
        if train_features != metadata["clip_large_embedding"]:
            raise ValueError("Training CLIP-L cache changed since CV")
        dev_vectors, dev_features = extract_clip_large(args.data_dir, dev["image"], args.cache_dir,
                                                      config["encoder"], args.device)
        seed = metadata["exp002_config"]["fit_seed"]
        large, fits = fit_head(train, dev, {**train_vectors, **dev_vectors}, seed)
        fit_info = {LARGE: fits}
        prior_csv = args.exp007_dir / "dev" / "predictions" / E50 / submission.FILENAME
        expected_sha = read_json(ROOT / "configs" / "e50_inference.json")["reference_csv_sha256"]["e50"]
        if file_sha256(prior_csv) != expected_sha:
            raise ValueError("Development E50 predictions differ from the frozen reference")
        e50 = submission.read_predictions(prior_csv, dev)
        siglip_pred = None
        if winner in (LARGE_SIGLIP, E50_LARGE):
            clip_pred, fit_info[REFERENCE] = fit_head(train, dev, clip, seed)
            siglip_pred, fit_info[SIGLIP] = fit_head(train, dev, siglip, seed)
            verify_reference(blend(clip_pred, siglip_pred, ALPHA), e50)
        if not all(info["converged"] for info in fit_info.values()):
            raise ValueError("A full-training head failed convergence")
        candidate = compose(winner, large, e50=e50, siglip=siglip_pred)
        write_development(stage, data.validation_gold(annotations, dev), {E50: e50, winner: candidate}, winner,
                          config, E50)
        archive = stage / "submission.zip"
        submission.pack(stage / "predictions" / winner / submission.FILENAME, archive, dev)
        submission.validate_archive(archive, dev)
        json_output(run_metadata({**provenance, "selection": decision, "fits": fit_info,
                                 "train_clip_large_embedding": train_features, "dev_clip_large_embedding": dev_features,
                                 "reference_predictions_sha256": expected_sha,
                                 "submission_sha256": file_sha256(archive)}, start), stage / "run.json")
    print(f"EXP-010 scored {winner} and validated {archive.name} for {len(dev)} development pairs", flush=True)
