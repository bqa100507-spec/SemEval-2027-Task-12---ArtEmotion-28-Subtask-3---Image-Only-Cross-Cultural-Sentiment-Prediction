"""EXP-004 grouped study and frozen development evaluation."""

from contextlib import contextmanager
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from . import data, submission
from .ablation import paired_bootstrap
from .baseline import MAX_ITER, design_matrix, file_sha256, fit_lr
from .cli import json_output
from .contrast import PairwiseHuber, PaintingPearson, fit_contrast, objective
from .low_rank import split_lr
from .low_rank_study import (FOLD_FILE, dense_logits, grid_predictions, image_index, run_metadata, score_result,
                             study_inputs, training_grid, verify_folds)
from .metrics import score


@contextmanager
def new_output(destination):
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / "output"
        stage.mkdir()
        yield stage
        stage.rename(destination)


def auxiliaries(grid, config):
    _, positives, totals = grid
    return {"pairwise": PairwiseHuber(positives, totals, config["huber_delta"]),
            "pearson": PaintingPearson(positives, totals, config["pearson_epsilon"])}


def fit_c(train, held, embeddings, seed):
    start = time.perf_counter()
    languages = sorted(train["language"].unique())
    grid = training_grid(train, embeddings, languages)
    matrix = design_matrix(train["image"], train["language"], embeddings, languages)
    c = fit_lr(matrix, train, seed)
    del matrix
    indices = image_index(held, embeddings, languages)
    initial = split_lr(c, grid[0].shape[1], len(languages))
    q = c.predict_proba(design_matrix(held["image"], held["language"], embeddings, languages))[:, 1]
    value, gradient, nll, _ = objective(np.r_[initial[0], initial[1].ravel()], *grid, None, 0.)
    iterations = int(c.n_iter_[0])
    info = {"seed": seed, "converged": iterations < MAX_ITER, "status": "official L-BFGS",
            "iterations": iterations, "seconds": time.perf_counter() - start, "objective": value,
            "train_nll": nll, "gradient_max_abs": float(np.max(np.abs(gradient)))}
    return grid, indices, initial, q, info


def zero_control(grid, indices, initial, q, config):
    weights, info = fit_contrast(grid, initial, None, 0., config)
    if weights is None or not info["converged"]:
        raise ValueError(f"Lambda-zero optimizer failed: {info}")
    shifted = expit(dense_logits(indices[0], *weights)[indices[1], indices[2]])
    error = float(np.max(np.abs(q - shifted)))
    if error > config["zero_probability_atol"]:
        raise ValueError(f"Lambda-zero control exceeds raw-probability tolerance: {error:.12g}")
    return {**info, "max_probability_error": error}


def fit_variant(held, grid, indices, initial, auxiliary, variant, config, seed):
    start = time.perf_counter()
    weights, info = fit_contrast(grid, initial, auxiliary, variant["lambda"], config)
    pred = grid_predictions(held, dense_logits(indices[0], *weights), indices) if weights is not None else None
    return pred, {**info, **variant, "seed": seed, "seconds": time.perf_counter() - start}


def summarize(rows):
    reference = rows[rows.model.eq("C")].set_index("fold")
    for metric in ("contrast", "log_loss"):
        rows[f"{metric}_delta"] = rows[metric] - rows["fold"].map(reference[metric])
    return rows.groupby("model", sort=False).agg(
        contrast_mean=("contrast", "mean"), contrast_sd=("contrast", "std"),
        log_loss_mean=("log_loss", "mean"), log_loss_sd=("log_loss", "std"),
        contrast_gain=("contrast_delta", "mean"), log_loss_delta=("log_loss_delta", "mean"),
        fold_wins=("contrast_delta", lambda x: int(x.gt(0).sum())),
        converged=("converged", "sum"), scored=("contrast", "count"), loss_scored=("log_loss", "count"),
        folds=("fold", "size"), total_seconds=("seconds", "sum"))


def choose(summary, config, expected_folds):
    candidates = summary.drop(index="C")
    valid = candidates[
        candidates["converged"].eq(expected_folds) & candidates["folds"].eq(expected_folds)
        & candidates["scored"].eq(expected_folds) & candidates["loss_scored"].eq(expected_folds)
        & candidates["contrast_gain"].ge(config["minimum_contrast_gain"])
        & candidates["fold_wins"].ge(config["minimum_fold_wins"])
        & candidates["log_loss_delta"].le(config["log_loss_guardrail_bits"])]
    tied = valid[valid["contrast_mean"].ge(valid["contrast_mean"].max() - config["contrast_tie_band"])]
    model = next((name for name in config["preference_order"] if name in tied.index), "C")
    return {"model": model, "eligible_candidates": list(valid.index),
            "maximum_log_loss": float(summary.loc["C", "log_loss_mean"] + config["log_loss_guardrail_bits"])}


def run_contrast_cv(args, dev):
    start = time.perf_counter()
    with new_output(args.run_dir / "cv") as stage:
        config, _, train, embeddings, provenance = study_inputs(args, dev, "exp004")
        saved = pd.read_csv(args.exp002_dir / FOLD_FILE, float_precision="round_trip")
        folds = verify_folds(train, saved, provenance["exp002_config"])
        seed = provenance["exp002_config"]["fit_seed"]
        rows, zero, support = [], [], []
        records = {key: [] for key in ("predictions", "paintings", "languages")}
        for fold, (fit_idx, held_idx) in enumerate(folds, 1):
            print(f"EXP-004 train-only fold {fold}/{len(folds)}", flush=True)
            fit, held = train.iloc[fit_idx], train.iloc[held_idx]
            grid, indices, initial, q, info = fit_c(fit, held, embeddings, seed)
            original = held[data.KEYS].merge(saved[saved.fold.eq(fold)], on=data.KEYS, validate="one_to_one")
            if not info["converged"] or not np.array_equal(q.round(6), original["p_C"]):
                raise ValueError(f"Fold {fold} Model C does not reproduce EXP-002")
            zero.append({"fold": fold, **zero_control(grid, indices, initial, q, config)})
            support.append(held.assign(exactly_five=held["n"].eq(5)).groupby("painting")
                           .agg(fraction_exactly_five=("exactly_five", "mean")).reset_index().assign(fold=fold))
            context = {"fold": fold, "model": "C", "seed": seed}
            baseline = held[data.KEYS].assign(p_positive=q.round(6))
            contrast, log_loss = score_result(held, baseline, context, records)
            rows.append({**context, **info, "contrast": contrast, "log_loss": log_loss})
            aux = auxiliaries(grid, config)
            for name, variant in config["variants"].items():
                context = {"fold": fold, "model": name, "seed": seed}
                pred, info = fit_variant(held, grid, indices, initial, aux[variant["loss"]], variant, config, seed)
                contrast, log_loss = score_result(held, pred, context, records) if pred is not None else (None, None)
                rows.append({**context, **info, "contrast": contrast, "log_loss": log_loss})
                print(f"  {name}: {info['iterations']} iterations, {info['status']}", flush=True)
        table = pd.DataFrame(rows)
        summary = summarize(table)
        decision = choose(summary, config, len(folds))
        for name, frames in records.items():
            pd.concat(frames, ignore_index=True).to_csv(stage / f"{name}.csv", index=False, lineterminator="\n")
        for name, frame in (("fold_metrics", table), ("zero_controls", pd.DataFrame(zero)),
                            ("painting_support", pd.concat(support, ignore_index=True))):
            frame.to_csv(stage / f"{name}.csv", index=False, lineterminator="\n")
        summary.to_csv(stage / "summary.csv", lineterminator="\n")
        json_output(run_metadata(provenance, start), stage / "run.json")
        decision.update({name: file_sha256(stage / file) for name, file in
                         (("summary_sha256", "summary.csv"), ("run_sha256", "run.json"))})
        json_output(decision, stage / "selection.json")
    print(f"Frozen EXP-004 decision: {decision['model']}", flush=True)


def write_development(stage, gold, predictions, candidate, config):
    records = {key: [] for key in ("predictions", "paintings", "languages")}
    metrics = {}
    for name, pred in predictions.items():
        metrics[name] = score(gold, pred)
        score_result(gold, pred, {"model": name}, records)
        output = stage / "predictions" / name
        output.mkdir(parents=True)
        pred.to_csv(output / submission.FILENAME, index=False, lineterminator="\n")
    c, model = records["paintings"]
    paired = c.drop(columns="model").rename(columns={"contrast": "contrast_C"})
    paired["contrast_candidate"] = paired["painting"].map(model.set_index("painting")["contrast"])
    paired["contrast_candidate_minus_C"] = paired["contrast_candidate"] - paired["contrast_C"]
    draws, intervals = paired_bootstrap(paired, config["bootstrap_replicates"], config["bootstrap_seed"],
                                        columns=("contrast_candidate_minus_C",))
    c, model = records["languages"]
    langs = c.drop(columns="model").rename(columns={"log_loss": "log_loss_C"})
    langs["log_loss_candidate"] = langs["language"].map(model.set_index("language")["log_loss"])
    langs["log_loss_candidate_minus_C"] = langs["log_loss_candidate"] - langs["log_loss_C"]
    pairs = gold.merge(predictions["C"].rename(columns={"p_positive": "p_C"}), on=data.KEYS, validate="one_to_one")
    pairs = pairs.merge(predictions[candidate].rename(columns={"p_positive": "p_candidate"}), on=data.KEYS,
                        validate="one_to_one")
    for name, frame in (("per_painting", paired), ("per_language", langs), ("observed_vs_predicted", pairs),
                        ("bootstrap_distribution", draws)):
        frame.to_csv(stage / f"{name}.csv", index=False, lineterminator="\n")
    json_output({"models": metrics, "paired_bootstrap": intervals}, stage / "metrics.json")


def run_contrast_dev(args, dev):
    start = time.perf_counter()
    with new_output(args.run_dir / "dev") as stage:
        cv = args.run_dir / "cv"
        decision = json.loads((cv / "selection.json").read_text(encoding="utf-8"))
        metadata = json.loads((cv / "run.json").read_text(encoding="utf-8"))
        if decision["summary_sha256"] != file_sha256(cv / "summary.csv") or decision["run_sha256"] != file_sha256(cv / "run.json"):
            raise ValueError("Frozen EXP-004 CV artifacts have changed")
        summary = pd.read_csv(cv / "summary.csv", index_col="model", float_precision="round_trip")
        expected = choose(summary, metadata["config"], metadata["exp002_config"]["cv_folds"])
        if any(decision[key] != value for key, value in expected.items()):
            raise ValueError("Decision disagrees with the frozen CV selection rule")
        candidate = decision["model"]
        result = {"selection": decision, "selection_sha256": file_sha256(cv / "selection.json"), "status": "no_go"}
        if candidate != "C":
            config, annotations, train, embeddings, provenance = study_inputs(args, dev, "exp004")
            if any(metadata[key] != value for key, value in provenance.items()):
                raise ValueError("Development inputs differ from frozen CV")
            seed = provenance["exp002_config"]["fit_seed"]
            grid, indices, initial, q, c_info = fit_c(train, dev, embeddings, seed)
            baseline = dev[data.KEYS].assign(p_positive=q.round(6))
            original = submission.read_predictions(args.exp002_dir / "predictions" / "C" / submission.FILENAME, dev)
            if not c_info["converged"] or not np.array_equal(baseline["p_positive"], original["p_positive"]):
                raise ValueError("Development Model C does not reproduce EXP-002")
            zero = zero_control(grid, indices, initial, q, config)
            variant = config["variants"][candidate]
            pred, info = fit_variant(dev, grid, indices, initial, auxiliaries(grid, config)[variant["loss"]],
                                     variant, config, seed)
            result.update({**run_metadata(provenance, start), "fit": info, "reference_fit": c_info, "zero": zero,
                           "status": "fit_failed" if not info["converged"] else "scored"})
            if info["converged"]:
                gold = data.validation_gold(annotations, dev)
                write_development(stage, gold, {"C": baseline, candidate: pred}, candidate, config)
        json_output(result, stage / "run.json")
    print(f"EXP-004 development stage: {result['status']}", flush=True)
