"""Conduct one frozen, CPU-only comparison of local cultural residuals."""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .baseline import file_sha256
from .cli import ROOT, json_output
from .competition_challenger import check_hash, component_predictions, fit_heads, frame_sha
from .competition_ridge import painting_splits
from .competition_stack import ENCODERS, aligned_values, blend, compare
from .competition_track1 import load_p2, peak_memory_bytes, save_csv
from .contrast_study import new_output
from .data import KEYS
from .e50_heads import read_json
from .metrics import painting_scores, score

CONFIG = ROOT / "configs" / "competition_neighborhood_residual.json"
TRACK1_CONFIG = ROOT / "configs" / "competition_track1.json"
SYSTEMS = ("ridge", "corrected")


@dataclass(frozen=True)
class ResidualBank:
    visual: pd.DataFrame
    numerator: pd.DataFrame
    votes: pd.DataFrame


def make_bank(train, crossfit, visual, minimum_languages):
    base = aligned_values(train, crossfit, ["p_positive"]).ravel()
    cells = train[KEYS + ["n"]].assign(unweighted_error=train.k / train.n - base)
    cells["weighted_error"] = cells.n * cells.unweighted_error
    grouped = cells.groupby("painting")
    center = grouped.weighted_error.transform("sum") / grouped.n.transform("sum")
    cells["weighted_error"] -= cells.n * center
    cells = cells.loc[grouped.n.transform("size").ge(minimum_languages)]
    if cells.empty:
        raise ValueError("No fitting paintings have sufficient languages for cultural residuals")
    numerator = cells.pivot(index="painting", columns="language", values="weighted_error").fillna(0)
    votes = cells.pivot(index="painting", columns="language", values="n").fillna(0)
    return ResidualBank(visual.loc[numerator.index], numerator, votes)


def predict_correction(bank, keys, visual, base, rule):
    """Use only query image vectors, baseline probabilities and target languages."""
    if list(keys.columns) != KEYS or set(keys.painting) & set(bank.visual.index):
        raise ValueError("Prediction requires painting/language keys disjoint from the residual bank")
    baseline = aligned_values(keys, base, ["p_positive"]).ravel()
    query = visual.loc[sorted(keys.painting.unique())]
    similarity = np.clip(query.to_numpy() @ bank.visual.to_numpy().T, -1, 1)
    # Bank order is sorted by painting, making ties deterministic.
    neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :rule["neighbors"]]
    distances = np.take_along_axis(similarity, neighbors, axis=1) - 1
    w = np.exp(distances / rule["kernel_temperature"])
    error_sums, masses = [], []
    numerator, votes = bank.numerator.to_numpy(), bank.votes.to_numpy()
    for idx, weights in zip(neighbors, w):
        error_sums.append(weights @ numerator[idx])
        masses.append(weights @ votes[idx])
    row = query.index.get_indexer(keys.painting)
    col = bank.votes.columns.get_indexer(keys.language)
    supported = col >= 0
    residual, used_votes = np.zeros(len(keys)), np.zeros(len(keys))
    at = (row[supported], col[supported])
    used_votes[supported] = np.array(masses)[at]
    residual[supported] = np.array(error_sums)[at] / (used_votes[supported] + rule["zero_residual_pseudovotes"])
    return keys.assign(ridge=baseline, corrected=np.clip(baseline + residual, 0, 1).round(6),
                       correction=residual, weighted_neighbor_votes=used_votes)


def fitting_predictions(train, features, track1, fold):
    names = sorted(train.painting.unique())
    assignments = painting_splits(names, track1["inner_seed"] + fold, track1["inner_folds"])
    index = train.painting.map(dict(zip(names, assignments)))
    pieces, receipts = [], []
    for split in range(track1["inner_folds"]):
        fit, query = train.loc[index.ne(split)], train.loc[index.eq(split)]
        if query.empty or fit.empty or set(fit.painting) & set(query.painting):
            raise ValueError("Inner ridge fold has invalid painting separation")
        heads, opts = fit_heads(fit, features, track1)
        pieces.append(component_predictions(heads, query[KEYS], features))
        receipts.append({"inner_fold": split, "fit_paintings": fit.painting.nunique(),
                         "query_paintings": query.painting.nunique(), "fit_sha256": frame_sha(fit),
                         "query_keys_sha256": frame_sha(query[KEYS]), "optimizers": opts})
    parts = pd.concat(pieces, ignore_index=True).sort_values(KEYS, ignore_index=True)
    q = blend(aligned_values(train.sort_values(KEYS), parts, ENCODERS), track1["fixed_weights"], fixed=True)
    return parts[KEYS].assign(p_positive=q), {"components_sha256": frame_sha(parts), "inner_folds": receipts}


def selection_metrics(gold, outputs, config, excluded):
    frame = gold.merge(outputs, on=KEYS, validate="one_to_one")
    aligned_values(gold, frame, SYSTEMS)
    frame["p"] = frame.k / frame.n
    report, tables = {}, []
    for scope, subset in (("full", frame), ("filtered", frame.loc[~frame.language.isin(excluded)])):
        paintings = subset[["painting", "fold"]].drop_duplicates().set_index("painting")

        def evaluate(block, name):
            return score(block[KEYS + ["k", "n"]], block[KEYS].assign(p_positive=block[name]))

        pooled = {name: evaluate(subset, name) for name in SYSTEMS}
        folds = {int(fold): {name: evaluate(part, name) for name in SYSTEMS}
                 for fold, part in subset.groupby("fold")}
        for name in SYSTEMS:
            paintings[name] = painting_scores(subset, prediction=name).set_index("painting").contrast
        paintings = paintings.dropna().assign(delta=lambda table: table.corrected - table.ridge)
        tables.append(paintings.reset_index().assign(scope=scope))
        intervals = compare(paintings.delta, config["decision"])
        differences = {i: x["corrected"]["cultural_contrast"] - x["ridge"]["cultural_contrast"] for i, x in folds.items()}
        deltas = paintings.delta.to_numpy()
        top = min(int(np.ceil(.01 * len(deltas))), int((deltas > 0).sum()), len(deltas) - 1)
        strongest_wins = np.argsort(-deltas, kind="stable")[:top]
        drop_wins = np.ones(len(deltas), dtype=bool)
        drop_wins[strongest_wins] = False
        report[scope] = {"pooled": pooled, "folds": folds, "corrected_minus_ridge": intervals | {
            "log_loss_delta": pooled["corrected"]["log_loss"] - pooled["ridge"]["log_loss"],
            "fold_deltas": differences, "fold_wins": sum(x > 0 for x in differences.values()),
            "delta_after_dropping_top_one_percent_of_wins": float(deltas[drop_wins].mean()),
            "leave_one_fold_out": {i: float(paintings.loc[paintings.fold.ne(i), "delta"].mean()) for i in folds},
            "probabilities_changed": int(subset.ridge.ne(subset.corrected).sum()),
            "mean_abs_probability_change": float((subset.corrected - subset.ridge).abs().mean())}}
    rule = config["decision"]
    full, filtered = (report[scope]["corrected_minus_ridge"] for scope in ("full", "filtered"))
    passed = (full["contrast_delta"] >= 0 and filtered["contrast_delta"] >= rule["minimum_filtered_gain"]
              and filtered["bootstrap_95"][0] > 0 and filtered["fold_wins"] >= rule["minimum_fold_wins"]
              and filtered["trimmed_contrast_delta"] > 0
              and all(x["log_loss_delta"] <= rule["maximum_log_loss_delta"] for x in (full, filtered)))
    report["recommendation"] = "ADD CORRECTION" if passed else "RETAIN RIDGE"
    return report, pd.concat(tables, ignore_index=True)


def run(args):
    config, track1 = read_json(CONFIG), read_json(TRACK1_CONFIG)
    if (args.run_dir / "results").exists():
        raise ValueError("Completed neighborhood results already exist")
    start = time.perf_counter()
    check_hash(TRACK1_CONFIG, config["track1_config_sha256"])
    for name, digest in config["track1_files_sha256"].items():
        check_hash(args.track1_dir / name, digest)
    gold, features, _ = load_p2(args.inputs, track1)
    reference = pd.read_csv(args.track1_dir / "results" / "oof.csv")
    aligned_values(gold, reference, ["fixed"])
    prior = gold[KEYS + ["fold"]].merge(reference[KEYS + ["fold", "fixed"]], on=KEYS,
                                          validate="one_to_one", suffixes=("", "_track1"))
    if not prior.fold.eq(prior.fold_track1).all():
        raise ValueError("Track 1 reference used different outer folds")
    visual = pd.concat([features[name] for name in config["similarity_features"]], axis=1, ignore_index=True)
    visual = visual / np.sqrt(len(config["similarity_features"]))
    folds, receipts = [], {}
    with threadpool_limits(limits=track1["threads"]):
        for fold, held in gold.groupby("fold"):
            tick = time.perf_counter()
            train, keys = gold.loc[gold.fold.ne(fold)], held[KEYS]
            if set(train.painting) & set(keys.painting):
                raise ValueError("Outer-fold painting overlap")
            crossfit, details = fitting_predictions(train, features, track1, int(fold))
            expected = read_json(args.track1_dir / f"fold_{fold}" / "fit.json")["stack"]["inner_oof_sha256"]
            if details["components_sha256"] != expected:
                raise ValueError(f"Fold {fold} failed exact replay of Track 1 inner probabilities")
            bank = make_bank(train, crossfit, visual, config["minimum_bank_languages"])
            base = prior.loc[prior.fold.eq(fold), KEYS + ["fixed"]].rename(columns={"fixed": "p_positive"})
            before = time.perf_counter()
            folds.append(predict_correction(bank, keys, visual, base, config))
            receipts[int(fold)] = details | {"training_sha256": frame_sha(train), "outer_keys_sha256": frame_sha(keys),
                                           "bank_paintings": len(bank.visual),
                                           "correction_cpu_seconds": time.perf_counter() - before,
                                           "total_seconds": time.perf_counter() - tick}
            print(f"Fold {fold}: inner ridge OOF replay matched; corrected {len(keys)} held-out pairs", flush=True)

    output = pd.concat(folds, ignore_index=True).sort_values(KEYS, ignore_index=True)
    metrics, table = selection_metrics(gold, output, config, track1["test_like_excluded"])
    incumbent = read_json(args.track1_dir / "results" / "metrics.json")
    for scope in ("full", "filtered"):
        for key in ("cultural_contrast", "log_loss"):
            if abs(metrics[scope]["pooled"]["ridge"][key] - incumbent[scope]["systems"]["fixed"]["pooled"][key]) > 1e-12:
                raise ValueError("Fixed ridge reference scores changed")
    metrics |= {"protocol": config, "config_sha256": file_sha256(CONFIG), "source_sha256": file_sha256(__file__),
                "fold_receipts": receipts, "threads": track1["threads"], "cpu_wall_seconds": time.perf_counter() - start,
                "peak_rss_bytes": peak_memory_bytes()}
    with new_output(args.run_dir / "results") as stage:
        save_csv(output.merge(gold[KEYS + ["fold"]], on=KEYS, validate="one_to_one"), stage / "oof.csv")
        save_csv(table, stage / "painting_deltas.csv")
        json_output(metrics, stage / "metrics.json")
        json_output({p.name: file_sha256(p) for p in stage.iterdir()}, stage / "seal.json")
    print(metrics["recommendation"], flush=True)
