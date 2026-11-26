"""Painting-disjoint Corrected Ridge offsets and scoring for EXP-035."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .competition_challenger import component_predictions, fit_heads, frame_sha
from .competition_cultural_concepts import reference_rows
from .competition_neighborhood import fitting_predictions, make_bank, predict_correction
from .competition_ridge import painting_splits
from .competition_stack import ENCODERS, aligned_values, blend, compare
from .data import KEYS, require_same_pairs
from .metrics import painting_scores, score


def split_groups(table, seed, groups):
    names = sorted(table.painting.unique())
    if len(names) < groups:
        raise ValueError("Insufficient distinct paintings for grouped cross-fitting")
    idx = table.painting.map(dict(zip(names, painting_splits(names, seed, groups))))
    return [(table[idx.ne(f)], table[idx.eq(f)]) for f in range(groups)]


def corrected_offsets(fit, keys, features, track, correction, seed):
    if list(keys) != KEYS or set(fit.painting) & set(keys.painting):
        raise ValueError("Corrected Ridge fit and query paintings must be disjoint")
    visual = pd.concat([features[name] for name in correction["similarity_features"]], axis=1,
                       ignore_index=True) / np.sqrt(len(correction["similarity_features"]))
    crossfit, _ = fitting_predictions(fit, features, track, seed)
    bank = make_bank(fit, crossfit, visual, correction["minimum_bank_languages"])
    heads, _ = fit_heads(fit, features, track)
    parts = component_predictions(heads, keys, features)
    q = blend(aligned_values(keys, parts, ENCODERS), track["fixed_weights"], fixed=True)
    result = predict_correction(bank, keys, visual, keys.assign(p_positive=q), correction)
    return keys.assign(prior=result.corrected), {"fit_sha256": frame_sha(fit), "keys_sha256": frame_sha(keys)}


def prepare_frames(gold, features, reference, track, correction, base, config):
    outer = gold[gold.fold.eq(config["fold"])]
    train = gold[gold.fold.ne(config["fold"])]
    fit, inner = split_groups(train, config["seed"], 5)[0]
    frozen = reference_rows(gold, reference, base)
    with threadpool_limits(limits=track["threads"]):
        pieces, receipts = [], []
        for i, (within, query) in enumerate(split_groups(fit, config["seed"] + 1, 3)):
            part, receipt = corrected_offsets(within, query[KEYS], features, track, correction, config["seed"] + i)
            pieces.append(part)
            receipts.append(receipt)
        inner_offsets, inner_receipt = corrected_offsets(fit, inner[KEYS], features, track, correction, config["seed"])
    outer_offsets = frozen.rename(columns={"corrected_ridge": "prior"})[KEYS + ["prior"]]
    output = {}
    for name, frame, offsets in (("fit", fit, pd.concat(pieces)), ("inner", inner, inner_offsets),
                                 ("outer", outer, outer_offsets)):
        if name == "outer":
            offsets = offsets.merge(frame[KEYS], on=KEYS, validate="one_to_one")
        require_same_pairs(frame, offsets)
        output[name] = frame.merge(offsets, on=KEYS, validate="one_to_one").sort_values(KEYS, ignore_index=True)
    return output, {"fit_offsets": receipts, "inner_offsets": inner_receipt, "held_incumbent": base["incumbent_oof_sha256"]}


def scores(frame, names, exclude):
    result = {}
    for scope, rows in (("full", frame), ("test_like", frame[~frame.language.isin(exclude)])):
        result[scope] = {}
        for name in names:
            metrics = score(rows[KEYS + ["k", "n"]], rows[KEYS].assign(p_positive=rows[name]))
            result[scope][name] = {k: metrics[k] for k in ("cultural_contrast", "log_loss", "contrast_paintings")}
            result[scope][name]["mean_language_variance"] = float(rows.groupby("painting")[name].var(ddof=0).mean())
    return result


def select_epoch(history, margin):
    def eligible(row):
        return all(scores["candidate"]["log_loss"] <= scores["prior"]["log_loss"] + margin
                   for scores in row["inner"].values())

    allowed = [row for row in history if eligible(row)]
    if allowed:
        return max(allowed, key=lambda r: (r["inner"]["test_like"]["candidate"]["cultural_contrast"], -r["epoch"])), True
    # A failed loss guardrail is diagnostic. Still score one predetermined checkpoint.
    return min(history, key=lambda r: (r["inner"]["test_like"]["candidate"]["log_loss"], r["epoch"])), False


def metric_deltas(a, b, c):
    """Signed, matched changes. Positive Log Loss deltas are regressions."""
    return {f"{name}_delta": {label: lhs[name] - rhs[name] for label, lhs, rhs in
                            (("B_minus_A", b, a), ("C_minus_A", c, a), ("C_minus_B", c, b))}
            for name in ("cultural_contrast", "log_loss")}


def paired_epoch_curves(arms):
    b, c = (arms[x]["history"] for x in ("frozen", "merger"))
    results = []
    for left, right in zip(b, c, strict=True):
        if left["epoch"] != right["epoch"]:
            raise ValueError("Inner curves have mismatched epochs")
        scopes = {}
        for scope, values in left["inner"].items():
            other = right["inner"][scope]
            if values["prior"] != other["prior"]:
                raise ValueError("Inner curves do not share an identical Ridge reference")
            scopes[scope] = metric_deltas(values["prior"], values["candidate"], other["candidate"])
        results.append({"epoch": left["epoch"], "inner": scopes})
    return results


def held_metrics(frame, track):
    report = scores(frame, ("prior", "frozen", "merger"), track["test_like_excluded"])
    for name, rows in (("full", frame), ("test_like", frame[~frame.language.isin(track["test_like_excluded"])])):
        rows = rows.assign(p=rows.k / rows.n)
        contrasts = {n: painting_scores(rows, n).set_index("painting").contrast.dropna()
                     for n in ("prior", "frozen", "merger")}
        report[name].update(metric_deltas(*(report[name][key] for key in ("prior", "frozen", "merger"))))
        report[name]["paired_contrast"] = {f"{a}_minus_{b}": compare(contrasts[a] - contrasts[b], track["decision"])
                                           for a, b in (("frozen", "prior"), ("merger", "prior"), ("merger", "frozen"))}
    return report
