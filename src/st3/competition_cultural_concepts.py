"""One grouped CPU test of observable concept and concept-language vote associations."""

import hashlib
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from .baseline import file_sha256
from .cli import ROOT, json_output
from .competition_challenger import check_hash, frame_sha
from .competition_semantic_evaluation import sealed_recovery
from .competition_semantic_generation import CONFIG as SEMANTIC, TRACK1
from .competition_stack import aligned_values, compare
from .competition_track1 import load_p2, peak_memory_bytes
from .contrast_study import new_output
from .data import KEYS
from .e50_heads import read_json
from .metrics import painting_scores, score

CONFIG = ROOT / "configs/competition_cultural_concepts.json"
UNCERTAIN = re.compile(r"\b(?:no|not|without|uncertain|unknown|possible|possibly|maybe|perhaps|may|might|could|appears?|seems?|suggests?)\b")
MAIN = ("fixed_ridge", "corrected_ridge", "concept_global", "concept_culture", "concept_mismatched")
SUPPLEMENTS = ("ridge_global", "ridge_static", "ridge_culture", "ridge_mismatched")


def concept_hits(description, config):
    fragments = [" ".join(re.findall(r"[a-z]+", text.lower()))
                 for field in config["fields"] for text in re.split(r"[,;.!?]", description.get(field, ""))]
    visible = [f" {part} " for part in fragments if part and not UNCERTAIN.search(part)]
    return [concept for concept, aliases in config["concepts"].items()
            if any(f" {word} " in fragment for word in aliases for fragment in visible)]


def feature_matrix(documents, vocabulary):
    rows = np.asarray([[name in doc for name in vocabulary] for doc in documents], dtype=float)
    return rows / np.sqrt(np.maximum(1, rows.sum(axis=1, keepdims=True)))


def fit_binomial(design, k, n, penalty, config):
    mass = n.sum()

    def objective(theta):
        z = design @ theta
        loss = (n @ np.logaddexp(0, z) - k @ z + .5 * (penalty @ (theta ** 2))) / mass
        gradient = (design.T @ (n * expit(z) - k) + penalty * theta) / mass
        return loss, gradient

    start = np.zeros(design.shape[1])
    start[0] = logit((k.sum() + .5) / (mass + 1))
    fitted = minimize(objective, start, method="L-BFGS-B", jac=True, options={"maxiter": config["fit_maxiter"],
                                                                            "gtol": config["fit_gtol"], "ftol": 1e-13})
    norm = float(np.linalg.norm(fitted.jac, ord=np.inf))
    if not fitted.success or not np.isfinite(fitted.x).all() or norm > 5e-7:
        raise RuntimeError(f"Concept head did not converge: {fitted.message}, |gradient|={norm:.3g}")
    return fitted.x, {"iterations": int(fitted.nit), "max_gradient": norm}


def association_design(x, language, support):
    rows = np.arange(len(x))
    count, width = support.shape
    prior = sparse.csr_matrix((np.ones(len(x)), (rows, language)), shape=(len(x), count))
    i, j = np.nonzero(x)
    keep = support[language[i], j]
    i, j = i[keep], j[keep]
    interaction = sparse.csr_matrix((x[i, j], (i, language[i] * width + j)), shape=(len(x), count * width))
    return sparse.hstack([np.ones((len(x), 1)), x, prior, interaction], format="csr")


def mismatch_map(support, votes):
    """Derange only nonempty, fitting-supported language slots for each concept."""
    result = np.broadcast_to(np.arange(len(support))[:, None], support.shape).copy()
    for j in range(support.shape[1]):
        indices = np.flatnonzero(support[:, j])
        if len(indices) == 1:
            raise ValueError("A mismatch requires two supported languages for the same concept")
        ordered = indices[np.argsort(votes[indices, j], kind="stable")]
        result[ordered, j] = np.roll(ordered, -1)
    return result


def fit_fold(train, keys, hits, config):
    if list(keys.columns) != KEYS or set(train.painting) & set(keys.painting):
        raise ValueError("Concept fitting and queries must be painting-disjoint and query keys only")
    if train.duplicated(KEYS).any() or not train.n.ge(5).all() or not train.k.between(0, train.n).all():
        raise ValueError("Invalid fitting vote panels")
    names = sorted(train.painting.unique())
    # Only the fitting descriptions can determine whether a vocabulary item survives.
    frequency = pd.Series([name for p in names for name in hits[p]], dtype=object).value_counts()
    concepts = [c for c in config["concepts"] if frequency.get(c, 0) >= config["minimum_concept_paintings"]]
    if not concepts:
        raise ValueError("No observable concepts have enough independent fitting paintings")
    groups = sorted(train.language.unique())
    x = feature_matrix([hits[p] for p in train.painting], concepts)
    xq = feature_matrix([hits[p] for p in keys.painting], concepts)
    k, n = train.k.to_numpy(float), train.n.to_numpy(float)
    seen = pd.Index(groups)
    lang = seen.get_indexer(train.language)
    query_lang = seen.get_indexer(keys.language)
    support_paintings = np.zeros((len(groups), len(concepts)), dtype=int)
    votes = np.zeros(support_paintings.shape)
    active = x > 0
    np.add.at(support_paintings, lang, active)
    np.add.at(votes, lang, n[:, None] * active)
    support = ((support_paintings >= config["minimum_concept_language_paintings"])
               & (votes >= config["minimum_concept_language_votes"]))
    support[:, support.sum(0) < 2] = False
    wrong = mismatch_map(support, votes)

    penalty_b = np.r_[config["ridge_bias"], np.full(len(concepts), config["ridge_shared"])]
    coeff_b, solve_b = fit_binomial(np.column_stack([np.ones(len(x)), x]), k, n, penalty_b, config)
    penalties = np.r_[penalty_b, np.full(len(groups), config["ridge_language"]),
                      np.full(support.size, config["ridge_interaction"])]
    coeff_c, solve_c = fit_binomial(association_design(x, lang, support), k, n, penalties, config)
    prior_end = len(concepts) + 1 + len(groups)
    shared = coeff_c[0] + xq @ coeff_c[1:len(concepts) + 1]
    prior = np.r_[coeff_c[len(concepts) + 1:prior_end], 0.0]
    effects = coeff_c[prior_end:].reshape(support.shape)
    # Off-support coefficients are exactly unused, including unseen target languages.
    effects *= support
    wrong_effects = effects[wrong, np.arange(len(concepts))[None, :]] * support
    shift = (xq * np.vstack([effects, np.zeros(len(concepts))])[query_lang]).sum(1)
    mismatch = (xq * np.vstack([wrong_effects, np.zeros(len(concepts))])[query_lang]).sum(1)
    static = shared + prior[query_lang]
    return keys.assign(concept_global=expit(coeff_b[0] + xq @ coeff_b[1:]), concept_culture=expit(static + shift),
                       concept_mismatched=expit(static + mismatch), concept_static=expit(static)), {
        "fitting_sha256": frame_sha(train), "held_keys_sha256": frame_sha(keys), "fitted_vocabulary": concepts,
        "supported_concept_language_cells": int(support.sum()), "total_concept_language_cells": support.size,
        "min_supported_paintings": int(support_paintings[support].min()) if support.any() else None,
        "fit_paintings_with_concepts": len({p for p in names if set(hits[p]) & set(concepts)}),
        "held_pairs_no_concepts": int((xq.sum(1) == 0).sum()), "held_pairs_with_cultural_support": int(np.count_nonzero(
            (xq * np.vstack([support, np.zeros(len(concepts), dtype=bool)])[query_lang]).any(1))),
        "optimizer_global": solve_b, "optimizer_cultural": solve_c}


def reference_rows(gold, reference, base):
    check_hash(reference / "oof.csv", base["incumbent_oof_sha256"])
    check_hash(reference / "metrics.json", base["incumbent_metrics_sha256"])
    raw = pd.read_csv(reference / "oof.csv", float_precision="round_trip")
    aligned_values(gold, raw, ["ridge", "corrected"])
    join = gold[KEYS + ["fold"]].merge(raw[KEYS + ["fold", "ridge", "corrected"]], on=KEYS,
                                       validate="one_to_one", suffixes=("", "_source"))
    if not join.fold.eq(join.fold_source).all():
        raise ValueError("The neighborhood reference uses different outer folds")
    return join[KEYS + ["ridge", "corrected"]].rename(columns={"ridge": MAIN[0], "corrected": MAIN[1]})


def evaluate(gold, outputs, config, excluded):
    frame = gold.merge(outputs, on=KEYS, validate="one_to_one").assign(p=lambda d: d.k / d.n)
    comparisons = [("corrected_ridge", "fixed_ridge"), ("concept_culture", "concept_mismatched"),
                   ("concept_culture", "concept_static"), ("ridge_culture", "corrected_ridge"),
                   ("ridge_culture", "fixed_ridge"), ("ridge_global", "corrected_ridge"),
                   ("ridge_culture", "ridge_global"), ("ridge_culture", "ridge_mismatched"),
                   ("ridge_culture", "ridge_static")]
    rule = config["decision"]
    report = {}
    for name, part in (("full", frame), ("filtered", frame.loc[~frame.language.isin(excluded)])):
        systems = (*MAIN, "concept_static", *SUPPLEMENTS)

        def measure(d, arm):
            result = score(d[KEYS + ["k", "n"]], d[KEYS].assign(p_positive=d[arm]))
            result.pop("per_language")
            return result

        pooled = {a: measure(part, a) for a in systems}
        folds = {int(i): {a: measure(subset, a) for a in systems} for i, subset in part.groupby("fold")}
        painting = {a: painting_scores(part, prediction=a).set_index("painting").contrast for a in systems}
        paired = {}
        for a, b in comparisons:
            delta = (painting[a] - painting[b]).dropna()
            changes = {i: group[a]["cultural_contrast"] - group[b]["cultural_contrast"] for i, group in folds.items()}
            strongest = min(int(np.ceil(.01 * len(delta))), int(delta.gt(0).sum()))
            paired[f"{a}_minus_{b}"] = compare(delta, rule) | {
                "fold_deltas": changes, "fold_wins": sum(x > 1e-12 for x in changes.values()),
                "exclude_top_1pct_wins_delta": float(delta.sort_values(ascending=False).iloc[strongest:].mean()),
                "log_loss_delta": pooled[a]["log_loss"] - pooled[b]["log_loss"]}
        report[name] = {"pairs": len(part), "pooled": pooled, "folds": folds, "paired": paired}
    c, f = [report[s]["paired"]["ridge_culture_minus_corrected_ridge"] for s in ("full", "filtered")]
    d = report["filtered"]["paired"]
    passed = (c["contrast_delta"] >= 0 and f["contrast_delta"] >= rule["minimum_filtered_gain"]
              and f["fold_wins"] >= rule["minimum_fold_wins"] and f["bootstrap_95"][0] > 0
              and f["trimmed_contrast_delta"] > 0 and f["exclude_top_1pct_wins_delta"] > 0
              and all(x["log_loss_delta"] <= rule["maximum_log_loss_delta"] for x in (c, f))
              and all(d[f"ridge_culture_minus_ridge_{name}"]["bootstrap_95"][0] > 0 for name in ("mismatched", "static")))
    report["decision"] = "PROMISING_PENDING_ELIGIBILITY" if passed else "RETAIN CORRECTED RIDGE"
    return report


def run(args):
    started = time.perf_counter()
    config, track, base = read_json(CONFIG), read_json(TRACK1), read_json(SEMANTIC)
    check_hash(args.semantic_run / "extraction.json", config["semantic_extraction_sha256"])
    descriptions, extraction = sealed_recovery(args.semantic_args, base, track)
    hits = {name: concept_hits(item, config) for name, item in descriptions.items()}
    gold, _, _ = load_p2(args.inputs, track)
    if set(hits) != set(gold.painting):
        raise ValueError("Concept descriptions do not match the exact P2 paintings")
    reference = reference_rows(gold, args.reference, base)
    receipts, parts = {}, []
    with threadpool_limits(limits=track["threads"]):
        for i, held in gold.groupby("fold"):
            train = gold.loc[gold.fold.ne(i)]
            part, receipts[int(i)] = fit_fold(train, held[KEYS], hits, config)
            parts.append(part)
            print(f"Fold {i}: {len(train)} fitting and {len(held)} held pairs; associations sealed as fold diagnostics", flush=True)
    outputs = reference.merge(pd.concat(parts, ignore_index=True), on=KEYS, validate="one_to_one")
    for name, column in zip(SUPPLEMENTS, ("concept_global", "concept_static", "concept_culture", "concept_mismatched"), strict=True):
        outputs[name] = (1 - config["blend_share"]) * outputs.corrected_ridge + config["blend_share"] * outputs[column]
    metrics = evaluate(gold, outputs, config, track["test_like_excluded"])
    sample = sorted(hits, key=lambda p: hashlib.sha256(f"{config['grounding_audit_seed']}:{p}".encode()).hexdigest())[:config["grounding_audit_count"]]
    metrics |= {"fold_receipts": receipts, "extraction_sha256": extraction, "config_sha256": file_sha256(CONFIG),
                "source_sha256": file_sha256(Path(__file__)), "cpu_seconds": time.perf_counter() - started,
                "peak_rss_bytes": peak_memory_bytes(), "paintings_with_concepts": sum(bool(h) for h in hits.values()),
                "total_paintings": len(hits), "competition_eligibility": "UNRESOLVED"}
    with new_output(args.destination) as stage:
        outputs.merge(gold[KEYS + ["fold"]], on=KEYS, validate="one_to_one").to_csv(
            stage / "oof.csv", index=False, lineterminator="\n", float_format="%.17g")
        json_output(metrics, stage / "metrics.json")
        json_output([{"painting": p, "concepts": hits[p], "description": descriptions[p]} for p in sample], stage / "grounding_sample.json")
        json_output({p.name: file_sha256(p) for p in stage.iterdir()}, stage / "seal.json")
    print(metrics["decision"], flush=True)
