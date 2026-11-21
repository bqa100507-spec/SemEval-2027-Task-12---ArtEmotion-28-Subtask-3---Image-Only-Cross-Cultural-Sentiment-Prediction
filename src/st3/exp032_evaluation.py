"""Evaluation and frozen decisions for EXP-032."""

import numpy as np
from scipy.stats import rankdata

from .data import KEYS
from .metrics import MIN_LOSS_PAIRS, painting_scores, score

REFERENCE, CONTROL, CANDIDATE = "E50+L", "no_graph", "learned_graph"


def core_score(gold, prediction):
    return score(gold[KEYS + ["k", "n"]], prediction[KEYS + ["p_positive"]])


def within_painting(joined):
    eligible = painting_scores(joined).set_index("painting").dropna(subset=["contrast"])
    details = []
    for name, rows in joined.groupby("painting", sort=True):
        if name not in eligible.index:
            continue
        p, q = rows["p"].to_numpy(), rows["p_positive"].to_numpy()
        i, j = np.triu_indices(len(rows), k=1)
        truth, pred = np.sign(p[i] - p[j]), np.sign(q[i] - q[j])
        take = truth != 0
        wins = (pred[take] == truth[take]).sum()
        ties = (pred[take] == 0).sum()
        rho = float(np.corrcoef(rankdata(p), rankdata(q))[0, 1]) if np.ptp(q) > 0 else 0.0
        details.append({
            "painting": name, "contrast": float(eligible.loc[name, "contrast"]),
            "spearman": rho, "concordance": float((wins + 0.5 * ties) / take.sum()),
            "languages": len(rows), "fold": int(rows["fold"].iat[0]),
        })
    return details


def calibration(joined, bins):
    """Calibration uses empirical shares and equal pairs within eligible languages."""
    result, bins_out = {}, []
    for language, group in joined.groupby("language", sort=True):
        p, q = group["p"].to_numpy(), group["p_positive"].to_numpy()
        loss_q = np.clip(q, 0.01, 0.99)
        losses = -(p * np.log2(loss_q) + (1 - p) * np.log2(1 - loss_q))
        bin_indices = np.minimum((q * bins).astype(int), bins - 1)
        deviations = p - q
        ece = 0.0
        for index in np.unique(bin_indices):
            take = bin_indices == index
            gap = float(deviations[take].mean())
            ece += float(take.mean()) * abs(gap)
            bins_out.append({
                "language": language, "bin": int(index), "pairs": int(take.sum()),
                "mean_probability": float(q[take].mean()), "observed_share": float(p[take].mean()),
            })
        result[language] = {
            "pairs": len(group), "included": len(group) >= MIN_LOSS_PAIRS,
            "log_loss": float(losses.mean()), "brier_share": float(np.mean((p - q)**2)),
            "ece": ece, "bias": float(-deviations.mean()),
            "extreme_probability_fraction": float(((q <= 0.1) | (q >= 0.9)).mean()),
        }
    fields = ("log_loss", "brier_share", "ece", "bias", "extreme_probability_fraction")
    included = [v for v in result.values() if v["included"]]
    macro = {field: float(np.mean([v[field] for v in included])) if included else None for field in fields}
    return {"macro": macro, "by_language": result, "reliability_bins": bins_out}


def paired(deltas, replicas, seed):
    if not len(deltas):
        raise ValueError("No paired eligible paintings")
    generator = np.random.default_rng(seed)
    estimates = []
    for begin in range(0, replicas, 256):
        picks = generator.integers(0, len(deltas), size=(min(256, replicas - begin), len(deltas)))
        estimates.extend(deltas[picks].mean(axis=1))
    return {
        "gain": float(deltas.mean()), "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "wins": int((deltas > 0).sum()), "losses": int((deltas < 0).sum()),
        "ties": int((deltas == 0).sum()), "paintings": len(deltas),
    }


def choose_inner(trials, tolerance, baseline_loss=None):
    """Maximize training-only Contrast within budgeted loss, ties favor lower loss and simpler settings."""
    if any(trial["log_loss"] is None or trial["contrast"] is None for trial in trials):
        raise ValueError("Inner folds lack sufficient scoring coverage")
    best_loss = min(t["log_loss"] for t in trials)
    within = [t for t in trials if t["log_loss"] <= (best_loss if baseline_loss is None else baseline_loss) + tolerance]
    if within:
        return min(within, key=lambda t: (-t["contrast"], t["log_loss"], t["value"])), True
    return min(trials, key=lambda t: (t["log_loss"], -t["contrast"], t["value"])), False


def decision(comparisons, spec):
    checks = {}
    for comparator in (REFERENCE, CONTROL):
        entry = comparisons[comparator]
        checks[comparator] = {
            "contrast_gain": entry["pooled"]["gain"] >= spec["minimum_pooled_contrast_gain_vs_both"],
            "paired_ci": not spec["require_positive_paired_ci_lower_bound"] or entry["pooled"]["ci95"][0] > 0,
            "fold_wins": entry["fold_wins"] >= spec["minimum_winning_folds_vs_both"],
            "log_loss_guardrail": entry["pooled_log_loss_delta"] <= spec["maximum_pooled_log_loss_regression_vs_both"],
        }
    return {"status": "EXPLORATORY_GO" if all(all(v.values()) for v in checks.values()) else "NO_GO_OOF",
            "checks": checks}


def evaluate(gold, predictions, spec):
    summaries, by_painting, by_support = {}, {}, {}
    for name, predicted in predictions.items():
        joined = gold.merge(predicted[KEYS + ["p_positive"]], on=KEYS, validate="one_to_one")
        joined["p"] = joined["k"] / joined["n"]
        details = within_painting(joined)
        by_painting[name] = details
        pooled = core_score(gold, predicted)
        folds = {
            str(fold): core_score(rows, predicted.merge(rows[KEYS], on=KEYS, validate="one_to_one"))
            for fold, rows in gold.groupby("fold")
        }
        for fold, result in folds.items():
            eligible = [r for r in details if str(r["fold"]) == fold]
            result["spearman"] = float(np.mean([row["spearman"] for row in eligible]))
            result["concordance"] = float(np.mean([row["concordance"] for row in eligible]))
        pooled["spearman"] = float(np.mean([row["spearman"] for row in details]))
        pooled["concordance"] = float(np.mean([row["concordance"] for row in details]))
        summaries[name] = {"pooled": pooled, "folds": folds, "calibration": calibration(joined, spec["calibration_bins"])}
        support = {}
        buckets = np.searchsorted(spec["support_bin_edges"], joined["fitting_paintings"], side="right")
        # Loss slices use official >=20 per language recalculated within the slice.
        # Contrast slices similarly recompute painting eligibility within each support bucket.
        for bucket, rows in joined.groupby(buckets):
            support[str(bucket)] = core_score(rows, rows) | {"pairs": len(rows)}
        by_support[name] = support
    comparisons = {}
    for other in (REFERENCE, CONTROL):
        results = {}
        primary = {r["painting"]: r["contrast"] for r in by_painting[CANDIDATE]}
        matched = {r["painting"]: r["contrast"] for r in by_painting[other]}
        if primary.keys() != matched.keys():
            raise ValueError("Models have different Contrast eligibility")
        delta = np.array([primary[k] - matched[k] for k in sorted(primary)])
        results["pooled"] = paired(delta, spec["bootstrap_replicates"], spec["bootstrap_seed"])
        results["pooled_log_loss_delta"] = summaries[CANDIDATE]["pooled"]["log_loss"] - summaries[other]["pooled"]["log_loss"]
        results["folds"] = {}
        for fold, result in summaries[CANDIDATE]["folds"].items():
            reference = summaries[other]["folds"][fold]
            results["folds"][fold] = {
                "contrast_delta": result["cultural_contrast"] - reference["cultural_contrast"],
                "log_loss_delta": result["log_loss"] - reference["log_loss"],
            }
        results["fold_wins"] = sum(row["contrast_delta"] > 0 for row in results["folds"].values())
        comparisons[other] = results
    return {
        "systems": summaries, "comparisons": comparisons, "painting_diagnostics": by_painting,
        "support_slices": by_support, "decision": decision(comparisons, spec),
    }
