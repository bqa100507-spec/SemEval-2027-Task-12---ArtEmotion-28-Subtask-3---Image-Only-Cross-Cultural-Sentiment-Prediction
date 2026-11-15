"""Fixed seed aggregation, painting-paired uncertainty, diagnostics and immutable gates."""

import numpy as np
import pandas as pd

from . import data
from .metrics import score, painting_scores
from .submission import validate_predictions


def aggregate(inputs, raw_seeds, baseline):
    if len(raw_seeds) != 3:
        raise ValueError("Exactly three independent seeds are required")
    for values in raw_seeds:
        if len(values) != len(inputs) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid seed predictions")
    student = inputs[data.KEYS].assign(p_positive=np.round(np.mean(raw_seeds, axis=0), 6))
    validate_predictions(baseline, inputs)
    r0 = inputs[data.KEYS].merge(baseline, on=data.KEYS, validate="one_to_one").p_positive.to_numpy()
    ensemble = student.assign(p_positive=np.round(.5 * student.p_positive.to_numpy() + .5 * r0, 6))
    return student, ensemble


def paired_interval(differences, replicates=10000, seed=232027):
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No eligible painting-paired differences")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for first in range(0, replicates, 100):
        end = min(first + 100, replicates)
        means[first:end] = values[rng.integers(len(values), size=(end - first, len(values)))].mean(axis=1)
    return {"difference": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "replicates": replicates, "seed": seed, "paintings": len(values)}


def analyze(gold, predictions, config, *, has_folds=True):
    summary, fold_metrics, language_rows, painting_rows = {}, [], [], []
    for name, prediction in predictions.items():
        metrics = score(gold, prediction)
        joined = gold[[*data.KEYS, "k", "n"]].merge(prediction, on=data.KEYS, validate="one_to_one")
        paintings = painting_scores(joined.assign(p=joined.k / joined.n))
        painting_rows.append(paintings.assign(model=name))
        for language, values in metrics["per_language"].items():
            language_rows.append({"model": name, "language": language, **values})
        fold_values = []
        if has_folds:
            for fold in sorted(gold.fold.unique()):
                held = gold[gold.fold.eq(fold)]
                pred = held[data.KEYS].merge(prediction, on=data.KEYS, validate="one_to_one")
                m = score(held, pred)
                fold_values.append(m)
                fold_metrics.append({"model": name, "fold": int(fold), "contrast": m["cultural_contrast"], "macro_log_loss": m["log_loss"]})
        summary[name] = {"contrast": metrics["cultural_contrast"], "macro_log_loss": metrics["log_loss"],
                         "fold_mean_contrast": float(np.mean([m["cultural_contrast"] for m in fold_values])) if fold_values else None,
                         "fold_mean_macro_log_loss": float(np.mean([m["log_loss"] for m in fold_values])) if fold_values else None,
                         "prediction_variance": float(np.var(prediction.p_positive)),
                         "contrast_paintings": metrics["contrast_paintings"]}
    table = pd.concat(painting_rows, ignore_index=True)
    pivot = table.pivot(index="painting", columns="model", values="contrast").dropna()
    comparisons = {}
    for right in ("R0", "EC0", "EC1"):
        delta = pivot.EC2 - pivot[right]
        comparisons["EC2-" + right] = {**paired_interval(delta, config["bootstrap_replicates"], config["bootstrap_seed"]),
                                      "wins": int((delta > 0).sum()), "regressions": int((delta < 0).sum())}
    return {"summary": summary, "comparisons": comparisons}, pd.DataFrame(fold_metrics), table, pd.DataFrame(language_rows)


def scientific_gate(analysis, folds, config, *, verified_models, inputs_verified, confirmation=False):
    summary = analysis["summary"]
    cfg = config["confirmation_gate" if confirmation else "cv_gate"]
    expected = 9 if confirmation else 45
    checks = {"complete_independent_replay": verified_models == expected, "input_source_partitions": bool(inputs_verified)}
    for right in ("R0", "EC0", "EC1"):
        checks["gain_" + right] = summary["EC2"]["contrast"] - summary[right]["contrast"] >= cfg["gain_" + right]
    for right in (("R0",) if confirmation else ("R0", "EC1")):
        checks["ci_lower_" + right] = analysis["comparisons"]["EC2-" + right]["ci95"][0] > cfg["ci_lower_" + right]
    loss_key = "macro_log_loss" if confirmation else "fold_mean_macro_log_loss"
    margin = cfg["pooled_loss_guardrail" if confirmation else "fold_mean_loss_guardrail"]
    checks["loss_guardrail"] = summary["EC2"][loss_key] <= summary["R0"][loss_key] + margin
    checks["finite_noncollapsed"] = all(np.isfinite([v["contrast"], v["macro_log_loss"], v["prediction_variance"]]).all()
                                          and v["prediction_variance"] > 0 for v in summary.values())
    if not confirmation:
        table = folds.pivot(index="fold", columns="model", values="contrast")
        checks["all_five_folds"] = set(table.index) == {1, 2, 3, 4, 5} and not table.isna().any().any()
        for right in ("R0", "EC1"):
            checks["fold_wins_" + right] = int((table.EC2 > table[right]).sum()) >= cfg["fold_wins_" + right]
        checks["worst_fold_R0"] = bool(((table.EC2 - table.R0) >= cfg["worst_fold_R0"]).all())
    passed = all(checks.values())
    return {"status": ("CONFIRMATION_GO" if confirmation else "CV_GO") if passed else ("CONFIRMATION_NO_GO" if confirmation else "CV_NO_GO"),
            "checks": {k: bool(v) for k, v in checks.items()}, "production": "E50+L", "automatic_promotion": False}
