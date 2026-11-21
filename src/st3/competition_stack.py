"""Vote-aware global probability stacking and paired ST3 evaluation."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import KEYS, require_same_pairs
from .metrics import MIN_CONTRAST_LANGUAGES, MIN_LOSS_PAIRS, painting_scores, score

ENCODERS = ("clip", "siglip2", "clip_large")


def aligned_values(keys, predictions, columns):
    require_same_pairs(keys, predictions)
    values = keys[KEYS].merge(predictions[KEYS + list(columns)], on=KEYS, validate="one_to_one")
    values = values[list(columns)].to_numpy(dtype=np.float64)
    if not (np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()):
        raise ValueError("Predictions must be finite probabilities")
    return values


def blend(values, weights, *, fixed):
    if fixed:
        # EXP-032 and E50+L round the E50 midpoint before the final midpoint.
        return (0.5 * (0.5 * (values[:, 0] + values[:, 1])).round(6) + 0.5 * values[:, 2]).round(6)
    return (values @ weights).round(6)


def stack_objective(gold, components, config):
    values = aligned_values(gold, components, ENCODERS)
    frame = gold[KEYS].assign(p=gold["k"] / gold["n"], **dict(zip(ENCODERS, values.T)))
    frame = frame.loc[~frame.language.isin(config["test_like_excluded"])]
    groups = frame.groupby("painting")
    counts = groups.p.transform("size")
    use = counts.ge(MIN_CONTRAST_LANGUAGES) & groups.p.transform("nunique").gt(1)
    if not use.any():
        raise ValueError("No eligible stack-training paintings")
    columns = ["p", *ENCODERS]
    centered = (frame[columns] - groups[columns].transform("mean")).loc[use]
    scale = np.sqrt(counts[use].to_numpy() * frame.loc[use, "painting"].nunique())
    y = centered.p.to_numpy() / scale
    x = centered[list(ENCODERS)].to_numpy() / scale[:, None]
    gram, target, constant = x.T @ x, x.T @ y, y @ y

    group_sizes = frame.groupby("language").p.transform("size")
    ll = frame.loc[group_sizes.ge(MIN_LOSS_PAIRS)]
    if ll.empty:
        raise ValueError("No eligible stack-training Log Loss languages")
    ll_x = ll[list(ENCODERS)].to_numpy().clip(.01, .99)
    ll_y = ll.p.to_numpy()
    mass = 1 / (group_sizes.loc[ll.index].to_numpy() * ll.language.nunique() * np.log(2))
    anchor = np.asarray(config["fixed_weights"])
    rule = config["stack"]

    def value_gradient(w):
        p = ll_x @ w
        ll_value = -(mass @ (ll_y * np.log(p) + (1 - ll_y) * np.log1p(-p)))
        ll_gradient = ll_x.T @ (mass * ((1 - ll_y) / (1 - p) - ll_y / p))
        d = w - anchor
        mse = w @ gram @ w - 2 * (w @ target) + constant
        loss = mse + rule["log_loss_weight"] * ll_value + rule["ridge_weight"] * (d @ d)
        grad = 2 * (gram @ w - target) + rule["log_loss_weight"] * ll_gradient + 2 * rule["ridge_weight"] * d
        return loss, grad

    return value_gradient


def fit_stack(gold, components, config):
    objective = stack_objective(gold, components, config)
    anchor = np.asarray(config["fixed_weights"])
    rule = config["stack"]
    result = minimize(objective, anchor, method="SLSQP", jac=True, bounds=[(0., 1.)] * len(anchor),
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                    "jac": lambda w: np.ones_like(w)}],
                      options={"maxiter": rule["maxiter"], "ftol": rule["ftol"]})
    weights = result.x
    if (not result.success or not np.isfinite(result.fun) or (weights < 0).any() or (weights > 1).any()
            or abs(weights.sum() - 1) > 1e-10 or result.fun > objective(anchor)[0] + 1e-12):
        raise RuntimeError(f"Stack weights failed simplex or convergence checks: {result.message}")
    return {"weights": weights.tolist(), "iterations": int(result.nit), "objective": float(result.fun),
            "anchor_objective": float(objective(anchor)[0])}


def compare(delta, rule):
    values = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(rule["bootstrap_seed"])
    boot = np.concatenate([rng.choice(values, (min(256, rule["bootstrap_draws"] - offset), len(values))).mean(axis=1)
                           for offset in range(0, rule["bootstrap_draws"], 256)])
    trim = min(len(values) - 1, int(np.ceil(rule["trim_fraction"] * len(values))))
    keep = np.argsort(np.abs(values), kind="stable")[:len(values) - trim]
    return {"contrast_delta": float(values.mean()), "bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
            "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "losses": int((values < 0).sum()),
            "quantiles_05_25_50_75_95": np.quantile(values, [.05, .25, .50, .75, .95]).tolist(),
            "trimmed_count": trim, "trimmed_contrast_delta": float(values[keep].mean())}


def evaluate(gold, outputs, config):
    frame = gold[KEYS + ["fold", "k", "n"]].merge(outputs, on=KEYS, validate="one_to_one")
    require_same_pairs(gold, frame)
    frame["p"] = frame.k / frame.n
    reference, fixed, learned = "reference", "fixed", "learned"
    names = (reference, fixed, learned)
    rule = config["decision"]
    results, painting_details = {}, []
    for scope, block in (("full", frame), ("filtered", frame.loc[~frame.language.isin(config["test_like_excluded"])])):
        results[scope] = {"systems": {}, "comparisons": {}}
        outcome = results[scope]
        painted = block[["painting", "fold"]].drop_duplicates().set_index("painting")
        for name in names:
            def evaluate_block(part):
                metrics = score(part[KEYS + ["k", "n"]], part[KEYS].assign(p_positive=part[name]))
                metrics.pop("per_language")
                return metrics

            outcome["systems"][name] = {"pooled": evaluate_block(block),
                                        "folds": {int(fold): evaluate_block(part) for fold, part in block.groupby("fold")}}
            painted[name] = painting_scores(block, prediction=name).set_index("painting").contrast
        painted = painted.dropna()
        painting_details.append(painted.reset_index().assign(scope=scope))
        for a, b in ((fixed, reference), (learned, reference), (learned, fixed)):
            sa, sb = outcome["systems"][a], outcome["systems"][b]
            pairs = {i: {"contrast_delta": x["cultural_contrast"] - sb["folds"][i]["cultural_contrast"],
                         "log_loss_delta": x["log_loss"] - sb["folds"][i]["log_loss"]}
                     for i, x in sa["folds"].items()}
            outcome["comparisons"][f"{a}_minus_{b}"] = compare(painted[a] - painted[b], rule) | {
                "log_loss_delta": sa["pooled"]["log_loss"] - sb["pooled"]["log_loss"],
                "fold_wins": sum(x["contrast_delta"] > 0 for x in pairs.values()), "folds": pairs}

    def over_reference(name):
        full, filtered = [results[s]["comparisons"][f"{name}_minus_{reference}"] for s in ("full", "filtered")]
        return (all(x["contrast_delta"] >= rule["minimum_over_reference"]
                    and x["log_loss_delta"] <= rule["maximum_log_loss_regression"] for x in (full, filtered))
                and filtered["bootstrap_95"][0] > 0 and filtered["trimmed_contrast_delta"] > 0
                and filtered["fold_wins"] >= rule["reference_fold_wins"])

    full, filtered = [results[s]["comparisons"]["learned_minus_fixed"] for s in ("full", "filtered")]
    incremental = (full["contrast_delta"] >= 0 and filtered["contrast_delta"] >= rule["minimum_over_fixed"]
                   and all(x["log_loss_delta"] <= rule["maximum_log_loss_regression"] for x in (full, filtered))
                   and filtered["bootstrap_95"][0] > 0 and filtered["trimmed_contrast_delta"] > 0
                   and filtered["fold_wins"] >= rule["incremental_fold_wins"])
    decision = {"selected": "learned" if incremental and over_reference(learned) else "fixed",
                "fixed_reference_gate": over_reference(fixed), "learned_reference_gate": over_reference(learned),
                "learned_incremental_gate": incremental}
    return results | {"decision": decision}, pd.concat(painting_details, ignore_index=True)
