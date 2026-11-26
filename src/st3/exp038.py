"""EXP-038 frozen-feature interaction heads, candidate mixes, neighborhood banks and scoring.

Every per-encoder head is the unchanged Track 1 component (``competition_ridge.fit_head``). E4 adds one
language-independent linear term on the pair-level instruction delta, regularized with the same ridge
strength as the shared slope. Mixes, banks and corrections reuse the frozen Track 1 and neighborhood code.
"""

import hashlib
import io
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import expit, logit

from .competition_challenger import component_predictions, fit_heads, frame_sha
from .competition_neighborhood import make_bank, predict_correction
from .competition_ridge import Head, make_panel, painting_splits
from .competition_ridge import fit_head as track1_fit_head
from .competition_stack import ENCODERS, aligned_values, blend, compare
from .data import KEYS, read_exact_csv
from .exp036 import rounded
from .exp037_diagnostics import decompose, pair_diversity, paired_bootstrap
from .metrics import painting_scores, score
from .submission import COLUMNS, validate_predictions

ARMS = ("E1", "E2", "E3", "E4")
IMAGE_ARMS = ("E1", "E2", "E3")
R0_SYSTEMS = ("R0_fixed", "R0_corrected")


class ReplayError(RuntimeError):
    """The exact R0 reference could not be reproduced. The run must stop."""


def image_encoder(arm):
    return "E3" if arm == "E4" else arm


def candidate_encoders(arms):
    """Image encoders feeding a candidate, in a fixed order (base three first)."""
    names = list(ENCODERS)
    for arm in arms:
        if image_encoder(arm) not in names:
            names.append(image_encoder(arm))
    return tuple(names)


@dataclass(frozen=True)
class PairDelta:
    """Pair-level instruction deltas e(i,l) - e_neutral(i), keyed by painting and language."""

    index: dict
    values: np.ndarray

    @classmethod
    def build(cls, keys, language_embeddings, neutral):
        """``language_embeddings`` rows align with ``keys``; ``neutral`` is the E3 frame by painting."""
        if len(keys) != len(language_embeddings) or keys.duplicated(KEYS).any():
            raise ValueError("E4 embeddings must align one-to-one with unique pair keys")
        base = neutral.loc[keys.painting].to_numpy(dtype=np.float32)
        values = np.asarray(language_embeddings, dtype=np.float32) - base
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite E4 language delta")
        index = {pair: i for i, pair in enumerate(keys[KEYS].itertuples(index=False, name=None))}
        return cls(index, values)

    def rows(self, keys):
        try:
            positions = [self.index[pair] for pair in keys[KEYS].itertuples(index=False, name=None)]
        except KeyError as error:
            raise ValueError(f"E4 delta missing for requested pair {error.args[0]}") from None
        return self.values[positions].astype(np.float64)


@dataclass(frozen=True)
class PairHead:
    """The E3 head design plus one shared linear term on the pair delta."""

    head: Head
    gamma: np.ndarray

    @property
    def languages(self):
        return self.head.languages


def pair_objective(panel, x, rows, cols, d, ridge, precision):
    """Track 1 weighted NLL plus ``d @ gamma`` on observed cells, with gradient."""
    width, count, extra = x.shape[1], len(panel.languages), d.shape[1]
    basis = helmert(count, full=False).T
    votes = panel.n.sum()
    head_size = count + width + (count - 1) * width

    def unpack(theta):
        intercept = theta[0]
        contrast = theta[1:count]
        shared = theta[count:count + width]
        deviations = basis @ theta[count + width:head_size].reshape(-1, width)
        return Head(intercept, basis @ contrast, shared, deviations, panel.languages), theta[head_size:]

    def loss_grad(theta):
        head, gamma = unpack(theta)
        z = (head.intercept + x @ head.shared)[:, None] + head.language_intercepts
        z += x @ head.deviations.T
        z[rows, cols] += d @ gamma
        nll = (panel.n * np.logaddexp(0, z) - panel.k * z).sum() / votes
        value = nll + 0.5 * ridge * (head.shared @ head.shared + head.language_intercepts @ head.language_intercepts)
        residual = (panel.n * expit(z) - panel.k) / votes
        penalty_grad = precision @ head.deviations
        value += 0.5 * np.sum(head.deviations * penalty_grad) + 0.5 * ridge * (gamma @ gamma)
        gradient = [
            np.array([residual.sum()]),
            basis.T @ (residual.sum(axis=0) + ridge * head.language_intercepts),
            x.T @ residual.sum(axis=1) + ridge * head.shared,
            (basis.T @ (residual.T @ x + penalty_grad)).ravel(),
            d.T @ residual[rows, cols] + ridge * gamma,
        ]
        return value, np.concatenate(gradient)

    return loss_grad, unpack, head_size + extra


def fit_pair_head(gold, x_frame, delta, rule):
    panel = make_panel(gold)
    x = np.ascontiguousarray(x_frame.loc[list(panel.paintings)].to_numpy(dtype=np.float64))
    rows = gold.painting.map({name: i for i, name in enumerate(panel.paintings)}).to_numpy()
    cols = gold.language.map({name: i for i, name in enumerate(panel.languages)}).to_numpy()
    d = np.ascontiguousarray(delta.rows(gold))
    ridge = 1 / (rule["reference_c"] * panel.n.sum())
    precision = (ridge * rule["deviation_ratio"]) * np.eye(len(panel.languages))
    loss_grad, unpack, size = pair_objective(panel, x, rows, cols, d, ridge, precision)
    theta = np.zeros(size)
    theta[0] = logit((panel.k.sum() + 0.5) / (panel.n.sum() + 1))
    started = perf_counter()
    fitted = minimize(loss_grad, theta, jac=True, method="L-BFGS-B",
                      options={"gtol": rule["gtol"], "ftol": rule["ftol"], "maxiter": rule["maxiter"], "maxls": 40})
    norm = float(np.linalg.norm(fitted.jac, ord=np.inf))
    if not fitted.success or not np.isfinite(fitted.fun) or not np.isfinite(norm) or norm > rule["max_final_gradient"]:
        raise RuntimeError(f"E4 head did not converge: {fitted.message}, gradient infinity norm {norm:.3g}")
    head, gamma = unpack(fitted.x)
    receipt = {"parameters": size, "converged": True, "iterations": int(fitted.nit),
               "evaluations": int(fitted.nfev), "reason": str(fitted.message), "seconds": perf_counter() - started,
               "max_abs_gradient": norm, "objective": float(fitted.fun),
               "shared_l2": float(np.linalg.norm(head.shared)), "pair_term_l2": float(np.linalg.norm(gamma)),
               "deviations_l2": float(np.linalg.norm(head.deviations)), "pair_term_ridge": ridge}
    return PairHead(head, gamma), receipt


def fit_arm(arm, gold, features, delta, rule):
    if arm == "E4":
        return fit_pair_head(gold, features["E3"], delta, rule)
    panel = make_panel(gold)
    x = np.ascontiguousarray(features[arm].loc[list(panel.paintings)].to_numpy(dtype=np.float64))
    return track1_fit_head(panel, x, rule)


def fit_all(gold, features, delta, arms, track1):
    """The three Fixed Ridge heads through the original code path, then each requested arm."""
    heads, receipts = fit_heads(gold, features, track1)
    for arm in arms:
        heads[arm], receipts[arm] = fit_arm(arm, gold, features, delta, track1["ridge"])
    return heads, receipts


def arm_probabilities(head, keys, features, delta, arm):
    paintings, languages = sorted(keys.painting.unique()), sorted(keys.language.unique())
    row = keys.painting.map({name: i for i, name in enumerate(paintings)}).to_numpy()
    col = keys.language.map({name: i for i, name in enumerate(languages)}).to_numpy()
    base = head.head if isinstance(head, PairHead) else head
    x = np.ascontiguousarray(features[image_encoder(arm)].loc[paintings].to_numpy(dtype=np.float64))
    z = base.logits(x, languages)[row, col]
    if isinstance(head, PairHead):
        z = z + delta.rows(keys) @ head.gamma
    return expit(z).round(6)


def mix_columns(frame, candidates):
    frame = frame.assign(fixed=blend(frame[list(ENCODERS)].to_numpy(dtype=np.float64), None, fixed=True))
    for name, arms in candidates.items():
        values = frame[list(arms)].to_numpy(dtype=np.float64)
        frame[name] = (0.5 * frame.fixed.to_numpy() + 0.5 * values.mean(axis=1)).round(6)
    return frame


def predict_frame(heads, keys, features, delta, candidates):
    base = {name: heads[name] for name in ENCODERS}
    frame = component_predictions(base, keys[KEYS], features)
    for arm in heads:
        if arm not in ENCODERS:
            frame[arm] = arm_probabilities(heads[arm], keys[KEYS].reset_index(drop=True), features, delta, arm)
    return mix_columns(frame, candidates)


def available_candidates(candidates, arms):
    return {name: tuple(parts) for name, parts in candidates.items() if set(parts) <= set(arms)}


def r0_space(features):
    names = ("siglip2", "clip_large")
    return pd.concat([features[name] for name in names], axis=1, ignore_index=True) / np.sqrt(len(names))


def candidate_space(features, arms):
    blocks = []
    for name in candidate_encoders(arms):
        x = features[name].to_numpy(dtype=np.float64)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if not np.isfinite(x).all() or (norms <= 0).any():
            raise ValueError(f"Invalid similarity features for {name}")
        blocks.append(pd.DataFrame(x / norms, index=features[name].index))
    return pd.concat(blocks, axis=1, ignore_index=True) / np.sqrt(len(blocks))


def spaces(features, candidates):
    return {"R0": r0_space(features)} | {name: candidate_space(features, arms) for name, arms in candidates.items()}


def crossfit(train, features, delta, arms, track1, fold, candidates):
    """Inner painting-grouped predictions inside one fitting set, exactly as the original bank."""
    names = sorted(train.painting.unique())
    assignments = painting_splits(names, track1["inner_seed"] + fold, track1["inner_folds"])
    index = train.painting.map(dict(zip(names, assignments)))
    pieces, receipts = [], []
    for split in range(track1["inner_folds"]):
        fit, query = train.loc[index.ne(split)], train.loc[index.eq(split)]
        if query.empty or fit.empty or set(fit.painting) & set(query.painting):
            raise ValueError("Inner ridge fold has invalid painting separation")
        heads, opts = fit_all(fit, features, delta, arms, track1)
        pieces.append(predict_frame(heads, query[KEYS], features, delta, {}))
        receipts.append({"inner_fold": split, "fit_paintings": fit.painting.nunique(),
                         "query_paintings": query.painting.nunique(), "fit_sha256": frame_sha(fit),
                         "query_keys_sha256": frame_sha(query[KEYS]), "optimizers": opts})
    parts = pd.concat(pieces, ignore_index=True).sort_values(KEYS, ignore_index=True)
    r0_sha = frame_sha(parts[KEYS + list(ENCODERS)])
    # Same order as the original bank: train sorted by key, aligned one-to-one with the inner parts.
    aligned_values(train.sort_values(KEYS), parts, ENCODERS)
    parts = mix_columns(parts.drop(columns="fixed"), candidates)
    return parts, {"r0_components_sha256": r0_sha, "inner_folds": receipts}


def banks(train, inner, visual, candidates, minimum_languages):
    systems = {"R0": "fixed"} | {name: name for name in candidates}
    return {system: make_bank(train, inner[KEYS + [column]].rename(columns={column: "p_positive"}),
                              visual[system], minimum_languages) for system, column in systems.items()}


def corrections(bank_set, keys, visual, outer, candidates, rule):
    columns = {"R0": "fixed"} | {name: name for name in candidates}
    result = outer[KEYS].copy()
    details = {}
    for system, column in columns.items():
        base = outer[KEYS + [column]].rename(columns={column: "p_positive"})
        corrected = predict_correction(bank_set[system], keys[KEYS], visual[system], base, rule)
        aligned = keys[KEYS].merge(corrected, on=KEYS, validate="one_to_one")
        label = "R0_corrected" if system == "R0" else system
        result = result.merge(aligned[KEYS + ["corrected"]].rename(columns={"corrected": label}), on=KEYS,
                              validate="one_to_one")
        details[system] = corrected
    return result, details


def systems_frame(outer, corrected, arms, candidates):
    frame = outer[KEYS].assign(R0_fixed=outer.fixed.to_numpy())
    frame = frame.merge(corrected, on=KEYS, validate="one_to_one")
    for arm in arms:
        frame[arm] = outer[arm].to_numpy()
    for name in candidates:
        frame[f"{name}_mix"] = outer[name].to_numpy()
    return frame


def evaluate_p2(gold, features, delta, arms, config, track1, neighborhood, expected_inner, log=print):
    """Five outer painting folds; each outer fit owns its heads, inner fits and neighbor bank."""
    candidates = available_candidates(config["candidates"], arms)
    visual = spaces(features, candidates)
    outputs, components, r0_details, receipts = [], [], [], {}
    for fold, held in gold.groupby("fold"):
        tick = perf_counter()
        train, keys = gold.loc[gold.fold.ne(fold)], held[KEYS]
        if set(train.painting) & set(keys.painting):
            raise ValueError("Outer-fold painting overlap")
        heads, fits = fit_all(train, features, delta, arms, track1)
        outer = predict_frame(heads, keys, features, delta, candidates)
        inner, details = crossfit(train, features, delta, arms, track1, int(fold), candidates)
        if details["r0_components_sha256"] != expected_inner[str(int(fold))]:
            raise ReplayError(f"Fold {fold} failed exact replay of the Track 1 inner component probabilities")
        bank_set = banks(train, inner, visual, candidates, neighborhood["minimum_bank_languages"])
        for system, bank in bank_set.items():
            if set(bank.visual.index) & set(keys.painting):
                raise ValueError(f"Held-out painting reached the {system} neighbor bank")
        corrected, correction_details = corrections(bank_set, keys, visual, outer, candidates, neighborhood)
        outputs.append(systems_frame(outer, corrected, arms, candidates))
        components.append(outer.assign(fold=int(fold)))
        r0_details.append(correction_details["R0"])
        receipts[int(fold)] = {"outer_fits": fits, "inner": details, "training_sha256": frame_sha(train),
                               "outer_keys_sha256": frame_sha(keys),
                               "bank_paintings": {k: len(v.visual) for k, v in bank_set.items()},
                               "seconds": perf_counter() - tick}
        log(f"P2 fold {fold}: R0 inner replay matched; scored {len(keys)} held-out pairs in "
            f"{perf_counter() - tick:.1f}s")
    frame = pd.concat(outputs, ignore_index=True).sort_values(KEYS, ignore_index=True)
    frame = frame.merge(gold[KEYS + ["fold"]], on=KEYS, validate="one_to_one")
    r0 = pd.concat(r0_details, ignore_index=True).sort_values(KEYS, ignore_index=True)
    r0 = r0.merge(gold[KEYS + ["fold"]], on=KEYS, validate="one_to_one")
    parts = pd.concat(components, ignore_index=True).sort_values(KEYS, ignore_index=True)
    return frame, parts, r0, receipts, candidates


def csv_text(frame):
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.6f")


def replay_r0_p2(gold, parts, r0_frame, track1_oof, corrected_oof_sha, metrics, excluded):
    """Fixed Ridge rows exact, Corrected Ridge CSV bytes exact, pooled metrics to 1e-12."""
    for column in (*ENCODERS, "fixed"):
        mine = aligned_values(gold, parts, [column]).ravel()
        saved = aligned_values(gold, track1_oof, [column]).ravel()
        if not np.array_equal(mine, saved):
            differences = int((mine != saved).sum())
            raise ReplayError(f"Fixed Ridge OOF replay differs in {differences} {column} rows")
    digest = hashlib.sha256(csv_text(r0_frame).encode()).hexdigest()
    if digest != corrected_oof_sha:
        raise ReplayError(f"Corrected Ridge OOF CSV replay differs: {digest}")
    track1_metrics, neighborhood_metrics = metrics
    observed = {}
    for scope, subset in scopes(gold, excluded):
        for system, column, expected in (
                ("fixed", "fixed", track1_metrics[scope]["systems"]["fixed"]["pooled"]),
                ("corrected", "corrected", neighborhood_metrics[scope]["pooled"]["corrected"])):
            source = parts if column == "fixed" else r0_frame
            values = subset[KEYS].merge(source[KEYS + [column]], on=KEYS, validate="one_to_one")[column].to_numpy()
            result = score(subset[KEYS + ["k", "n"]], subset[KEYS].assign(p_positive=values))
            for key in ("cultural_contrast", "log_loss"):
                if abs(result[key] - expected[key]) > 1e-12:
                    raise ReplayError(f"{system} {scope} {key} replay differs: {result[key]} vs {expected[key]}")
            observed[f"{system}_{scope}"] = {key: result[key] for key in ("cultural_contrast", "log_loss")}
    return {"fixed_rows": len(gold), "corrected_csv_sha256": digest, "pooled": observed}


def scopes(frame, excluded):
    return (("full", frame), ("filtered", frame.loc[~frame.language.isin(excluded)]))


def painting_contrasts(block, systems):
    table = block[["painting", "fold"]].drop_duplicates().set_index("painting") if "fold" in block else \
        pd.DataFrame(index=pd.Index(sorted(block.painting.unique()), name="painting"))
    for name in systems:
        table[name] = painting_scores(block, prediction=name).set_index("painting").contrast
    return table.dropna(subset=list(systems))


def pooled(block, name):
    result = score(block[KEYS + ["k", "n"]], block[KEYS].assign(p_positive=block[name].to_numpy()))
    result.pop("per_language")
    return result


def p2_report(gold, frame, systems, candidates, config):
    """Pooled, per-fold and painting-paired metrics, and the frozen gate per candidate."""
    rule = config["gate"]
    reference = rule["reference"]
    joined = gold[KEYS + ["fold", "k", "n"]].merge(frame.drop(columns="fold"), on=KEYS, validate="one_to_one")
    joined["p"] = joined.k / joined.n
    report, tables = {}, []
    for scope, block in scopes(joined, config["test_like_excluded"]):
        result = {"systems": {}, "comparisons": {}}
        for name in systems:
            result["systems"][name] = {"pooled": pooled(block, name),
                                       "folds": {int(f): pooled(part, name) for f, part in block.groupby("fold")}}
        table = painting_contrasts(block, systems)
        tables.append(table.reset_index().assign(scope=scope))
        for name in systems:
            if name == reference:
                continue
            a, b = result["systems"][name], result["systems"][reference]
            folds = {i: a["folds"][i]["cultural_contrast"] - b["folds"][i]["cultural_contrast"] for i in a["folds"]}
            delta = table[name] - table[reference]
            result["comparisons"][f"{name}_minus_{reference}"] = compare(delta, rule) | {
                "pooled_contrast_delta": a["pooled"]["cultural_contrast"] - b["pooled"]["cultural_contrast"],
                "log_loss_delta": a["pooled"]["log_loss"] - b["pooled"]["log_loss"],
                "fold_deltas": folds, "fold_wins": sum(v > 0 for v in folds.values()),
                "leave_one_fold_out": {i: float(delta[table.fold.ne(i)].mean()) for i in folds}}
        report[scope] = result
    gates = {name: gate(report, name, reference, rule) for name in candidates}
    return report, pd.concat(tables, ignore_index=True), gates


def gate(report, name, reference, rule):
    key = f"{name}_minus_{reference}"
    full, filtered = report["full"]["comparisons"][key], report["filtered"]["comparisons"][key]
    checks = {
        "filtered_gain": filtered["pooled_contrast_delta"] >= rule["minimum_filtered_gain"],
        "bootstrap_lower_positive": filtered["bootstrap_95"][0] > 0,
        "filtered_fold_wins": filtered["fold_wins"] >= rule["minimum_fold_wins"],
        "trimmed_gain_positive": filtered["trimmed_contrast_delta"] > 0,
        "full_no_regression": full["pooled_contrast_delta"] >= 0,
        "log_loss_full": full["log_loss_delta"] <= rule["maximum_log_loss_delta"],
        "log_loss_filtered": filtered["log_loss_delta"] <= rule["maximum_log_loss_delta"],
    }
    numbers = {"filtered_contrast_delta": filtered["pooled_contrast_delta"],
               "filtered_bootstrap_95": filtered["bootstrap_95"], "filtered_fold_wins": filtered["fold_wins"],
               "filtered_trimmed_delta": filtered["trimmed_contrast_delta"],
               "full_contrast_delta": full["pooled_contrast_delta"], "full_log_loss_delta": full["log_loss_delta"],
               "filtered_log_loss_delta": filtered["log_loss_delta"],
               "filtered_contrast": report["filtered"]["systems"][name]["pooled"]["cultural_contrast"]}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "passed": all(checks.values()),
            "checks": checks, "numbers": numbers}


def select(gates, order, tolerance):
    """Highest filtered Contrast among passing candidates; within the tolerance, the simpler one."""
    passing = [name for name in order if gates.get(name, {}).get("passed")]
    if not passing:
        return None
    best = max(gates[name]["numbers"]["filtered_contrast"] for name in passing)
    for name in passing:
        if best - gates[name]["numbers"]["filtered_contrast"] <= tolerance:
            return name
    raise AssertionError("unreachable")


def p2_diagnostics(gold, frame, arms, candidates, config):
    """Pre-registered, non-gating diagnostics."""
    joined = gold[KEYS + ["fold", "k", "n"]].merge(frame.drop(columns="fold"), on=KEYS, validate="one_to_one")
    joined["p"] = joined.k / joined.n
    seed = config["diagnostic_seed"]
    out, languages, paintings = {}, [], []
    names = ["R0_fixed", "R0_corrected", *arms, *[f"{c}_mix" for c in candidates], *candidates]
    for scope, block in scopes(joined, config["test_like_excluded"]):
        scored = painting_scores(block, prediction="R0_fixed")
        eligible = set(scored.loc[scored.status.eq("scored"), "painting"])
        result = {"diversity_vs_fixed": {}, "decomposition": {}}
        for name in names:
            if name == "R0_fixed":
                continue
            pair = pair_diversity(block, ["R0_fixed", name], eligible)[0]
            result["diversity_vs_fixed"][name] = {key: pair[key] for key in (
                "centered_residual_r", "order_agreement", "residual_r", "centered_prediction_r")} | {
                "below_0.95": pair["centered_residual_r"] < .95}
        for name in names:
            summary, per_painting, order = decompose(block, name)
            result["decomposition"][name] = {key: summary[key] for key in (
                "contrast", "ranking", "deviation_rmse", "deviation_calibration_slope", "global_rmse")}
            for language, (hits, total) in order.items():
                languages.append({"scope": scope, "system": name, "language": language,
                                  "pair_order_accuracy": hits / total if total else np.nan, "ordered_pairs": total})
        if {"E3", "E4"} <= set(arms):
            table = painting_contrasts(block, ["E3", "E4"])
            result["E4_minus_E3"] = paired_bootstrap(table.E4, table.E3, seed=seed)
        table = painting_contrasts(block, [n for n in names])
        for name in names:
            if name != "R0_corrected":
                table[f"{name}_minus_R0_corrected"] = table[name] - table["R0_corrected"]
        paintings.append(table.reset_index().assign(scope=scope))
        for language, part in block.groupby("language"):
            for name in names:
                q = part[name].clip(.01, .99)
                loss = float((-(part.p * np.log2(q) + (1 - part.p) * np.log2(1 - q))).mean())
                languages.append({"scope": scope, "system": name, "language": language, "pairs": len(part),
                                  "log_loss": loss})
        out[scope] = result
    language_table = pd.DataFrame(languages)
    return out, language_table, pd.concat(paintings, ignore_index=True)


# Full-training refit and Development.

def deployment_fit(train, features, delta, arms, track1, neighborhood, candidates):
    """Heads on all official training paintings, and banks cross-fitted like Corrected Ridge's bank."""
    heads, fits = fit_all(train, features, delta, arms, track1)
    inner, details = crossfit(train, features, delta, arms, track1, 0, candidates)
    visual = spaces(features, candidates)
    bank_set = banks(train, inner, visual, candidates, neighborhood["minimum_bank_languages"])
    return heads, bank_set, {"fits": fits, "crossfit": details, "training_inner_sha256": frame_sha(inner)}, visual


def development_predictions(heads, bank_set, visual, keys, features, delta, candidates, neighborhood):
    for system, bank in bank_set.items():
        if set(bank.visual.index) & set(keys.painting):
            raise ValueError(f"A Development painting is in the {system} bank")
    outer = predict_frame(heads, keys, features, delta, candidates)
    corrected, details = corrections(bank_set, keys, visual, outer, candidates, neighborhood)
    arms = [name for name in heads if name not in ENCODERS]
    frame = systems_frame(outer, corrected, arms, candidates)
    return keys[KEYS].merge(frame, on=KEYS, validate="one_to_one"), details


def system_frame(frame, name):
    return frame[KEYS].assign(p_positive=frame[name].to_numpy())


def canonical(frame, dev):
    """Six-decimal CSV text parsed by the official prediction reader, as every saved archive is read."""
    text = frame[COLUMNS].to_csv(index=False, lineterminator="\n", float_format="%.6f")
    parsed = read_exact_csv(io.StringIO(text), COLUMNS)
    return rounded(parsed.assign(p_positive=validate_predictions(parsed, dev)), dev)


def mix_pair(parent, candidate, alpha, dev):
    """EXP-036/037 composition: round6((1 - alpha) * parent + alpha * candidate)."""
    parent, candidate = canonical(parent, dev), canonical(candidate, dev)
    return candidate.assign(p_positive=((1 - alpha) * parent.p_positive + alpha * candidate.p_positive).round(6))


def mean_of(frames, dev):
    """CS-CVA-3: round6 of the plain mean of the canonical seed predictions."""
    aligned = [canonical(frame, dev) for frame in frames]
    values = sum(frame.p_positive.to_numpy() for frame in aligned) / len(aligned)
    return aligned[0].assign(p_positive=np.round(values, 6))


def blends(bases, cscva, c6, dev, config):
    """B1 = round6(.75 Cx + .25 CS-CVA), B2 = round6(.80 B1 + .20 C6) for each available CS-CVA."""
    weights = config["blends"]
    out = {}
    for base_name, base in bases.items():
        for label, cs in cscva.items():
            b1 = mix_pair(base, cs, weights["B1_cscva_weight"], dev)
            out[f"B1_{base_name}_{label}"] = b1
            out[f"B2_{base_name}_{label}"] = mix_pair(b1, c6, weights["B2_c6_weight"], dev)
    return out


def development_scores(gold, predictions, excluded):
    keep = ~gold.language.isin(excluded)
    result = {}
    for name, frame in predictions.items():
        full = score(gold, frame)
        subset = gold.loc[keep]
        filtered = score(subset, subset[KEYS].merge(frame, on=KEYS, validate="one_to_one"))
        result[name] = {"full": {k: full[k] for k in ("cultural_contrast", "log_loss")},
                        "evaluation_like": {k: filtered[k] for k in ("cultural_contrast", "log_loss")}}
    return result


def development_intervals(gold, predictions, references, excluded, seed):
    joined = gold.assign(p=gold.k / gold.n)
    out = {}
    for scope, block in scopes(joined, excluded):
        frame = block.copy()
        for name, prediction in (predictions | references).items():
            frame[name] = frame[KEYS].merge(prediction, on=KEYS, validate="one_to_one").p_positive.to_numpy()
        table = painting_contrasts(frame, list(predictions) + list(references))
        out[scope] = {f"{name}_minus_{ref}": paired_bootstrap(table[name], table[ref], seed=seed)
                      for name in predictions for ref in references}
    return out


def eligibility(name, config):
    return "VLM_FREE" if name in config["eligibility"]["vlm_free"] else config["eligibility"]["unconfirmed_label"]
