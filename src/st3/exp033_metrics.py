"""Paired five-fold analysis for the preregistered EXP-033 contrasts."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import KEYS
from .metrics import painting_scores, score


def summarize(frame, prediction):
    pooled = score(frame[KEYS + ["k", "n"]], prediction)
    joint = frame.merge(prediction, on=KEYS, validate="one_to_one")
    joint["p"] = joint["k"] / joint["n"]
    folds = {str(i): score(group[KEYS + ["k", "n"]], group[KEYS + ["p_positive"]])
             for i, group in joint.groupby("fold")}
    paintings = painting_scores(joint).dropna(subset=["contrast"]).set_index("painting")
    concordance, spearman = [], []
    for name, rows in joint.groupby("painting"):
        if name not in paintings.index:
            continue
        p, q = rows.p.to_numpy(), rows.p_positive.to_numpy()
        a, b = np.triu_indices(len(p), 1)
        true, predicted = np.sign(p[a] - p[b]), np.sign(q[a] - q[b])
        keep = true != 0
        concordance.append(np.mean((true[keep] == predicted[keep]) + .5 * (predicted[keep] == 0)))
        spearman.append(np.corrcoef(rankdata(p), rankdata(q))[0, 1] if np.ptp(q) else 0.)
    calibration = {}
    for language, rows in joint.groupby("language"):
        if not pooled["per_language"][language]["included"]:
            continue
        p, q = rows.p.to_numpy(), rows.p_positive.to_numpy()
        bucket = np.minimum((10 * q).astype(int), 9)
        calibration[language] = {
            "brier_share": float(np.mean((p - q)**2)), "tail_fraction": float(np.mean((q <= .1) | (q >= .9))),
            "bins": [{"bin": int(i), "pairs": int((bucket == i).sum()),
                      "observed": float(p[bucket == i].mean()), "predicted": float(q[bucket == i].mean())}
                     for i in np.unique(bucket)],
        }
    pooled.update(spearman=float(np.mean(spearman)), concordance=float(np.mean(concordance)))
    support = {}
    for i, held in joint.groupby("fold"):
        counts = frame.loc[frame.fold.ne(i)].groupby("language").painting.nunique()
        for name, rows in held.groupby(np.searchsorted([1, 50, 200], held.language.map(counts).fillna(0), side="right")):
            support.setdefault(str(name), []).append(rows)
    support_metrics = {}
    for name, parts in support.items():
        rows = pd.concat(parts, ignore_index=True)
        support_metrics[name] = score(rows[KEYS + ["k", "n"]], rows[KEYS + ["p_positive"]]) | {"pairs": len(rows)}
    return {"pooled": pooled, "folds": folds, "calibration": calibration, "fitting_support": support_metrics}, paintings.contrast


def compare(a, b, stats, paintings, config):
    if not paintings[a].index.equals(paintings[b].index):
        raise ValueError("Mismatched eligible paintings")
    delta = (paintings[a] - paintings[b]).to_numpy()
    generator = np.random.default_rng(config["bootstrap_seed"])
    draws = []
    for offset in range(0, config["bootstrap"], 128):
        size = min(128, config["bootstrap"] - offset)
        indices = generator.integers(len(delta), size=(size, len(delta)))
        draws.extend(delta[indices].mean(axis=1))
    by_fold = {i: stats[a]["folds"][i]["cultural_contrast"] - stats[b]["folds"][i]["cultural_contrast"]
               for i in stats[a]["folds"]}
    return {"gain": float(delta.mean()), "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
            "fold_gains": by_fold, "fold_wins": sum(v > 0 for v in by_fold.values()),
            "painting_wins": int((delta > 0).sum()), "painting_losses": int((delta < 0).sum()),
            "log_loss_delta": stats[a]["pooled"]["log_loss"] - stats[b]["pooled"]["log_loss"]}


def report(frame, standalone, fused, config):
    contrasts = [("B", "A"), ("C", "B"), ("D", "C"), ("probe", "B")]
    stats, comparisons = {}, {}
    for scope, series in (("standalone", standalone), ("fusion", fused)):
        score_map, paintings = {}, {}
        for name, preds in series.items():
            score_map[name], paintings[name] = summarize(frame, preds)
        pairs = contrasts + ([("D", "E50+L"), ("D", "ridge"), ("C", "E50+L"),
                              ("C", "ridge"), ("C", "A")] if scope == "fusion" else [])
        stats[scope] = score_map
        comparisons[scope] = {f"{a}-{b}": compare(a, b, score_map, paintings, config) for a, b in pairs}
    fused_comps = comparisons["fusion"]

    def meets(label, threshold):
        pair = fused_comps[label]
        return bool(pair["gain"] >= threshold and pair["ci95"][0] > 0 and pair["fold_wins"] >= 3
                    and pair["log_loss_delta"] <= config["guardrail"])

    checks = {}
    for arm, baselines in (("D", {"C": config["minimum_increment"]}),
                           ("C", {"B": config["minimum_increment"], "A": 0})):
        limits = baselines | {name: config["minimum_reference_gain"] for name in ("E50+L", "ridge")}
        checks[arm] = {k: meets(f"{arm}-{k}", v) for k, v in limits.items()}
    status = "EXPLORATORY_D_GO" if all(checks["D"].values()) else (
        "EXPLORATORY_C_GO" if all(checks["C"].values()) else "NO_GO_OOF"
    )
    return {"status": status, "metrics": stats, "paired": comparisons, "checks": checks,
            "scope": "REUSED_P2_EXPLORATORY"}
