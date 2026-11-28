#!/usr/bin/env python3
"""H1 Gate 0 (CPU only): en/ar/zh deviation transfer, oracle ceiling over C2, and visual learnability.

Inputs are official Train annotations (from the verified EXP-037 base), the pinned P2 folds (EXP-038 overlay)
and the sealed EXP-038 run-4 C2 OOF and E1/E3 features. Development is not read. Nothing external is used.
"""

import argparse
import hashlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3 import h1_gate0 as gate  # noqa: E402
from st3.competition_stack import compare  # noqa: E402
from st3.exp036_io import read, sha  # noqa: E402
from st3.exp037_io import verify_inputs  # noqa: E402
from st3.exp038_io import restore_overlay, training_population, unpack_overlay  # noqa: E402
from st3.metrics import painting_scores  # noqa: E402

PINS = {"base_manifest_sha256": "026720f406c02236aeaf5bfdc806853816323b66267f81f0e3e2dbbfe1ce7e0e",
        "training_pair_sha256": "c7cf5585752d451c7ae13f427337a37cbe9c20ae0b28c7f686afed87a57a3e69",
        "exp038_results_zip_sha256": "b80e7cda21ee9381b947c2a1707742d4bbf7dff44c864130185b71ccefbe9e09",
        "exp038_overlay_zip_sha256": "c3adc3b5104bc914f9aa1a003d6c5f41b706a16dc460a0c36f4781b24598370d",
        "exp038_overlay_manifest_sha256": "b3303f035a5765941284d2a8398f0cb6ead2f67f0286c246ea50f2b607a2d202",
        "p2_protocol_sha256": "869b2fe227fb614d83827d8124a80bd02c5f7677482bf64ae7098856b9c70eed",
        "p2_pairs_sha256": "50798ddac6d9bda709c53b0ab7fc5ff2bfc6fbc33a4e581c1ab5c45f94d57534"}
SEED = 20261128
FRACTIONS = (.25, .5, .75, 1.)
ALPHAS_ORACLE = (.1, 1., 10., 100., 1000.)
ALPHAS_CURVE = tuple(np.logspace(-1, 4, 11))


def results_member(z, manifest, name):
    content = z.read(name)
    if hashlib.sha256(content).hexdigest() != manifest["files"][name]:
        raise ValueError(f"EXP-038 results member differs: {name}")
    return content


def load(args):
    if sha(args.exp038_results) != PINS["exp038_results_zip_sha256"] or sha(args.exp038_overlay) != PINS["exp038_overlay_zip_sha256"]:
        raise ValueError("EXP-038 archive checksum differs")
    base_manifest = verify_inputs(args.base, PINS["base_manifest_sha256"])
    train, _ = training_population(args.base, PINS["training_pair_sha256"])
    scratch = Path(tempfile.mkdtemp())
    mounted = unpack_overlay(args.exp038_overlay, scratch / "m", PINS["exp038_overlay_manifest_sha256"],
                             PINS["exp038_overlay_zip_sha256"])
    overlay = restore_overlay(mounted, PINS["exp038_overlay_manifest_sha256"], PINS["base_manifest_sha256"], scratch / "r")
    if sha(overlay / "p2/protocol.csv") != PINS["p2_protocol_sha256"] or sha(overlay / "p2/pairs.csv") != PINS["p2_pairs_sha256"]:
        raise ValueError("P2 fold assignment differs")
    gold = pd.read_csv(overlay / "p2/pairs.csv")[["painting", "language", "k", "n", "fold"]]
    with ZipFile(args.exp038_results) as z:
        manifest = json.loads(z.read("export_manifest.json"))
        oof = pd.read_csv(io.BytesIO(results_member(z, manifest, "p2/oof.csv")))
        features = {}
        for arm in ("E1", "E3"):
            with np.load(io.BytesIO(results_member(z, manifest, f"features/{arm}/features.npz"))) as cache:
                features[arm] = pd.DataFrame(cache["values"], index=cache["keys"].tolist())
    images = train[["painting", "image"]].drop_duplicates().set_index("painting").image
    digest = images.map(lambda name: Path(base_manifest["images"][name]).stem)
    return train, gold, oof, features, digest


def analysis_b(gold, oof, excluded, folds):
    p2 = gate.logits(gold)
    c2 = oof[["painting", "language", "C2"]].assign(
        c2_logit=lambda f: np.log(f.C2.clip(1e-6, 1 - 1e-6) / (1 - f.C2.clip(1e-6, 1 - 1e-6))))
    features = gate.source_features(p2, c2)
    rows = p2[~p2.language.isin(excluded)].merge(c2, on=["painting", "language"], validate="one_to_one")
    rows = rows.assign(residual=rows.y - rows.c2_logit, fold=rows.painting.map(folds))
    outputs = {"oracle": [], "control": []}
    alphas = {}
    for fold in sorted(rows.fold.unique()):
        train, held = rows[rows.fold.ne(fold)], rows[rows.fold.eq(fold)]
        for label, values in (("oracle", True), ("control", False)):
            models = gate.fit_oracle(train, features, ALPHAS_ORACLE, 3, SEED + fold, use_values=values)
            outputs[label].append(gate.apply_oracle(models, held, features))
            alphas.setdefault(label, {})[int(fold)] = {k: m[2] for k, m in models.items()}
    oracle = pd.concat(outputs["oracle"]).rename(columns={"oracle": "oracle_q"})
    control = pd.concat(outputs["control"])[["painting", "language", "oracle"]].rename(columns={"oracle": "control_q"})
    frame = oracle.merge(control, on=["painting", "language"]).assign(p=lambda f: f.k / f.n)
    subsets = {"i_at_least_one_source": features.index[features.filter(like="has_").sum(axis=1) >= 1],
               "ii_all_three_sources": features.index[features.filter(like="has_").sum(axis=1) == 3]}
    rule = {"bootstrap_seed": SEED, "bootstrap_draws": 20000, "trim_fraction": .05}
    out = {"alphas": alphas}
    for name, paintings in subsets.items():
        block = frame[frame.painting.isin(paintings)]
        table = pd.DataFrame(index=sorted(block.painting.unique()))
        for column in ("C2", "oracle_q", "control_q"):
            table[column] = painting_scores(block, prediction=column).set_index("painting").contrast
        table = table.dropna()
        out[name] = {"paintings": len(table), "C2": float(table.C2.mean()),
                     "oracle_minus_C2": compare(table.oracle_q - table.C2, rule),
                     "control_minus_C2": compare(table.control_q - table.C2, rule),
                     "oracle_minus_control": compare(table.oracle_q - table.control_q, rule)}
    return out


def analysis_c(gold, features, digest, folds):
    p2 = gate.logits(gold)
    y, v = gate.wide(p2)
    v_exact = gate.wide(p2[["painting", "language", "k", "n"]], variance="exact")[1]
    blocks = []
    for arm in ("E3", "E1"):
        x = features[arm].loc[digest.loc[y.index]].to_numpy(dtype=float)
        blocks.append(x / np.linalg.norm(x, axis=1, keepdims=True))
    x = pd.DataFrame(np.concatenate(blocks, axis=1) / np.sqrt(2), index=y.index)
    out, curves = {}, []
    for source in gate.SOURCES:
        rows = []
        for painting in y.index:
            if np.isnan(y.at[painting, source]):
                continue
            others = [c for c in y.columns if c != source and not np.isnan(y.at[painting, c])]
            if len(others) < 3:
                continue
            rows.append((painting, y.at[painting, source] - y.loc[painting, others].mean(),
                         v.at[painting, source] + v.loc[painting, others].sum() / len(others) ** 2,
                         v_exact.at[painting, source] + v_exact.loc[painting, others].sum() / len(others) ** 2))
        frame = pd.DataFrame(rows, columns=["painting", "d", "v", "v_exact"])
        reliability = 1 - frame.v.mean() / frame.d.var(ddof=0)
        reliability_exact = 1 - frame.v_exact.mean() / frame.d.var(ddof=0)
        curve = gate.learning_curve(x, frame, folds, FRACTIONS, (0, 1, 2), ALPHAS_CURVE)
        curve = curve.assign(source=source,
                             r_disattenuated=curve.r / np.sqrt(reliability) if reliability > 0 else np.nan,
                             r_disattenuated_exact=curve.r / np.sqrt(reliability_exact) if reliability_exact > 0 else np.nan)
        curves.append(curve)
        out[source] = {"paintings": len(frame), "target_reliability": float(reliability),
                       "target_reliability_exact_sensitivity": float(reliability_exact),
                       "slope_r_per_log_n": gate.slope(curve),
                       "slope_disattenuated_per_log_n": gate.slope(curve.assign(r=curve.r_disattenuated)),
                       "slope_disattenuated_exact_per_log_n": gate.slope(curve.assign(r=curve.r_disattenuated_exact)),
                       "by_fraction": curve.groupby("fraction").agg(
                           r=("r", "mean"), r_sd=("r", "std"), r_disattenuated=("r_disattenuated", "mean"),
                           r_disattenuated_exact=("r_disattenuated_exact", "mean"),
                           n=("mean_train_paintings", "mean")).reset_index().to_dict(orient="records")}
    return out, pd.concat(curves, ignore_index=True)


def figure(curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for source, part in curves.groupby("source"):
        means = part.groupby("mean_train_paintings")
        line = ax.errorbar(means.r.mean().index, means.r.mean(), yerr=means.r.std(), marker="o", capsize=3,
                           label=f"{source} raw")
        exact = means.r_disattenuated_exact.mean()
        if exact.notna().any():
            ax.plot(exact.index, exact, ls="--", color=line[0].get_color(), alpha=.7,
                    label=f"{source} disattenuated (exact-variance sensitivity)")
    ax.set_xscale("log")
    ax.axhline(0, color="grey", lw=.8)
    ax.set_xlabel("Training paintings per fold (log scale)")
    ax.set_ylabel("CV Pearson r (mean of 3 subsamples, +/-1 sd)")
    ax.set_title("H1 Gate 0: visual learnability of en/ar/zh deviations", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--exp038-results", type=Path, required=True)
    parser.add_argument("--exp038-overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    tick = time.time()
    train, gold, oof, features, digest = load(args)
    excluded = list(gate.SOURCES)
    targets = sorted(set(train.language) - set(excluded))
    y, v = gate.wide(train)
    v_exact = gate.wide(train[["painting", "language", "k", "n"]], variance="exact")[1]
    eligible = {lang: int(y[lang].notna().sum()) for lang in gate.SOURCES}
    all_three = int(y[list(gate.SOURCES)].notna().all(axis=1).sum())
    a, cells = {}, {}
    for mode in ("shared", "split"):
        a[mode], cells[mode] = gate.analysis_a(y, v, targets, mode, seed=SEED)
        a[f"{mode}_exact_variance_sensitivity"], cells[f"{mode}_exact"] = gate.analysis_a(
            y, v_exact, targets, mode, seed=SEED)
    a["shared"]["expected_null_r_mean"] = float(gate.shared_reference_null(cells["shared"].mean_others).mean())
    a["sampling_variance_mean"] = {"delta": float(np.nanmean(v.to_numpy())),
                                   "exact": float(np.nanmean(v_exact.to_numpy()))}
    a["cells_with_both_reliabilities_positive"] = {
        m: int(((t.reliability_source > 0) & (t.reliability_target > 0)).sum()) for m, t in cells.items()}
    folds = gold.drop_duplicates("painting").set_index("painting").fold
    b = analysis_b(gold, oof, excluded, folds)
    c, curves = analysis_c(gold, features, digest, folds)
    positive = sum(c[s]["slope_r_per_log_n"] > 0 for s in gate.SOURCES)
    decisions, clauses = {}, {}
    oracle = b["i_at_least_one_source"]["oracle_minus_C2"]
    for mode in ("shared", "split", "shared_exact_variance_sensitivity", "split_exact_variance_sensitivity"):
        pooled = a[mode]["all"]
        decisions[mode] = gate.decide(pooled["pooled_disattenuated"], pooled["pooled_disattenuated_ci95"],
                                      oracle["contrast_delta"], oracle["bootstrap_95"], positive)
        clauses[mode] = gate.stop_clauses(pooled["pooled_disattenuated"], pooled["pooled_disattenuated_ci95"],
                                          oracle["contrast_delta"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode, table in cells.items():
        table.to_csv(args.output.parent / f"analysis_a_{mode}.csv", index=False)
    curves.to_csv(args.output.parent / "analysis_c_curves.csv", index=False)
    figure(curves, args.figure)
    result = {"pins": PINS, "inputs_sha256": {"exp038_results": sha(args.exp038_results),
                                              "exp038_overlay": sha(args.exp038_overlay),
                                              "base_manifest": sha(args.base / "exp037_manifest.json")},
              "population": {"train_paintings": int(train.painting.nunique()), "train_pairs": len(train),
                             "eligible_paintings_per_source": eligible, "paintings_with_all_three": all_three,
                             "targets": targets, "p2_paintings": int(gold.painting.nunique())},
              "analysis_a": a, "analysis_b": b, "analysis_c": c, "positive_slopes": int(positive),
              "decision_preregistered_shared_reference": decisions["shared"],
              "decision_split_reference_bias_check": decisions["split"],
              "decision_by_variant": decisions, "stop_clauses_by_variant": clauses,
              "seed": SEED, "runtime_seconds": time.time() - tick, "development_used": False}
    args.output.write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"decision_shared": decisions["shared"], "decision_split": decisions["split"],
                      "A_shared": a["shared"]["all"], "A_split": a["split"]["all"],
                      "A_split_exact": a["split_exact_variance_sensitivity"]["all"],
                      "A_shared_exact": a["shared_exact_variance_sensitivity"]["all"], "clauses": clauses,
                      "B_i": b["i_at_least_one_source"]["oracle_minus_C2"]["contrast_delta"],
                      "runtime": result["runtime_seconds"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
