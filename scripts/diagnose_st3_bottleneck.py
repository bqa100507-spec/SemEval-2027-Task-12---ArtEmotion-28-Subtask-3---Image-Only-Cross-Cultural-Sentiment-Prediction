#!/usr/bin/env python3
"""Refit the missing nested additive control and audit existing P2 predictions."""

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.cli import json_output
from st3.competition_cultural_concepts import reference_rows
from st3.competition_challenger import component_predictions, frame_sha
from st3.competition_ridge import make_panel, objective
from st3.competition_stack import ENCODERS, aligned_values, blend, compare
from st3.competition_track1 import load_p2, load_protocol, verify_archive
from st3.contrast_study import new_output
from st3.data import KEYS
from st3.e50_heads import read_json
from st3.metrics import painting_scores, score


def additive(train, keys, features, track):
    panel = make_panel(train)
    rule = track["ridge"]
    heads, receipts = {}, {}
    for name in ENCODERS:
        x = np.ascontiguousarray(features[name].loc[list(panel.paintings)], dtype=float)
        loss, unpack, size = objective(panel, x, 1 / (rule["reference_c"] * panel.n.sum()))
        initial = np.zeros(size)
        initial[0] = logit((panel.k.sum() + .5) / (panel.n.sum() + 1))
        fit = minimize(loss, initial, jac=True, method="L-BFGS-B", options={
            "maxiter": rule["maxiter"], "ftol": rule["ftol"], "gtol": rule["gtol"], "maxls": 40})
        gradient = float(np.linalg.norm(fit.jac, ord=np.inf))
        if not fit.success or not np.isfinite(gradient) or gradient > rule["max_final_gradient"]:
            raise ValueError(f"Additive fit did not converge for {name}: {fit.message}")
        heads[name] = unpack(fit.x)
        receipts[name] = {"iterations": int(fit.nit), "max_gradient": gradient}
    q = component_predictions(heads, keys, features)[list(ENCODERS)].to_numpy()
    return blend(q, track["fixed_weights"], fixed=True), receipts


def diagnostics(frame, track):
    out = {}
    names = ["prior", "additive", "interaction", "fixed_ridge", "corrected_ridge"]
    for scope, d in (("full", frame), ("test_like", frame[~frame.language.isin(track["test_like_excluded"])])):
        scores = {a: score(d[KEYS + ["k", "n"]], d[KEYS].assign(p_positive=d[a])) for a in names}
        r = {a: painting_scores(d, a).set_index("painting").contrast for a in names}
        paired = {f"{a}_minus_{b}": compare((r[a] - r[b]).dropna(), track["decision"])
                  for a, b in [("interaction", "additive"), ("corrected_ridge", "fixed_ridge")]}
        scored = d[d.painting.isin(r["corrected_ridge"].dropna().index)]
        parts = []
        for name, g in scored.groupby("painting"):
            y, q, a = [g[k].to_numpy() for k in ("p", "corrected_ridge", "additive")]
            rho = r["corrected_ridge"].at[name]
            ix, jx = np.triu_indices(len(g), 1)
            gap, predicted = y[ix] - y[jx], q[ix] - q[jx]
            good = ((np.sign(gap) == np.sign(predicted)) + 0.5 * (predicted == 0)).astype(float)
            centered = lambda x: x - x.mean()
            interaction = centered(q - a)
            parts.append({"painting": name, "style": g["style"].iloc[0], "rho": rho,
                          "order": good[gap != 0].mean(),
                          "order_small": float(np.mean(good[(abs(gap) > 0) & (abs(gap) <= .200001)]))
                          if ((abs(gap) > 0) & (abs(gap) <= .200001)).any() else np.nan,
                          "order_large": float(np.mean(good[abs(gap) >= .399999])) if (abs(gap) >= .399999).any() else np.nan,
                          "mean_bias_mse": (q.mean() - y.mean()) ** 2, "spread_mse": (q.std() - y.std()) ** 2,
                          "ordering_mse": 2 * q.std() * y.std() * (1 - rho),
                          "centered_mse": np.mean(centered(y - q) ** 2), "additive_centered_mse": np.mean(centered(y - a) ** 2),
                          "increment_residual_alignment": np.mean(centered(y - a) * interaction),
                          "vote_noise": g.noise.mean(), "sample_variance": y.var(ddof=1)})
        detail = pd.DataFrame(parts)
        quality = detail.assign(noise_dominated=detail.vote_noise >= detail.sample_variance).groupby("noise_dominated")
        style = detail.groupby("style").agg(paintings=("rho", "size"), contrast=("rho", "mean"), noise=("vote_noise", "mean"))
        rows = d.assign(bias=d.corrected_ridge - d.p, centered_error=lambda f: (f.bias - f.groupby("painting").bias.transform("mean")) ** 2)
        languages = rows.groupby("language").agg(pairs=("n", "size"), mean_votes=("n", "mean"), y=("p", "mean"),
                                                 noise=("noise", "mean"), bias=("bias", "mean"), centered_mse=("centered_error", "mean"))
        all_p = d.groupby("painting").agg(languages=("p", "size"), variance=("p", "var"), noise=("noise", "mean"))
        all_p = all_p[all_p.languages.ge(3)]
        out[scope] = {"scores": scores, "paired": paired,
                      "fold_interaction_gain": {int(i): float((r["interaction"]-r["additive"]).reindex(g.painting.unique()).mean())
                                                for i, g in d.groupby("fold")},
                      "order_and_error": detail.select_dtypes("number").mean().to_dict(),
                      "noise_groups": quality.agg(paintings=("rho", "size"), contrast=("rho", "mean"), order=("order", "mean")).to_dict("index"),
                      "style_slices": style[style.paintings.ge(60)].to_dict("index"),
                      "languages": languages.to_dict("index"), "noise_fraction": float(all_p.noise.sum() / all_p.variance.sum()),
                      "observed_variance": float(all_p.variance.mean()), "sampling_variance": float(all_p.noise.mean())}
    return out


def main():
    start = perf_counter()
    runs = ROOT / "runs"
    inputs = runs / "exp023_budget_delivery/release/restore_check"
    track = read_json(ROOT / "configs/competition_track1.json")
    base = read_json(ROOT / "configs/competition_semantic_vlm.json")
    gold, features, e50 = load_p2(inputs, track)
    archive = runs / "exp032/p2_verified"
    refs = reference_rows(gold, runs / "competition_neighborhood_residual/results", base)
    refit, receipts = [], {}
    with threadpool_limits(limits=track["threads"]):
        verify_archive(archive, gold, features, e50, track)
        for fold, held in gold.groupby("fold"):
            train = gold.loc[gold.fold.ne(fold)]
            assert not set(train.painting) & set(held.painting)
            counts = train.groupby("language")[["k", "n"]].sum()
            prior = held.language.map((counts.k + .5) / (counts.n + 1)).fillna((train.k.sum() + .5) / (train.n.sum() + 1))
            q, opts = additive(train, held[KEYS], features, track)
            receipts[int(fold)] = {"fit_sha256": frame_sha(train), "held_keys_sha256": frame_sha(held[KEYS]), "fits": opts}
            refit.append(held[KEYS].assign(prior=prior, additive=q))
            print(f"Fold {fold} additive fits converged", flush=True)
    interaction = aligned_values(gold, pd.read_csv(archive / "oof_no_graph.csv"), ["p_positive"]).ravel()
    refs = gold[KEYS].merge(refs, on=KEYS, validate="one_to_one")
    np.testing.assert_array_equal(interaction, refs.fixed_ridge)
    style = load_protocol(inputs, track).set_index("painting").image.str.split("/").str[0]
    frame = gold.merge(pd.concat(refit), on=KEYS, validate="one_to_one").merge(refs, on=KEYS, validate="one_to_one")
    frame = frame.assign(interaction=interaction, p=lambda d: d.k/d.n, noise=lambda d: d.p*(1-d.p)/(d.n-1), style=lambda d: d.painting.map(style))
    metrics = diagnostics(frame, track) | {"fits": receipts, "seconds": perf_counter() - start}
    with new_output(runs / "st3_bottleneck_diagnosis") as stage:
        frame.drop(columns=["k", "n", "p", "noise", "style"]).to_csv(stage / "oof.csv", index=False, float_format="%.17g")
        json_output(metrics, stage / "metrics.json")


if __name__ == "__main__":
    main()
