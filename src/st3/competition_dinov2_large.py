"""Matched P2 complementarity evaluation for frozen DINOv2-L."""

import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .baseline import file_sha256
from .cli import ROOT, json_output
from .competition_challenger import component_predictions, frame_sha
from .competition_cultural_concepts import reference_rows
from .competition_ridge import fit_head, make_panel
from .competition_semantic_generation import CONFIG as SEMANTIC, TRACK1
from .competition_stack import compare
from .competition_track1 import load_p2, peak_memory_bytes
from .contrast_study import new_output
from .data import KEYS, require_same_pairs
from .e50_heads import read_json
from .frozen_features import cls_features, extract_frozen
from .metrics import painting_scores, score

CONFIG = ROOT / "configs/competition_dinov2_large.json"
SYSTEMS = ("corrected_ridge", "dinov2_large", "blend")


def extract(root, scope, cache, checkpoint, rule, device):
    def load(path):
        from transformers import AutoModel

        return AutoModel.from_pretrained(path, use_safetensors=True, attn_implementation="eager",
                                         low_cpu_mem_usage=True, local_files_only=True)

    images, receipt = extract_frozen(root, scope.image, cache, rule | {"local_checkpoint": checkpoint}, device,
                                     prefix="dinov2-large", load_model=load,
                                     encode=lambda m, inputs, width: cls_features(m(**inputs).last_hidden_state, width))
    return pd.DataFrame([images[name] for name in scope.image], index=scope.painting), receipt


def predict(fit, keys, descriptors, rule):
    if list(keys.columns) != KEYS or set(fit.painting) & set(keys.painting):
        raise ValueError("Only disjoint painting/language query keys are allowed")
    panel = make_panel(fit)
    matrix = np.ascontiguousarray(descriptors.loc[list(panel.paintings)], dtype=float)
    head, receipt = fit_head(panel, matrix, rule)
    frame = component_predictions({"dinov2_large": head}, keys, {"dinov2_large": descriptors})
    return frame, receipt


def evaluate(gold, predictions, track, config):
    require_same_pairs(gold, predictions)
    frame = gold.merge(predictions, on=KEYS, validate="one_to_one").assign(p=lambda f: f.k / f.n)
    report, painting_outputs = {}, []
    for scope, part in (("full", frame), ("test_like", frame.loc[~frame.language.isin(track["test_like_excluded"])])):
        def measure(data, name):
            return score(data[KEYS + ["k", "n"]], data[KEYS].assign(p_positive=data[name]))

        scores = {name: measure(part, name) for name in SYSTEMS}
        folds = {int(i): {n: measure(g, n) for n in SYSTEMS} for i, g in part.groupby("fold")}
        paints = pd.DataFrame({n: painting_scores(part, n).set_index("painting").contrast for n in SYSTEMS}).dropna()
        painting_outputs.append(paints.reset_index().assign(scope=scope))
        changes = {}
        for name in SYSTEMS[1:]:
            diff = paints[name] - paints.corrected_ridge
            fold_delta = {i: s[name]["cultural_contrast"] - s["corrected_ridge"]["cultural_contrast"] for i, s in folds.items()}
            top = min(int(np.ceil(.01 * len(diff))), int(diff.gt(0).sum()))
            changes[name] = compare(diff, track["decision"]) | {
                "log_loss_delta": scores[name]["log_loss"] - scores["corrected_ridge"]["log_loss"],
                "fold_deltas": fold_delta, "fold_wins": sum(d > 1e-12 for d in fold_delta.values()),
                "drop_top_1pct_wins_delta": float(diff.sort_values(ascending=False).iloc[top:].mean())}
        selected = part[part.painting.isin(paints.index)]
        by = selected.groupby("painting")
        columns = ["p", *SYSTEMS]
        centered = selected[columns] - by[columns].transform("mean")
        w = 1 / by.p.transform("size")
        residual, alternative = centered.p - centered.corrected_ridge, centered.p - centered.dinov2_large
        rms = np.sqrt(np.average(residual**2, weights=w) * np.average(alternative**2, weights=w))
        report[scope] = {"pooled": scores, "folds": folds, "paired": changes, "pairs": len(part),
                         "centered_residual_correlation": float(np.average(residual * alternative, weights=w) / rms) if rms else None}
    qualified = []
    for name in SYSTEMS[1:]:
        a, b = (report[s]["paired"][name] for s in ("full", "test_like"))
        if (a["contrast_delta"] >= 0 and b["contrast_delta"] >= config["min_test_like_gain"]
                and b["bootstrap_95"][0] > 0 and b["fold_wins"] >= config["min_test_like_fold_wins"]
                and b["trimmed_contrast_delta"] > 0 and b["drop_top_1pct_wins_delta"] > 0
                and max(a["log_loss_delta"], b["log_loss_delta"]) <= config["max_log_loss_delta"]):
            qualified.append(name)
    report["selected"] = max(qualified, key=lambda a: report["test_like"]["pooled"][a]["cultural_contrast"],
                             default="corrected_ridge")
    return report, pd.concat(painting_outputs, ignore_index=True)


def run(inputs, descriptors, receipt, reference, destination):
    start = time.perf_counter()
    track, config = read_json(TRACK1), read_json(CONFIG)
    gold, _, _ = load_p2(inputs, track)
    if set(descriptors.index) != set(gold.painting):
        raise ValueError("DINOv2-L cache differs from the pinned P2 painting population")
    incumbent = reference_rows(gold, reference, read_json(SEMANTIC))
    pieces, fits = [], {}
    with threadpool_limits(limits=track["threads"]):
        for i, held in gold.groupby("fold"):
            stage = time.perf_counter()
            train = gold.loc[gold.fold.ne(i)]
            part, fits[int(i)] = predict(train, held[KEYS], descriptors, track["ridge"])
            fits[int(i)].update(train_sha256=frame_sha(train), held_keys_sha256=frame_sha(held[KEYS]),
                                seconds=time.perf_counter() - stage)
            pieces.append(part)
            print(f"Fold {i}: {len(held)} held pairs scored", flush=True)
    out = incumbent.merge(pd.concat(pieces, ignore_index=True), on=KEYS, validate="one_to_one").drop(columns="fixed_ridge")
    w = config["dino_share"]
    out["blend"] = ((1 - w) * out.corrected_ridge + w * out.dinov2_large).round(6)
    report, paints = evaluate(gold, out, track, config)
    report |= {"fits": fits, "feature_receipt": receipt, "wall_seconds": time.perf_counter() - start,
               "rss_bytes": peak_memory_bytes(), "source_sha256": file_sha256(ROOT / "src/st3/competition_dinov2_large.py"),
               "config_sha256": file_sha256(CONFIG)}
    with new_output(destination) as stage:
        out.merge(gold[KEYS + ["fold"]], on=KEYS, validate="one_to_one").to_csv(stage / "oof.csv", index=False, float_format="%.6f")
        paints.to_csv(stage / "painting_contrast.csv", index=False, float_format="%.17g")
        json_output(report, stage / "metrics.json")
        json_output({p.name: file_sha256(p) for p in stage.iterdir()}, stage / "seal.json")
    print(f"P2 selection: {report['selected']}", flush=True)
