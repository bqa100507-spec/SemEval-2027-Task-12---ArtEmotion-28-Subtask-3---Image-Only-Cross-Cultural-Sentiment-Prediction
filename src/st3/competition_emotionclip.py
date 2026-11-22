"""Matched P2 evaluation for the frozen EmotionCLIP branch."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .baseline import file_sha256
from .cli import ROOT, json_output
from .competition_challenger import frame_sha
from .competition_cultural_concepts import reference_rows
from .competition_dinov2_large import predict
from .competition_semantic_generation import CONFIG as SEMANTIC
from .competition_stack import compare
from .competition_track1 import load_p2, peak_memory_bytes
from .contrast_study import new_output
from .data import KEYS, require_same_pairs
from .e50_heads import read_json
from .metrics import painting_scores, score

CONFIG = ROOT / "configs/competition_emotionclip.json"
BRANCHES = ("clip_b", "emotion_scores", "emotion_embedding")


def evaluate(gold, predictions, track, config):
    require_same_pairs(gold, predictions)
    data = gold.merge(predictions, on=KEYS, validate="one_to_one").assign(y=lambda f: f.k / f.n)
    candidates = ("emotion_scores", "emotion_embedding", "blend")
    systems = ("fixed_ridge", "corrected_ridge", *BRANCHES, "blend")
    contrasts = [*((name, "corrected_ridge") for name in (*BRANCHES, "blend")),
                 ("emotion_embedding", "clip_b"), ("emotion_scores", "clip_b")]
    result, painting_rows = {}, []
    for scope, block in (("full", data), ("test_like", data[~data.language.isin(track["test_like_excluded"])])):
        def scores(rows):
            return {name: score(rows[KEYS + ["k", "n"]], rows[KEYS].assign(p_positive=rows[name]))
                    for name in systems}

        pooled = scores(block)
        folds = {int(f): scores(rows) for f, rows in block.groupby("fold")}
        paintings = pd.DataFrame({name: painting_scores(block, name).set_index("painting").contrast
                                  for name in systems})
        if paintings.empty or paintings.isna().any().any():
            raise ValueError("Systems must share the same Contrast-eligible paintings")
        painting_rows.append(paintings.reset_index().assign(scope=scope))
        paired = {}
        for lhs, rhs in contrasts:
            diff = paintings[lhs] - paintings[rhs]
            deltas = {fold: s[lhs]["cultural_contrast"] - s[rhs]["cultural_contrast"] for fold, s in folds.items()}
            top = min(int(np.ceil(.01 * len(diff))), int(diff.gt(0).sum()))
            paired[f"{lhs}_minus_{rhs}"] = compare(diff, track["decision"]) | {
                "loss_delta": pooled[lhs]["log_loss"] - pooled[rhs]["log_loss"],
                "fold_deltas": deltas, "fold_wins": sum(d > 1e-12 for d in deltas.values()),
                "drop_top_wins": float(diff.sort_values(ascending=False).iloc[top:].mean()),
            }
        eligible = block[block.painting.isin(paintings.index)]
        columns = ["y", "corrected_ridge", *BRANCHES]
        grouped = eligible.groupby("painting")
        centered = eligible[columns] - grouped[columns].transform("mean")
        weights = 1 / grouped.y.transform("size")
        residual = centered.y - centered.corrected_ridge
        correlations = {}
        for name in BRANCHES:
            branch = centered.y - centered[name]
            scale = np.sqrt(np.average(residual**2, weights=weights) * np.average(branch**2, weights=weights))
            correlations[name] = float(np.average(residual * branch, weights=weights) / scale) if scale else None
        result[scope] = {"pooled": pooled, "folds": folds, "paired": paired, "pairs": len(block),
                         "centered_error_correlation": correlations}

    qualified = []
    for name in candidates:
        a, b = (result[scope]["paired"][f"{name}_minus_corrected_ridge"] for scope in ("full", "test_like"))
        if (a["contrast_delta"] >= 0 and b["contrast_delta"] >= config["min_test_like_gain"]
                and b["bootstrap_95"][0] > 0 and b["fold_wins"] >= config["min_fold_wins"]
                and b["trimmed_contrast_delta"] > 0 and b["drop_top_wins"] > 0
                and max(a["loss_delta"], b["loss_delta"]) <= config["max_loss_regression"]):
            qualified.append(name)
    result["selected"] = max(qualified, key=lambda name: result["test_like"]["pooled"][name]["cultural_contrast"],
                             default="corrected_ridge")
    return result, pd.concat(painting_rows, ignore_index=True)


def run(inputs, descriptors, receipt, reference, destination, track, config):
    start = time.perf_counter()
    gold, features, _ = load_p2(inputs, track)
    descriptors = {"clip_b": features["clip"], **descriptors}
    if any(not frame.index.is_unique or set(frame.index) != set(gold.painting) for frame in descriptors.values()):
        raise ValueError("EmotionCLIP and CLIP-B must match the pinned P2 artwork keys")
    reference = reference_rows(gold, reference, read_json(SEMANTIC))
    fits, pieces = {}, []
    with threadpool_limits(limits=track["threads"]):
        for fold, held in gold.groupby("fold"):
            train = gold[gold.fold.ne(fold)]
            keys = held[KEYS]
            fit = {"train_sha256": frame_sha(train), "held_keys_sha256": frame_sha(keys), "branches": {}}
            part = keys.copy()
            for name, matrix in descriptors.items():
                predicted, stats = predict(train, keys, matrix, track["ridge"])
                part = part.merge(predicted.rename(columns={"dinov2_large": name}), on=KEYS, validate="one_to_one")
                fit["branches"][name] = stats
            print(f"Fold {fold}: {len(held)} held pairs predicted by all three heads", flush=True)
            fits[int(fold)] = fit
            pieces.append(part)
    output = reference.merge(pd.concat(pieces, ignore_index=True), on=KEYS, validate="one_to_one")
    share = config["blend_share"]
    output["blend"] = ((1 - share) * output.corrected_ridge + share * output.emotion_embedding).round(6)
    result, painted = evaluate(gold, output, track, config)
    result.update(fits=fits, features=receipt, config=config, config_sha256=file_sha256(CONFIG),
                  source_sha256=file_sha256(Path(__file__)), wall_seconds=time.perf_counter() - start,
                  peak_rss_bytes=peak_memory_bytes())
    with new_output(destination) as stage:
        output.merge(gold[KEYS + ["fold"]], on=KEYS, validate="one_to_one").to_csv(
            stage / "oof.csv", index=False, float_format="%.6f")
        painted.to_csv(stage / "painting_contrast.csv", index=False, float_format="%.17g")
        json_output(result, stage / "metrics.json")
        json_output({f.name: file_sha256(f) for f in stage.iterdir()}, stage / "seal.json")
    print(f"P2 selection: {result['selected']}", flush=True)
