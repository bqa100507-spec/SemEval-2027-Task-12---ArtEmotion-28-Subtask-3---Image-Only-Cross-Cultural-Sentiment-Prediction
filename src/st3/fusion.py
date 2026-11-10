"""Fixed probability and feature fusion for the existing Model C head."""

import numpy as np

from .data import KEYS, require_same_pairs
from .submission import validate_predictions
from .visual_study import REFERENCE

EARLY = "EarlyFusion"


def blend(clip, siglip, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("CLIP blend weight must lie in [0, 1]")
    clip_q = validate_predictions(clip, clip)
    siglip_q = validate_predictions(siglip, clip)
    matched = clip[KEYS].merge(siglip.assign(p_positive=siglip_q), on=KEYS, validate="one_to_one")
    return clip[KEYS].assign(p_positive=(alpha * clip_q + (1 - alpha) * matched["p_positive"].to_numpy()).round(6))


def combine_features(clip, siglip):
    names = sorted(clip)
    if not names or clip.keys() != siglip.keys():
        raise ValueError("Fusion requires identical, nonempty image coverage")
    left, right = np.stack([clip[n] for n in names]), np.stack([siglip[n] for n in names])
    for matrix, width in ((left, 512), (right, 768)):
        if matrix.shape != (len(names), width) or not np.allclose(np.linalg.norm(matrix, axis=1), 1., rtol=0., atol=3e-6):
            raise ValueError(f"Expected unit-normalized image features of width {width}")
    features = np.hstack((left, right)).astype(np.float32) / np.float32(np.sqrt(2.))
    return dict(zip(names, features))


def verify_reference(predictions, expected):
    require_same_pairs(expected, predictions)
    aligned = predictions[KEYS].merge(expected, on=KEYS, validate="one_to_one")
    if not np.array_equal(predictions["p_positive"].to_numpy(), aligned["p_positive"].to_numpy()):
        raise ValueError("Reference probabilities differ from the saved experiment")


def choose_fusion(summary, config, folds):
    baseline = summary.loc[REFERENCE]
    ceiling = float(baseline["log_loss_mean"] + config["log_loss_guardrail_bits"])
    alternatives = summary.drop(index=REFERENCE)
    eligible = alternatives[
        alternatives["pooled_contrast"].gt(baseline["pooled_contrast"])
        & alternatives["log_loss_mean"].le(ceiling) & alternatives["converged"].eq(folds)
        & alternatives["folds"].eq(folds) & alternatives["scored"].eq(folds) & alternatives["loss_scored"].eq(folds)]
    leader = winner = REFERENCE
    if len(eligible):
        eligible = eligible.sort_values("pooled_contrast", ascending=False, kind="stable")
        leader = eligible.index[0]
        tied = eligible[
            (eligible.loc[leader, "pooled_contrast"] - eligible["pooled_contrast"]).lt(config["contrast_tie_band"])
            & (eligible["log_loss_mean"] - eligible.loc[leader, "log_loss_mean"]).abs().le(
                config["negligible_log_loss_bits"])]
        winner = min(tied.index, key=config["complexity"].__getitem__)
    return {"model": winner, "pooled_leader": leader, "eligible_candidates": list(eligible.index),
            "maximum_mean_log_loss": ceiling}
