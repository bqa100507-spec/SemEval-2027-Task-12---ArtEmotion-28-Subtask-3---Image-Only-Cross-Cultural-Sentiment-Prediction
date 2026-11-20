"""EXP-031 cultural ranking and vote-aware losses (research-only fold 3)."""
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .exp030 import EMOTIONS, count_votes, fold3, matched_reference, best_epoch, loss_sums, positive_probability


def training_priors(train, *, pseudocount=0.5):
    """Fixed train-only Beta-smoothed language priors, with explicit language order."""
    if not 0 < pseudocount <= 1:
        raise ValueError("Expected a positive, small prior pseudocount")
    if not {"language", "k", "n"} <= set(train.columns):
        raise ValueError("Training prior requires vote counts")
    sums = train.groupby("language", sort=True)[["k", "n"]].sum()
    if sums.empty or (sums["n"] <= 0).any() or (sums["k"] > sums["n"]).any():
        raise ValueError("Invalid fitting-only vote totals")
    probs = (sums["k"].to_numpy(dtype=float) + pseudocount) / (
        sums["n"].to_numpy(dtype=float) + 2 * pseudocount)
    logits = np.log(probs) - np.log1p(-probs)
    return list(sums.index), torch.tensor(logits, dtype=torch.float32)


def evidence_rank_loss(binary_logits, counts, *, min_z=1.0, z_ramp=1.5):
    """Within-painting sign ranking, downweighting differences unsupported by votes.

    Jeffreys posterior variance suppresses noisy small-n comparisons; only
    distinct positive-share observations provide supervision. One painting is
    one equally weighted ranking task, irrespective of its language count.
    """
    if binary_logits.ndim != 1 or counts.shape != (len(binary_logits), 8):
        raise ValueError("Ranking logits/counts have incompatible shapes")
    if len(binary_logits) < 2:
        return binary_logits.sum() * 0
    votes = counts.sum(1).float()
    k = counts[:, :4].sum(1).float()
    if (votes <= 0).any() or (k > votes).any():
        raise ValueError("Invalid vote totals")
    p = k / votes
    a, b = k + 0.5, votes - k + 0.5
    variance = (a * b) / ((a + b).square() * (a + b + 1))
    delta = p[:, None] - p[None, :]
    standard_error = torch.sqrt(variance[:, None] + variance[None, :]).clamp_min(1e-6)
    z = delta.abs() / standard_error
    confidence = ((z - min_z) / z_ramp).clamp(0, 1)
    upper = torch.triu(torch.ones_like(confidence, dtype=torch.bool), diagonal=1)
    weights = torch.where(upper & (delta != 0), confidence, 0.)
    margin = (binary_logits[:, None] - binary_logits[None, :]) * delta.sign()
    return (F.softplus(-margin) * weights).sum() / weights.sum().clamp_min(1e-8)


def grouped_objective(log_probs, binary_logits, interaction, counts, config):
    ce, bce = loss_sums(log_probs, counts)
    ranking = evidence_rank_loss(binary_logits, counts, min_z=config["ranking_min_z"],
                                 z_ramp=config["ranking_z_ramp"])
    # Centered per painting: do not penalize train-derived prior or shared art sentiment.
    centered = interaction - interaction.mean()
    regularizer = centered.square().mean()
    return ce, bce, ranking, regularizer


def painting_batches(frame, *, seed, epoch, max_pairs):
    """Yield deterministic updates containing complete paintings, never split one."""
    if max_pairs < 1:
        raise ValueError("Invalid accumulation batch")
    groups = frame.groupby("painting", sort=True).indices
    if epoch is None:
        names = sorted(groups)
    else:
        names = sorted(groups)
        rng = np.random.default_rng(seed + epoch)
        names = [names[i] for i in rng.permutation(len(names))]
    pending, size = [], 0
    for name in names:
        indices = np.asarray(groups[name], dtype=int)
        if pending and size + len(indices) > max_pairs:
            yield [frame.iloc[idx] for idx in pending]
            pending, size = [], 0
        pending.append(indices)
        size += len(indices)
    if pending:
        yield [frame.iloc[idx] for idx in pending]


def validate_source_groups(groups):
    for batch in groups:
        for frame in batch:
            if frame.painting.nunique() != 1 or frame.language.duplicated().any():
                raise ValueError("A painting was split or its language keys repeated")
