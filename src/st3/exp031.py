"""EXP-031 hierarchical supervision and complete-painting updates."""
import numpy as np
import torch
from torch.nn import functional as F


def training_priors(train, *, pseudocount=0.5):
    """Fixed fitting-only Beta-smoothed language priors in sorted order."""
    sums = train.groupby("language", sort=True)[["k", "n"]].sum()
    if (not 0 < pseudocount <= 1 or sums.empty or not np.isfinite(sums).all().all()
            or (sums.n <= 0).any() or ((sums.k < 0) | (sums.k > sums.n)).any()):
        raise ValueError("Invalid fitting-only vote totals or pseudocount")
    return list(sums.index), torch.tensor(np.log((sums.k + pseudocount) /
                                                (sums.n - sums.k + pseudocount)).to_numpy(),
                                          dtype=torch.float32)


def evidence_rank_loss(binary_logits, counts, *, min_z=1., z_ramp=1.5):
    """Confidence-shrunk margin loss, averaged over observed non-tied pairs."""
    if binary_logits.ndim != 1 or counts.shape != (len(binary_logits), 8):
        raise ValueError("Ranking logits/counts have incompatible shapes")
    votes = counts.sum(1).float()
    k = counts[:, :4].sum(1).float()
    p = k / votes
    a, b = k + .5, votes - k + .5
    variance = (a * b) / ((a + b).square() * (a + b + 1))
    left, right = torch.triu_indices(len(p), len(p), offset=1, device=p.device)
    delta = p[left] - p[right]
    z = delta.abs() / (variance[left] + variance[right]).sqrt()
    weight = ((z - min_z) / z_ramp).clamp(0, 1)
    margin = (binary_logits[left] - binary_logits[right]) * delta.sign()
    return (F.softplus(-margin) * weight).sum() / delta.ne(0).sum().clamp_min(1)


def binary_terms(binary, all_interactions, counts, config):
    positive = counts[:, :4].sum(1)
    negative = counts[:, 4:].sum(1)
    bce = (positive * F.softplus(-binary) + negative * F.softplus(binary)).sum()
    ranking = evidence_rank_loss(binary, counts, min_z=config["ranking_min_z"],
                                 z_ramp=config["ranking_z_ramp"])
    return bce, ranking, all_interactions.square().mean()


def conditional_ce(logits, counts):
    return -sum((counts[:, part] * F.log_softmax(logits[:, part], dim=1)).sum()
                for part in (slice(0, 4), slice(4, 8)))


def painting_batches(frame, *, seed, epoch, max_pairs):
    """Yield deterministic updates containing complete paintings, never split one."""
    if max_pairs < 1:
        raise ValueError("Invalid accumulation batch")
    groups = list(frame.groupby("painting", sort=True).indices.values())
    if epoch is not None:
        order = np.random.default_rng(seed + epoch).permutation(len(groups))
        groups = [groups[i] for i in order]
    pending, size = [], 0
    for indices in groups:
        if pending and size + len(indices) > max_pairs:
            yield pending
            pending, size = [], 0
        pending.append(frame.iloc[indices])
        size += len(indices)
    if pending:
        yield pending
