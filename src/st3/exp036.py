"""Caption-supervised visual adaptation for ST3 and Development selection."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import INPUT_COLUMNS, KEYS, require_same_pairs
from .metrics import score
from .submission import COLUMNS, pack, validate_archive, validate_predictions


class LoRA(nn.Module):
    def __init__(self, base, rank, scale):
        super().__init__()
        self.base = base
        self.left = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.right = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.left, a=math.sqrt(5))
        self.multiplier = scale / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.left), self.right) * self.multiplier


def adapt(vision, cfg):
    vision.requires_grad_(False)
    layers = vision.encoder.layers
    if len(layers) != cfg["layers"]:
        raise ValueError("Unexpected SigLIP layer count")
    last = cfg["adapter_layers"]
    if not 0 < last <= len(layers):
        raise ValueError("Invalid adapter tail length")
    names = []
    for i in range(len(layers) - last, len(layers)):
        for path in cfg["adapter_modules"]:
            parent, name = path.rsplit(".", 1)
            container = layers[i].get_submodule(parent)
            base = getattr(container, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Unexpected SigLIP adapter target {i}.{path}")
            setattr(container, name, LoRA(base, cfg["rank"], cfg["scale"]))
            names.append(f"encoder.layers.{i}.{path}")
    return names


class Candidate(nn.Module):
    def __init__(self, vision, languages, cfg, *, auxiliary=True):
        super().__init__()
        self.vision = vision
        self.targets = adapt(vision, cfg)
        width = cfg["hidden_size"]
        self.shared = nn.Linear(width, 1, bias=False)
        self.interaction = nn.Linear(width, languages)
        self.shared.weight.data.zero_()
        self.interaction.weight.data.zero_()
        self.interaction.bias.data.zero_()
        if auxiliary:
            hidden = cfg["caption_width"]
            self.caption = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))
            self.caption_language = nn.Embedding(languages, hidden)

    def features(self, pixels):
        return F.normalize(self.vision(pixel_values=pixels).pooler_output.float(), dim=-1)

    def logits(self, features, groups, languages):
        w = self.interaction.weight
        interactions = F.linear(features, w - w.mean(dim=0), self.interaction.bias)
        # Center the interaction across fitting languages to identify the shared effect.
        effects = interactions[groups, languages.clamp_min(0)]
        effects = torch.where(languages < 0, self.interaction.bias.mean(), effects)
        return self.shared(features).flatten()[groups] + effects

    def caption_vectors(self, features, groups, languages):
        projected = self.caption[0](features)[groups] + self.caption_language(languages)
        return self.caption[2](self.caption[1](projected))

    def predict(self, pixels, groups, languages):
        return self.logits(self.features(pixels), groups, languages).float().sigmoid()


def centered_targets(vectors, captions, groups, languages, epsilon):
    """Sequential caption-count-weighted language and painting centering."""
    raw = np.zeros_like(vectors, dtype=np.float32)
    keep = captions > 0
    if not keep.any():
        raise ValueError("The training scope contains no caption targets")
    raw[keep] = vectors[keep] / captions[keep, None]

    def subtract_mean(values, ids):
        result = values.copy()
        for i in np.unique(ids[keep]):
            rows = np.flatnonzero((ids == i) & keep)
            result[rows] -= np.average(values[rows], axis=0, weights=captions[rows])
        return result

    lang = subtract_mean(raw, languages)
    painting = subtract_mean(raw, groups)
    joint = subtract_mean(lang, groups)
    norms = np.linalg.norm(joint, axis=1)
    valid = keep & (norms > epsilon)
    joint[valid] /= norms[valid, None]
    joint[~valid] = 0
    summary = {"caption_pairs": int(keep.sum()), "target_pairs": int(valid.sum())}
    for key, value in (("raw", raw), ("language", lang), ("painting", painting)):
        summary[f"{key}_mean_norm"] = float(np.linalg.norm(value[keep], axis=1).mean())
    summary["joint_mean_norm"] = float(norms[keep].mean())
    return joint, valid, summary


def loss_parts(model, pixels, batch, vote_mass, caption_mass, epsilon):
    groups, languages, positive, counts, targets, weight = batch
    features = model.features(pixels)
    logits = model.logits(features, groups, languages).float()
    bce = (F.binary_cross_entropy_with_logits(logits, positive / counts, reduction="none") * counts).sum()
    vectors = model.caption_vectors(features, groups, languages).float()
    cosine = F.cosine_similarity(vectors, targets, dim=1, eps=epsilon)
    return bce / vote_mass, ((1 - cosine) * weight).sum() / caption_mass


def inference_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()
            if p.requires_grad and not name.startswith(("caption.", "caption_language."))}


def rounded(frame, dev):
    p = validate_predictions(frame, dev)
    aligned = dev[KEYS].merge(frame.assign(p_positive=p), on=KEYS, validate="one_to_one")
    return aligned.assign(p_positive=aligned.p_positive.round(6))


def choose_epoch(history):
    return max(history, key=lambda row: (row["metrics"]["cultural_contrast"],
                                         -row["metrics"]["log_loss"], -row["epoch"]))


def finish(dev, gold, standalone, ridge, output, cfg):
    if list(dev) != INPUT_COLUMNS:
        raise ValueError("Development inference expects only the official input schema")
    require_same_pairs(dev, gold)
    if not set(gold.columns) == {*KEYS, "k", "n"}:
        raise ValueError("Expected scoring-only Development counts")
    p, q = [rounded(values, dev) for values in (standalone, ridge)]
    reference = score(gold, q)
    rows, predictions = [], {}
    if not {0, 1} <= set(cfg["ensemble_grid"]):
        raise ValueError("The ensemble grid needs both endpoints")
    for alpha in cfg["ensemble_grid"]:
        values = ((1 - alpha) * q.p_positive + alpha * p.p_positive).round(6)
        combined = p.assign(p_positive=values)[COLUMNS]
        metrics = score(gold, combined)
        allowed = metrics["log_loss"] <= reference["log_loss"] + cfg["ensemble_max_log_loss_regression"]
        rows.append({"alpha": alpha, "eligible": allowed, "metrics": metrics})
        predictions[alpha] = combined
    best = max((row for row in rows if row["eligible"]), key=lambda row: (
        row["metrics"]["cultural_contrast"], -row["metrics"]["log_loss"], -row["alpha"]))
    for name, frame in (("standalone", p), ("ensemble", predictions[best["alpha"]])):
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        csv = folder / "predictions.csv"
        archive = output / f"exp036_{name}_submission.zip"
        frame.to_csv(csv, index=False, lineterminator="\n", float_format="%.6f")
        pack(csv, archive, dev)
        valid = validate_archive(archive, dev)
        if not np.array_equal(valid.p_positive.to_numpy(), frame.p_positive.to_numpy()):
            raise ValueError(f"Submission round trip changed {name} probabilities")
    return {"reference": reference, "standalone": score(gold, p), "ensemble": best, "grid": rows}
