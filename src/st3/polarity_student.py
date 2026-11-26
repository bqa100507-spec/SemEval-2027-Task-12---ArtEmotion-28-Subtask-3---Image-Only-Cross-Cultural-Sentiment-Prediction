"""Matched polarity-query students and fixed, painting-balanced cosine losses."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import data
from .caption_student import CaptionStudent, make_rows


class PolarityStudent(CaptionStudent):
    def __init__(self, languages, config):
        super().__init__(languages, config)
        width = config["hidden_width"]
        self.polarity = nn.Embedding(2, width)
        nn.init.normal_(self.polarity.weight, std=config["polarity_embedding_std"])
        self.auxiliary = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                       nn.Linear(width, config["target_dimensions"]))

    def forward(self, images, language, *, auxiliary=False):
        hidden = self.visual(images)
        known = language.ge(0)
        index = language.clamp_min(0)
        embedding = self.language_visual(index) * known[:, None]
        offset = self.language_bias(index).squeeze(-1)
        logits = self.shared(hidden).squeeze(-1) + torch.where(
            known, offset + (embedding * hidden).sum(-1) / self.scale, 0.)
        predicted = None
        if auxiliary:
            inputs = torch.cat([hidden[:, None].expand(-1, 2, -1), embedding[:, None].expand(-1, 2, -1),
                                self.polarity.weight[None].expand(len(hidden), -1, -1)], dim=-1)
            predicted = F.normalize(self.auxiliary(inputs), dim=-1)
        return logits, predicted


def seed_polarity_model(languages, seed, config, device):
    torch.manual_seed(seed)
    model = PolarityStudent(languages, config).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    return model, generator, optimizer


def polarity_rows(frame, features, languages, device, targets=None, weights=None):
    rows = make_rows(frame, features, languages, device)
    if targets is None:
        return rows
    example = next(iter(targets.values()), (np.zeros((2, 64), dtype=np.float32), np.zeros(2, dtype=bool)))
    empty = (np.zeros_like(example[0]), np.zeros(2, dtype=bool))
    keys = list(frame[data.KEYS].itertuples(index=False, name=None))
    items = [targets.get(key, empty) for key in keys]
    rows["text"] = torch.as_tensor(np.stack([item[0] for item in items]), device=device)
    rows["text_mask"] = torch.as_tensor(np.stack([item[1] for item in items]), device=device)
    rows["text_weights"] = torch.tensor([weights.get(key, 0.) for key in keys], dtype=torch.float32, device=device)
    return rows


def balanced_channels(losses, mask, weights):
    channels = mask.sum(dim=1)
    pair_loss = (losses * mask).sum(dim=1) / channels.clamp_min(1)
    effective = weights * channels.gt(0)
    return (pair_loss * effective).sum() / effective.sum().clamp_min(1e-12)


def cosine_loss(predicted, targets, mask, weights):
    cosine = (predicted * targets).sum(dim=-1).clamp(-1., 1.)
    return balanced_channels(1. - cosine, mask, weights)


def polarity_epoch(model, rows, rng, optimizer, config, text_weight):
    model.train()
    total = rows["votes"].new_zeros(())
    auxiliary_total = total.clone()
    auxiliary_weights = total.clone()
    order = torch.randperm(len(rows["image"]), device=rows["image"].device, generator=rng)
    for batch in order.split(config["batch_size"]):
        optimizer.zero_grad(set_to_none=True)
        logits, prediction = model(rows["features"][rows["image"][batch]], rows["language"][batch], auxiliary=text_weight > 0)
        votes = rows["votes"][batch]
        nll = (F.binary_cross_entropy_with_logits(logits, rows["labels"][batch], reduction="none") * votes).sum()
        loss = nll / votes.sum()
        if text_weight > 0:
            mask, weights = rows["text_mask"][batch], rows["text_weights"][batch]
            auxiliary = cosine_loss(prediction, rows["text"][batch], mask, weights)
            loss = loss + text_weight * auxiliary
            effective = (weights * mask.any(dim=1)).sum()
            auxiliary_total += (auxiliary * effective).detach()
            auxiliary_weights += effective
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        total += nll.detach()
    return float(total / rows["votes"].sum()), float(auxiliary_total / auxiliary_weights.clamp_min(1e-12))


def gradient_diagnostic(model, rows, batch, coefficient):
    model.train()
    logits, predicted = model(rows["features"][rows["image"][batch]], rows["language"][batch], auxiliary=True)
    votes = rows["votes"][batch]
    bce = (F.binary_cross_entropy_with_logits(logits, rows["labels"][batch], reduction="none") * votes).sum() / votes.sum()
    targets, mask, weights = rows["text"][batch], rows["text_mask"][batch], rows["text_weights"][batch]
    auxiliary = cosine_loss(predicted, targets, mask, weights)
    parameters = tuple(model.visual[0].parameters())
    bce_vector = torch.cat([value.flatten() for value in torch.autograd.grad(bce, parameters, retain_graph=True)])
    auxiliary_vector = torch.cat([value.flatten() for value in torch.autograd.grad(auxiliary, parameters, retain_graph=True)])
    polarity_grad = torch.autograd.grad(auxiliary, model.polarity.weight)[0]
    if not all(bool(torch.isfinite(value).all()) for value in (bce, auxiliary, bce_vector, auxiliary_vector, polarity_grad)):
        raise ValueError("Numerically invalid polarity gradient diagnostic")
    bce_norm, auxiliary_norm = bce_vector.norm(), auxiliary_vector.norm()
    branch_norms = polarity_grad.norm(dim=1)
    if float(bce_norm) == 0 or float(auxiliary_norm) == 0 or not bool(branch_norms.gt(0).all()):
        raise ValueError("The representative pilot must exercise both polarity branches and shared gradients")
    mse = balanced_channels((predicted - targets).square().mean(dim=-1), mask, weights)
    return {"bce_nats": float(bce.detach()), "cosine_auxiliary": float(auxiliary.detach()),
            "scaled_auxiliary": float(coefficient * auxiliary.detach()),
            "shared_layer": "visual[0] weight and bias", "bce_gradient_norm": float(bce_norm),
            "auxiliary_gradient_norm": float(auxiliary_norm),
            "scaled_auxiliary_gradient_norm": float(coefficient * auxiliary_norm),
            "unscaled_gradient_ratio": float(auxiliary_norm / bce_norm),
            "scaled_gradient_ratio": float(coefficient * auxiliary_norm / bce_norm),
            "gradient_cosine": float(F.cosine_similarity(bce_vector[None], auxiliary_vector[None])),
            "polarity_gradient_norms": branch_norms.tolist(), "finite": True,
            "normalized_mse": float(mse.detach()), "cosine_to_mse_ratio": float((auxiliary / mse).detach()),
            "available_channels": mask.sum(dim=0).tolist()}
