"""Matched EXP-017 students, adapted from archived EXP-013 at cd321f1."""

import math
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

from . import data
from .caption_data import require_isolation


def concatenate_features(banks):
    names = sorted(banks[0])
    widths = (512, 768, 768)
    blocks = []
    for bank, width in zip(banks, widths):
        if set(bank) != set(names):
            raise ValueError("Visual feature coverage differs between encoders")
        values = np.stack([bank[name] for name in names])
        if (values.shape != (len(names), width) or values.dtype != np.float32
                or not np.allclose(np.linalg.norm(values, axis=1), 1., atol=3e-6, rtol=0.)):
            raise ValueError("Invalid pinned visual features")
        blocks.append(values)
    return dict(zip(names, np.concatenate(blocks, axis=1) / np.float32(math.sqrt(3))))


class CaptionStudent(nn.Module):
    def __init__(self, languages, config):
        super().__init__()
        self.languages = tuple(languages)
        width = config["hidden_width"]
        self.visual = nn.Sequential(nn.Linear(2048, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(config["dropout"]))
        self.shared = nn.Linear(width, 1)
        self.language_bias = nn.Embedding(len(languages), 1)
        self.language_visual = nn.Embedding(len(languages), width)
        nn.init.zeros_(self.language_bias.weight)
        nn.init.zeros_(self.language_visual.weight)
        self.scale = math.sqrt(width)
        # Allocate the same branch even for BCE-only controls to match RNG state.
        self.auxiliary = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                       nn.Linear(width, config["target_dimensions"]))

    def forward(self, images, language, *, auxiliary=False):
        h = self.visual(images)
        known = language.ge(0)
        indices = language.clamp_min(0)
        embedding = self.language_visual(indices) * known[:, None]
        offset = self.language_bias(indices).squeeze(-1)
        logits = self.shared(h).squeeze(-1) + torch.where(known, offset + (embedding * h).sum(-1) / self.scale, 0.)
        target = F.normalize(self.auxiliary(torch.cat([h, embedding], dim=1)), dim=1) if auxiliary else None
        return logits, target


def make_rows(frame, features, languages, device, targets=None, weights=None, *, training=True):
    images = sorted(frame.image.unique())
    rows = {"features": torch.as_tensor(np.stack([features[key] for key in images]), device=device),
            "image": torch.as_tensor(pd.Index(images).get_indexer(frame.image), device=device),
            "language": torch.as_tensor(pd.Index(languages).get_indexer(frame.language), device=device)}
    if training:
        votes = frame.n.to_numpy(dtype=np.float32)
        labels = frame.k.to_numpy(dtype=np.float32) / votes
        if not np.isfinite([votes, labels]).all() or (votes < 5).any() or ((labels < 0) | (labels > 1)).any():
            raise ValueError("Invalid original eligible vote counts")
        rows.update({"votes": torch.as_tensor(votes, device=device), "labels": torch.as_tensor(labels, device=device)})
    if targets is not None:
        keys = list(frame[data.KEYS].itertuples(index=False, name=None))
        width = len(next(iter(targets.values()), np.zeros(64)))
        rows["text"] = torch.as_tensor(np.stack([targets.get(key, np.zeros(width, dtype=np.float32)) for key in keys]), device=device)
        rows["text_weights"] = torch.tensor([weights.get(key, 0.) for key in keys], dtype=torch.float32, device=device)
    return rows


def text_loss(predicted, target, weights):
    dimensions = (predicted - target).square().mean(dim=1)
    return (dimensions * weights).sum() / weights.sum().clamp_min(1e-12)


def seed_model(languages, seed, config, device):
    torch.manual_seed(seed)
    model = CaptionStudent(languages, config).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    return model, generator, optimizer


def train_epoch(model, rows, rng, optimizer, config, text_weight):
    model.train()
    bce_total = rows["votes"].new_zeros(())
    auxiliary_total = bce_total.clone()
    auxiliary_weight = bce_total.clone()
    order = torch.randperm(len(rows["image"]), device=rows["image"].device, generator=rng)
    for batch in order.split(config["batch_size"]):
        optimizer.zero_grad(set_to_none=True)
        logits, predicted = model(rows["features"][rows["image"][batch]], rows["language"][batch], auxiliary=text_weight > 0)
        votes = rows["votes"][batch]
        nll = (F.binary_cross_entropy_with_logits(logits, rows["labels"][batch], reduction="none") * votes).sum()
        loss = nll / votes.sum()
        if text_weight > 0:
            weights = rows["text_weights"][batch]
            auxiliary = text_loss(predicted, rows["text"][batch], weights)
            loss = loss + text_weight * auxiliary
            auxiliary_total += (auxiliary * weights.sum()).detach()
            auxiliary_weight += weights.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        bce_total += nll.detach()
    return float(bce_total / rows["votes"].sum()), float(auxiliary_total / auxiliary_weight.clamp_min(1e-12))


@torch.inference_mode()
def validation_bce(model, rows, batch_size):
    model.eval()
    total = rows["votes"].new_zeros(())
    for batch in torch.arange(len(rows["image"]), device=rows["image"].device).split(batch_size):
        logits, _ = model(rows["features"][rows["image"][batch]], rows["language"][batch])
        total += (F.binary_cross_entropy_with_logits(logits, rows["labels"][batch], reduction="none") * rows["votes"][batch]).sum()
    return float(total / rows["votes"].sum())


def fit_student(partition, inner_fit, stopping, features, prepared, seed, config, device, *,
                model_factory=seed_model, epoch_function=train_epoch, training_rows=make_rows, progress=None):
    require_isolation(inner_fit, stopping)
    data.require_same_pairs(partition, pd.concat([inner_fit, stopping]))
    if not partition.split.eq("train").all():
        raise ValueError("Student fitting requires original training rows")
    start = time.perf_counter()
    weight = config["text_weight"] if prepared else 0.
    inner_targets, inner_weights = prepared["inner"][:2] if prepared else (None, None)
    languages = sorted(inner_fit.language.unique())
    training = training_rows(inner_fit, features, languages, device, inner_targets, inner_weights)
    validation = make_rows(stopping, features, languages, device)
    model, rng, optimizer = model_factory(languages, seed, config, device)
    best, selected = math.inf, 0
    curves = []
    for epoch in range(1, config["max_epochs"] + 1):
        if progress is not None:
            progress("selection", epoch)
        nll, auxiliary = epoch_function(model, training, rng, optimizer, config, weight)
        value = validation_bce(model, validation, config["batch_size"])
        if not np.isfinite([nll, auxiliary, value]).all():
            raise ValueError("Nonfinite student training or validation loss")
        curves.append({"phase": "selection", "epoch": epoch, "train_bce": nll, "train_auxiliary": auxiliary,
                       "validation_bce": value})
        if value < best:
            best, selected = value, epoch
        if epoch - selected >= config["patience"]:
            break
    selection_epochs = epoch
    del model, optimizer, training, validation
    languages = sorted(partition.language.unique())
    full_targets, full_weights = prepared["full"][:2] if prepared else (None, None)
    training = training_rows(partition, features, languages, device, full_targets, full_weights)
    model, rng, optimizer = model_factory(languages, seed, config, device)
    for epoch in range(1, selected + 1):
        if progress is not None:
            progress("refit", epoch)
        nll, auxiliary = epoch_function(model, training, rng, optimizer, config, weight)
        if not np.isfinite([nll, auxiliary]).all():
            raise ValueError("Nonfinite student refit loss")
        curves.append({"phase": "refit", "epoch": epoch, "train_bce": nll, "train_auxiliary": auxiliary,
                       "validation_bce": None})
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Nonfinite student parameters")
    return model, curves, {"selected_epoch": selected, "selection_epochs": selection_epochs,
                           "best_validation_bce": best, "fit_seconds": time.perf_counter() - start,
                           "parameters": sum(p.numel() for p in model.parameters()), "known_languages": languages}


@torch.inference_mode()
def predict_student(model, inputs, features, baseline, device, batch_size):
    # Intentionally admit only artwork lookup keys and target language.
    inputs = inputs[[*data.KEYS, "image"]]
    rows = make_rows(inputs, features, model.languages, device, training=False)
    model.eval()
    parts = []
    for batch in torch.arange(len(inputs), device=device).split(batch_size):
        logits, _ = model(rows["features"][rows["image"][batch]], rows["language"][batch])
        parts.append(logits.sigmoid().cpu().numpy())
    values = np.concatenate(parts).astype(np.float64)
    reference = inputs[data.KEYS].merge(baseline, on=data.KEYS, validate="one_to_one")["p_positive"].to_numpy()
    values[~inputs.language.isin(model.languages)] = reference[~inputs.language.isin(model.languages)]
    if not np.isfinite(values).all():
        raise ValueError("Student returned nonfinite probabilities")
    return values
