"""The fixed EXP-013 nonlinear head and painting-grouped training loop."""

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Deterministic CUDA matrix products require this before Torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

from . import data
from .residual_crossfit import disjoint

WIDTHS = (512, 768, 768)
INPUT_WIDTH = sum(WIDTHS)


def concatenate_features(clip, siglip, large):
    names = sorted(clip)
    if not names or clip.keys() != siglip.keys() or clip.keys() != large.keys():
        raise ValueError("The three visual caches must cover the same nonempty image set")
    blocks = []
    for bank, width in zip((clip, siglip, large), WIDTHS):
        block = np.stack([bank[name] for name in names])
        if (block.shape != (len(names), width) or block.dtype != np.float32
                or not np.allclose(np.linalg.norm(block, axis=1), 1., rtol=0., atol=3e-6)):
            raise ValueError(f"Expected unit-normalized FP32 features of width {width}")
        blocks.append(block)
    return dict(zip(names, np.concatenate(blocks, axis=1) / np.float32(math.sqrt(3))))


class NonlinearHead(nn.Module):
    def __init__(self, languages, width, dropout):
        super().__init__()
        self.languages = tuple(languages)
        if not self.languages:
            raise ValueError("At least one training language is required")
        self.visual = nn.Sequential(nn.Linear(INPUT_WIDTH, width), nn.LayerNorm(width), nn.GELU(),
                                    nn.Dropout(dropout))
        self.shared = nn.Linear(width, 1)
        self.language_bias = nn.Embedding(len(self.languages), 1)
        self.language_visual = nn.Embedding(len(self.languages), width)
        nn.init.zeros_(self.language_bias.weight)
        nn.init.zeros_(self.language_visual.weight)
        self.scale = math.sqrt(width)

    def forward(self, images, language):
        h = self.visual(images)
        known = language.ge(0)
        index = language.clamp_min(0)
        offset = self.language_bias(index).squeeze(-1)
        interaction = (self.language_visual(index) * h).sum(dim=-1) / self.scale
        return self.shared(h).squeeze(-1) + torch.where(known, offset + interaction, 0.)


@dataclass
class Rows:
    features: torch.Tensor
    image: torch.Tensor
    language: torch.Tensor
    targets: torch.Tensor | None
    votes: torch.Tensor | None

    def logits(self, model, indices):
        return model(self.features[self.image[indices]], self.language[indices])


def make_rows(frame, features, languages, device, *, training=False):
    images = sorted(set(frame["image"]))
    values = np.stack([features[image] for image in images])
    image = pd.Index(images).get_indexer(frame["image"])
    language = pd.Index(languages).get_indexer(frame["language"])
    targets = votes = None
    if training:
        k, n = frame["k"].to_numpy(dtype=np.float32), frame["n"].to_numpy(dtype=np.float32)
        if not np.isfinite([k, n]).all() or ((n <= 0) | (k < 0) | (k > n)).any():
            raise ValueError("Invalid original training vote counts")
        targets, votes = (torch.as_tensor(array, device=device) for array in (k / n, n))
    return Rows(*(torch.as_tensor(array, device=device) for array in (values, image, language)), targets, votes)


def vote_loss(logits, targets, votes):
    """Weighted BCE sum. Divide by vote count for optimization and metrics."""
    return (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * votes).sum()


def early_split(partition, forbidden, config, fold):
    if not partition["split"].eq("train").all():
        raise ValueError("Epoch selection requires original training paintings")
    splitter = GroupShuffleSplit(n_splits=1, test_size=config["early_stop_fraction"],
                                  random_state=config["early_stop_seed"] + fold)
    fit, validation = next(splitter.split(partition, groups=partition["painting"]))
    inner_fit, inner_validation = partition.iloc[fit], partition.iloc[validation]
    disjoint(inner_fit, inner_validation, forbidden)
    return inner_fit, inner_validation


def configure_device(requested):
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return torch.device(device)


def make_model(languages, seed, config, device):
    torch.manual_seed(seed)
    model = NonlinearHead(languages, config["hidden_width"], config["dropout"]).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    return model, generator, optimizer


def train_epoch(model, rows, generator, optimizer, config):
    model.train()
    total = rows.votes.new_zeros(())
    order = torch.randperm(len(rows.image), device=rows.image.device, generator=generator)
    for batch in order.split(config["batch_size"]):
        optimizer.zero_grad(set_to_none=True)
        weights = rows.votes[batch]
        loss = vote_loss(rows.logits(model, batch), rows.targets[batch], weights)
        (loss / weights.sum()).backward()
        nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        total += loss.detach()
    return float(total / rows.votes.sum())


@torch.inference_mode()
def evaluate_bce(model, rows, batch_size):
    model.eval()
    total = rows.votes.new_zeros(())
    for batch in torch.arange(len(rows.image), device=rows.image.device).split(batch_size):
        total += vote_loss(rows.logits(model, batch), rows.targets[batch], rows.votes[batch])
    return float(total / rows.votes.sum())


def fit_seed(partition, inner_fit, inner_validation, features, seed, config, device):
    """Choose the epoch on an internal split and reinitialize before the full refit."""
    for frame in (partition, inner_fit, inner_validation):
        if frame.empty or not frame["split"].eq("train").all():
            raise ValueError("Nonlinear fitting requires nonempty original training rows")
    disjoint(inner_fit, inner_validation)
    data.require_same_pairs(partition, pd.concat([inner_fit, inner_validation]))
    start = time.perf_counter()
    inner_languages = sorted(inner_fit["language"].unique())
    training = make_rows(inner_fit, features, inner_languages, device, training=True)
    validation = make_rows(inner_validation, features, inner_languages, device, training=True)
    model, rng, optimizer = make_model(inner_languages, seed, config, device)
    curves = []
    best, epoch_selected = math.inf, 0
    for epoch in range(1, config["max_epochs"] + 1):
        train_loss = train_epoch(model, training, rng, optimizer, config)
        val_loss = evaluate_bce(model, validation, config["batch_size"])
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise ValueError(f"Nonfinite epoch-selection loss for seed {seed}, epoch {epoch}")
        curves.append({"phase": "selection", "epoch": epoch, "train_bce": train_loss, "validation_bce": val_loss})
        if val_loss < best:
            best, epoch_selected = val_loss, epoch
        if epoch - epoch_selected >= config["patience"]:
            break
    selection_seconds = time.perf_counter() - start
    selection_epochs = epoch
    del model, optimizer, training, validation

    languages = sorted(partition["language"].unique())
    full = make_rows(partition, features, languages, device, training=True)
    model, rng, optimizer = make_model(languages, seed, config, device)
    for epoch in range(1, epoch_selected + 1):
        train_loss = train_epoch(model, full, rng, optimizer, config)
        if not math.isfinite(train_loss):
            raise ValueError(f"Nonfinite refit loss for seed {seed}, epoch {epoch}")
        curves.append({"phase": "refit", "epoch": epoch, "train_bce": train_loss, "validation_bce": None})
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError(f"Nonfinite refit parameters for seed {seed}")
    return model, curves, {"selected_epoch": epoch_selected, "best_validation_bce": best,
                           "selection_epochs": selection_epochs, "selection_seconds": selection_seconds,
                           "fit_seconds": time.perf_counter() - start,
                           "parameters": sum(param.numel() for param in model.parameters())}


@torch.inference_mode()
def predict_seed(model, frame, features, device, batch_size):
    rows = make_rows(frame, features, model.languages, device)
    model.eval()
    parts = [torch.sigmoid(rows.logits(model, batch))
             for batch in torch.arange(len(rows.image), device=device).split(batch_size)]
    probabilities = torch.cat(parts).cpu().numpy().astype(np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonlinear model produced nonfinite predictions")
    return probabilities


def train_partition(partition, held, features, config, fold, device):
    """Fit every seed before using its held-out image and language inputs."""
    inner_fit, inner_validation = early_split(partition, held, config, fold)
    audit = {"fit_paintings": sorted(inner_fit["painting"].unique()),
             "validation_paintings": sorted(inner_validation["painting"].unique()),
             "held_paintings": sorted(held["painting"].unique()),
             "fit_pairs": len(inner_fit), "validation_pairs": len(inner_validation), "held_pairs": len(held)}
    probabilities, runs, curves = [], [], []
    inputs = held[[*data.KEYS, "image"]]
    for seed in config["seeds"]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, trace, info = fit_seed(partition, inner_fit, inner_validation, features, seed, config, device)
        tick = time.perf_counter()
        probabilities.append(predict_seed(model, inputs, features, device, config["batch_size"]))
        info["predict_seconds"] = time.perf_counter() - tick
        info["known_languages"] = list(model.languages)
        info["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        info["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        runs.append({"fold": fold, "seed": seed, **info})
        curves.extend({"fold": fold, "seed": seed, **row} for row in trace)
        print(f"EXP-013 fold {fold}, seed {seed}: epoch {info['selected_epoch']}/{info['selection_epochs']}, "
              f"fit {info['fit_seconds']:.2f}s", flush=True)
        del model
    return np.stack(probabilities), runs, curves, audit
