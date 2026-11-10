"""Painting batches, frozen LR readout, and adapter-only epoch selection."""

import math
import time

import numpy as np
import pandas as pd

from .ablation import predict_lr
from .baseline import design_matrix
from .clip_lora import adapter_parameters
from .e50_heads import fit_frozen_head
from .nonlinear import early_split, torch, vote_loss
from .residual_crossfit import disjoint

from torch import nn


class FrozenReadout(nn.Module):
    def __init__(self, model, languages, width=768):
        super().__init__()
        self.languages = tuple(languages)
        coef = model.coef_[0]
        split = width + len(languages)
        for name, value in (("visual", coef[:width]), ("offset", coef[width:split]),
                            ("interactions", coef[split:].reshape(len(languages), width)),
                            ("intercept", model.intercept_)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def forward(self, features, image, language):
        selected = features[image]
        index = language.clamp_min(0)
        interaction = self.offset[index] + (self.interactions[index] * selected).sum(dim=-1)
        return self.intercept + selected @ self.visual + torch.where(language >= 0, interaction, 0.)


def check_readout(model, readout, frame, vectors, device, tolerance):
    names = sorted(set(frame["image"]))
    features = torch.as_tensor(np.stack([vectors[name] for name in names]), device=device)
    image = torch.as_tensor(pd.Index(names).get_indexer(frame["image"]), device=device)
    language = torch.as_tensor(pd.Index(readout.languages).get_indexer(frame["language"]), device=device)
    with torch.inference_mode():
        actual = readout(features, image, language).cpu().numpy()
    expected = model.decision_function(design_matrix(frame["image"], frame["language"], vectors, readout.languages))
    delta = float(np.max(np.abs(expected - actual)))
    if delta > tolerance:
        raise ValueError(f"Frozen PyTorch readout differs from sklearn by {delta}")
    return delta


class PaintingRows:
    def __init__(self, frame, index, languages, device):
        self.groups = list(frame.groupby("painting", sort=True).indices.values())
        if not self.groups or frame.groupby("painting")["image"].nunique().gt(1).any():
            raise ValueError("Each painting batch needs one image per painting")
        self.image = np.array([index[frame["image"].iloc[group[0]]] for group in self.groups])
        self.language = pd.Index(languages).get_indexer(frame["language"])
        k, n = frame["k"].to_numpy(), frame["n"].to_numpy()
        if not np.isfinite([k, n]).all() or ((n <= 0) | (k < 0) | (k > n)).any():
            raise ValueError("Invalid original training votes")
        self.votes = np.array([n[group].sum() for group in self.groups])
        self.targets = (k / n).astype(np.float32)
        self.counts = n.astype(np.float32)
        self.device = device

    def batch(self, paintings):
        groups = [self.groups[i] for i in paintings]
        rows = np.concatenate(groups)
        pair_image = np.repeat(np.arange(len(groups)), [len(group) for group in groups])
        arrays = self.image[paintings], pair_image, self.language[rows], self.targets[rows], self.counts[rows]
        return tuple(torch.as_tensor(values, device=self.device) for values in arrays)


def microbatches(order, size):
    return [order[i:i + size] for i in range(0, len(order), size)]


def step(tail, readout, bank, rows, batches, optimizer, config, *, gradients=False):
    """Accumulate summed losses then divide by votes across the entire step."""
    optimizer.zero_grad(set_to_none=True)
    votes = int(sum(rows.votes[batch].sum() for batch in batches))
    total = bank.new_zeros(())
    for batch in batches:
        image, pair_image, language, targets, counts = rows.batch(batch)
        with torch.autocast(device_type=bank.device.type, dtype=torch.bfloat16,
                            enabled=config["training_precision"] == "bf16" and bank.device.type == "cuda"):
            features = tail(bank[image])
        loss = vote_loss(readout(features, pair_image, language), targets, counts)
        (loss / votes).backward()
        total += loss.detach()
    parameters = adapter_parameters(tail)
    norms = {}
    if gradients:
        norms = {name: float(param.grad.norm()) for name, param in parameters.items()}
    nn.utils.clip_grad_norm_(parameters.values(), config["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    return total, norms


def optimizer_for(tail, config):
    return torch.optim.AdamW(adapter_parameters(tail).values(), lr=config["learning_rate"],
                             weight_decay=config["weight_decay"])


def gradient_control(tail, readout, bank, rows, config):
    batches = microbatches(np.arange(len(rows.groups)), config["image_batch_size"])
    accumulation = config["accumulation"]
    traces, states = [], []
    for _ in range(2):
        tail.reset(config["seed"])
        optimizer = optimizer_for(tail, config)
        steps = []
        for start in (0, accumulation):
            loss, norms = step(tail, readout, bank, rows, batches[start:start + accumulation],
                               optimizer, config, gradients=True)
            steps.append({"weighted_bce_sum": float(loss), "gradient_norms": norms})
        traces.append(steps)
        states.append({name: value.detach().clone() for name, value in adapter_parameters(tail).items()})
    if traces[0] != traces[1] or not all(torch.equal(value, states[1][name]) for name, value in states[0].items()):
        raise ValueError("Identical seeded pilot optimization steps did not reproduce")
    initial, second = (entry["gradient_norms"] for entry in traces[0])
    if (not all(value == 0. if name.endswith(".a") else value > 0. for name, value in initial.items())
            or not all(math.isfinite(value) and value > 0. for value in second.values())):
        raise ValueError("Unexpected LoRA gradient flow in the pilot")
    tail.reset(config["seed"])
    return traces[0]


@torch.inference_mode()
def validation_loss(tail, readout, bank, rows, batch_size):
    total = bank.new_zeros(())
    for batch in microbatches(np.arange(len(rows.groups)), batch_size):
        image, pair_image, language, target, counts = rows.batch(batch)
        total += vote_loss(readout(tail(bank[image]), pair_image, language), target, counts)
    return float(total / rows.votes.sum())


def train_phase(tail, readout, bank, rows, config, epochs, validation=None):
    tail.reset(config["seed"])
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    optimizer = optimizer_for(tail, config)
    curves, best, selected = [], math.inf, 0
    for epoch in range(1, epochs + 1):
        tick = time.perf_counter()
        total = bank.new_zeros(())
        batches = microbatches(rng.permutation(len(rows.groups)), config["image_batch_size"])
        accumulation = config["accumulation"]
        for start in range(0, len(batches), accumulation):
            loss, _ = step(tail, readout, bank, rows, batches[start:start + accumulation], optimizer, config)
            total += loss
        train_bce = float(total / rows.votes.sum())
        val_bce = validation_loss(tail, readout, bank, validation, config["image_batch_size"]) if validation else None
        if not math.isfinite(train_bce) or (val_bce is not None and not math.isfinite(val_bce)):
            raise ValueError("Nonfinite vote-weighted BCE in LoRA training")
        curves.append({"epoch": epoch, "train_bce": train_bce, "validation_bce": val_bce,
                       "seconds": time.perf_counter() - tick, "image_batches": len(batches)})
        print(f"  epoch {epoch}: train BCE {train_bce:.6f}, validation {val_bce}, "
              f"{curves[-1]['seconds']:.1f}s", flush=True)
        if val_bce is not None and val_bce < best:
            best, selected = val_bce, epoch
        if validation is not None and epoch - selected >= config["patience"]:
            break
    return curves, selected if validation is not None else epochs


@torch.inference_mode()
def predict_vectors(tail, frame, bank, index, batch_size):
    names = sorted(set(frame["image"]))
    vectors = []
    for batch in microbatches(np.array([index[name] for name in names]), batch_size):
        vectors.append(tail(bank[batch]).cpu().numpy())
    joined = np.concatenate(vectors)
    if not np.isfinite(joined).all() or not np.allclose(np.linalg.norm(joined, axis=1), 1., rtol=0., atol=2e-6):
        raise ValueError("Invalid normalized CLIP-L output")
    return dict(zip(names, joined))


def adapt_partition(tail, bank, index, partition, held, original, config, fold, device, pilot):
    inner_fit, validation = early_split(partition, held, config, fold)
    disjoint(inner_fit, validation, held)
    audit = {name: sorted(frame["painting"].unique()) for name, frame in
             (("inner_fit", inner_fit), ("validation", validation), ("held", held))}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    inner = fit_frozen_head(inner_fit, original, sorted(inner_fit["language"].unique()), config["seed"])
    readout = FrozenReadout(inner, sorted(inner_fit["language"].unique())).to(device)
    inner_delta = check_readout(inner, readout, validation, original, device, config["head_logit_atol"])
    rows = PaintingRows(inner_fit, index, readout.languages, device)
    stopping = PaintingRows(validation, index, readout.languages, device)
    control = gradient_control(tail, readout, bank, rows, config) if pilot else []
    curve, selected = train_phase(tail, readout, bank, rows, config, config["max_epochs"], stopping)
    languages = sorted(partition["language"].unique())
    model = fit_frozen_head(partition, original, languages, config["seed"])
    full = FrozenReadout(model, languages).to(device)
    full_delta = check_readout(model, full, held, original, device, config["head_logit_atol"])
    refit, _ = train_phase(tail, full, bank, PaintingRows(partition, index, languages, device), config, selected)
    tick = time.perf_counter()
    vectors = predict_vectors(tail, held, bank, index, config["image_batch_size"])
    prediction = predict_lr(model, held, vectors, languages)
    info = {"audit": audit, "selected_epoch": selected, "stopping_epochs": len(curve),
            "inner_readout_max_error": inner_delta, "full_readout_max_error": full_delta,
            "gradient_control": control, "predict_seconds": time.perf_counter() - tick,
            "fit_seconds": tick - start, "languages": languages}
    if device.type == "cuda":
        info["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        info["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    curves = [{"phase": phase, **row} for phase, trace in (("selection", curve), ("refit", refit)) for row in trace]
    return prediction, model, curves, info
