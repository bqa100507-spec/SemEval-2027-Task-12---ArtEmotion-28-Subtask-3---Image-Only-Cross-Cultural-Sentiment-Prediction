"""EXP-018 optimizer steps with observational telemetry and an early refit snapshot."""

import hashlib
import time

import torch
from torch import nn
from torch.nn import functional as F

from .polarity_student import cosine_loss, seed_polarity_model


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TrainingRecorder:
    def __init__(self, directory, budget, early_epoch):
        self.directory, self.budget, self.early_epoch = directory, budget, early_epoch
        self.phase, self.epoch = None, 0
        self.records = []

    def progress(self, phase, epoch):
        self.phase, self.epoch = phase, epoch
        self.budget.check(phase, epoch)

    def epoch_function(self, model, rows, rng, optimizer, config, text_weight):
        start = time.perf_counter()
        model.train()
        total = rows["votes"].new_zeros(())
        auxiliary_total, auxiliary_weights = total.clone(), total.clone()
        order = torch.randperm(len(rows["image"]), device=rows["image"].device, generator=rng)
        norms, clipped, update_norm = [], 0, None
        for number, batch in enumerate(order.split(config["batch_size"])):
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
            norm = nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            norms.append(float(norm))
            clipped += int(norm > config["gradient_clip"])
            before = [p.detach().clone() for p in model.parameters()] if number == 0 else None
            optimizer.step()
            if before is not None:
                update_norm = float(torch.stack([(p.detach()-old).norm() for p, old in zip(model.parameters(), before)]).norm())
            total += nll.detach()
        self.records.append({"phase": self.phase, "epoch": self.epoch, "batches": len(norms),
            "clipped_batches": clipped, "clipping_frequency": clipped/len(norms),
            "mean_pre_clip_norm": sum(norms)/len(norms), "max_pre_clip_norm": max(norms),
            "mean_post_clip_norm": sum(min(x, config["gradient_clip"]) for x in norms)/len(norms),
            "first_batch_parameter_update_norm": update_norm, "training_epoch_seconds": time.perf_counter()-start})
        if self.phase == "refit" and self.epoch == self.early_epoch:
            torch.save({"languages": list(model.languages), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                       self.directory / "early.pt")
        return float(total / rows["votes"].sum()), float(auxiliary_total / auxiliary_weights.clamp_min(1e-12))


def matched_factory(languages, seed, config, device):
    reference, _, _ = seed_polarity_model(languages, seed, {**config, "text_weight": .1}, device)
    expected = state_hash(reference)
    del reference
    model, rng, optimizer = seed_polarity_model(languages, seed, config, device)
    if state_hash(model) != expected:
        raise ValueError("C2 initialization differs from EXP-018 C1")
    return model, rng, optimizer
