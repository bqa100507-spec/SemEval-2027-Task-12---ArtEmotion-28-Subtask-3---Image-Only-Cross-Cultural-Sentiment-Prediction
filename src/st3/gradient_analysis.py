"""Training-only gradient decomposition; frozen-feature inputs are not encoders."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from .polarity_student import cosine_loss

BLOCKS = (("clip_b", 0, 512), ("siglip2", 512, 1280), ("clip_l", 1280, 2048))


def deterministic_batches(length, batch_size, fold, seed, count=3):
    if length < 1 or count < 1:
        raise ValueError("Diagnostics require nonempty training rows")
    order = torch.randperm(length, generator=torch.Generator().manual_seed(seed + fold))
    chunks = list(order.split(batch_size))
    return [chunks[i % len(chunks)] for i in range(count)]


def decompose(bce, text, coefficient, clip_factor=1.):
    """Opposing fraction is negative parallel length / auxiliary length."""
    bce, text = bce.detach().double().flatten(), text.detach().double().flatten()
    if bce.shape != text.shape or not bool(torch.isfinite(torch.cat([bce, text])).all()):
        raise ValueError("Invalid gradient vectors")
    bn, tn = float(bce.norm()), float(text.norm())
    dot = float(bce @ text)
    cosine = max(-1., min(1., dot / (bn * tn))) if bn and tn else None
    combined = float((bce + coefficient * text).norm())
    return {"bce_norm": bn, "text_norm": tn, "scaled_text_norm": coefficient * tn,
            "scaled_ratio": coefficient * tn / bn if bn else None,
            "cosine": cosine, "combined_norm": combined,
            "orthogonal_fraction": math.sqrt(max(0., 1. - cosine**2)) if cosine is not None else None,
            "opposing_fraction": max(0., -cosine) if cosine is not None else None,
            "zero_bce": bn == 0., "zero_text": tn == 0.,
            "post_clip_bce_norm": clip_factor * bn,
            "post_clip_scaled_text_norm": clip_factor * coefficient * tn,
            "post_clip_combined_norm": clip_factor * combined}


def diagnostic(model, rows, batch, coefficient, clip_limit):
    """One eval-mode forward for both losses, preserving RNG and parameter grads."""
    device = rows["features"].device
    mode = model.training
    captured = []
    hook = model.visual.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            model.eval()
            batch = batch.to(device)
            z = rows["features"][rows["image"][batch]].detach().clone().requires_grad_(True)
            if z.shape[1] != 2048:
                raise ValueError("Expected frozen feature blocks 512 + 768 + 768")
            logits, predicted = model(z, rows["language"][batch], auxiliary=True)
            hidden = captured[0]
            votes = rows["votes"][batch]
            bce = (F.binary_cross_entropy_with_logits(logits, rows["labels"][batch], reduction="none") * votes).sum() / votes.sum()
            text = cosine_loss(predicted, rows["text"][batch], rows["text_mask"][batch], rows["text_weights"][batch])
            params = tuple(model.parameters())
            objects = (*params, hidden, z)
            bg = torch.autograd.grad(bce, objects, retain_graph=True, allow_unused=True)
            tg = torch.autograd.grad(text, objects, allow_unused=True)
            b = [torch.zeros_like(p) if g is None else g for p, g in zip(objects, bg)]
            t = [torch.zeros_like(p) if g is None else g for p, g in zip(objects, tg)]
            global_norm = float(torch.stack([(x + coefficient*y).norm() for x, y in zip(b[:len(params)], t[:len(params)])]).norm())
            if not np.isfinite([float(bce.detach()), float(text.detach()), global_norm]).all():
                raise ValueError("Nonfinite gradient diagnostic")
            factor = min(1., clip_limit / (global_norm + 1e-6))
            names = [name for name, _ in model.named_parameters()]
            groups = {"first_visual": [i for i, name in enumerate(names) if name.startswith("visual.0.")],
                      "language_interaction": [i for i, name in enumerate(names) if name.startswith(("language_visual.", "language_bias."))]}
            result = {}
            for name, indices in groups.items():
                result[name] = decompose(torch.cat([b[i].flatten() for i in indices]),
                                         torch.cat([t[i].flatten() for i in indices]), coefficient, factor)
                result[name]["bce_unused_parameters"] = [names[i] for i in indices if bg[i] is None]
                result[name]["text_unused_parameters"] = [names[i] for i in indices if tg[i] is None]
            result["shared_representation"] = decompose(b[-2], t[-2], coefficient)
            for name, lo, hi in BLOCKS:
                result[name] = decompose(b[-1][:, lo:hi], t[-1][:, lo:hi], coefficient)
                result[name]["dimensions"] = hi-lo
                result[name]["text_rms_per_coordinate"] = result[name]["text_norm"] / math.sqrt(z.shape[0]*(hi-lo))
            # Parameter clipping does not directly clip leaf/activation gradients.
            for name in ("shared_representation", *(x[0] for x in BLOCKS)):
                for key in list(result[name]):
                    if key.startswith("post_clip_"):
                        del result[name][key]
            return {"bce": float(bce.detach()), "text_loss": float(text.detach()),
                    "coefficient": coefficient, "scaled_loss": coefficient * float(text.detach()),
                    "global_combined_parameter_norm": global_norm, "clip_factor": factor,
                    "clipped": factor < 1., "finite": True, "groups": result,
                    "mode": "eval", "feature_gradient_scope": "dL/dZ only; no pretrained encoder graph"}
    finally:
        hook.remove()
        model.train(mode)
