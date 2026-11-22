"""A neutral Qwen3 hidden state and a bounded cultural log-odds correction."""

import torch
from torch import nn

from .competition_direct_qwen import LowRankUpdate, count_bce, last_valid


class ResidualHead(nn.Module):
    def __init__(self, hidden, languages, config):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config["seed"])
            self.layers = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, config["head_width"]),
                                        nn.GELU(), nn.Linear(config["head_width"], languages))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        self.limit = config["residual_limit"]

    def forward(self, hidden):
        return self.limit * self.layers(hidden.float()).tanh()


def add_merger_lora(backbone, config):
    """Adapt the final image merger projection. DeepStack mergers stay frozen."""
    merger = backbone.visual.merger
    base = merger.linear_fc2
    if not isinstance(base, nn.Linear) or isinstance(base, LowRankUpdate):
        raise TypeError(f"Unsupported Qwen merger projection: {type(base).__name__}")
    adapter = LowRankUpdate(base, config["adapter_rank"], config["adapter_alpha"],
                            torch.Generator().manual_seed(config["seed"] + 1))
    merger.linear_fc2 = adapter
    if {id(p) for p in backbone.parameters() if p.requires_grad} != {id(adapter.left), id(adapter.right)}:
        raise ValueError("Only visual.merger.linear_fc2 LoRA may be trainable")
    return adapter


def image_hidden(backbone, encoded):
    # The native image placeholder, multimodal RoPE and DeepStack path stay intact.
    output = backbone(**encoded, use_cache=False).last_hidden_state
    return last_valid(output, encoded["attention_mask"])


def select_languages(deltas, language_ids):
    return torch.where(language_ids >= 0, deltas[language_ids.clamp_min(0)], 0.)


def odds_update(prior, delta):
    """Equivalent to sigmoid(logit(prior) + delta), with exact identity at zero."""
    shift = delta.to(prior.dtype).expm1()
    return prior + prior * (1 - prior) * shift / (1 + prior * shift)


def objective(prior, delta, positive, votes, mean_votes, config):
    """One complete painting. Absent language pairs are not part of the vectors."""
    # Clamp only the BCE logit to keep loss and derivatives finite at exact endpoints.
    logits = torch.logit(prior.clamp(1e-6, 1 - 1e-6)) + delta
    bce = count_bce(logits, positive, votes, vote_mass=mean_votes)
    q = odds_update(prior, delta)
    target = positive / votes
    error = q - target
    centered = (error - error.mean()).square().mean()
    penalty = delta.square().mean()
    loss = bce + config["center_weight"] * centered + config["residual_weight"] * penalty
    return loss, (bce.detach(), centered.detach(), penalty.detach())


def gradients(parameters):
    parts = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(parts).norm()) if parts else 0.


def update(head, adapter, optimizer, scaler, feature, prior, ids, k, n, mean_votes, config):
    optimizer.zero_grad(set_to_none=True)
    delta = select_languages(head(feature).squeeze(0), ids)
    loss, terms = objective(prior, delta, k, n, mean_votes, config)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite residual training objective")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    adapter_params = (adapter.left, adapter.right) if adapter is not None else ()
    heads, lora = gradients(head.parameters()), gradients(adapter_params)
    norm = nn.utils.clip_grad_norm_([*head.parameters(), *adapter_params], config["clip_norm"],
                                    error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return dict(zip(("bce", "centered_mse", "residual_penalty"), map(float, terms))) | {
        "loss": float(loss.detach()), "head_gradient": heads, "adapter_gradient": lora,
        "preclip": float(norm), "clipped": bool(norm > config["clip_norm"]),
        "language_variance": float(odds_update(prior, delta.detach()).var(unbiased=False)),
        "delta_rms": float(delta.detach().square().mean().sqrt()),
    }
