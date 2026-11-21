"""Native Qwen3 image-and-language hidden states with matched frozen/QLoRA readouts."""

import math
import re
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VisionTokens:
    merged: tuple
    deepstack: tuple
    grid: torch.Tensor


def language_prompt(processor, language, template):
    if not isinstance(language, str) or not re.fullmatch(r"[A-Za-z][A-Za-z -]{0,47}", language):
        raise ValueError("Expected a target language name from the ST3 input")
    message = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": template.format(language=language)}]}]
    return processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)


def encode_text(processor, languages, image_token_counts, template, device):
    texts = []
    image_token = processor.image_token
    for language, count in zip(languages, image_token_counts, strict=True):
        text = language_prompt(processor, language, template)
        if text.count(image_token) != 1 or count < 1:
            raise ValueError("The chat input must contain exactly one image placeholder")
        texts.append(text.replace(image_token, image_token * count))
    return processor.tokenizer(texts, padding=True, padding_side="left", return_tensors="pt").to(device)


def last_valid(hidden, mask):
    if hidden.shape[:2] != mask.shape or mask.ndim != 2 or not mask.bool().any(1).all():
        raise ValueError("A nonempty attention mask must accompany each hidden sequence")
    positions = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~mask.bool(), -1).max(1).values
    return hidden[torch.arange(len(mask), device=mask.device), last]


class LowRankUpdate(nn.Module):
    def __init__(self, base, rank, alpha, generator):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.left = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.right = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.left, a=math.sqrt(5), generator=generator)
        self.left.data = self.left.data.to(base.weight.device)
        self.right.data = self.right.data.to(base.weight.device)

    def forward(self, states):
        base = self.base(states)
        # FP32 parameters and update math, including under an enclosing autocast.
        with torch.autocast(device_type=states.device.type, enabled=False):
            delta = F.linear(F.linear(states.float(), self.left), self.right)
        return base + (delta * self.scale).to(base.dtype)

    def squared_delta(self):
        # Exact ||B A||_F^2 via two rank x rank Gram matrices.
        return ((self.right.T @ self.right) * (self.left @ self.left.T)).sum() * self.scale ** 2


class CulturalQwen(nn.Module):
    def __init__(self, backbone, config, *, adapted):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.adapters = {}
        decoder = backbone.language_model
        if len(decoder.layers) != config["decoder_layers"]:
            raise ValueError("Unexpected Qwen text layer count")
        if adapted:
            start = len(decoder.layers) - config["lora_tail_layers"]
            targets = config["lora_targets"]
            if not 0 <= start < len(decoder.layers) or tuple(targets) != ("q_proj", "v_proj"):
                raise ValueError("LoRA is restricted to Q and V of the pinned decoder tail")
            generator = torch.Generator().manual_seed(config["seed"] + 1)
            for i in range(start, len(decoder.layers)):
                attention = decoder.layers[i].self_attn
                for target in targets:
                    name = f"language_model.layers.{i}.self_attn.{target}"
                    adapter = LowRankUpdate(getattr(attention, target), config["lora_rank"], config["lora_alpha"], generator)
                    setattr(attention, target, adapter)
                    self.adapters[name] = adapter
        width = decoder.config.hidden_size
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config["seed"])
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1)).to(backbone.get_input_embeddings().weight.device)
        permitted = {id(p) for p in self.head.parameters()} | {id(p) for p in self.adapter_parameters()}
        actual = {id(p) for p in self.parameters() if p.requires_grad}
        if actual != permitted:
            raise RuntimeError("Only the sentiment head and named LoRA matrices may be trainable")

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def adapter_parameters(self):
        return [p for adapter in self.adapters.values() for p in (adapter.left, adapter.right)]

    @property
    def device(self):
        return next(self.head.parameters()).device

    @torch.no_grad()
    def encode_vision(self, image_inputs):
        grid = image_inputs["image_grid_thw"].to(self.device)
        pixels = image_inputs["pixel_values"].to(self.device)
        merged, taps = self.backbone.get_image_features(pixels, grid)
        sizes = [len(tokens) for tokens in merged]
        return VisionTokens(tuple(merged), tuple(tuple(t.split(sizes)) for t in taps), grid)

    def features(self, vision, rows, encoded):
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        if len(rows) != len(ids) or any(not 0 <= i < len(vision.merged) for i in rows):
            raise ValueError("Every language row must identify a valid artwork in the vision batch")
        embeds = self.backbone.get_input_embeddings()(ids)
        counts = (ids == self.backbone.config.image_token_id).sum(1)
        if counts.tolist() != [len(vision.merged[i]) for i in rows]:
            raise ValueError("Image placeholder count differs from the matching per-row vision grid")
        merged = torch.cat([vision.merged[i] for i in rows]).to(embeds.dtype)
        image_mask, _ = self.backbone.get_placeholder_mask(ids, embeds, image_features=merged)
        embeds = embeds.masked_scatter(image_mask, merged)
        positions, _ = self.backbone.get_rope_index(ids, vision.grid[list(rows)], attention_mask=mask)
        hidden = self.backbone.language_model(
            inputs_embeds=embeds, attention_mask=mask, position_ids=positions, use_cache=False,
            visual_pos_masks=image_mask[..., 0],
            deepstack_visual_embeds=[torch.cat([tap[i] for i in rows]) for tap in vision.deepstack],
        ).last_hidden_state
        return last_valid(hidden, mask)

    def logits(self, features):
        return self.head(features.float()).squeeze(-1)

    def forward(self, vision, rows, encoded):
        return self.logits(self.features(vision, rows, encoded))

    def penalty(self):
        return torch.stack([adapter.squared_delta() for adapter in self.adapters.values()]).mean()


def count_bce(logits, positive, total, *, vote_mass=None):
    """Mean negative binomial log-likelihood per observed vote, without constants."""
    if logits.ndim != 1 or positive.shape != logits.shape or total.shape != logits.shape:
        raise ValueError("Vote and prediction vectors must align one to one")
    if not ((total >= 5) & (positive >= 0) & (positive <= total) & (positive == positive.floor())
            & (total == total.floor()) & total.isfinite()).all():
        raise ValueError("Require at least five integer votes and valid positive counts")
    raw = F.softplus(-logits.float()) * positive + F.softplus(logits.float()) * (total - positive)
    return raw.sum() / (total.sum() if vote_mass is None else vote_mass)


def optimizer_for(model, config):
    groups = [{"params": list(model.head.parameters()), "lr": config["head_lr"]}]
    if model.adapters:
        groups.append({"params": model.adapter_parameters(), "lr": config["adapter_lr"]})
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"], foreach=False)


def train_step(model, optimizer, scaler, config, minibatches, vote_mass):
    """Accumulate exact vote-normalized gradients across an artwork's language chunks."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    objective = torch.zeros((), device=model.device)
    for forward, k, n in minibatches:
        loss = count_bce(forward(), k, n, vote_mass=vote_mass)
        scaler.scale(loss).backward()
        objective += loss.detach()
    anchor = model.penalty() if model.adapters else objective.new_zeros(())
    if model.adapters:
        scaler.scale(anchor * config["lora_update_penalty"]).backward()
    scaler.unscale_(optimizer)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(params, config["clip_grad_norm"], error_if_nonfinite=True)
    head_grad = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.head.parameters()]))
    adapters = model.adapter_parameters()
    adapter_grad = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in adapters])) if adapters else norm.new_zeros(())
    scaler.step(optimizer)
    scaler.update()
    return {"bce": float(objective), "anchor": float(anchor.detach()), "preclip_grad_norm": float(norm),
            "head_grad_norm": float(head_grad), "adapter_grad_norm": float(adapter_grad),
            "clipped": bool(norm > config["clip_grad_norm"])}
