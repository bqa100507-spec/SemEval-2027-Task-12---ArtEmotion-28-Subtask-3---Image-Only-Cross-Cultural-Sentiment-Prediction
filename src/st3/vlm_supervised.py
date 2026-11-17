"""Direct Qwen decoder states and decoder-only LoRA. Never generates text."""

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DecoderLoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, values):
        original = self.base(values)
        if not self.enabled:
            return original
        update = F.linear(F.linear(values.float(), self.A), self.B) * self.scale
        return original + update.to(original.dtype)


def attach_adapters(model, config):
    model.requires_grad_(False)
    names = [name for name, _ in model.named_modules() if re.fullmatch(r"model\.layers\.\d+\.self_attn\.(q_proj|v_proj)", name)]
    expected = {f"model.layers.{i}.self_attn.{name}" for i in range(config["decoder_layers"]) for name in config["lora"]["targets"]}
    if set(names) != expected or len(names) != config["lora"]["expected_modules"]:
        raise ValueError("Decoder LoRA target matching differs; vision/text head must not be adapted")
    # Adapter allocation cannot advance the sentiment-head initialization RNG.
    devices = [torch.cuda.current_device()] if next(model.parameters()).is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config["seed"])
        for name in names:
            parent_name, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child, DecoderLoRA(getattr(parent, child), config["lora"]["rank"], config["lora"]["alpha"]))
    return names


def adapter_parameters(model):
    return [parameter for module in model.modules() if isinstance(module, DecoderLoRA) for parameter in (module.A, module.B)]


@contextmanager
def adapters_disabled(model):
    modules = [m for m in model.modules() if isinstance(m, DecoderLoRA)]
    previous = [m.enabled for m in modules]
    try:
        for module in modules:module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous):module.enabled = enabled


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class SentimentHead(nn.Module):
    def __init__(self, languages, config):
        super().__init__()
        self.languages = tuple(languages)
        width = config["head_width"]
        self.visual = nn.Sequential(nn.LayerNorm(config["hidden_size"]), nn.Linear(config["hidden_size"], width), nn.GELU())
        self.shared = nn.Linear(width, 1)
        self.language_bias = nn.Embedding(len(languages), 1)
        self.language_visual = nn.Embedding(len(languages), width)
        nn.init.zeros_(self.language_bias.weight);nn.init.zeros_(self.language_visual.weight)
        self.scale = math.sqrt(width)

    def indices(self, languages, device):
        lookup = {name: i for i, name in enumerate(self.languages)}
        return torch.tensor([lookup.get(name, -1) for name in languages], device=device)

    def forward(self, hidden, language):
        values = self.visual(hidden.float())
        known, indices = language.ge(0), language.clamp_min(0)
        residual = self.language_bias(indices).squeeze(-1) + (values * self.language_visual(indices)).sum(-1) / self.scale
        return self.shared(values).squeeze(-1) + residual * known


def vote_loss(logits, positive, votes, *, denominator=None):
    if (not torch.isfinite(logits).all() or not torch.isfinite(votes).all() or not torch.isfinite(positive).all()
            or (votes <= 0).any() or (positive < 0).any() or (positive > votes).any()):
        raise ValueError("Invalid count-aware BCE input")
    if denominator is not None and (not math.isfinite(float(denominator)) or float(denominator)<=0):
        raise ValueError("Invalid accumulation vote denominator")
    values = F.binary_cross_entropy_with_logits(logits.float(), positive.float() / votes.float(), reduction="none")
    return (values * votes).sum() / (votes.sum() if denominator is None else denominator)


def make_head(languages, config, device="cpu"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        return SentimentHead(languages, config).to(device)


class QwenStates:
    def __init__(self, model, processor, config):
        self.model, self.processor, self.config = model, processor, config
        self.processor.tokenizer.padding_side = "left"
        self.device = next(model.parameters()).device

    def text(self, language=None):
        if language is not None and (not language.strip() or any(ch in language for ch in "\n\r\0")):
            raise ValueError("Invalid target-language identifier")
        prompt = self.config["neutral_prompt"] if language is None else self.config["language_prompt"].format(language=language)
        return self.processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}],
                                                  tokenize=False, add_generation_prompt=True)

    def vision(self, image):
        batch = self.processor(text=[self.text()], images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            tokens = self.model.visual(batch.pixel_values.to(self.device, self.model.visual.dtype), grid_thw=batch.image_grid_thw.to(self.device))
        return {"tokens":tokens.detach().cpu(), "grid":batch.image_grid_thw[0].cpu()}

    def batch(self, visions, languages):
        texts = []
        token = self.processor.image_token
        for vision, language in zip(visions, languages):
            texts.append(self.text(language).replace(token, token * len(vision["tokens"])))
        encoded = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        ids, mask = encoded.input_ids.to(self.device), encoded.attention_mask.to(self.device)
        with torch.no_grad():
            embeddings = self.model.model.embed_tokens(ids)
            for i, vision in enumerate(visions):
                locations = ids[i].eq(self.model.config.image_token_id)
                if int(locations.sum()) != len(vision["tokens"]):raise ValueError("Image token count differs")
                embeddings[i, locations] = vision["tokens"].to(self.device, embeddings.dtype)
        grid = torch.stack([v["grid"] for v in visions]).to(self.device)
        positions, _ = self.model.get_rope_index(ids, image_grid_thw=grid, attention_mask=mask)
        return ids, mask, embeddings, positions

    def hidden(self, visions, languages, *, train=False):
        _, mask, embeddings, positions = self.batch(visions, languages)
        if train:embeddings.requires_grad_(True)
        output = self.model.model(input_ids=None, inputs_embeds=embeddings, position_ids=positions,
                                  attention_mask=mask, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state[:, -1, :]
        if not torch.isfinite(hidden).all():raise ValueError("Nonfinite multimodal hidden state")
        return hidden

    def parity(self, image, vision):
        """Compare the official image route with cached vision injection; no decoding."""
        batch = self.processor(text=[self.text()], images=[image], return_tensors="pt", padding=True).to(self.device)
        ids, _, _, _ = self.batch([vision], [None])
        if not torch.equal(ids, batch.input_ids):raise ValueError("Cached image-token formatting differs from official processor")
        original = self.model.lm_head
        try:
            # Reference official multimodal assembly, avoiding a vocabulary-logit tensor.
            self.model.lm_head = nn.Identity()
            with torch.no_grad(), adapters_disabled(self.model):
                reference = self.model(**batch, use_cache=False, return_dict=True).logits[:, -1, :]
                cached = self.hidden([vision], [None])
        finally:self.model.lm_head = original
        error = float((reference.float() - cached.float()).abs().max())
        if not torch.allclose(reference, cached, atol=2e-3, rtol=2e-3):
            raise ValueError(f"Official/cached multimodal state parity failed: {error}")
        return {"status":"PASS","max_abs_error":error,"cosine":float(F.cosine_similarity(reference.float(),cached.float()).item()),
                "hidden_dimensions":reference.shape[-1],"vision_tokens":len(vision["tokens"])}
