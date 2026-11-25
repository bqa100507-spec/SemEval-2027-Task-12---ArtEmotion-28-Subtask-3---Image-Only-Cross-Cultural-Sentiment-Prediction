"""Progressive fusion of SigLIP patches and contextual Qwen image tokens."""

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .competition_direct_qwen import LowRankUpdate
from .data import KEYS, POSITIVE, NEGATIVE
from .exp036 import Candidate, LoRA, adapt, rounded
from .metrics import score

EMOTIONS = tuple(sorted(POSITIVE)) + tuple(sorted(NEGATIVE))
STAGES = ("A", "B", "C")


def counts_for(annotations, pairs):
    if set(annotations.split) != {"train"}:
        raise ValueError("Eight-emotion targets require official training annotations only")
    counts = annotations[annotations.emotion.isin(EMOTIONS)].groupby([*KEYS, "emotion"]).size()
    counts = counts.unstack(fill_value=0).reindex(columns=EMOTIONS, fill_value=0)
    counts = counts.reindex(list(pairs[KEYS].itertuples(index=False, name=None)))
    values = counts.to_numpy(dtype=np.float32)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.array_equal(values.sum(1), pairs.n)
            or not np.array_equal(values[:, :4].sum(1), pairs.k)):
        raise ValueError("Eight-way and binary vote aggregation disagree")
    return values


def image_tokens(hidden, ids, attention, image_id, grid, merge):
    """Select only active image placeholders in each native Qwen decoder row."""
    if hidden.shape[:2] != ids.shape or ids.shape != attention.shape or grid.shape != (len(ids), 3):
        raise ValueError("Unexpected Qwen token or grid layout")
    image = ids.eq(image_id) & attention.bool()
    volume = grid.prod(dim=1)
    expected = volume // merge**2
    if (volume % merge**2).any() or (expected < 1).any() or not torch.equal(image.sum(1), expected):
        raise ValueError("Active Qwen image tokens disagree with the native image grid")
    values = pad_sequence([row[m] for row, m in zip(hidden, image, strict=True)], batch_first=True)
    padding = torch.arange(values.shape[1], device=hidden.device)[None, :] >= expected[:, None]
    return values, padding


class CrossBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2*width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, padding):
        kv = self.context_norm(context)
        update, _ = self.attention(self.query_norm(x), kv, kv, key_padding_mask=padding, need_weights=False)
        x = x + self.dropout(update)
        return x + self.dropout(self.ff(x))


class FusionHeads(nn.Module):
    logits = Candidate.logits
    caption_vectors = Candidate.caption_vectors

    def __init__(self, width, languages, cfg, *, auxiliary):
        super().__init__()
        dim = cfg["fusion_width"]
        self.visual_projection = nn.Linear(width, dim)
        self.qwen_projection = nn.Linear(cfg["qwen_width"], dim)
        self.blocks = nn.ModuleList(CrossBlock(dim, cfg["fusion_heads"], cfg["dropout"])
                                    for _ in range(cfg["fusion_blocks"]))
        self.pool_norm = nn.LayerNorm(dim)
        self.shared = nn.Linear(width + dim, 1, bias=False)
        self.interaction = nn.Linear(width + dim, languages)
        nn.init.zeros_(self.shared.weight)
        nn.init.zeros_(self.interaction.weight)
        nn.init.zeros_(self.interaction.bias)
        if auxiliary:
            hidden = 128
            self.caption = nn.Sequential(nn.Linear(width+dim, hidden), nn.GELU(), nn.Linear(hidden, width))
            self.caption_language = nn.Embedding(languages, hidden)
            self.emotion = nn.Linear(width+dim, 8)
            self.emotion_language = nn.Embedding(languages, 8)

    def forward(self, patches, pooled, context, padding):
        queries = self.visual_projection(patches.float())
        keys = self.qwen_projection(context.float())
        for block in self.blocks:
            queries = block(queries, keys, padding)
        return torch.cat([F.normalize(pooled.float(), dim=-1),
                          F.normalize(self.pool_norm(queries.mean(dim=1)).float(), dim=-1)], dim=-1)


class Progressive(nn.Module):
    def __init__(self, vision, qwen, visual_cfg, cfg, languages, *, auxiliary=True):
        super().__init__()
        self.vision = vision
        self.qwen = qwen.requires_grad_(False)
        self.visual_targets = adapt(vision, visual_cfg)
        self.heads = FusionHeads(visual_cfg["hidden_size"], languages, cfg, auxiliary=auxiliary)
        layers = qwen.language_model.layers
        if len(layers) != cfg["qwen_layers"] or qwen.config.text_config.hidden_size != cfg["qwen_width"]:
            raise ValueError("Unexpected pinned Qwen decoder architecture")
        rng = torch.Generator().manual_seed(cfg["seed"] + 1)
        self.qwen_targets = []
        for i in range(len(layers) - cfg["qwen_tail"], len(layers)):
            attn = layers[i].self_attn
            for projection in ("q_proj", "v_proj"):
                base = getattr(attn, projection)
                if not isinstance(base, nn.Linear):
                    raise TypeError(f"Unexpected Qwen target {i}.{projection}")
                setattr(attn, projection, LowRankUpdate(base, cfg["qwen_rank"], cfg["qwen_alpha"], rng))
                self.qwen_targets.append(f"language_model.layers.{i}.self_attn.{projection}")
        self.stage("A")

    def groups(self):
        return {"fusion": list(self.heads.parameters()),
                "vision": [p for m in self.vision.modules() if isinstance(m, LoRA) for p in (m.left, m.right)],
                "qwen": [p for m in self.qwen.modules() if isinstance(m, LowRankUpdate) for p in (m.left, m.right)]}

    def stage(self, label):
        if label not in STAGES:
            raise ValueError(f"Unknown stage {label}")
        self.active = label
        groups = self.groups()
        for name, values in groups.items():
            enabled = name == "fusion" or (name == "vision" and label in "BC") or (name == "qwen" and label == "C")
            for p in values:
                p.requires_grad_(enabled)
                p.grad = None
        expected = {id(p) for values in groups.values() for p in values if p.requires_grad}
        if {id(p) for p in self.parameters() if p.requires_grad} != expected:
            raise RuntimeError("Unexpected trainable base weights")
        self.train(self.training)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode=True):
        super().train(mode)
        self.vision.train(mode and self.active in "BC")
        self.qwen.eval()
        self.qwen.language_model.train(mode and self.active == "C")
        return self

    def features(self, pixels, encoded):
        with nullcontext() if self.active in "BC" else torch.no_grad():
            visual = self.vision(pixel_values=pixels)
        with nullcontext() if self.active == "C" else torch.no_grad():
            hidden = self.qwen(**encoded, use_cache=False).last_hidden_state
            context, padding = image_tokens(hidden, encoded["input_ids"], encoded["attention_mask"],
                                            self.qwen.config.image_token_id, encoded["image_grid_thw"],
                                            self.qwen.config.vision_config.spatial_merge_size)
        return self.heads(visual.last_hidden_state, visual.pooler_output, context, padding)

    def learned(self, *, inference=False):
        ids = {id(p) for values in self.groups().values() for p in values}
        return {n:p for n,p in self.named_parameters() if id(p) in ids and (
            not inference or not n.startswith(("heads.caption", "heads.emotion")))}

    def snapshot(self, *, inference=False):
        return {n:p.detach().cpu().clone() for n,p in self.learned(inference=inference).items()}

    def restore(self, tensors, *, inference=False):
        expected = self.learned(inference=inference)
        if set(expected) != set(tensors) or any(tensors[n].shape != p.shape for n,p in expected.items()):
            raise ValueError("Learned tensor identities or shapes differ")
        with torch.no_grad():
            for n,p in expected.items():
                p.copy_(tensors[n])


def losses(heads, features, batch, vote_mass, caption_mass):
    groups, languages, k, n, emotion_counts, targets, caption_weights = batch
    logits = heads.logits(features, groups, languages).float()
    sentiment = (F.binary_cross_entropy_with_logits(logits, k/n, reduction="none") * n).sum() / vote_mass
    emo = heads.emotion(features)[groups] + heads.emotion_language(languages)
    emotion = -(emo.float().log_softmax(dim=1) * emotion_counts).sum() / vote_mass
    vectors = heads.caption_vectors(features, groups, languages)
    caption = ((1-F.cosine_similarity(vectors.float(), targets, dim=1, eps=1e-6)) * caption_weights).sum() / caption_mass
    return torch.stack((sentiment, emotion, caption))


def diagnostics(gold, pred, excluded):
    keep = ~gold.language.isin(excluded)
    subset = gold.loc[keep]
    filtered = subset[KEYS].merge(pred, on=KEYS, validate="one_to_one")
    return {"full": score(gold, pred), "evaluation_like": score(subset, filtered)}


def blends(gold, dev, parent, candidate, cfg):
    parent, candidate = rounded(parent, dev), rounded(candidate, dev)
    rows, frames = [], {}
    for alpha in cfg["ensemble_grid"]:
        frame = candidate.assign(p_positive=((1-alpha)*parent.p_positive + alpha*candidate.p_positive).round(6))
        frames[alpha] = frame
        rows.append({"alpha": alpha, **diagnostics(gold, frame, cfg["test_like_excluded"])})
    winner = max(rows, key=lambda row: (row["full"]["cultural_contrast"], -row["full"]["log_loss"], -row["alpha"]))
    return rows, winner, frames[winner["alpha"]]
