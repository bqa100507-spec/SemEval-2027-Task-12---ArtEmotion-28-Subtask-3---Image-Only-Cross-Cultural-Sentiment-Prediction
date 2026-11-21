"""NaFlex patch representations, grouped count loss and matched EXP-033 heads."""

import math

import torch
from scipy.linalg import helmert
from torch import nn
from torch.nn import functional as F


def patch_coordinates(mask, shapes):
    index = torch.arange(mask.shape[1], device=mask.device)[None, :]
    height, width = shapes.long().unbind(dim=1)
    if (shapes <= 0).any() or not torch.equal(mask.bool(), index < (height * width)[:, None]):
        raise ValueError("NaFlex mask must mark the contiguous row-major grid")
    return torch.stack(((index // width[:, None] + .5) / height[:, None],
                        (index % width[:, None] + .5) / width[:, None]), dim=-1)


def attention_mask(mask, tokens):
    return torch.zeros((len(mask), 1, mask.shape[1], mask.shape[1]), device=tokens.device, dtype=tokens.dtype).masked_fill(
        ~mask.bool()[:, None, None, :], torch.finfo(tokens.dtype).min,
    )


def prefix(vision, pixels, mask, shapes, blocks):
    tokens = vision.embeddings(pixels, shapes)
    attention = attention_mask(mask, tokens)
    for layer in vision.encoder.layers[:-blocks]:
        tokens = layer(tokens, attention_mask=attention)[0]
    return tokens


class Adapter(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens):
        return self.up(F.gelu(self.down(tokens)))


def suffix(vision, tokens, mask, blocks, adapters=None):
    attention = attention_mask(mask, tokens)
    for index, layer in enumerate(vision.encoder.layers[-blocks:]):
        if adapters is None:
            tokens = layer(tokens, attention_mask=attention)[0]
        else:
            residual, _ = layer.self_attn(layer.layer_norm1(tokens), attention_mask=attention)
            y = tokens + residual
            u = layer.layer_norm2(y)
            tokens = y + layer.mlp(u) + adapters[index](u)
    return vision.post_layernorm(tokens)


class SpatialHead(nn.Module):
    def __init__(self, dim, languages, config):
        super().__init__()
        if not languages:
            raise ValueError("Training needs at least one observed language")
        self.languages = tuple(languages)
        self.radius = config["pool_radius"]
        query_width = config["query_width"]
        self.key = nn.Linear(dim, query_width, bias=False)
        self.queries = nn.Parameter(torch.zeros(4, query_width))
        self.register_buffer("centers", torch.tensor([[.25, .25], [.25, .75], [.75, .25], [.75, .75]]))
        self.register_buffer("basis", torch.tensor(helmert(len(languages), full=False).T, dtype=torch.float32))
        self.intercept = nn.Parameter(torch.zeros(()))
        self.shared = nn.Parameter(torch.zeros(5 * dim))
        self.language = nn.Parameter(torch.zeros(len(languages) - 1))
        self.deviations = nn.Parameter(torch.zeros(len(languages) - 1, 5 * dim))

    def representation(self, tokens, global_vector, mask, shapes, arm):
        coords = patch_coordinates(mask, shapes)
        tokens = tokens.float().masked_fill(~mask.bool()[:, :, None], 0.)
        if arm == "A":
            affinities = torch.zeros_like(tokens[:, None, :, 0]).expand(-1, 4, -1)
        else:
            affinities = -(coords[:, None] - self.centers[None, :, None]).square().sum(-1) / (2 * self.radius**2)
        if arm in ("C", "D"):
            keys = self.key(F.layer_norm(tokens, (tokens.shape[-1],)))
            affinities = affinities + torch.einsum("sr,bpr->bsp", self.queries, keys) / math.sqrt(keys.shape[-1])
        weights = affinities.masked_fill(~mask.bool()[:, None, :], -torch.inf).softmax(dim=-1)
        slots = torch.einsum("bsp,bpd->bsd", weights, tokens)

        return F.normalize(torch.cat((global_vector.float()[:, None], slots), dim=1), dim=-1).flatten(1) / math.sqrt(5)

    def logits(self, h):
        return (self.intercept + h @ self.shared)[:, None] + (self.basis @ self.language)[None] + h @ (
            self.basis @ self.deviations
        ).T

    def ridge(self, coefficient, deviation_ratio):
        return .5 * coefficient * (self.shared.square().sum() + self.language.square().sum()
                                   + deviation_ratio * self.deviations.square().sum())

    def pool_state_names(self):
        return {"key.weight", "queries"}


def vote_sum(logits, k, n):
    logits = logits.float()
    return (n * F.softplus(logits) - k * logits).sum()


def drift_sum(tokens, frozen, mask):
    error = (tokens.float() - frozen.float()).square().mean(-1).masked_fill(~mask.bool(), 0.)
    return (error.sum(1) / mask.sum(1)).sum()
