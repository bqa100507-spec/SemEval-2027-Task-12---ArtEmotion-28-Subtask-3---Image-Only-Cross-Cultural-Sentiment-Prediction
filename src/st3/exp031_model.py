"""Native Qwen states with separable binary and conditional emotion heads."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .exp030_model import NativeStates


class HierarchicalHead(nn.Module):
    def __init__(self, hidden_size, prior_logits, rank):
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.visual = nn.Linear(hidden_size, rank, bias=False)
        self.language = nn.Embedding(len(prior_logits), rank)
        self.emotions = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 8))
        nn.init.zeros_(self.shared[-1].weight)
        nn.init.zeros_(self.shared[-1].bias)
        nn.init.zeros_(self.language.weight)
        self.scale = math.sqrt(rank)
        self.register_buffer("prior_logits", prior_logits.detach().clone().float())
        self.register_buffer("anchor", torch.zeros(hidden_size), persistent=False)

    def normalize(self, state):
        return F.layer_norm(state.float(), self.anchor.shape)

    def binary(self, state, ids):
        shared = self.shared(state.float()).squeeze(-1)
        visual = self.visual(self.normalize(state) - self.anchor)
        languages = self.language.weight - self.language.weight.mean(dim=0)
        interaction = languages @ visual / self.scale
        return self.prior_logits[ids] + shared + interaction[ids], interaction

    def conditionals(self, states):
        return self.emotions(states.float())

    def distribution(self, binary, conditional_logits):
        return torch.cat([F.logsigmoid(sign * binary)[:, None] + F.log_softmax(conditional_logits[:, part], dim=1)
                          for sign, part in ((1, slice(0, 4)), (-1, slice(4, 8)))], dim=1)


class CulturalEmotionModel(nn.Module):
    def __init__(self, pretrained, processor, config, languages, priors):
        super().__init__()
        self.native = NativeStates(pretrained, processor, config)
        self.decoder = pretrained.model
        self.languages = tuple(languages)
        self.lookup = {name: i for i, name in enumerate(languages)}
        self.head = HierarchicalHead(pretrained.config.hidden_size, priors, config["cultural_rank"]).to(self.native.device)

    def train(self, mode=True):
        super().train(mode)
        self.native.model.visual.eval()
        return self

    def state(self, vision, languages):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.native.device.type == "cuda"):
            return self.native.hidden([vision] * len(languages), languages, train=self.training)

    def binary(self, vision, languages):
        ids = torch.tensor([self.lookup[name] for name in languages], device=self.native.device)
        return self.head.binary(self.state(vision, [None])[0], ids)

    def conditionals(self, vision, languages):
        return self.head.conditionals(self.state(vision, languages))

    @torch.no_grad()
    def set_anchor(self, vision):
        self.head.anchor.copy_(self.head.normalize(self.state(vision, [None])[0]))
