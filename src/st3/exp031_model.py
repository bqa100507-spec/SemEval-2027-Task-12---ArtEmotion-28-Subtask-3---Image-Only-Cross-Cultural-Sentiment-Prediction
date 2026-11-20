"""EXP-031: image-level sentiment and low-rank culture adjustments.

A neutral native-Qwen visual-language state represents an image regardless of
which target languages happen to be requested. Culture interaction acts on that
state, not on held-out batch composition. Language-specific hidden states still
supervise the conditional eight-emotion distribution.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .exp030_model import NativeStates


class HierarchicalHead(nn.Module):
    def __init__(self, hidden_size, prior_logits, rank):
        super().__init__()
        languages = len(prior_logits)
        if languages == 0 or rank < 1:
            raise ValueError("Empty language prior or invalid low rank")
        self.shared = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.cultural = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, rank))
        self.language = nn.Embedding(languages, rank)
        self.emotions = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 8))
        nn.init.zeros_(self.language.weight)
        self.scale = math.sqrt(rank)
        self.register_buffer("prior_logits", prior_logits.detach().clone().float(), persistent=True)

    def forward(self, neutral_state, language_states, language_ids):
        if language_states.shape[0] != language_ids.numel():
            raise ValueError("Language states/IDs mismatch")
        image = neutral_state.float().reshape(1, -1)
        shared = self.shared(image).reshape(())
        visual = self.cultural(image).reshape(-1)
        # Centering gives the shared painting offset and interaction identifiable roles.
        centered_languages = self.language.weight - self.language.weight.mean(dim=0)
        interaction = (centered_languages[language_ids] * visual).sum(dim=-1) / self.scale
        binary = self.prior_logits[language_ids] + shared + interaction

        emotion_logits = self.emotions(language_states.float())
        positive = F.logsigmoid(binary).unsqueeze(1) + F.log_softmax(emotion_logits[:, :4], dim=1)
        negative = F.logsigmoid(-binary).unsqueeze(1) + F.log_softmax(emotion_logits[:, 4:], dim=1)
        return torch.cat([positive, negative], dim=1), binary, interaction


class CulturalEmotionModel(nn.Module):
    def __init__(self, pretrained, processor, config, languages, priors):
        super().__init__()
        self.native = NativeStates(pretrained, processor, config)
        self.decoder = pretrained.model
        self.languages = tuple(languages)
        self._lookup = {name: i for i, name in enumerate(languages)}
        self.head = HierarchicalHead(pretrained.config.hidden_size, priors, config["cultural_rank"]).to(self.native.device)

    def train(self, mode=True):
        super().train(mode)
        self.native.model.visual.eval()
        return self

    def forward_painting(self, vision, languages, microbatch_pairs):
        if not languages or len(set(languages)) != len(languages):
            raise ValueError("Require distinct target languages from one painting")
        if any(name not in self._lookup for name in languages):
            raise ValueError("Validation has an unseen language; no held-out prior may be fit")
        device = self.native.device
        ids = torch.tensor([self._lookup[name] for name in languages], device=device)
        neutral = self.native.hidden([vision], ["unspecified"], train=self.training).squeeze(0)
        states = []
        for start in range(0, len(languages), microbatch_pairs):
            subset = languages[start:start + microbatch_pairs]
            states.append(self.native.hidden([vision] * len(subset), subset, train=self.training))
        conditioned = torch.cat(states, dim=0)
        return self.head(neutral, conditioned, ids)
