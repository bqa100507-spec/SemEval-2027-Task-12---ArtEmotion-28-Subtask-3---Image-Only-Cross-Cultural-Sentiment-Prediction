"""Frozen native Qwen vision tokens and a trainable eight-emotion head."""
import torch
from torch import nn

from .vlm_supervised import QwenStates


class NativeStates(QwenStates):
    def text(self, language=None):
        instruction = self.config["prompt"].format(language=language or "unspecified")
        return self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}],
            tokenize=False, add_generation_prompt=True)


class EmotionModel(nn.Module):
    def __init__(self, model, processor, config):
        super().__init__()
        self.native = NativeStates(model, processor, config)
        self.head = nn.Sequential(nn.LayerNorm(model.config.hidden_size),
                                  nn.Linear(model.config.hidden_size, 8)).to(self.native.device)
        self.decoder = model.model

    def train(self, mode=True):
        super().train(mode)
        self.native.model.visual.eval()
        return self

    def forward(self, visions, languages):
        features = self.native.hidden(visions, languages, train=self.training)
        return self.head(features.float())
