"""Frozen tensor-only inference for the pinned publisher's EmotionCLIP release."""

import numpy as np
from PIL import Image


def preprocess(image):
    """Reproduce the publisher's OpenAI CLIP PIL/torchvision transform."""
    import torch

    width, height = image.size
    size = 224
    if min(width, height) != size:
        shape = (size, size * height // width) if width < height else (size * width // height, size)
        image = image.resize(shape, Image.Resampling.BICUBIC)
    left, top = (round((side - size) / 2) for side in image.size)
    crop = image.crop((left, top, left + size, top + size)).convert("RGB")
    pixels = torch.from_numpy(np.array(crop)).permute(2, 0, 1).contiguous().float().div_(255)
    mean = pixels.new_tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
    std = pixels.new_tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
    return pixels.sub_(mean).div_(std)


def prompt_tokens(tokenizer_file, labels, template):
    from tokenizers import Tokenizer
    import torch

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    rows = [token.ids for token in tokenizer.encode_batch([template.format(label) for label in labels])]
    if any(row[0] != 49406 or row[-1] != 49407 or len(row) > 77 for row in rows):
        raise ValueError("Unexpected OpenAI CLIP prompt tokens")
    return torch.tensor([row + [0] * (77 - len(row)) for row in rows], dtype=torch.long)


class EmotionCLIP:
    """Inference operations matching EmotionCLIP.py for its stored K/V prefixes."""

    def __init__(self, path, device):
        import torch

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        # The publisher's inference code consumes the compiled prefixes. Its
        # training-only MLP and token index tensors are unused in that forward.
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.used = set()
        self.state = {name: tensor for name, tensor in state.items()
                      if not name.endswith(".prefix_tokens")
                      and ".prefix.embedding." not in name and ".prefix.transformer." not in name}
        self.state = {name: tensor.to(device, dtype=self.dtype) for name, tensor in self.state.items()}
        for name in ("visual.transformer", "transformer"):
            prefix = self.state[f"{name}.prefix.prefix_embedding"]
            width = 768 if name.startswith("visual.") else 512
            if tuple(prefix.shape) != (1, 20, 24 * width):
                raise ValueError(f"Unexpected prefix tensor for {name}")

    def weight(self, name):
        self.used.add(name)
        return self.state[name]

    def verified_forward_keys(self):
        if self.used != set(self.state):
            raise ValueError(f"Unexpected tensors outside the pinned forward: {sorted(set(self.state) - self.used)}")

    def norm(self, x, name):
        from torch.nn.functional import layer_norm

        return layer_norm(x, (x.shape[-1],), self.weight(name + ".weight"), self.weight(name + ".bias"))

    def linear(self, x, name):
        from torch.nn.functional import linear

        return linear(x, self.weight(name + ".weight"), self.weight(name + ".bias"))

    def transformer(self, x, name, heads):
        import torch
        from torch.nn.functional import scaled_dot_product_attention

        batch, _, width = x.shape
        prefix = self.weight(name + ".prefix.prefix_embedding").reshape(20, 12, 2, width)
        prefix = prefix.permute(1, 2, 0, 3)

        def split(t):
            return t.reshape(batch, -1, heads, width // heads).transpose(1, 2)

        for index, (pk, pv) in enumerate(prefix.unbind()):
            layer = f"{name}.resblocks.{index}"
            attn = layer + ".attn"
            normalized = self.norm(x, layer + ".ln_1")
            qkv = normalized @ self.weight(attn + ".in_proj_weight").T + self.weight(attn + ".in_proj_bias")
            q, k, v = qkv.chunk(3, dim=-1)
            k = torch.cat((pk.expand(batch, -1, -1), k), dim=1)
            v = torch.cat((pv.expand(batch, -1, -1), v), dim=1)
            attention = scaled_dot_product_attention(split(q), split(k), split(v))
            attention = attention.transpose(1, 2).reshape(batch, x.shape[1], width)
            x = x + self.linear(attention, attn + ".out_proj")
            h = self.linear(self.norm(x, layer + ".ln_2"), layer + ".mlp.c_fc")
            h32 = h.float()
            x = x + self.linear((h32 * torch.sigmoid(1.702 * h32)).to(h.dtype), layer + ".mlp.c_proj")
        return x

    def encode_image(self, pixels):
        import torch
        from torch.nn.functional import conv2d

        x = conv2d(pixels.to(self.device, self.dtype), self.weight("visual.conv1.weight"), stride=32)
        x = x.flatten(2).transpose(1, 2)
        cls = self.weight("visual.class_embedding").expand(len(pixels), 1, -1)
        x = self.norm(torch.cat((cls, x), dim=1) + self.weight("visual.positional_embedding"), "visual.ln_pre")
        x = self.transformer(x, "visual.transformer", heads=12)
        # Preserve the publisher's token readout, including its index of 20.
        return self.norm(x[:, 20, :], "visual.ln_post") @ self.weight("visual.proj")

    def encode_text(self, tokens):
        import torch

        tokens = tokens.to(self.device)
        x = self.weight("token_embedding.weight")[tokens] + self.weight("positional_embedding")[None, :tokens.shape[1]]
        x = self.norm(self.transformer(x, "transformer", heads=8), "ln_final")
        return x[torch.arange(len(x), device=x.device), tokens.argmax(dim=-1)] @ self.weight("text_projection")

    def features(self, pixels, text):
        from torch.nn.functional import normalize

        vision = self.encode_image(pixels)
        image_unit = vision / vision.norm(dim=-1, keepdim=True)
        text_unit = text / text.norm(dim=-1, keepdim=True)
        logits = (self.weight("logit_scale").exp() * image_unit) @ text_unit.T
        return normalize(vision.float(), dim=-1), logits.float().softmax(dim=-1)
