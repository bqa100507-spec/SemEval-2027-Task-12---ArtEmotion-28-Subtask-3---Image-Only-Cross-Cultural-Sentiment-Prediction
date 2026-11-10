"""Pinned CLIP-L prefix and final-block LoRA modules for EXP-014."""

import hashlib
import importlib.metadata
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .baseline import file_sha256
from .cli import json_output
from .clip_large import load_vision
from .contrast_study import new_output
from .data import image_path
from .e50_heads import read_json
from .e50l_heads import clip_large_config
from .frozen_features import cache_location, verify_asset_hashes
from .nonlinear import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LoRAProjection(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        self.scale = alpha / rank

    def reset(self, generator):
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5), generator=generator)
        nn.init.zeros_(self.b)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def vision_prefix(model, pixels, blocks):
    vision = model.vision_model
    x = vision.pre_layrnorm(vision.embeddings(pixels))
    for layer in vision.encoder.layers[:-blocks]:
        x = layer(x, None, None)[0]
    return x


class ClipTail(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        model.requires_grad_(False)
        self.blocks = nn.ModuleList(model.vision_model.encoder.layers[-config["blocks"]:])
        self.norm = model.vision_model.post_layernorm
        self.projection = model.visual_projection
        self.use_checkpoint = config["gradient_checkpointing"]
        self.targets = []
        first = len(model.vision_model.encoder.layers) - len(self.blocks)
        for number, layer in enumerate(self.blocks, first):
            for name in config["projections"]:
                base = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, LoRAProjection(base, config["rank"], config["alpha"]))
                self.targets.append(f"vision_model.encoder.layers.{number}.self_attn.{name}")
        self.reset(config["seed"])

    def reset(self, seed):
        generator = torch.Generator(device=self.projection.weight.device).manual_seed(seed)
        for layer in self.modules():
            if isinstance(layer, LoRAProjection):
                layer.reset(generator)

    def forward(self, x):
        for layer in self.blocks:
            if self.use_checkpoint and torch.is_grad_enabled():
                x = checkpoint(layer, x, None, None, use_reentrant=False)[0]
            else:
                x = layer(x, None, None)[0]
        projected = self.projection(self.norm(x[:, 0]))
        return F.normalize(projected.float(), dim=-1)


def adapter_parameters(tail):
    return {name: value for name, value in tail.named_parameters() if value.requires_grad}


def frozen_digest(tail):
    digest = hashlib.sha256()
    for name, parameter in tail.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_adapter(tail, path):
    np.savez(path, **{name: value.detach().cpu().numpy() for name, value in adapter_parameters(tail).items()})


def load_adapter(tail, path):
    parameters = adapter_parameters(tail)
    with np.load(path, allow_pickle=False) as saved, torch.no_grad():
        if set(saved.files) != set(parameters):
            raise ValueError("Adapter checkpoint has unexpected parameter names")
        for name, target in parameters.items():
            value = saved[name]
            if value.shape != tuple(target.shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid adapter parameter {name}")
            target.copy_(torch.from_numpy(value))


def load_encoder(cache_dir):
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor

    config = clip_large_config()
    if importlib.metadata.version("transformers") != config["transformers"]:
        raise ValueError("EXP-014 requires the pinned CLIP-L Transformers runtime")
    assets = {name: Path(hf_hub_download(config["model"], name, revision=config["revision"],
                                        cache_dir=cache_dir / "huggingface"))
              for name in config["assets_sha256"]}
    verify_asset_hashes({name: file_sha256(path) for name, path in assets.items()}, config)
    directory = assets["config.json"].parent
    return (load_vision(directory).eval().requires_grad_(False),
            AutoImageProcessor.from_pretrained(directory, use_fast=False))


def prefix_cache(args, names, model, processor, config, device, split):
    """Store original FP32 outputs before the first adapted block, keyed by image bytes."""
    names = sorted(set(names))
    identity = {"encoder": clip_large_config(), "torch": str(torch.__version__), "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "untouched_blocks": len(model.vision_model.encoder.layers) - config["blocks"],
                "batch_size": config["prefix_batch_size"], "dtype": "float32",
                "pillow": importlib.metadata.version("Pillow")}
    fingerprint = cache_location(args.data_dir, names, args.run_dir, identity, "prefix").name
    output = args.run_dir / f"prefix-{split}"
    if output.exists():
        manifest = read_json(output / "manifest.json")
        if (manifest["identity"] != identity or manifest["fingerprint"] != fingerprint or manifest["names"] != names
                or file_sha256(output / "features.npy") != manifest["features_sha256"]):
            raise ValueError("Frozen prefix cache provenance or checksum differs")
    else:
        start = time.perf_counter()
        with new_output(output) as stage:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.to(device)
            batch_size = config["prefix_batch_size"]
            width = model.config.hidden_size
            tokens = 1 + (model.config.image_size // model.config.patch_size) ** 2
            path = stage / "features.npy"
            shape = (len(names), tokens, width)
            bank = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
            with torch.inference_mode():
                for start_at in range(0, len(names), batch_size):
                    images = []
                    for name in names[start_at:start_at + batch_size]:
                        with Image.open(image_path(args.data_dir, name)) as image:
                            images.append(image.convert("RGB"))
                    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
                    hidden = vision_prefix(model, pixels, config["blocks"]).cpu().numpy()
                    if not np.isfinite(hidden).all():
                        raise ValueError("Nonfinite original prefix activations")
                    stop_at = start_at + len(images)
                    bank[start_at:stop_at] = hidden
                    if start_at % (batch_size * 10) == 0 or stop_at == len(names):
                        print(f"EXP-014 {split} prefix {stop_at}/{len(names)} images", flush=True)
            bank.flush()
            del bank
            manifest = {"identity": identity, "fingerprint": fingerprint, "names": names, "shape": list(shape),
                        "features_sha256": file_sha256(path), "wall_seconds": time.perf_counter() - start}
            if device.type == "cuda":
                manifest["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                manifest["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            json_output(manifest, stage / "manifest.json")
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    bank = np.load(output / "features.npy", mmap_mode="c", allow_pickle=False)
    if bank.dtype != np.float32 or list(bank.shape) != manifest["shape"]:
        raise ValueError("Frozen prefix cache shape or dtype differs")
    return torch.from_numpy(bank).to(device), {name: i for i, name in enumerate(names)}, manifest
