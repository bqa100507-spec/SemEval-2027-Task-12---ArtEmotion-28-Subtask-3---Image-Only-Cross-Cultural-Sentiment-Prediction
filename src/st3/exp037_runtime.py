"""Pinned backbones and painting-wise multi-task optimization."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import data
from .competition_semantic_generation import verified_assets
from .competition_semantic_recovery import load_model
from .exp036_train import Paintings, amp, backbone, model_assets
from .exp037 import Progressive, losses


class BatchMemoryError(RuntimeError):
    """An OOM before unscaling or applying an optimizer update."""


class Processor:
    def __init__(self, siglip, qwen, cfg, root, files):
        self.siglip, self.qwen = siglip, qwen
        self.root, self.files = root, files
        message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg["prompt"]}]}]
        self.prompt = qwen.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    def batch(self, names, device):
        images = []
        for name in names:
            path = self.root / self.files[name] if self.files is not None else data.image_path(self.root, name)
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
        pixels = self.siglip(images=images, return_tensors="pt").pixel_values.to(device)
        qwen = self.qwen(text=[self.prompt]*len(images), images=images, padding=True, return_tensors="pt").to(device)
        return pixels, qwen


def pinned_model(visual, qwen, cfg, visual_path, qwen_path, languages, *, auxiliary, root, images):
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor

    visual_path = model_assets(visual, visual_path, text_assets=False)
    siglip = backbone(visual_path, visual, text=False)
    siglip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    siglip_processor = AutoImageProcessor.from_pretrained(visual_path, local_files_only=True, use_fast=False)
    if qwen_path is None:
        qwen_path = Path(snapshot_download(qwen["model"], revision=qwen["revision"], token=False,
                         allow_patterns=[*qwen["weight_sha256"], *qwen["git_blobs"]]))
    verified_assets(qwen_path, qwen)
    _, loaded, qwen_processor, hardware = load_model(qwen_path, qwen | {"attention": "sdpa"})
    loaded.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decoder = loaded.model
    del loaded  # The LM head and generation wrapper are outside the forward path.
    model = Progressive(siglip.vision_model, decoder, visual, cfg, len(languages), auxiliary=auxiliary)
    model.vision.to("cuda")
    model.heads.to("cuda")
    return model, Processor(siglip_processor, qwen_processor, cfg, root, images), hardware


class TrainingPaintings(Paintings):
    def __init__(self, pairs, languages, emotion, values, mask):
        super().__init__(pairs, languages, values=values, mask=mask)
        self.emotion = emotion
        if (self.languages < 0).any():
            raise ValueError("Training language missing from the fitted mapping")

    def targets(self, ids, groups, device):
        arrays = (groups, self.languages[ids], self.k[ids], self.n[ids], self.emotion[ids],
                  self.values[ids], self.mask[ids]*self.n[ids])
        return tuple(torch.as_tensor(x, device=device) for x in arrays)


def train_step(model, panel, indices, processor, micro, config, opt, scaler, device):
    ids, _ = panel.groups(indices)
    votes = float(panel.n[ids].sum())
    captions = max(1., float((panel.mask[ids]*panel.n[ids]).sum()))
    params = [p for group in opt.param_groups for p in group["params"]]
    weights = torch.tensor([1., config["emotion_lambda"], config["caption_lambda"]], device=device)
    groups = {name:[p for p in values if p.requires_grad] for name, values in model.groups().items()}
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        total = torch.zeros(3, device=device)
        for i in range(0, len(indices), micro):
            paintings = indices[i:i+micro]
            ids, group = panel.groups(paintings)
            try:
                pixels, qwen = processor.batch([panel.images[k] for k in paintings], device)
                with amp(device):
                    hidden = model.features(pixels, qwen)
                    components = losses(model.heads, hidden, panel.targets(ids, group, device), votes, captions)
                if not torch.isfinite(components).all():
                    raise FloatingPointError("Nonfinite progressive multi-task loss")
                scaler.scale((components*weights).sum()).backward()
            except torch.cuda.OutOfMemoryError as error:
                raise BatchMemoryError(str(error)) from None
            total += components.detach()
        scaler.unscale_(opt)
        norms = {}
        for name, values in groups.items():
            if values and any(p.grad is None for p in values):
                raise RuntimeError(f"Disconnected trainable parameters in {name}")
            norms[name] = (torch.stack([p.grad.detach().float().norm() for p in values]).norm().item()
                           if values else 0.)
        norm = torch.nn.utils.clip_grad_norm_(params, config["gradient_clip"])
        if torch.isfinite(norm):
            scaler.step(opt)
            scaler.update()
            terms = total.cpu().tolist()
            return [*terms, (total*weights).sum().item(), *norms.values(), float(norm>config["gradient_clip"])]
        scaler.update(new_scale=scaler.get_scale()/2)
    raise FloatingPointError("Repeated nonfinite gradients at one optimizer boundary")


def optimizer(model, config, stage, steps):
    rule = config["stages"][stage]
    opt = torch.optim.AdamW([{"params": values, "lr": rule[f"{name}_lr"]}
                             for name, values in model.groups().items() if f"{name}_lr" in rule],
                            weight_decay=config["weight_decay"], foreach=False)
    total = steps*rule["epochs"]
    warmup = max(1, int(total*config["warmup_fraction"]))
    plan = torch.optim.lr_scheduler.LambdaLR(opt, lambda i:
        (i+1)/warmup if i < warmup else .1+.9*(1+math.cos(math.pi*(i-warmup)/max(1,total-warmup)))/2)
    return opt, plan
