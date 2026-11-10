"""Encoder worker for resumable image-only inference in the pinned runtimes."""

import importlib.metadata
import io
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .baseline import clip_projection, file_sha256
from .cli import json_output
from .e50_heads import inference_config, read_json
from .frozen_features import normalize_rows
from .incremental import collect_features, digest_bytes


def load_rgb(digest, path):
    payload = path.read_bytes()
    if digest_bytes(payload) != digest:
        raise ValueError(f"Image changed during inference: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def encoder_loader(name, config, cache_dir, device):
    import torch
    from huggingface_hub import hf_hub_download

    files = {}
    for filename, expected in config["assets_sha256"].items():
        path = Path(hf_hub_download(config["model"], filename, revision=config["revision"],
                                    cache_dir=cache_dir / "huggingface"))
        if file_sha256(path) != expected:
            raise ValueError(f"Pinned {name} asset checksum mismatch for {filename}")
        files[filename] = path
    checkpoint = files["config.json"].parent
    if name == "clip":
        from transformers import CLIPImageProcessor, CLIPModel

        model = CLIPModel.from_pretrained(checkpoint, use_safetensors=False, low_cpu_mem_usage=True)
        processor = CLIPImageProcessor.from_pretrained(checkpoint)
    else:
        from transformers import AutoImageProcessor

        from .siglip2 import load_vision

        model = load_vision(checkpoint)
        processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=False)
    model.eval().requires_grad_(False).to(device)

    def encode(items):
        images = [load_rgb(digest, path) for digest, path in items]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            if name == "clip":
                return clip_projection(model, inputs).cpu().numpy()
            from .siglip2 import pooled_features

            return normalize_rows(pooled_features(model, inputs, config["width"]).cpu().numpy())

    return encode


def run_worker(name, request_path, output_dir):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    import torch

    start = time.perf_counter()
    request = read_json(request_path)
    config = inference_config()["encoders"][name]
    versions = {package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "Pillow", "numpy")}
    if versions["transformers"] != config["transformers"]:
        raise ValueError(f"{name} requires Transformers {config['transformers']} in its pinned environment")
    device = request["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA was requested for {name} but is unavailable")
    identity = {"version": 1, "encoder": name, "width": config["width"], "model": config["model"],
                "revision": config["revision"], "feature": config["feature"], "assets_sha256": config["assets_sha256"],
                "runtime": versions, "device": device,
                "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
                "cuda_runtime": torch.version.cuda if device == "cuda" else None}
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cache_dir = Path(request["cache_dir"])
    images = {digest: Path(path) for digest, path in request["images"].items()}
    vectors, meta = collect_features(
        images, cache_dir / "e50-incremental-v1", identity, request["batch_size"],
        lambda: encoder_loader(name, config, cache_dir, device),
        progress=lambda count: print(f"{name}: committed {count} new image features", flush=True))
    np.savez(output_dir / "vectors.npz", digests=np.array(list(images)),
             features=np.stack([vectors[digest] for digest in images]))
    json_output({**meta, "wall_seconds": time.perf_counter() - start}, output_dir / "features.json")
