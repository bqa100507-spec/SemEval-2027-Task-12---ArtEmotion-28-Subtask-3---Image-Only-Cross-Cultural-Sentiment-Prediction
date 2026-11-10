"""Pinned DINOv2 CLS extraction with a checksum-validated feature cache."""

import gc
import importlib.metadata
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .baseline import feature_cache_path, file_sha256
from .data import image_path


def cls_features(hidden_states, width):
    if hidden_states.ndim != 3 or hidden_states.shape[1] < 1 or hidden_states.shape[2] != width:
        raise ValueError(f"Expected a batched token sequence of width {width}")
    return hidden_states[:, 0, :]


def normalize_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Expected finite, nonzero image features")
    return features / norms


def cache_location(root, names, cache_dir, identity):
    fingerprint = feature_cache_path(root, names, cache_dir, identity).stem.removeprefix("clip-")
    return cache_dir / f"dino-{fingerprint}"


def load_dino_cache(location, names, identity):
    manifest = json.loads((location / "manifest.json").read_text(encoding="utf-8"))
    path = location / "features.npz"
    if manifest["identity"] != identity or manifest["features_sha256"] != file_sha256(path):
        raise ValueError(f"DINOv2 cache identity or feature hash mismatch: {location}")
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"]
        if archive["names"].tolist() != names or features.shape != (len(names), identity["width"]):
            raise ValueError(f"DINOv2 cache has unexpected coverage: {location}")
        if features.dtype != np.float32 or not np.allclose(np.linalg.norm(features, axis=1), 1., rtol=0., atol=2e-6):
            raise ValueError(f"DINOv2 cache contains invalid normalized features: {location}")
    return dict(zip(names, features)), {**manifest, "path": str(location)}


def extract_dino(root, names, cache_dir, config, requested_device="auto"):
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "huggingface"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    start = time.perf_counter()
    initial_device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    if initial_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if config["batch_size"] < 1:
        raise ValueError("Batch size must be positive")
    names = sorted(set(names))
    if not names:
        raise ValueError("Expected at least one image")
    identity = {"model": config["dino_model"], "revision": config["dino_revision"], "width": config["dino_width"],
                "batch_size": config["batch_size"], "feature": "final-cls-l2-fp32-v1",
                "attention": "eager", "processor": "slow", "initial_device": initial_device,
                "device_name": torch.cuda.get_device_name() if initial_device == "cuda" else "CPU",
                "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "Pillow", "numpy")}}
    location = cache_location(root, names, cache_dir, identity)
    if location.is_dir():
        print(f"Reused {len(names)} checksum-validated DINOv2 features from {location}", flush=True)
        return load_dino_cache(location, names, identity)

    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, AutoModel

    assets = {name: Path(hf_hub_download(identity["model"], name, revision=identity["revision"]))
              for name in ("config.json", "preprocessor_config.json", "model.safetensors")}
    checkpoint = assets["model.safetensors"].parent
    model = AutoModel.from_pretrained(checkpoint, use_safetensors=True, attn_implementation="eager",
                                     low_cpu_mem_usage=True).eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=False)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = initial_device
    batch_size = identity["batch_size"]
    changes = []
    if initial_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        model.to(device)
    except torch.OutOfMemoryError:
        if requested_device != "auto" or device != "cuda":
            raise
        changes.append({"completed_images": 0, "reason": "CUDA OOM loading model", "device": "cpu"})
        device = "cpu"
    if device != initial_device:
        model.to(device)
        gc.collect()
        torch.cuda.empty_cache()

    def embed(paths):
        images = []
        for name in paths:
            with Image.open(image_path(root, name)) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            return normalize_rows(cls_features(hidden, identity["width"]).cpu().numpy())

    encoder_start = time.perf_counter()
    features = np.empty((len(names), identity["width"]), dtype=np.float32)
    offset = 0
    print(f"Extracting DINOv2 on {device}, batch size {batch_size}", flush=True)
    next_progress = 160
    while offset < len(names):
        end = min(offset + batch_size, len(names))
        try:
            features[offset:end] = embed(names[offset:end])
        except torch.OutOfMemoryError:
            if device != "cuda" or (batch_size == 1 and requested_device != "auto"):
                raise
        else:
            offset = end
            if offset >= next_progress or offset == len(names):
                print(f"Embedded {offset}/{len(names)} images", flush=True)
                next_progress = offset + 160
            continue
        gc.collect()
        torch.cuda.empty_cache()
        if batch_size > 1:
            batch_size = max(1, batch_size // 2)
        else:
            device = "cpu"
            model.to(device)
            torch.cuda.empty_cache()
        changes.append({"completed_images": offset, "reason": "CUDA OOM", "device": device, "batch_size": batch_size})
        print(f"Retrying from image {offset} on {device}, batch size {batch_size}", flush=True)

    extraction_seconds = time.perf_counter() - encoder_start
    memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()} if initial_device == "cuda" else None
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as scratch:
        stage = Path(scratch) / "output"
        stage.mkdir()
        path = stage / "features.npz"
        np.savez(path, names=np.array(names), features=features)
        manifest = {"identity": identity, "features_sha256": file_sha256(path), "images": len(names),
                    "assets_sha256": {name: file_sha256(file) for name, file in assets.items()},
                    "processor_class": type(processor).__name__,
                    "processor_config": json.loads(assets["preprocessor_config.json"].read_text(encoding="utf-8")),
                    "final_device": device, "final_batch_size": batch_size, "fallbacks": changes,
                    "extraction_seconds": extraction_seconds, "wall_seconds": time.perf_counter() - start,
                    "cuda_memory": memory}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(location)
    return load_dino_cache(location, names, identity)


def run_visual_embed(args, dev):
    from .low_rank_study import study_inputs

    config, _, train, _, _ = study_inputs(args, dev, "exp005")
    _, metadata = extract_dino(args.data_dir, set(train["image"]) | set(dev["image"]), args.cache_dir, config, args.device)
    print(json.dumps(metadata, indent=2))
