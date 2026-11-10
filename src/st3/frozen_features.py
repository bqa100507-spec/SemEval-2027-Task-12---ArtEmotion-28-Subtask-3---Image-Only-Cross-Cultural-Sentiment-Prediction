"""Image-only frozen extraction with checksum-validated feature caches."""

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


def cache_location(root, names, cache_dir, identity, prefix="dino"):
    fingerprint = feature_cache_path(root, names, cache_dir, identity).stem.removeprefix("clip-")
    return cache_dir / f"{prefix}-{fingerprint}"


def load_frozen_cache(location, names, identity):
    manifest = json.loads((location / "manifest.json").read_text(encoding="utf-8"))
    path = location / "features.npz"
    if manifest["identity"] != identity or manifest["features_sha256"] != file_sha256(path):
        raise ValueError(f"Frozen cache identity or feature hash mismatch: {location}")
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"]
        if archive["names"].tolist() != names or features.shape != (len(names), identity["width"]):
            raise ValueError(f"Frozen cache has unexpected coverage: {location}")
        if features.dtype != np.float32 or not np.allclose(np.linalg.norm(features, axis=1), 1., rtol=0., atol=2e-6):
            raise ValueError(f"Frozen cache contains invalid normalized features: {location}")
    return dict(zip(names, features)), {**manifest, "path": str(location)}


def extract_frozen(root, names, cache_dir, config, requested_device, *, prefix, load_model, encode):
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
    identity = {"model": config["model"], "revision": config["revision"], "width": config["width"],
                "batch_size": config["batch_size"], "feature": config["feature"],
                "attention": "eager", "processor": "slow", "initial_device": initial_device,
                "device_name": torch.cuda.get_device_name() if initial_device == "cuda" else "CPU",
                "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "Pillow", "numpy")}}
    required_version = config.get("transformers_version")
    if required_version and identity["runtime"]["transformers"] != required_version:
        raise ValueError(f"Extraction requires Transformers {required_version} in the dedicated environment")
    location = cache_location(root, names, cache_dir, identity, prefix)
    if location.is_dir():
        vectors, manifest = load_frozen_cache(location, names, identity)
        if config.get("weights_sha256") and manifest["assets_sha256"]["model.safetensors"] != config["weights_sha256"]:
            raise ValueError("Cached features disagree with the pinned weight hash")
        print(f"Reused {len(names)} checksum-validated {prefix} features from {location}", flush=True)
        return vectors, manifest

    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor

    assets = {name: Path(hf_hub_download(identity["model"], name, revision=identity["revision"]))
              for name in ("config.json", "preprocessor_config.json", "model.safetensors")}
    asset_hashes = {name: file_sha256(file) for name, file in assets.items()}
    if config.get("weights_sha256") and asset_hashes["model.safetensors"] != config["weights_sha256"]:
        raise ValueError("Downloaded checkpoint disagrees with the pinned weight hash")
    checkpoint = assets["model.safetensors"].parent
    model = load_model(checkpoint).eval().requires_grad_(False)
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
            features = encode(model, inputs, identity["width"])
            if features.shape != (len(paths), identity["width"]):
                raise ValueError(f"Unexpected image feature shape: {features.shape}")
            return normalize_rows(features.cpu().numpy())

    encoder_start = time.perf_counter()
    features = np.empty((len(names), identity["width"]), dtype=np.float32)
    offset = 0
    print(f"Extracting {prefix} on {device}, batch size {batch_size}", flush=True)
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
                    "assets_sha256": asset_hashes,
                    "processor_class": type(processor).__name__,
                    "processor_config": json.loads(assets["preprocessor_config.json"].read_text(encoding="utf-8")),
                    "final_device": device, "final_batch_size": batch_size, "fallbacks": changes,
                    "extraction_seconds": extraction_seconds, "wall_seconds": time.perf_counter() - start,
                    "cuda_memory": memory}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(location)
    return load_frozen_cache(location, names, identity)
