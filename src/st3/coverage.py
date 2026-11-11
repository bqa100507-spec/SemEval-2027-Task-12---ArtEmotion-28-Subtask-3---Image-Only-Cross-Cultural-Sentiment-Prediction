"""Label-free artwork geometry and frozen CLIP views for EXP-008."""

import importlib.metadata
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import data
from .baseline import file_sha256
from .frozen_features import cache_location, load_frozen_cache, normalize_rows


def letterbox(image, size, padding):
    """Fit the full RGB frame on a centered square using a single bicubic resize."""
    width, height = image.size
    scale = size / max(width, height)
    fitted = tuple(max(1, round(length * scale)) for length in (width, height))
    resized = image.convert("RGB").resize(fitted, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), tuple(padding))
    canvas.paste(resized, tuple((size - length) // 2 for length in fitted))
    return canvas


def dual_view(center, full):
    """Average the two unit projections and return a unit projection."""
    return normalize_rows((center + full) * np.float32(0.5))


def center_retained(width, height, size):
    # Transformers sets the short edge exactly and floors the resized long edge.
    short, long = sorted((width, height))
    return size / (size * long // short)


def image_geometry(root, paintings, config):
    rows = []
    for name in paintings["image_name"]:
        with Image.open(data.image_path(root, name)) as image:
            rows.append(image.size)
    result = paintings.reset_index(drop=True).join(pd.DataFrame(rows, columns=["width", "height"]))
    result["aspect_ratio"] = result["width"] / result["height"]
    size = config["canvas_size"]
    result["retained_fraction"] = [center_retained(w, h, size) for w, h in rows]
    result["substantially_cropped"] = result["retained_fraction"].lt(config["substantial_crop_below"])
    near_square = config["orientation_near_square_min"]
    result["orientation"] = np.select(
        [result["aspect_ratio"].lt(near_square), result["aspect_ratio"].gt(1 / near_square)],
        ["portrait", "landscape"], default="near-square")
    result["style_folder"] = result["image_name"].str.split("/").str[0]
    result["coverage_band"] = pd.cut(
        result["retained_fraction"], [0, *config["coverage_bin_edges"], np.inf],
        labels=["<65%", "65–<80%", "80–<95%", ">=95%"], right=False).astype(str)
    return result


def clip_assets(cache_dir, baseline, config, *, weights=False):
    from huggingface_hub import hf_hub_download

    files = {"preprocessor_config.json": config["clip_processor_sha256"]}
    if weights:
        files["pytorch_model.bin"] = config["clip_weights_sha256"]
    paths = {}
    for filename, digest in files.items():
        path = Path(hf_hub_download(baseline["clip_model"], filename, revision=baseline["clip_revision"],
                                    cache_dir=cache_dir / "huggingface"))
        if file_sha256(path) != digest:
            raise ValueError(f"Pinned CLIP asset hash differs for {filename}")
        paths[filename] = path
    return paths


def load_processor(checkpoint, config):
    from transformers import CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained(checkpoint)
    size = config["canvas_size"]
    if (not processor.do_resize or not processor.do_center_crop or not processor.do_normalize
            or not processor.do_rescale or processor.size != {"shortest_edge": size}
            or processor.crop_size != {"height": size, "width": size}
            or config["resample"] != "bicubic" or processor.resample != Image.Resampling.BICUBIC
            or [round(mean / processor.rescale_factor) for mean in processor.image_mean] != config["padding_rgb"]):
        raise ValueError("CLIP processor differs from the frozen EXP-008 preprocessing")
    return processor


def extract_coverage(root, names, original, cache_dir, baseline, config, center_sha256):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise ValueError("EXP-008 feature extraction requires CUDA")
    start = time.perf_counter()
    names = sorted(set(names))
    center = np.stack([original[name] for name in names])
    identity = {
        "model": baseline["clip_model"], "revision": baseline["clip_revision"], "width": 512,
        "batch_size": baseline["batch_size"], "device": torch.cuda.get_device_name(),
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "Pillow", "numpy")},
        "preprocessing": {key: config[key] for key in ("canvas_size", "resample", "padding_rgb")},
        "feature": "normalized-clip-letterbox-and-dual-projections-fp32-v1",
        "clip_weights_sha256": config["clip_weights_sha256"], "clip_processor_sha256": config["clip_processor_sha256"],
        "center_cache_sha256": center_sha256,
    }
    location = cache_location(root, names, cache_dir, identity, prefix="exp008-coverage")
    if not location.exists():
        from transformers import CLIPModel

        assets = clip_assets(cache_dir, baseline, config, weights=True)
        checkpoint = assets["pytorch_model.bin"].parent
        processor = load_processor(checkpoint, config)
        torch.manual_seed(baseline["seed"])
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.cuda.reset_peak_memory_stats()
        model = CLIPModel.from_pretrained(checkpoint, use_safetensors=False, low_cpu_mem_usage=True)
        model = model.eval().requires_grad_(False).to("cuda")
        batch_size = identity["batch_size"]
        decode_seconds = preprocess_seconds = encoder_seconds = 0.
        batches = []
        with torch.inference_mode():
            for offset in range(0, len(names), batch_size):
                tick = time.perf_counter()
                images = []
                for name in names[offset:offset + batch_size]:
                    with Image.open(data.image_path(root, name)) as image:
                        images.append(image.convert("RGB"))
                decode_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                frames = [letterbox(image, config["canvas_size"], config["padding_rgb"]) for image in images]
                inputs = processor(images=frames, do_resize=False, do_center_crop=False, return_tensors="pt")
                preprocess_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                output = model.get_image_features(**inputs.to("cuda"))
                batches.append(torch.nn.functional.normalize(output, dim=-1).cpu().numpy())
                encoder_seconds += time.perf_counter() - tick
                done = offset + len(images)
                if offset == 0 or (offset // batch_size + 1) % 10 == 0 or done == len(names):
                    print(f"EXP-008 full-image CLIP {done}/{len(names)}", flush=True)

        full = np.concatenate(batches)
        tick = time.perf_counter()
        dual = dual_view(center, full)
        dual_seconds = time.perf_counter() - tick
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_dir) as scratch:
            stage = Path(scratch) / "output"
            stage.mkdir()
            path = stage / "features.npz"
            np.savez(path, names=np.array(names), features=full, dual=dual)
            manifest = {"identity": identity, "features_sha256": file_sha256(path), "images": len(names),
                        "decode_seconds": decode_seconds, "preprocess_seconds": preprocess_seconds,
                        "encoder_seconds": encoder_seconds, "dual_seconds": dual_seconds,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "wall_seconds": time.perf_counter() - start}
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            stage.rename(location)

    full, manifest = load_frozen_cache(location, names, identity)
    with np.load(location / "features.npz", allow_pickle=False) as archive:
        dual = archive["dual"]
    if (dual.shape != center.shape or dual.dtype != np.float32
            or not np.allclose(np.linalg.norm(dual, axis=1), 1., rtol=0., atol=2e-6)):
        raise ValueError("Invalid EXP-008 dual-view feature cache")
    return full, dict(zip(names, dual)), manifest
