"""Pinned vision-only NaFlex loading and fold-independent cache construction."""

import tempfile
from time import monotonic
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .exp033 import patch_coordinates, prefix, suffix
from .exp033_io import read_json, save_json, sha256


def load_model(config, directory, device):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import Siglip2Config, Siglip2ImageProcessor, Siglip2VisionModel

    if directory is None:
        directory = Path(snapshot_download(config["model"], revision=config["revision"],
                                           allow_patterns=list(config["assets"])))
    for name, fingerprint in config["assets"].items():
        if sha256(directory / name) != fingerprint:
            raise ValueError(f"Model asset checksum differs: {name}")
    vision_config = Siglip2Config.from_pretrained(directory, local_files_only=True).vision_config
    vision_config._attn_implementation = "eager"
    if (vision_config.hidden_size, vision_config.num_hidden_layers, vision_config.patch_size) != (
        config["width"], config["layers"], 16,
    ):
        raise ValueError("Unexpected pinned vision shape")
    model = Siglip2VisionModel(vision_config)
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as weights:
        state = {key: weights.get_tensor(key) for key in weights.keys() if key.startswith("vision_model.")}
    model.load_state_dict(state, strict=True)
    processor = Siglip2ImageProcessor.from_pretrained(directory, local_files_only=True)
    return model.vision_model.eval().requires_grad_(False).to(device), processor


@torch.no_grad()
def extract_batch(vision, processor, paths, config, device, *, parity):
    pictures = []
    started = monotonic()
    for path in paths:
        with Image.open(path) as image:
            pictures.append(image.convert("RGB"))
    batch = processor(images=pictures, max_num_patches=config["patches"], return_tensors="pt")
    load_seconds = monotonic() - started
    batch = batch.to(device)
    pixels, mask, shapes = batch["pixel_values"], batch["pixel_attention_mask"], batch["spatial_shapes"]
    patch_coordinates(mask, shapes)
    blocks = config["tail_blocks"]
    delta = {}
    with torch.autocast("cuda", dtype=torch.float16, enabled=torch.device(device).type == "cuda"):
        pre = prefix(vision, pixels, mask, shapes, blocks)
        frozen = suffix(vision, pre.float(), mask, blocks)
        global_vector = vision.head(frozen, mask)
        if parity:
            native = vision(pixel_values=pixels, attention_mask=mask, spatial_shapes=shapes)
            for name, a, b in (("tokens", frozen[mask.bool()], native.last_hidden_state[mask.bool()]),
                               ("global", global_vector, native.pooler_output)):
                torch.testing.assert_close(a.float(), b.float(), atol=2e-4, rtol=2e-4)
                delta[name] = float((a.float() - b.float()).abs().max())
    arrays = {"prefix": pre.float(), "tokens": frozen.float(), "global": global_vector.float(),
              "mask": mask.bool(), "shapes": shapes.long()}
    return {k: v.cpu().numpy() for k, v in arrays.items()}, delta, load_seconds


def cache_arrays(folder, identity):
    seal = read_json(folder / "seal.json")
    expected = {"prefix", "tokens", "global", "mask", "shapes"}
    if seal["identity"] != identity or set(seal["files"]) != expected:
        raise ValueError("Frozen cache identity differs")
    arrays = {}
    for key, fingerprint in seal["files"].items():
        path = folder / f"{key}.npy"
        if sha256(path) != fingerprint:
            raise ValueError(f"Corrupt cache: {key}")
        arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    return arrays


def build_cache(folder, identity, paths, vision, processor, config, device, microbatch, deadline):
    if folder.exists():
        return cache_arrays(folder, identity)
    with tempfile.TemporaryDirectory(dir=folder.parent) as scratch:
        staged = Path(scratch) / "complete"
        staged.mkdir()
        maps = {}
        for start in range(0, len(paths), microbatch):
            if monotonic() >= deadline:
                raise TimeoutError("Session budget reached during frozen extraction")
            batch, _, _ = extract_batch(vision, processor, paths[start:start + microbatch],
                                        config, device, parity=start == 0)
            for key, array in batch.items():
                if key not in maps:
                    maps[key] = np.lib.format.open_memmap(
                        staged / f"{key}.npy", mode="w+", dtype=array.dtype, shape=(len(paths), *array.shape[1:]),
                    )
                if not np.isfinite(array).all():
                    raise FloatingPointError(f"Nonfinite frozen cache block: {key}")
                maps[key][start:start + len(array)] = array
        keys = list(maps)
        for mapping in maps.values():
            mapping.flush()
        del mapping
        maps.clear()
        save_json(staged / "seal.json", {"identity": identity,
                                       "files": {key: sha256(staged / f"{key}.npy") for key in keys}})
        staged.rename(folder)
    return cache_arrays(folder, identity)
