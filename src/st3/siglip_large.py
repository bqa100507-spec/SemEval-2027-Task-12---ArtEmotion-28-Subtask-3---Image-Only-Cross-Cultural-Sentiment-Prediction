"""Vision-only SigLIP 2 Large extraction through the existing frozen cache."""

import os

from .frozen_features import cache_location, extract_frozen, load_frozen_cache, verify_asset_hashes
from .siglip2 import load_vision, pooled_features

PREFIX = "siglip2-large"


def load_large_vision(checkpoint, config):
    import torch

    model = load_vision(checkpoint)
    if (model.config.hidden_size != config["width"] or model.config.image_size != config["image_size"]
            or model.config.patch_size != 16
            or sum(p.numel() for p in model.parameters()) != config["vision_parameters"]
            or hasattr(model, "text_model") or any(p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError("SigLIP 2 Large vision-only model differs from the pinned FP32 configuration")
    return model


def extract_large(root, names, cache_dir, config, *, feature_dir=None, probe=None):
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "huggingface"))

    def encode(model, inputs, width):
        values = pooled_features(model, inputs, width)
        if probe is not None:
            import torch

            repeated = pooled_features(model, inputs, width)
            if not torch.equal(values, repeated):
                raise ValueError("Repeated SigLIP 2 Large FP32 forwards are not bit-identical")
            if tuple(inputs["pixel_values"].shape[1:]) != (3, config["image_size"], config["image_size"]):
                raise ValueError("Pinned SigLIP 2 Large processor returned an unexpected image size")
            probe.update({"repeat_exact": True, "input_shape": list(inputs["pixel_values"].shape),
                          "output_shape": list(values.shape), "output_dtype": str(values.dtype)})
        return values

    return extract_frozen(root, names, feature_dir or cache_dir, config, "cuda", prefix=PREFIX,
                          load_model=lambda path: load_large_vision(path, config), encode=encode)


def load_large_cache(root, names, cache_dir, config, saved):
    names = sorted(set(names))
    identity = saved["identity"]
    pins = ("model", "revision", "width", "batch_size", "feature", "assets_sha256")
    if (any(identity[key] != config[key] for key in pins)
            or identity["runtime"]["transformers"] != config["transformers_version"]
            or identity["initial_device"] != "cuda" or identity["attention"] != "eager"
            or identity["processor"] != "slow"):
        raise ValueError("SigLIP 2 Large extraction manifest differs from the pinned encoder")
    location = cache_location(root, names, cache_dir, identity, PREFIX)
    vectors, manifest = load_frozen_cache(location, names, identity)
    verify_asset_hashes(manifest["assets_sha256"], config)
    if manifest["features_sha256"] != saved["features_sha256"] or manifest["final_device"] != "cuda":
        raise ValueError("SigLIP 2 Large feature checksum or execution device differs")
    return vectors, manifest
