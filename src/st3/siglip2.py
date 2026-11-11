"""The fixed-resolution SigLIP 2 checkpoint's documented visual embedding."""

import json

from .cli import json_output
from .contrast_study import new_output
from .frozen_features import cache_location, extract_frozen, load_frozen_cache
from .low_rank_study import study_inputs

FEATURE = "siglip-attention-pool-l2-fp32-v1"
PREFIX = "siglip2"


def load_vision(checkpoint):
    from transformers import SiglipVisionModel

    model, info = SiglipVisionModel.from_pretrained(checkpoint, use_safetensors=True, attn_implementation="eager",
                                                   low_cpu_mem_usage=True, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"] or not model.vision_model.use_head:
        raise ValueError(f"SigLIP vision tower or trained pooling head did not load: {info}")
    return model


def pooled_features(model, inputs, width):
    # In the pinned SiglipModel source, get_image_features returns this directly.
    return model(**inputs).pooler_output


def extract_siglip2(root, names, cache_dir, config, requested_device="auto"):
    return extract_frozen(root, names, cache_dir, {**config, "feature": FEATURE}, requested_device,
                          prefix=PREFIX, load_model=load_vision, encode=pooled_features)


def load_siglip2_features(args, dev, train, config):
    pointer = args.run_dir / "extraction" / "features.json"
    saved = json.loads(pointer.read_text(encoding="utf-8"))
    identity = saved["identity"]
    names = sorted(set(train["image"]) | set(dev["image"]))
    expected = {"model": config["model"], "revision": config["revision"], "width": config["width"],
                "batch_size": config["batch_size"], "feature": FEATURE}
    if any(identity[key] != value for key, value in expected.items()) or \
            identity["runtime"]["transformers"] != config["transformers_version"]:
        raise ValueError("Extraction manifest differs from the pinned SigLIP 2 settings")
    location = cache_location(args.data_dir, names, args.cache_dir, identity, PREFIX)
    features, manifest = load_frozen_cache(location, names, identity)
    if manifest["features_sha256"] != saved["features_sha256"] or \
            manifest["assets_sha256"]["model.safetensors"] != config["weights_sha256"]:
        raise ValueError("SigLIP 2 cache or checkpoint hash differs from its pinned manifest")
    return features, manifest


def run_siglip_embed(args, dev):
    with new_output(args.run_dir / "extraction") as stage:
        config, _, train, _, _ = study_inputs(args, dev, "exp006")
        _, metadata = extract_siglip2(args.data_dir, set(train["image"]) | set(dev["image"]),
                                     args.cache_dir, config, args.device)
        json_output(metadata, stage / "features.json")
