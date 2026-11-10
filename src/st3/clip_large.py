"""Frozen CLIP ViT-L/14 vision projection and checksum-validated extraction."""

from .frozen_features import extract_frozen


def load_vision(checkpoint):
    from transformers import CLIPVisionModelWithProjection

    model, info = CLIPVisionModelWithProjection.from_pretrained(
        checkpoint, use_safetensors=True, low_cpu_mem_usage=True, attn_implementation="eager",
        output_loading_info=True)
    if (info["missing_keys"] or info["mismatched_keys"] or
            any(key.startswith(("vision_model.", "visual_projection.")) for key in info["unexpected_keys"])):
        raise ValueError(f"CLIP-L vision tower or trained projection did not load: {info}")
    return model


def projected_features(model, inputs, width):
    return model(**inputs).image_embeds


def extract_clip_large(root, names, cache_dir, config, device="cuda"):
    return extract_frozen(root, names, cache_dir, config, device, prefix="clip-large",
                          load_model=load_vision, encode=projected_features)
