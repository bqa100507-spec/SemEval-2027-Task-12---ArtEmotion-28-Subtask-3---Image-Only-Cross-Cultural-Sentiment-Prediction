"""Pinned DINOv2 CLS features used in EXP-005."""

import json

from .frozen_features import cache_location, cls_features, extract_frozen, load_frozen_cache as load_dino_cache, normalize_rows


def extract_dino(root, names, cache_dir, config, requested_device="auto"):
    def load(checkpoint):
        from transformers import AutoModel

        return AutoModel.from_pretrained(checkpoint, use_safetensors=True, attn_implementation="eager",
                                         low_cpu_mem_usage=True)

    def encode(model, inputs, width):
        return cls_features(model(**inputs).last_hidden_state, width)

    spec = {"model": config["dino_model"], "revision": config["dino_revision"], "width": config["dino_width"],
            "batch_size": config["batch_size"], "feature": "final-cls-l2-fp32-v1"}
    return extract_frozen(root, names, cache_dir, spec, requested_device, prefix="dino", load_model=load, encode=encode)


def run_visual_embed(args, dev):
    from .low_rank_study import study_inputs

    config, _, train, _, _ = study_inputs(args, dev, "exp005")
    _, metadata = extract_dino(args.data_dir, set(train["image"]) | set(dev["image"]), args.cache_dir, config, args.device)
    print(json.dumps(metadata, indent=2))
