"""Frozen training-caption teachers with progressive checksum-validated caches."""

import gc
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import file_sha256
from .caption_data import allowed_captions, text_digest
from .cli import json_output
from .contrast_study import new_output
from .incremental import collect_features, valid_vectors

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def teacher_texts(texts, name):
    return ["passage: " + text for text in texts] if name == "e5" else list(texts)


def masked_mean(hidden, mask):
    masked = hidden.masked_fill(~mask[..., None].bool(), 0.)
    return masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1)[..., None]


def text_projection(model, inputs, name):
    import torch.nn.functional as functional

    output = model(**inputs)
    values = output.text_embeds if name == "clip" else masked_mean(output.last_hidden_state, inputs["attention_mask"])
    return functional.normalize(values.float(), dim=1)


def teacher_assets(cache_dir, config):
    from huggingface_hub import hf_hub_download

    snapshot = cache_dir / "huggingface" / ("models--" + config["model"].replace("/", "--")) / "snapshots" / config["revision"]
    for filename, expected in config["assets_sha256"].items():
        path = snapshot / filename
        if not path.is_file():
            path = Path(hf_hub_download(config["model"], filename, revision=config["revision"],
                                        cache_dir=cache_dir / "huggingface"))
        if file_sha256(path) != expected:
            raise ValueError(f"Pinned text teacher asset checksum differs: {filename}")
    return snapshot


def load_text_model(snapshot, name):
    from transformers import AutoModel, CLIPTextModelWithProjection

    cls = CLIPTextModelWithProjection if name == "clip" else AutoModel
    model, info = cls.from_pretrained(snapshot, use_safetensors=True, low_cpu_mem_usage=True,
                                      attn_implementation="eager", output_loading_info=True)
    relevant = ("text_model.", "text_projection.") if name == "clip" else ("",)
    if (info["missing_keys"] or info["mismatched_keys"]
            or any(key.startswith(relevant) for key in info["unexpected_keys"])):
        raise ValueError(f"Incomplete text teacher loading: {info}")
    if name == "clip" and hasattr(model, "vision_model"):
        raise ValueError("Expected CLIP text-only loading")
    return model


def tokenize_diagnostics(tokenizer, texts, name, limit):
    records = []
    for start in range(0, len(texts), 256):
        selected = texts[start:start + 256]
        encoded = tokenizer(teacher_texts(selected, name), truncation=False, add_special_tokens=True)["input_ids"]
        for text, ids in zip(selected, encoded):
            content = ids[1:-1]
            records.append({"caption_id": text_digest(text), "tokens": len(ids), "truncated": len(ids) > limit,
                            "unknown_tokens": content.count(tokenizer.unk_token_id),
                            "tokens_per_character": len(content) / max(len(text), 1),
                            "tokens_per_whitespace_word": len(content) / max(len(text.split()), 1),
                            "replacement_characters": text.count("\ufffd"), "characters": len(text)})
    return pd.DataFrame(records)


def audit_training_tokens(args, captions, config, output):
    """Descriptive token counts for all training annotations, without model fitting."""
    from transformers import AutoTokenizer

    name = args.teacher
    snapshot = teacher_assets(args.cache_dir, config)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=True)
    valid = captions[captions.valid_caption]
    texts = valid[["caption_id", "caption"]].drop_duplicates().sort_values("caption_id")
    diagnostic = tokenize_diagnostics(tokenizer, texts.caption.tolist(), name, config["max_length"])
    joined = valid[["annotation_id", "language", "caption_id"]].merge(diagnostic, on="caption_id", validate="many_to_one")
    grouped = joined.groupby("language").agg(annotations=("tokens", "size"), tokens_mean=("tokens", "mean"),
        tokens_median=("tokens", "median"), tokens_max=("tokens", "max"), truncation_rate=("truncated", "mean"),
        unknown_tokens=("unknown_tokens", "sum"), token_character_ratio=("tokens_per_character", "mean"),
        token_word_ratio=("tokens_per_whitespace_word", "mean"))
    grouped.to_csv(output, lineterminator="\n")
    return grouped


def extract_teacher(args, captions, pairs, study_config, audit, output, *, feature_directory=None, progress_hook=None):
    import torch
    from transformers import AutoTokenizer

    start = time.perf_counter()
    name = args.teacher
    config = study_config["teachers"][name]
    selected = allowed_captions(captions, pairs)
    texts = selected[["caption_id", "caption"]].drop_duplicates().sort_values("caption_id")
    mapping = dict(texts.itertuples(index=False, name=None))
    snapshot = teacher_assets(args.cache_dir, config)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=True)
    diagnostics = tokenize_diagnostics(tokenizer, list(mapping.values()), name, config["max_length"])
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Text extraction requested CUDA without an available GPU")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    identity = {"encoder": "caption-" + name, **config, "device": device, "attention": "eager",
                "tokenizer": type(tokenizer).__name__, "prefix": "passage: " if name == "e5" else "",
                "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                "tokenizers": importlib.metadata.version("tokenizers"), "precision": "float32"}
    probe = {}
    fallbacks = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def load_encoder():
        model = load_text_model(snapshot, name).eval().requires_grad_(False).to(device)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("Frozen teacher must use FP32 weights")
        batch_size = config["batch_size"]

        @torch.inference_mode()
        def encode(batch):
            nonlocal batch_size
            values = teacher_texts([text for _, text in batch], name)
            if not probe:
                inputs = tokenizer(values[:2], padding=True, truncation=True, max_length=config["max_length"],
                                   return_tensors="pt").to(device)
                first = text_projection(model, inputs, name)
                second = text_projection(model, inputs, name)
                if not torch.equal(first, second):
                    raise ValueError("Repeated frozen text forwards differ")
                probe.update({"repeat_exact": True, "parameters": sum(p.numel() for p in model.parameters()),
                              "shape": list(first.shape), "dtype": str(first.dtype)})
            while True:
                try:
                    chunks = []
                    for index in range(0, len(values), batch_size):
                        inputs = tokenizer(values[index:index + batch_size], padding=True, truncation=True,
                                           max_length=config["max_length"], return_tensors="pt").to(device)
                        chunks.append(text_projection(model, inputs, name).cpu().numpy())
                    return np.concatenate(chunks)
                except torch.OutOfMemoryError:
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    fallbacks.append(batch_size)
                    del chunks
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

        return encode

    def progress(count):
        if progress_hook is not None:
            progress_hook(count)
        if count % 2048 < config["batch_size"] or count == len(mapping):
            print(f"{name}: cached {count}/{len(mapping)} captions", flush=True)

    with new_output(output) as stage:
        vectors, info = collect_features(mapping, feature_directory or args.cache_dir / "exp017_text", identity,
                                         config["batch_size"], load_encoder,
                                         progress=progress)
        keys = list(mapping)
        values = np.stack([vectors[key] for key in keys])
        valid_vectors(values, len(keys), config["width"])
        np.savez(stage / "features.npz", keys=np.array(keys), features=values)
        rows = selected[["annotation_id", "painting", "language", "emotion", "caption_id"]].merge(
            diagnostics, on="caption_id", validate="many_to_one")
        rows.to_csv(stage / "tokens.csv", index=False, lineterminator="\n")
        per_language = rows.groupby("language").agg(annotations=("tokens", "size"),
            tokens_mean=("tokens", "mean"), tokens_max=("tokens", "max"), truncation_rate=("truncated", "mean"),
            unknown_tokens=("unknown_tokens", "sum"), token_character_ratio=("tokens_per_character", "mean"),
            token_word_ratio=("tokens_per_whitespace_word", "mean"))
        per_language.to_csv(stage / "languages.csv", lineterminator="\n")
        center = values.mean(axis=0)
        norms = np.linalg.norm(values, axis=1)
        details = {"identity": identity, "config_sha256": audit["config_sha256"],
                   "protocol_sha256": audit["protocol_sha256"], "fit_paintings": sorted(pairs.painting.unique()),
                   "unique_texts": len(keys), "annotation_rows": len(selected), "probe": probe,
                   "features_sha256": file_sha256(stage / "features.npz"),
                   "database_sha256": file_sha256(Path(info["database"])),
                   "encoded": info["encoded"], "cache_hits": info["incremental_hits"],
                   "vector_norm_min": float(norms.min()), "vector_norm_max": float(norms.max()),
                   "centroid_norm": float(np.linalg.norm(center)),
                   "mean_component_variance": float(np.mean((values - center) ** 2)),
                   "duplicate_embeddings": len(values) - len({text_digest(row.tobytes().hex()) for row in values}),
                   "batch_fallbacks": fallbacks, "wall_seconds": time.perf_counter() - start,
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None}
        json_output(details, stage / "run.json")
    print(f"{name}: extracted {len(keys)} exact texts with unit FP32 features", flush=True)


def load_teacher(output, config, audit, allowed_paintings):
    manifest = json.loads((output / "run.json").read_text())
    if (manifest["features_sha256"] != file_sha256(output / "features.npz")
            or manifest["config_sha256"] != audit["config_sha256"]
            or manifest["protocol_sha256"] != audit["protocol_sha256"]
            or set(manifest["fit_paintings"]) != set(allowed_paintings)
            or any(manifest["identity"][key] != value for key, value in config.items())):
        raise ValueError("Text feature pins, checksum or allowed painting coverage differ")
    with np.load(output / "features.npz", allow_pickle=False) as archive:
        keys, vectors = archive["keys"].tolist(), archive["features"]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate caption cache keys")
    valid_vectors(vectors, len(keys), config["width"])
    return dict(zip(keys, vectors)), manifest
