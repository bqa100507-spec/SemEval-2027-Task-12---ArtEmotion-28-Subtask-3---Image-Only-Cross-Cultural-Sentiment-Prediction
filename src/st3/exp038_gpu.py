"""Pinned frozen encoders for EXP-038 and chunked, sealed feature extraction.

Heavy imports (torch, transformers, qwen_vl_utils) stay inside functions so CPU tests can import this
module with mocked encoders.
"""

import hashlib
import time
import unicodedata
from pathlib import Path

import numpy as np

from .exp036_io import check
from .exp038_io import BlockedCache, BudgetStop, load_features, save_features

SIGLIP_FILES = ("model.safetensors",)


def git_blob_sha1(path):
    content = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content, usedforsecurity=False).hexdigest()


def verify_snapshot(directory, spec):
    directory = Path(directory)
    check(directory, spec["weights_sha256"])
    for name, expected in spec["metadata_git_blobs"].items():
        path = directory / name
        if not path.is_file() or git_blob_sha1(path) != expected:
            raise ValueError(f"Pinned metadata differs for {spec['model']}: {name}")
    return {"model": spec["model"], "revision": spec["revision"], "weights_sha256": spec["weights_sha256"],
            "metadata_git_blobs": spec["metadata_git_blobs"], "path": str(directory)}


def fetch_snapshot(spec, local=None):
    """Download only the pinned files at the pinned commit, then verify every hash."""
    if local is None:
        from huggingface_hub import snapshot_download

        local = snapshot_download(spec["model"], revision=spec["revision"], token=False,
                                  allow_patterns=[*spec["weights_sha256"], *spec["metadata_git_blobs"]])
    return Path(local), verify_snapshot(local, spec)


def load_rgb(path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def deterministic_cuda():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalized(tensor):
    import torch.nn.functional as F

    values = F.normalize(tensor.float(), dim=-1).cpu().numpy().astype(np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite embedding")
    return values


class SiglipEncoder:
    """E1: EXP-036's SigLIP2 So400m processor and pooled vision output."""

    def __init__(self, spec, directory, device):
        from transformers import AutoImageProcessor

        from .exp036_train import backbone

        visual = {"hidden_size": spec["width"], "layers": 27, "image_size": 384, "patch_size": 14}
        self.model = backbone(directory, visual, text=False).to(device)
        self.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True, use_fast=False)
        self.device = device

    def __call__(self, images, instructions=None):
        import torch

        from .exp036_train import amp

        pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        with torch.inference_mode(), amp(self.device):
            pooled = self.model.vision_model(pixel_values=pixels).pooler_output
        return normalized(pooled)


class MetaClipEncoder:
    """E2: MetaCLIP 2 worldwide ViT-H/14 projected image embeddings."""

    def __init__(self, spec, directory, device):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        model, info = AutoModel.from_pretrained(directory, dtype=torch.float16, attn_implementation="sdpa",
                                                local_files_only=True, output_loading_info=True)
        if type(model).__name__ != "MetaClip2Model" or any(info.get(k) for k in ("missing_keys", "mismatched_keys",
                                                                                  "error_msgs")):
            raise ValueError(f"MetaCLIP 2 loading failed: {type(model).__name__} {info}")
        if model.config.projection_dim != spec["width"]:
            raise ValueError("Unexpected MetaCLIP 2 projection width")
        self.model = model.to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True, use_fast=False)
        self.device = device

    def __call__(self, images, instructions=None):
        import torch

        pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device, torch.float16)
        with torch.inference_mode():
            embeds = self.model.get_image_features(pixel_values=pixels)
        return normalized(embeds)


def official_instruction(text):
    """The pinned script appends a period when an instruction lacks final punctuation."""
    text = text.strip()
    if text and not unicodedata.category(text[-1]).startswith("P"):
        text += "."
    return text


def qwen_conversation(image, instruction, spec):
    return [{"role": "system", "content": [{"type": "text", "text": official_instruction(instruction)}]},
            {"role": "user", "content": [{"type": "image", "image": image, "min_pixels": spec["min_pixels"],
                                          "max_pixels": spec["max_pixels"]}]}]


def last_token(hidden, attention):
    """Official last-token pooling for right-padded batches."""
    import torch

    flipped = attention.flip(dims=[1])
    column = attention.shape[1] - flipped.argmax(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), column]


class QwenEmbedder:
    """E3/E4: Qwen3-VL-Embedding-2B following scripts/qwen3_vl_embedding.py at the pinned revision."""

    def __init__(self, spec, directory, device):
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        placement = spec.get("placement", "single")
        options = {"dtype": torch.float16, "attn_implementation": spec["attention"], "local_files_only": True,
                   "output_loading_info": True}
        if placement == "two_gpu":
            # FP16 weights split across both visible GPUs; the only change from E3 is model size.
            options |= {"device_map": "auto",
                        "max_memory": {i: f"{spec['max_memory_gib']}GiB" for i in range(torch.cuda.device_count())}}
        elif placement == "nf4":
            from transformers import BitsAndBytesConfig

            options |= {"device_map": {"": 0}, "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)}
        elif placement != "single":
            raise ValueError(f"Unknown Qwen placement {placement}")
        model, info = Qwen3VLForConditionalGeneration.from_pretrained(directory, **options)
        if any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Qwen3-VL-Embedding loading failed: {info}")
        if model.config.text_config.hidden_size != spec["width"]:
            raise ValueError("Unexpected Qwen3-VL-Embedding width")
        self.model = (model.to(device) if placement == "single" else model).eval()
        if placement != "single":
            device = model.get_input_embeddings().weight.device
        self.placement = placement
        self.processor = Qwen3VLProcessor.from_pretrained(directory, local_files_only=True, padding_side="right")
        self.spec, self.device = spec, device

    def __call__(self, images, instructions):
        import torch
        from qwen_vl_utils.vision_process import process_vision_info

        conversations = [qwen_conversation(image, text, self.spec) for image, text in zip(images, instructions,
                                                                                         strict=True)]
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        pictures, videos, video_kwargs = process_vision_info(conversations, image_patch_size=16,
                                                             return_video_metadata=True, return_video_kwargs=True)
        if videos is not None:
            raise ValueError("Unexpected video input")
        inputs = self.processor(text=text, images=pictures, videos=None, video_metadata=None, truncation=True,
                                max_length=self.spec["max_length"], padding=True, do_resize=False,
                                return_tensors="pt", **video_kwargs)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = self.model.model(**inputs).last_hidden_state
        return normalized(last_token(hidden, inputs["attention_mask"]))


def encoder(arm, spec, directory, device):
    kind = spec["kind"]
    if kind == "siglip2":
        return SiglipEncoder(spec, directory, device)
    if kind == "metaclip2":
        return MetaClipEncoder(spec, directory, device)
    if kind in {"qwen3_vl_embedding", "qwen3_vl_embedding_language", "qwen3_vl_embedding_pool"}:
        return QwenEmbedder(spec, directory, device)
    raise ValueError(f"Unknown encoder kind for {arm}")


def instructions_for(arm, spec, items):
    if arm == "E4":
        return [spec["instruction_template"].format(language=item[2]) for item in items]
    if spec.get("instructions"):
        # Instruction pools: item[2] is the index of the instruction for that item.
        return [spec["instructions"][int(item[2])] for item in items]
    if spec.get("instruction"):
        return [spec["instruction"]] * len(items)
    return None


def embed_items(arm, spec, embed, items, batch):
    rows = []
    for start in range(0, len(items), batch):
        part = items[start:start + batch]
        images = [load_rgb(item[1]) for item in part]
        rows.append(embed(images, instructions_for(arm, spec, part)))
    values = np.concatenate(rows) if rows else np.zeros((0, spec["width"]), dtype=np.float32)
    if values.shape != (len(items), spec["width"]) or not np.isfinite(values).all():
        raise FloatingPointError(f"{arm} returned invalid embeddings")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1, atol=1e-3):
        raise FloatingPointError(f"{arm} embeddings are not unit norm")
    return values


def extract(arm, spec, embed, items, folder, identity, chunk_items, deadline, on_chunk=None, cap_deadline=None):
    """Sealed chunks, then one sealed cache. Completed chunks are reused; corrupt ones block."""
    keys = [item[0] for item in items]
    if (folder / "seal.json").is_file():
        return load_features(folder, identity, keys)
    if folder.exists() and not (folder / "chunks").is_dir() and any(folder.iterdir()):
        raise BlockedCache(f"Unsealed feature cache without chunks: {folder}")
    started = time.monotonic()
    pieces = []
    for start in range(0, len(items), chunk_items):
        part = items[start:start + chunk_items]
        target = folder / "chunks" / f"{start:07d}"
        chunk_identity = identity | {"chunk_start": start, "chunk_rows": len(part)}
        if target.exists():
            pieces.append(load_features(target, chunk_identity, [item[0] for item in part]))
            continue
        now = time.time()
        if cap_deadline is not None and now >= cap_deadline:
            raise BudgetStop(f"{arm} reached its extraction cap after {start} of {len(items)} items")
        if now >= deadline:
            raise TimeoutError(f"Session allowance reached during {arm} after {start} of {len(items)} items")
        values = embed_items(arm, spec, embed, part, spec["batch"])
        save_features(target, [item[0] for item in part], values, chunk_identity)
        pieces.append(values)
        if on_chunk is not None:
            on_chunk(arm, start + len(part), len(items), time.monotonic() - started)
    values = np.concatenate(pieces) if pieces else np.zeros((0, spec["width"]), dtype=np.float32)
    import shutil

    shutil.rmtree(folder / "chunks", ignore_errors=True)
    save_features(folder, keys, values, identity)
    return load_features(folder, identity, keys)


def benchmark(arm, spec, embed, items, device=None):
    """Two timed passes over the same representative real items; returns throughput and determinism."""
    peak = None
    if device is not None:
        import torch

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    tick = time.monotonic()
    first = embed_items(arm, spec, embed, items, spec["batch"])
    middle = time.monotonic()
    second = embed_items(arm, spec, embed, items, spec["batch"])
    end = time.monotonic()
    if device is not None:
        import torch

        torch.cuda.synchronize(device)
        peak = {"allocated_bytes": torch.cuda.max_memory_allocated(device),
                "reserved_bytes": torch.cuda.max_memory_reserved(device),
                "total_bytes": torch.cuda.get_device_properties(device).total_memory}
    difference = float(np.abs(first - second).max()) if len(items) else 0.
    return {"arm": arm, "items": len(items), "first_pass_seconds": middle - tick, "second_pass_seconds": end - middle,
            "items_per_second": len(items) / max(end - middle, 1e-9), "bitwise_deterministic": bool(
                np.array_equal(first, second)), "max_abs_difference": difference,
            "deterministic_within_1e-3": difference <= 1e-3, "peak_memory": peak}
