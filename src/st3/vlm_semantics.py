"""Image-only descriptions and complete-token CLIP-L projections, cached per artwork."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def semantic_identity(config):
    return {key: config[key] for key in ("vlm", "prompt", "decoding", "text_encoder", "runtime")}


class SemanticCache:
    def __init__(self, directory, config):
        self.identity = semantic_identity(config)
        self.directory = Path(directory) / fingerprint(self.identity)[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "identity.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != self.identity:
            raise ValueError("Semantic cache identity differs")
        atomic_json(path, self.identity)

    def read(self, digest):
        path = self.directory / (digest + ".json")
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row["image_sha256"] != digest or row["identity"] != fingerprint(self.identity)
                or row["description_sha256"] != hashlib.sha256(row["description"].encode()).hexdigest()
                or not row["description"].strip()):
            raise ValueError("Corrupt or incompatible semantic record: " + digest)
        if "embedding" in row:
            vector = np.asarray(row["embedding"], dtype=np.float32)
            if (vector.shape != (768,) or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1., atol=1e-5)
                    or hashlib.sha256(vector.tobytes()).hexdigest() != row["embedding_sha256"]
                    or row["tokens_encoded"] != row["clip_tokens"]):
                raise ValueError("Invalid semantic embedding/token coverage")
        return row

    def save(self, digest, row):
        value = {**row, "image_sha256": digest, "identity": fingerprint(self.identity),
                 "description_sha256": hashlib.sha256(row["description"].encode()).hexdigest()}
        if "embedding" in value:
            value["embedding_sha256"] = hashlib.sha256(np.asarray(value["embedding"], dtype=np.float32).tobytes()).hexdigest()
        atomic_json(self.directory / (digest + ".json"), value)
        return self.read(digest)


def training_images(inputs, expected_hash):
    """Read image identifiers only, never the annotation/caption files."""
    inputs = Path(inputs)
    protocol = inputs / "training/protocol.csv"
    if file_hash(protocol) != expected_hash:
        raise ValueError("Original P2 protocol checksum differs")
    rows = pd.read_csv(protocol)
    rows = rows[rows.role.eq("development")].sort_values("painting").reset_index(drop=True)
    if rows.painting.duplicated().any() or rows.image_sha256.duplicated().any() or len(rows) != 1335:
        raise ValueError("Expected exactly 1335 distinct P2 training paintings/images")
    sizes = []
    for row in rows.itertuples():
        path = (inputs / "artwork/images" / row.image).resolve()
        if not path.is_relative_to((inputs / "artwork/images").resolve()) or file_hash(path) != row.image_sha256:
            raise ValueError("Artwork path/hash mismatch")
        with Image.open(path) as image:
            sizes.append(image.width / image.height)
    return rows.assign(aspect_ratio=sizes)


def smoke_sample(images, count=30, seed=252027):
    """Aspect-ratio strata plus fixed hash sampling; no labels or art metadata."""
    ranked = images.sort_values(["aspect_ratio", "image_sha256"])
    groups = np.array_split(np.arange(len(ranked)), count)
    return pd.DataFrame([min((ranked.iloc[i] for i in group),
                            key=lambda r: fingerprint([seed, r.image_sha256])) for group in groups]).reset_index(drop=True)


def token_chunks(token_ids, capacity=75):
    if not token_ids or capacity <= 0:
        raise ValueError("Cannot encode empty text or an invalid token capacity")
    chunks = [token_ids[i:i + capacity] for i in range(0, len(token_ids), capacity)]
    if [token for chunk in chunks for token in chunk] != token_ids:
        raise AssertionError("Text token coverage lost")
    return chunks


def encode_description(description, tokenizer, model):
    import torch
    ids = tokenizer(description, add_special_tokens=False, truncation=False)["input_ids"]
    chunks = token_chunks(ids)
    tokens = [[tokenizer.bos_token_id, *part, tokenizer.eos_token_id] for part in chunks]
    padded = [row + [tokenizer.pad_token_id] * (77 - len(row)) for row in tokens]
    mask = [[1] * len(row) + [0] * (77 - len(row)) for row in tokens]
    with torch.inference_mode():
        vectors = model(input_ids=torch.tensor(padded), attention_mask=torch.tensor(mask)).text_embeds
        vectors = vectors / vectors.norm(dim=1, keepdim=True)
        weights = torch.tensor([len(chunk) for chunk in chunks], dtype=vectors.dtype)
        result = (vectors * weights[:, None]).sum(0) / weights.sum()
        result = result / result.norm()
    return result.numpy().astype(np.float32), {"clip_tokens": len(ids), "tokens_encoded": sum(map(len, chunks)), "chunks": len(chunks)}


def load_vlm(config, model_path):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    vlm = config["vlm"]
    if vlm["quantization"] not in ("nf4", "fp16"):
        raise ValueError("Unsupported precision; no automatic fallback")
    if vlm["quantization"] == "fp16" and torch.cuda.mem_get_info()[0] < 10 * 1024 ** 3:
        raise RuntimeError("FP16 needs a safe 10GiB free-VRAM margin; use Colab L4 or explicitly choose the separate NF4 config")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=vlm["double_quant"], bnb_4bit_compute_dtype=torch.float16,
                                     llm_int8_skip_modules=vlm["skip_modules"]) if vlm["quantization"] == "nf4" else None
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False,
                         min_pixels=vlm["min_pixels"], max_pixels=vlm["max_pixels"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16,
                            device_map={"": 0}, quantization_config=quantization, attn_implementation=vlm["attention"]).eval()
    return processor, model


def describe(image, processor, model, config):
    import torch
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": config["prompt"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(**batch, **config["decoding"])
    output = generated[0, batch.input_ids.shape[1]:]
    description = processor.tokenizer.decode(output, skip_special_tokens=True).strip()
    if not description:
        raise ValueError("VLM produced an empty description")
    return description, {"generated_tokens": len(output), "generation_hit_limit": len(output) >= config["decoding"]["max_new_tokens"]}


def extract(inputs, cache_root, config, model_path, text_path, *, full=False, max_seconds=600):
    import gc
    import re
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Qwen inference; use the supplied Colab L4 notebook")
    for filename, expected in config["vlm"]["assets_sha256"].items():
        if file_hash(Path(model_path) / filename) != expected:
            raise ValueError("Pinned VLM checkpoint asset differs: " + filename)
    for filename, expected in config["vlm"]["assets_git_sha1"].items():
        raw = (Path(model_path) / filename).read_bytes()
        if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != expected:
            raise ValueError("Pinned VLM processor/tokenizer asset differs: " + filename)
    for filename, expected in config["text_encoder"]["assets_sha256"].items():
        if file_hash(Path(text_path) / filename) != expected:
            raise ValueError("Existing CLIP-L text encoder asset differs: " + filename)
    # Exact P2 metadata digest; no annotation file is consulted by extraction.
    protocol_hash = "869b2fe227fb614d83827d8124a80bd02c5f7677482bf64ae7098856b9c70eed"
    images = training_images(inputs, protocol_hash)
    smoke = smoke_sample(images, config["smoke_images"], config["smoke_seed"])
    selected = pd.concat([smoke, images]).drop_duplicates("image_sha256") if full else smoke
    cache = SemanticCache(cache_root, config)
    if full:
        reviewed = [cache.read(row.image_sha256) for row in smoke.itertuples()]
        if any(not r or "embedding" not in r or r["generation_hit_limit"] for r in reviewed):
            raise ValueError("Complete the 30-image smoke and text encoding before full extraction")
        report_path = cache.directory / "extraction_report.json"
        if not report_path.exists() or json.loads(report_path.read_text(encoding="utf-8"))["smoke_review_flags"]:
            raise ValueError("Smoke review flags block full extraction")
    processor = model = None
    durations, reused, generated_count = [], 0, 0
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    try:
        for row in selected.itertuples():
            previous = cache.read(row.image_sha256)
            if previous:
                reused += 1
                continue
            if time.perf_counter() - started > max_seconds - 60:
                raise TimeoutError("Extraction allocation exhausted; completed per-image records persist. Resume the same command/cache.")
            if model is None:
                processor, model = load_vlm(config, model_path)
            tick = time.perf_counter()
            with Image.open(Path(inputs) / "artwork/images" / row.image) as image:
                description, metadata = describe(image.convert("RGB"), processor, model, config)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            cache.save(row.image_sha256, {"description": description, **metadata, "seconds": elapsed,
                       "GPU": torch.cuda.get_device_name(), "prompt_sha256": hashlib.sha256(config["prompt"].encode()).hexdigest()})
            durations.append(elapsed); generated_count += 1
            print(json.dumps({"generated": generated_count, "reused": reused, "seconds": elapsed, "description": description}), flush=True)
    finally:
        telemetry = {"GPU": torch.cuda.get_device_name(), "total_VRAM": total, "free_VRAM_before_load": free,
                     "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
                     "generated": generated_count, "reused_descriptions": reused, "inference_seconds": durations,
                     "mean_image_seconds": float(np.mean(durations)) if durations else None,
                     "projected_1335_inference_seconds": float(np.mean(durations)) * 1335 if durations else None}
        atomic_json(cache.directory / f"telemetry_{time.time_ns()}.json", telemetry)
        del model, processor; gc.collect(); torch.cuda.empty_cache()
    # Text encoding runs on CPU after unloading Qwen, preserving local VRAM.
    tokenizer = AutoTokenizer.from_pretrained(text_path)
    text_model = None
    for row in selected.itertuples():
        record = cache.read(row.image_sha256)
        if "embedding" not in record:
            if text_model is None:
                torch.set_num_threads(4)
                text_model = CLIPTextModelWithProjection.from_pretrained(text_path).eval()
            vector, metadata = encode_description(record["description"], tokenizer, text_model)
            cache.save(row.image_sha256, {**record, **metadata, "embedding": vector.tolist()})
    records = [cache.read(row.image_sha256) for row in selected.itertuples()]
    smoke_records = [cache.read(row.image_sha256) for row in smoke.itertuples()]
    flags = [r["image_sha256"] for r in smoke_records if r["generation_hit_limit"] or re.search(
                 r"\b(Chinese|Japanese|American|European|African|nationality|artist named|painted by)\b", r["description"], re.I)]
    report = {**telemetry, "records": len(records), "smoke_images": len(smoke_records),
              "full_P2_coverage": len(records) == 1335, "empty_descriptions": 0,
              "token_coverage": "PASS", "multi_chunk_descriptions": sum(r["chunks"] > 1 for r in records),
              "smoke_review_flags": flags, "hallucination_audit": "Automated flags only; visual review is required for factual quality conclusions",
              "elapsed_seconds": time.perf_counter() - started, "cache": str(cache.directory)}
    atomic_json(cache.directory / f"extraction_report_{time.time_ns()}.json", report)
    atomic_json(cache.directory / "extraction_report.json", report)
    smoke[["painting", "image", "image_sha256", "aspect_ratio"]].assign(description=[r["description"] for r in smoke_records]).to_csv(
                    cache.directory / "smoke_descriptions.csv", index=False)
    if flags:
        raise ValueError("Smoke descriptions require review: generation length limit or unsupported cultural/artist wording; see extraction_report.json")
    return report
