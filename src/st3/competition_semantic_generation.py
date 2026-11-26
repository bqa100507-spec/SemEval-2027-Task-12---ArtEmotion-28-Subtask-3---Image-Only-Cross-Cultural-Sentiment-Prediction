"""Bounded image-only Qwen3-VL generation and a checksum-bound artwork cache."""

import hashlib
import importlib.metadata
import io
import json
import os
import time
from pathlib import Path

from PIL import Image

from .baseline import file_sha256
from .cli import ROOT, json_output
from .competition_challenger import check_hash
from .competition_track1 import load_protocol
from .e50_heads import read_json
from .e50_inference import safe_image_path

CONFIG = ROOT / "configs/competition_semantic_vlm.json"
TRACK1 = ROOT / "configs/competition_track1.json"


def parse_description(raw, fields):
    def unique(entries):
        result = dict(entries)
        if len(result) != len(entries):
            raise ValueError("Duplicate description field")
        return result

    try:
        values = json.loads(raw, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise ValueError("Output is not a single JSON object") from exc
    if not isinstance(values, dict) or set(values) != set(fields):
        raise ValueError("Output does not contain the frozen field set")
    if any(not isinstance(v, str) or not v.strip() or len(v) > 400 for v in values.values()):
        raise ValueError("Description fields must be nonempty short strings")
    # Preserve the model's diagnostics even if it exceeds the requested word count.
    values = {key: " ".join(values[key].split()) for key in fields}
    hedge = values["uncertain_context"].lower().rstrip(".")
    if hedge != "none" and not hedge.startswith("uncertain:"):
        raise ValueError("An interpretation was not marked uncertain")
    return values


def cache_identity(config):
    keys = ("model", "revision", "weight_sha256", "git_blobs", "runtime", "prompt", "fields", "vision",
            "quantization", "attention", "generation")
    return {key: config[key] for key in keys} | {"generator_sha256": file_sha256(Path(__file__))}


def verified_assets(path, config):
    assets = {}
    for name, expected in config["git_blobs"].items():
        payload = (path / name).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload, usedforsecurity=False).hexdigest()
        if git_blob != expected:
            raise ValueError(f"Checkpoint metadata does not match the pinned Git blob: {name}")
        assets[name] = hashlib.sha256(payload).hexdigest()
    for name, expected in config["weight_sha256"].items():
        check_hash(path / name, expected)
        assets[name] = expected
    return assets


def training_inventory(inputs, track1):
    protocol = load_protocol(inputs, track1)
    scope = protocol.loc[protocol.role.eq("development")].sort_values("painting", ignore_index=True)
    if len(scope) != len(track1["p2"]["fold_pair_counts"]) * track1["p2"]["paintings_per_fold"]:
        raise ValueError("P2 image inventory differs from the pinned protocol")
    return scope


def verified_rgb(path, digest):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"Artwork bytes differ from the pinned input: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def load_generator(checkpoint, config):
    versions = {key: importlib.metadata.version(key) for key in config["runtime"]}
    if versions != config["runtime"]:
        raise ValueError(f"Frozen VLM package versions differ: {versions}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL preflight requires a CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    processor.image_processor.size = {"shortest_edge": config["vision"]["min_pixels"],
                                      "longest_edge": config["vision"]["max_pixels"]}
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=True, quantization_config=quantization, device_map={"": 0},
        dtype=torch.float16, attn_implementation=config["attention"], use_safetensors=True)
    model.eval().requires_grad_(False)
    torch.cuda.synchronize()
    startup = {"gpu": torch.cuda.get_device_name(), "total_bytes": total, "initial_free_bytes": free,
               "load_seconds": time.perf_counter() - started, "runtime": versions}

    def generate(pixels):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": config["prompt"]}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        args = processor(text=[prompt], images=[pixels], return_tensors="pt").to("cuda")
        if "pixel_values" not in args or "image_grid_thw" not in args:
            raise RuntimeError("Qwen preprocessor omitted the image")
        tick = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(**args, **config["generation"], temperature=None, top_p=None, top_k=None)
        tokens = result[:, args.input_ids.shape[1]:]
        text = processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return text, {"seconds": time.perf_counter() - tick, "tokens": tokens.shape[1],
                      "image_grid_thw": args.image_grid_thw.tolist(),
                      "pixel_dimensions": list(pixels.size)}

    def memory():
        return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}

    return generate, startup, memory


def record_for(generate, path, digest, config):
    raw, timing = generate(verified_rgb(path, digest))
    record = {"image_sha256": digest, "identity": cache_identity(config), "raw": raw, "timing": timing}
    try:
        record["description"] = parse_description(raw, config["fields"])
    except ValueError as exc:
        record["parse_error"] = str(exc)
    return record


def read_record(path, digest, config):
    item = read_json(path)
    if item["image_sha256"] != digest or item["identity"] != cache_identity(config):
        raise ValueError(f"Semantic cache identity mismatch for {digest}")
    if item.get("description") != parse_description(item["raw"], config["fields"]):
        raise ValueError(f"Semantic cache description differs from its raw output for {digest}")
    return item


def store_record(directory, record):
    target = directory / (record["image_sha256"] + ".json")
    temporary = target.with_suffix(".tmp")
    json_output(record, temporary)
    temporary.replace(target)


def preflight(args, config, track1):
    started = time.perf_counter()
    path = args.run_dir / "preflight.json"
    if path.exists():
        raise ValueError(f"Preflight artifact already exists: {path}")
    scope = training_inventory(args.inputs, track1)
    rule = config["preflight"]
    ids = scope.painting.map(lambda p: hashlib.sha256(f"{rule['seed']}:{p}".encode()).hexdigest())
    sample = scope.assign(preflight_order=ids).sort_values("preflight_order").head(rule["images"])
    out = {"status": "FAIL", "identity": cache_identity(config), "records": []}
    try:
        out["asset_sha256"] = verified_assets(args.checkpoint, config)
        generate, startup, memory = load_generator(args.checkpoint, config)
        out["hardware"] = startup
        for item in sample.itertuples(index=False):
            if time.perf_counter() - started >= rule["max_seconds"]:
                raise TimeoutError("Preflight time limit reached")
            record = record_for(generate, safe_image_path(args.data_dir, item.image), item.image_sha256, config)
            out["records"].append({"painting": item.painting, "image": item.image, **record})
            if "description" in record:
                store_record(args.run_dir / "descriptions", record)
        first = sample.iloc[0]
        repeat = record_for(generate, safe_image_path(args.data_dir, first.image), first.image_sha256, config)
        out["repeat_exact"] = repeat["raw"] == out["records"][0]["raw"]
        out["memory"] = memory()
        seconds = sum(item["timing"]["seconds"] for item in out["records"])
        out["images_per_second"] = len(sample) / seconds
        out["projection_p2_seconds"] = len(scope) * seconds / len(sample)
        gpu_spare = startup["initial_free_bytes"] - out["memory"]["peak_reserved_bytes"]
        out["memory_headroom_bytes"] = gpu_spare
        out["status"] = ("TECHNICAL_PASS" if out["repeat_exact"] and gpu_spare >= rule["memory_headroom_mib"] * 1024**2
                         and all("description" in item for item in out["records"]) else "FAIL")
    except (RuntimeError, OSError, ValueError, TimeoutError, ImportError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        if "hardware" in out:
            out["memory"] = memory()
    out["wall_seconds"] = time.perf_counter() - started
    json_output(out, path)
    print(f"Semantic VLM preflight: {out['status']}; {len(out['records'])} training paintings", flush=True)
    return out


def extract(args, config, track1):
    if not args.authorize_full_extraction:
        raise PermissionError("Full extraction requires explicit authorization after image-grounding review")
    scope = training_inventory(args.inputs, track1)
    benchmark = read_json(args.run_dir / "preflight.json")
    if benchmark["identity"] != cache_identity(config) or benchmark["status"] != "TECHNICAL_PASS":
        raise ValueError("Full extraction requires a matching technical preflight")
    final_path = args.run_dir / "extraction.json"
    if final_path.exists():
        raise ValueError(f"Extraction is already sealed: {final_path}")
    directory = args.run_dir / "descriptions"
    missing = [row for row in scope.itertuples(index=False) if not (directory / f"{row.image_sha256}.json").exists()]
    if benchmark["projection_p2_seconds"] * len(missing) / len(scope) * 1.25 > config["max_full_extraction_seconds"]:
        raise TimeoutError("Measured projected extraction exceeds the frozen budget")
    started = time.perf_counter()
    for row in scope.itertuples(index=False):
        path = directory / f"{row.image_sha256}.json"
        if path.exists():
            read_record(path, row.image_sha256, config)
    if missing:
        verified_assets(args.checkpoint, config)
        generate, startup, memory = load_generator(args.checkpoint, config)
    for row in missing:
        if time.perf_counter() - started >= config["max_full_extraction_seconds"]:
            raise TimeoutError("Full extraction limit reached; completed artwork records were retained")
        record = record_for(generate, safe_image_path(args.data_dir, row.image), row.image_sha256, config)
        if "description" not in record:
            json_output(record, args.run_dir / "extraction_failure.json")
            raise ValueError("Malformed VLM description; inspect extraction_failure.json")
        store_record(directory, record)
    files = {row.image_sha256: file_sha256(directory / f"{row.image_sha256}.json") for row in scope.itertuples(index=False)}
    json_output({"identity": cache_identity(config), "images": files, "new": len(missing),
                 "hardware": (startup | memory()) if missing else {},
                 "wall_seconds": time.perf_counter() - started}, final_path)
