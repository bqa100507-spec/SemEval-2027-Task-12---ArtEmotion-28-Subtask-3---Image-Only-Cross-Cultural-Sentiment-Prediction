"""Profile the existing Qwen recipe and one compact batched recovery recipe."""

import importlib.metadata
import json
import os
import time
from collections import Counter
from pathlib import Path

from .baseline import file_sha256
from .cli import ROOT, json_output
from .competition_challenger import check_hash
from .competition_semantic_generation import parse_description, store_record, verified_assets, verified_rgb
from .e50_heads import read_json
from .e50_inference import safe_image_path

RECOVERY = ROOT / "configs/competition_semantic_recovery.json"
FIELDS = {"e": "entities", "s": "scene", "a": "style_atmosphere", "v": "visual_cues", "u": "uncertain_context"}


def parse_compact(raw, fields):
    def unique(entries):
        if len(dict(entries)) != len(entries):
            raise ValueError("Duplicate compact field")
        return dict(entries)

    try:
        item = json.loads(raw, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise ValueError("Incomplete or malformed compact JSON") from exc
    if not isinstance(item, dict) or set(item) != set(FIELDS):
        raise ValueError("Compact description must contain exactly e, s, a, v, u")
    if isinstance(item["e"], list):
        parts = item["e"]
        if not 1 <= len(parts) <= 6 or any(not isinstance(v, str) or not v.strip() for v in parts):
            raise ValueError("Entity list must contain one to six nonempty phrases")
        item["e"] = ", ".join(parts)
    if any(not isinstance(v, str) or not 1 <= len(v.split()) <= 14 for v in item.values()):
        raise ValueError("Each compact field must contain one to fourteen words")
    return parse_description(json.dumps({FIELDS[k]: v for k, v in item.items()}), fields)


def completion(ids, eos_ids, limit):
    for i, token in enumerate(ids):
        if token in eos_ids:
            return ids[:i + 1], True
    return ids, len(ids) < limit


def read_sample(historical, inventory, recovery):
    check_hash(historical, recovery["historical_preflight_sha256"])
    archived = read_json(historical)["records"]
    lookup = inventory.set_index("painting")
    for record in archived:
        row = lookup.loc[record["painting"]]
        if row.image != record["image"] or row.image_sha256 != record["image_sha256"]:
            raise ValueError("Historical artwork digest/path differs from the pinned training inventory")
    return archived


class ForwardProbe:
    def __init__(self, torch, module):
        self.torch = torch
        self.before = module.register_forward_pre_hook(self.start)
        self.after = module.register_forward_hook(self.end)
        self.calls = []

    def start(self, *_):
        gpu_start = self.torch.cuda.Event(enable_timing=True)
        gpu_start.record()
        self.current = (gpu_start, time.perf_counter())

    def end(self, _, __, output):
        gpu_end = self.torch.cuda.Event(enable_timing=True)
        gpu_end.record()
        gpu_start, host_start = self.current
        self.calls.append((gpu_start, gpu_end, time.perf_counter() - host_start,
                           getattr(output, "past_key_values", None) is not None))

    def snapshot(self):
        return [{"cuda_seconds": a.elapsed_time(b) / 1000, "host_call_seconds": host, "kv_cache": cache}
                for a, b, host, cache in self.calls]

    def close(self):
        self.before.remove()
        self.after.remove()


def load_model(checkpoint, base):
    versions = {key: importlib.metadata.version(key) for key in base["runtime"]}
    if versions != base["runtime"]:
        raise ValueError(f"Pinned runtime does not match installed packages: {versions}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    free, total = torch.cuda.mem_get_info()
    began = time.perf_counter()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    processor.image_processor.size = {"shortest_edge": base["vision"]["min_pixels"],
                                      "longest_edge": base["vision"]["max_pixels"]}
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=True, quantization_config=quant, device_map={"": 0},
        dtype=torch.float16, attn_implementation=base["attention"], use_safetensors=True,
    ).eval().requires_grad_(False)
    torch.cuda.synchronize()
    tensors = list(model.parameters()) + list(model.buffers())
    counts = Counter(f"{t.device}/{t.dtype}" for t in tensors)
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError(f"Unexpected parameter or buffer offload: {counts}")
    info = {"gpu": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda, "runtime": versions,
            "initial_free_bytes": free, "total_bytes": total, "load_seconds": time.perf_counter() - began,
            "device_map": model.hf_device_map, "tensor_counts": dict(counts),
            "quantization": "NF4 double quantization, FP16 compute", "kv_cache": model.generation_config.use_cache}
    return torch, model, processor, info


def batch_profile(torch, model, processor, data_dir, rows, base, recipe=None, *, trace=True):
    began = time.perf_counter()
    pixels = [verified_rgb(safe_image_path(data_dir, r["image"]), r["image_sha256"]) for r in rows]
    read_seconds = time.perf_counter() - began
    preparation = time.perf_counter()
    message = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": (recipe or base)["prompt"]}]}]
    prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    cpu = processor(text=[prompt] * len(rows), images=pixels, return_tensors="pt", padding=True,
                    padding_side="left")
    if "pixel_values" not in cpu or len(cpu["image_grid_thw"]) != len(rows):
        raise ValueError("Processor did not preserve one vision input per artwork")
    cpu_seconds = time.perf_counter() - preparation
    transfer = time.perf_counter()
    args = cpu.to("cuda")
    torch.cuda.synchronize()
    transfer_seconds = time.perf_counter() - transfer
    generation = base["generation"].copy()
    if recipe:
        generation["max_new_tokens"] = recipe["max_new_tokens"]
    main = ForwardProbe(torch, model) if trace else None
    vision = ForwardProbe(torch, model.model.visual) if trace else None
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            outputs = model.generate(**args, **generation, temperature=None, top_p=None, top_k=None)
        torch.cuda.synchronize()
    finally:
        if trace:
            main.close()
            vision.close()
    elapsed = time.perf_counter() - started
    decoded = outputs[:, args.input_ids.shape[1]:].tolist()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    records = []
    for row, ids, grid, length in zip(rows, decoded, args.image_grid_thw.tolist(), args.attention_mask.sum(1).tolist(), strict=True):
        ids, finished = completion(ids, eos, generation["max_new_tokens"])
        raw = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        item = {k: row[k] for k in ("painting", "image", "image_sha256")} | {
            "raw": raw, "output_tokens": len(ids), "prompt_tokens": length, "image_grid_thw": grid, "natural_end": finished}
        try:
            parser = parse_compact if recipe else parse_description
            item["description"] = parser(raw, base["fields"])
        except ValueError as error:
            item["parse_error"] = str(error)
        records.append(item)
    result = {"records": records, "jpeg_read_sha_rgb_seconds": read_seconds, "processor_cpu_seconds": cpu_seconds,
              "host_to_device_seconds": transfer_seconds, "generation_wall_seconds": elapsed,
              "wall_seconds": time.perf_counter() - began,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "attention": {k: getattr(model.config, k)._attn_implementation for k in ("text_config", "vision_config")}}
    if trace:
        passes, visual = main.snapshot(), vision.snapshot()
        if not passes or len(visual) != 1 or len(passes) != len(decoded[0]):
            raise RuntimeError("Unexpected forward count: expected one vision pass and one invocation per output-token step")
        result["forward_profile"] = {"prefill": passes[0], "vision": visual[0],
                                     "decode_cuda_seconds": sum(p["cuda_seconds"] for p in passes[1:]),
                                     "decode_host_call_seconds": sum(p["host_call_seconds"] for p in passes[1:]),
                                     "decode_steps": len(passes) - 1,
                                     "all_forwards_use_kv": all(p["kv_cache"] for p in passes)}
    return result


def recover_benchmark(args, base, recovery, sample):
    output = args.run_dir / "benchmark.json"
    if output.exists():
        raise FileExistsError(f"Benchmark already exists: {output}")
    began = time.perf_counter()
    receipt = {"status": "RUNNING", "baseline": [], "compact": [], "historical_sha256": file_sha256(args.historical),
               "recovery_sha256": file_sha256(RECOVERY), "source_sha256": file_sha256(Path(__file__)),
               "base_config_sha256": file_sha256(args.base_config)}

    def save():
        receipt["wall_seconds"] = time.perf_counter() - began
        json_output(receipt, output)

    def run(rows, recipe=None):
        if time.perf_counter() - began > recovery["benchmark_wall_seconds"]:
            raise TimeoutError("Bounded recovery benchmark time limit reached")
        return batch_profile(torch, model, processor, args.data_dir, rows, base, recipe)

    try:
        receipt["asset_sha256"] = verified_assets(args.checkpoint, base)
        torch, model, processor, receipt["hardware"] = load_model(args.checkpoint, base)
        save()
        for index in recovery["reprofile_sample_indices"]:
            result = run([sample[index]])
            result["historical_raw_exact"] = result["records"][0]["raw"] == sample[index]["raw"]
            receipt["baseline"].append(result)
            save()
            print(f"Original {index + 1}: {result['wall_seconds']:.2f} s", flush=True)
        model.set_attn_implementation(recovery["attention"])
        size = recovery["batch_size"]
        for i in range(0, len(sample), size):
            result = run(sample[i:i + size], recovery)
            receipt["compact"].append(result)
            save()
            print(f"Compact batch {i // size + 1}: {result['wall_seconds']:.2f} s", flush=True)
        repeated = run(sample[:size], recovery)
        receipt["repeated_batch"] = repeated
        receipt["repeat_exact"] = [row["raw"] for row in repeated["records"]] == [
            row["raw"] for row in receipt["compact"][0]["records"]]
        receipt["status"] = "MEASURED"
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
    return receipt
