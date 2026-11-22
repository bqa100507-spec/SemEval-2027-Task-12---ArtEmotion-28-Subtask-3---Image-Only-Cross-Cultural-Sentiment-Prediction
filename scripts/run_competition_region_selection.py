#!/usr/bin/env python3
"""Cache frozen DINOv2-L patch tokens and run the bounded CPU readout comparison."""

import argparse
import importlib.metadata
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.baseline import file_sha256
from st3.cli import json_output
from st3.competition_challenger import check_hash
from st3.competition_dinov2_large import CONFIG as BACKBONE, TRACK1
from st3.competition_semantic_generation import training_inventory
from st3.contrast_study import new_output
from st3.e50_heads import read_json
from st3.e50_inference import safe_image_path

RULE = ROOT / "configs/competition_region_selection.json"


def extract(args, scope, identity, rule, base):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoImageProcessor, AutoModel

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA extraction requested and unavailable")
    for name, sha in base["assets_sha256"].items():
        check_hash(args.checkpoint / name, sha)
    if importlib.metadata.version("transformers") != base["transformers_version"]:
        raise ValueError("Pinned Transformers runtime required")
    torch.set_num_threads(rule["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    gpu = args.device == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained(args.checkpoint, use_fast=False, local_files_only=True)
    model = AutoModel.from_pretrained(args.checkpoint, local_files_only=True, use_safetensors=True,
                                      attn_implementation="eager").eval().requires_grad_(False).to(args.device)
    if gpu:
        torch.cuda.synchronize()
    loaded = time.perf_counter()
    metadata = {"identity": identity, "model_load_seconds": loaded - start,
                "device_name": torch.cuda.get_device_name() if gpu else "CPU"}
    with new_output(args.run_dir / "patches") as stage:
        target = stage / "tokens.npy"
        tensor = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16,
                                           shape=(len(scope), rule["patch_count"], rule["patch_width"]))
        width = rule["batch_size"]
        for i in range(0, len(scope), width):
            rows = scope.iloc[i:i + width]
            images = []
            for row in rows.itertuples(index=False):
                path = safe_image_path(args.data_dir, row.image)
                check_hash(path, row.image_sha256)
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            with torch.inference_mode():
                inputs = proc(images=images, return_tensors="pt").to(args.device)
                values = model(**inputs).last_hidden_state[:, 1:]
                if tuple(values.shape[1:]) != (rule["patch_count"], rule["patch_width"]) or not values.isfinite().all():
                    raise ValueError(f"Unexpected patch output {values.shape}")
                tensor[i:i + len(rows)] = torch.nn.functional.normalize(values.float(), dim=-1).cpu().numpy()
            count = i + len(rows)
            if count == rule["probe_images"]:
                seconds = time.perf_counter() - loaded
                forecast = seconds / count * len(scope)
                metadata["probe"] = {"images": count, "seconds": seconds, "estimated_forward_seconds": forecast}
                print(f"Frozen probe: {count} images in {seconds:.2f}s; projected bank {forecast:.1f}s", flush=True)
                if gpu and forecast * 1.3 + loaded - start > rule["extract_limit_seconds"]:
                    raise TimeoutError("Patch extraction exceeds the frozen GPU ceiling by measured projection")
            if count % 256 == 0 or count == len(scope):
                print(f"Cached patch tokens for {count}/{len(scope)} artworks", flush=True)
        tensor.flush()
        del tensor
        metadata.update(images=len(scope), extraction_seconds=time.perf_counter() - loaded,
                        runtime={n: importlib.metadata.version(n) for n in ("torch", "transformers", "Pillow", "numpy")},
                        peak_reserved_bytes=torch.cuda.max_memory_reserved() if gpu else None,
                        tokens_sha256=file_sha256(target), token_bytes=target.stat().st_size)
        json_output(metadata, stage / "manifest.json")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "cv"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/upstream")
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/dinov2-large-47b73ee")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/competition_neighborhood_residual/results")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/competition_region_selection")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    rule, track, base = read_json(RULE), read_json(TRACK1), read_json(BACKBONE)
    if args.stage == "extract" and args.device == "cuda" and not args.worker:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                       timeout=rule["extract_limit_seconds"], check=True)
        return
    scope = training_inventory(args.inputs, track)
    identity = {"assets": base["assets_sha256"], "revision": base["revision"], "device": args.device,
                "processor": "official slow RGB 224 crop", "representation": "FP16 per-token unit final patches",
                "p2_protocol": track["p2_input_sha256"]["training/protocol.csv"], "extract_config": file_sha256(RULE),
                "extract_source": file_sha256(Path(__file__))}
    path = args.run_dir / "patches"
    if args.stage == "extract" and not path.exists():
        print(json_output(extract(args, scope, identity, rule, base)), end="")
        return
    meta = read_json(path / "manifest.json")
    if meta["identity"] != identity:
        raise ValueError("Cached patch-token identity changed")
    check_hash(path / "tokens.npy", meta["tokens_sha256"])
    patches = np.load(path / "tokens.npy", mmap_mode="r", allow_pickle=False)
    if patches.dtype != np.float16 or patches.shape != (len(scope), rule["patch_count"], rule["patch_width"]):
        raise ValueError("Patch cache shape or dtype changed")
    if args.stage == "cv":
        from st3.competition_region_selection import run

        run(args, scope.painting.tolist(), patches, meta, rule, track)
    else:
        print(f"Reused {len(patches)} checksum-verified token rows")


if __name__ == "__main__":
    main()
