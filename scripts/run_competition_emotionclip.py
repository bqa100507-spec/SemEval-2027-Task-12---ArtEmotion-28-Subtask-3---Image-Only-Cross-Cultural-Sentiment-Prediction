#!/usr/bin/env python3
"""Fetch the pinned checkpoint, cache frozen P2 features and score the CPU readouts."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.baseline import file_sha256
from st3.cli import json_output
from st3.competition_challenger import check_hash
from st3.competition_emotionclip import CONFIG, run
from st3.competition_semantic_generation import TRACK1, training_inventory
from st3.contrast_study import new_output
from st3.e50_heads import read_json
from st3.e50_inference import safe_image_path
from st3.emotionclip import EmotionCLIP, preprocess, prompt_tokens


def verify_assets(checkpoint, tokenizer, config):
    for name, digest in config["assets_sha256"].items():
        check_hash(checkpoint / name, digest)
    check_hash(tokenizer, config["clip_tokenizer"]["sha256"])


def read_features(path, names, width):
    with np.load(path, allow_pickle=False) as saved:
        embeddings, scores = saved["embeddings"], saved["scores"]
        if saved["paintings"].tolist() != names:
            raise ValueError("Cache artwork coverage or order differs from P2")
    if (embeddings.shape != (len(names), 512) or scores.shape != (len(names), width)
            or not np.isfinite(embeddings).all() or not np.isfinite(scores).all()
            or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)
            or (scores < 0).any() or (scores > 1).any() or not np.allclose(scores.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("Invalid EmotionCLIP embeddings or prompt probabilities")
    return {"emotion_embedding": pd.DataFrame(embeddings, index=names),
            "emotion_scores": pd.DataFrame(scores, index=names)}


def extract(args, scope, identity, config, destination):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    gpu = args.device == "cuda"
    if gpu and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verify_assets(args.checkpoint, args.tokenizer, config)
    for row in scope.itertuples(index=False):
        check_hash(safe_image_path(args.data_dir, row.image), row.image_sha256)
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = EmotionCLIP(args.checkpoint / "EmotionCLIP.pth", args.device)
    tokens = prompt_tokens(args.tokenizer, config["labels"], config["prompt"])
    with torch.inference_mode():
        text = model.encode_text(tokens)
        if gpu:
            torch.cuda.synchronize()
        loaded = time.perf_counter()
        embeds, scores = [], []
        probe = {}
        for start in range(0, len(scope), config["batch_size"]):
            part = scope.iloc[start:start + config["batch_size"]]
            pixels = []
            for row in part.itertuples(index=False):
                with Image.open(safe_image_path(args.data_dir, row.image)) as image:
                    pixels.append(preprocess(image))
            batch = torch.stack(pixels)
            outputs = model.features(batch, text)
            if start == 0:
                model.verified_forward_keys()
                if any(not t.isfinite().all() for t in outputs):
                    raise RuntimeError("Preflight produced nonfinite outputs")
                seconds = time.perf_counter() - loaded
                replay = model.features(batch, text)
                if any(not torch.equal(a, b) for a, b in zip(outputs, replay)):
                    raise RuntimeError("Preflight failed exact same-batch determinism")
                projected = seconds / len(part) * len(scope)
                probe = {"images": len(part), "seconds": seconds, "projected_seconds": projected,
                         "exact_same_batch_replay": True,
                         "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if gpu else None}
                print(f"Preflight: {len(part)} paintings in {seconds:.2f}s; projected {projected:.1f}s; exact replay",
                      flush=True)
                if gpu and projected * 1.3 + loaded - started > config["gpu_ceiling_seconds"]:
                    raise TimeoutError("Measured extraction forecast exceeds the bounded CUDA budget")
            embeds.append(outputs[0].cpu().numpy())
            scores.append(outputs[1].cpu().numpy())
            if (start + len(part)) % 256 == 0:
                print(f"Extracted {start + len(part)}/{len(scope)} paintings", flush=True)
    duration = time.perf_counter() - loaded
    with new_output(destination) as stage:
        path = stage / "features.npz"
        names = scope.painting.tolist()
        np.savez(path, paintings=names, embeddings=np.vstack(embeds), scores=np.vstack(scores))
        read_features(path, names, len(config["labels"]))
        receipt = {"identity": identity, "images": len(scope), "device": torch.cuda.get_device_name() if gpu else "CPU",
                   "probe": probe, "extraction_seconds": duration, "model_load_seconds": loaded - started,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated() if gpu else None,
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved() if gpu else None,
                   "features_sha256": file_sha256(path),
                   "versions": {p: importlib.metadata.version(p) for p in ("torch", "Pillow", "tokenizers")}}
        json_output(receipt, stage / "manifest.json")
    print(json_output(receipt), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "extract", "cv"))
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/upstream")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/competition_neighborhood_residual/results")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/emotionclip-367bd3d")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/competition_emotionclip")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config, track = read_json(CONFIG), read_json(TRACK1)
    tokenizer = config["clip_tokenizer"]
    tokenizer_dir = "models--" + tokenizer["model"].replace("/", "--")
    args.tokenizer = (args.tokenizer or ROOT / "cache/huggingface" / tokenizer_dir / "snapshots"
                      / tokenizer["revision"] / "tokenizer.json")
    if args.stage == "fetch":
        from huggingface_hub import snapshot_download

        snapshot_download(config["model"], revision=config["revision"], token=False, local_dir=args.checkpoint,
                          allow_patterns=list(config["assets_sha256"]), max_workers=2)
        if not args.tokenizer.exists():
            snapshot_download(tokenizer["model"], revision=tokenizer["revision"], token=False,
                              local_dir=args.tokenizer.parent, allow_patterns=["tokenizer.json"])
        verify_assets(args.checkpoint, args.tokenizer, config)
        return
    if args.stage == "extract" and args.device == "cuda" and not args.worker:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                       timeout=config["gpu_ceiling_seconds"], check=True)
        return
    scope = training_inventory(args.inputs, track)
    identity = {"config_sha256": file_sha256(CONFIG), "runner_sha256": file_sha256(Path(__file__)),
                "encoder_sha256": file_sha256(ROOT / "src/st3/emotionclip.py"),
                "model": config["model"], "revision": config["revision"], "device": args.device,
                "p2_protocol_sha256": track["p2_input_sha256"]["training/protocol.csv"]}
    cache = args.run_dir / "features"
    if args.stage == "extract" and not cache.exists():
        extract(args, scope, identity, config, cache)
        return
    receipt = read_json(cache / "manifest.json")
    if receipt["identity"] != identity:
        raise ValueError("Frozen EmotionCLIP cache identity changed")
    path = cache / "features.npz"
    check_hash(path, receipt["features_sha256"])
    features = read_features(path, scope.painting.tolist(), len(config["labels"]))
    if args.stage == "extract":
        print(f"Reused {len(scope)} checksum-verified artwork rows")
        return
    run(args.inputs, features, receipt, args.reference, args.run_dir / "results", track, config)


if __name__ == "__main__":
    main()
