"""One image-only semantic extraction command; default is the 30-painting smoke."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from st3.vlm_semantics import extract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp025.json")
    parser.add_argument("--cache", type=Path, default=Path("cache/exp025_semantics"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--text-path", type=Path)
    parser.add_argument("--model-cache", type=Path, default=Path("cache/exp025_models"), help="Local model downloads, separate from persistent semantic records")
    parser.add_argument("--full", action="store_true", help="After successful smoke, extract all1335 image-only descriptions")
    parser.add_argument("--max-seconds", type=float, default=600.)
    args = parser.parse_args()
    import math
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 60:
        raise ValueError("--max-seconds must be finite and exceed the60s cleanup allowance")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    import platform
    if platform.python_version() != config["runtime"]["python"]:
        raise RuntimeError("Use pinned Python " + config["runtime"]["python"])
    for name, expected in config["runtime"].items():
        if name != "python" and importlib.metadata.version(name) != expected:
            raise RuntimeError(f"{name} must remain pinned to {expected}; use the experiment environment")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; select an L4 runtime before model downloads")
    if config["vlm"]["quantization"] == "fp16" and torch.cuda.mem_get_info()[0] < 10 * 1024 ** 3:
        raise RuntimeError("FP16 requires10GiB free VRAM; use Colab L4 before downloading")
    from huggingface_hub import snapshot_download
    paths = []
    for key, explicit in (("vlm", args.model_path), ("text_encoder", args.text_path)):
        paths.append(explicit or Path(snapshot_download(config[key]["model"], revision=config[key]["revision"],
                         local_dir=args.model_cache / key, allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"])))
    torch.manual_seed(252027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    result = extract(args.inputs, args.cache, config, *paths, max_seconds=args.max_seconds)
    print(json.dumps(result, indent=2))
    if args.full:
        remaining = args.max_seconds - result["elapsed_seconds"]
        if remaining < 60:
            raise TimeoutError("Smoke completed; insufficient remaining allocation for full extraction")
        result = extract(args.inputs, args.cache, config, *paths, full=True, max_seconds=remaining)
        print(json.dumps(result, indent=2))
