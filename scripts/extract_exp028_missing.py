"""Prepare or fill an isolated missing-state cache; no fit or evaluation."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from st3.exp028_cache import audit_cache, expected_identity, read_protocol, state_key, verified_arrays
from st3.vlm_semantics import atomic_json, file_hash, fingerprint

ZIP_SHA = "2db4a53679940c7eb63c7c8272b91be8ec0618bb6c04a69ebaa15909e20f64e0"
MANIFEST_SHA = "b5315db3847c29c432e91510a3d8f99c4b1bf13aca8b17282a53059347a20aa9"


def extract_needed(archive, names, inputs):
    from hashlib import sha256

    with zipfile.ZipFile(archive) as source:
        if len(source.namelist()) != len(set(source.namelist())):
            raise ValueError("Duplicate input ZIP names")
        for name, digest in names.items():
            destination = (inputs / name).resolve()
            if not destination.is_relative_to(inputs.resolve()):
                raise ValueError("Nonlocal archive path")
            if destination.exists():
                if file_hash(destination) != digest:
                    raise ValueError(f"Changed restored file: {name}")
                continue
            raw = source.read(name)
            if sha256(raw).hexdigest() != digest:
                raise ValueError(f"Archive member checksum differs: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_suffix(".part")
            pending.write_bytes(raw)
            pending.replace(destination)


def setup_inputs(archive, inputs):
    from hashlib import sha256

    if file_hash(archive) != ZIP_SHA:
        raise ValueError("Use the unchanged exp023_handoff.zip; SHA-256 differs")
    with zipfile.ZipFile(archive) as source:
        if sha256(source.read("input_manifest.json")).hexdigest() != MANIFEST_SHA:
            raise ValueError("Original input manifest SHA-256 differs")
    config = json.loads((ROOT / "configs/exp026.json").read_text())
    names = ["training/pairs.csv", "training/protocol.csv"]
    extract_needed(archive, {n: config["input_sha256"][n] for n in names}, inputs)
    return config, read_protocol(ROOT, inputs)


def needed_by_image(rows, missing):
    mapping = {(r.painting, r.language): (r.image, r.image_sha256) for r in rows.itertuples()}
    groups = defaultdict(list)
    for painting, language in missing:
        groups[mapping[painting, language]].append(language)
    return sorted(groups.items())


def fill_missing(args, config, rows, missing, identity, delta_dir, directories):
    # Reuse EXP-026's pinned asset verifier and official processor, never checkpoints.
    from run_exp026 import build_backend
    from st3.vlm_supervised import adapters_disabled
    from st3.vlm_supervised_training import FeatureCache
    from PIL import Image
    import torch

    started = time.monotonic()
    restore = needed_by_image(rows, missing)
    extract_needed(args.input_zip, {f"artwork/images/{image}": digest
                                    for (image, digest), _ in restore}, args.inputs)
    cache = FeatureCache(delta_dir.parent, args.inputs, config, rows)
    if cache.root != delta_dir or cache.identity != identity:
        raise ValueError("Extractor identity differs")
    backend, _ = build_backend(SimpleNamespace(stage="extraction", model_path=None, model_cache=args.model_cache), config)
    backend.model.eval()
    backend.model.requires_grad_(False)
    new_vision = decoder_batches = decoder_sequences = 0

    def vision_for(image, digest):
        nonlocal new_vision
        for folder in directories:
            path = folder / (digest + "_vision.npz")
            if path.is_file():
                return {k: torch.from_numpy(v) for k, v in verified_arrays(path, identity, "vision").items()}
        new_vision += 1
        return cache.vision(image, backend)

    first_image = restore[0][0][0]
    with torch.inference_mode(), adapters_disabled(backend.model):
        with Image.open(args.inputs / "artwork/images" / first_image) as opened:
            fresh = backend.vision(opened.convert("RGB"))
            parity = backend.parity(opened.convert("RGB"), fresh)
        first_digest = restore[0][0][1]
        for folder in directories:
            old_vision = folder / (first_digest + "_vision.npz")
            if old_vision.is_file():
                import numpy as np
                arrays = verified_arrays(old_vision, identity, "vision")
                if any(not np.array_equal(arrays[k], v.numpy()) for k, v in fresh.items()):
                    raise ValueError("Existing representative vision tokens differ from the fresh pinned path")
                parity["original_cached_vision_exact"] = True
                break
        for (image, digest), langs in restore:
            if time.monotonic()-started >= args.max_seconds-90:
                raise TimeoutError("Extraction budget exhausted. Completed painting files persist")
            vision = vision_for(image, digest)
            path = cache.path(image, None)
            existing = verified_arrays(path, identity, "states") if path.exists() else {}
            additions = {}
            for first in range(0, len(langs), config["microbatch_pairs"]):
                if time.monotonic()-started >= args.max_seconds-90:
                    raise TimeoutError("Extraction budget exhausted before the next decoder batch")
                chunk = langs[first:first+config["microbatch_pairs"]]
                vectors = backend.hidden([vision] * len(chunk), chunk, train=False).float().cpu().numpy()
                additions.update({state_key(l): v for l, v in zip(chunk, vectors)})
                decoder_batches += 1
                decoder_sequences += len(chunk)
            if existing.keys() & additions.keys():
                raise ValueError("Compatible previously completed keys must never be overwritten")
            cache.write(path, **(existing | additions))
            verified_arrays(path, identity, "states")
            print(f"{image}: persisted {len(additions)} target languages", flush=True)
    return {"official_cache_parity": parity, "new_vision_forwards": new_vision,
            "parity_vision_forwards": 2, "parity_decoder_sequences": 2,
            "decoder_batches": decoder_batches, "decoder_sequences": decoder_sequences,
            "seconds": time.monotonic()-started, "GPU": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def run(args):
    from run_exp028_stage1 import check_pins
    # The audit module enforces CUDA hiding for its CLI. Restore an already-set
    # device selector here only for an explicitly authorized extraction worker.
    if args.stage == "extract":
        if args.original_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.original_cuda
    check_pins()
    config, rows = setup_inputs(args.input_zip, args.inputs)
    identity = expected_identity(ROOT)
    old = args.qwen_cache.resolve()
    delta_dir = (args.delta_root / fingerprint(identity)[:20]).resolve()
    if delta_dir == old or delta_dir.is_relative_to(old) or old.is_relative_to(delta_dir):
        raise ValueError("New cache must be separate from the original Drive cache")
    if not old.is_dir():
        raise ValueError("Original Drive cache directory is unavailable")
    directories = [old, delta_dir] if delta_dir.exists() else [old]
    if delta_dir.exists():
        expected_resume = {"source": args.revision, "identity_sha256": fingerprint(identity),
                           "inputs": config["input_sha256"], "scope": "EXP028_FROZEN_EXTRACTION_ONLY"}
        if json.loads((delta_dir.parent / "resume_identity.json").read_text()) != expected_resume:
            raise ValueError("Existing partial extraction has a different source/input identity")
    audit, _, missing = audit_cache(rows, directories, identity)
    atomic_json(args.output / "initial_audit.json", audit)
    atomic_json(args.output / "missing_pairs.json", missing)
    if audit["invalid_entries"]:
        raise ValueError("Original/additional cache audit failed; inspect the audit receipt")
    if args.stage == "audit" or not missing:
        return {"status": "COMPLETE_CACHE_VERIFIED" if not missing else "AUDIT_COMPLETE_MISSING_STATES"}
    if not delta_dir.exists():
        delta_dir.mkdir(parents=True)
        atomic_json(delta_dir / "identity.json", identity)
        atomic_json(delta_dir.parent / "resume_identity.json", {
            "source": args.revision, "identity_sha256": fingerprint(identity), "inputs": config["input_sha256"],
            "scope": "EXP028_FROZEN_EXTRACTION_ONLY",
        })
    telemetry = fill_missing(args, config, rows, missing, identity, delta_dir, [old, delta_dir])
    final, _, _ = audit_cache(rows, [old, delta_dir], identity)
    atomic_json(args.output / "final_audit.json", final)
    if not final["complete"]:
        raise ValueError("Independent CPU read-back found incomplete new cache")
    return {"status": "COMPLETE_CACHE_VERIFIED", "measured": telemetry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["audit", "extract"], default="audit")
    parser.add_argument("--authorize-extraction", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=9000)
    parser.add_argument("--revision", required=True)
    for name in ("input-zip", "inputs", "qwen-cache", "delta-root", "output", "model-cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "extract" and not args.authorize_extraction:
        parser.error("Separate owner GPU authorization is required for missing-state extraction")
    if not math.isfinite(args.max_seconds) or not 90 < args.max_seconds <= 9000:
        parser.error("Expected a finite extraction budget between 90 and 9000 seconds")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=ROOT).strip()
    if actual != args.revision or dirty:
        parser.error("A clean checkout of the exact pinned source is required")
    args.original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args)
        code = 0
    except Exception as error:
        result = {"status": "BLOCKED_OR_INTERRUPTED", "error": f"{type(error).__name__}: {error}"}
        code = 2
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
