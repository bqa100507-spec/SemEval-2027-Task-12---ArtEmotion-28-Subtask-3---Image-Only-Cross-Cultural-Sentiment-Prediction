#!/usr/bin/env python3
"""Prepare the matched direct-Qwen pilot and run a bounded synthetic-target preflight."""

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.baseline import file_sha256
from st3.competition_challenger import check_hash
from st3.competition_semantic_evaluation import write_atomic
from st3.competition_semantic_generation import CONFIG, TRACK1, verified_assets, verified_rgb
from st3.competition_semantic_recovery import load_model
from st3.competition_track1 import load_protocol
from st3.e50_heads import read_json
from st3.e50_inference import safe_image_path

DIRECT = ROOT / "configs/competition_direct_qwen.json"


def workload(pairs, width):
    counts = pairs.groupby("painting").size()
    return {"paintings": len(counts), "pairs": int(counts.sum()), "chunks": int(((counts + width - 1) // width).sum())}


def prepare(args, config, base):
    track = read_json(TRACK1)
    path = args.inputs / "training/pairs.csv"
    check_hash(path, track["p2_input_sha256"]["training/pairs.csv"])
    check_hash(args.reference / "oof.csv", base["incumbent_oof_sha256"])
    keys = pd.read_csv(path, usecols=["painting", "language", "fold", "role", "split"])
    protocol = load_protocol(args.inputs, track)
    scoped = protocol.loc[protocol.role.eq("development")].set_index("painting")
    if (keys.duplicated(["painting", "language"]).any() or not keys.role.eq("development").all()
            or not keys.split.eq("train").all() or not keys.painting.map(scoped.fold).eq(keys.fold).all()
            or keys.groupby("fold").size().tolist() != track["p2"]["fold_pair_counts"]):
        raise ValueError("Exact original grouped P2 keys are required")
    split = config["pilot_fold"]
    fit, held = keys.loc[keys.fold.ne(split)], keys.loc[keys.fold.eq(split)]
    if set(fit.painting) & set(held.painting) or set(keys.painting) != set(scoped.index):
        raise ValueError("Painting-level isolation failed")
    counts = fit.groupby("painting").size().rename("pairs").reset_index()
    counts["tie"] = counts.painting.map(lambda name: hashlib.sha256(f"{config['seed']}:{name}".encode()).hexdigest())
    order = counts.sort_values(["pairs", "tie"]).reset_index(drop=True)
    indices = [min(len(order) - 2, int(len(order) * q)) + offset for q in (.1, .5, .9) for offset in (0, 1)]
    sample = order.iloc[indices].painting.tolist()
    if len(set(sample)) != config["feasibility_paintings"]:
        raise ValueError("The probe requires six distinct fitting paintings")
    groups = [{"painting": p, "image": scoped.at[p, "image"], "image_sha256": scoped.at[p, "image_sha256"],
               "languages": sorted(fit.loc[fit.painting.eq(p), "language"])} for p in sample]
    width = config["pair_microbatch"]
    proof = {"fit": workload(fit, width), "held": workload(held, width),
             "all_folds": {str(i): {"fit": workload(keys.loc[keys.fold.ne(i)], width),
                                    "held": workload(keys.loc[keys.fold.eq(i)], width)} for i in sorted(keys.fold.unique())},
             "keys_sha256": hashlib.sha256(keys.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
             "corrected_ridge_oof_sha256": base["incumbent_oof_sha256"], "sample": groups}
    return groups, proof


def preflight(args, config, base, groups, proof):
    import torch

    from st3.competition_direct_qwen import CulturalQwen, encode_text, language_prompt, last_valid, optimizer_for, train_step

    began = time.perf_counter()
    output = args.run_dir / "feasibility.json"
    receipt = {"state": "RUNNING", "protocol": proof, "config_sha256": file_sha256(DIRECT),
               "base_config_sha256": file_sha256(CONFIG), "source_sha256": file_sha256(ROOT / "src/st3/competition_direct_qwen.py"),
               "runner_sha256": file_sha256(Path(__file__)), "frozen": [], "adapted": [], "adapted_inference": []}

    def save():
        receipt["wall_seconds"] = time.perf_counter() - began
        write_atomic(output, receipt)

    def timed(callback):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = callback()
        torch.cuda.synchronize()
        return value, time.perf_counter() - start

    def chunks(rows):
        width = config["pair_microbatch"]
        return [rows[i:i + width] for i in range(0, len(rows), width)]

    def prepare_image(group):
        pixels = verified_rgb(safe_image_path(args.data_dir, group["image"]), group["image_sha256"])
        return pixels, processor.image_processor(images=[pixels], return_tensors="pt")

    def language_batch(model, vision, languages):
        return encode_text(processor, languages, [len(vision.merged[0])] * len(languages), config["prompt"], model.device)

    def measure_artwork(model, group, adapting=False):
        start = time.perf_counter()
        (_, image), cpu_seconds = timed(lambda: prepare_image(group))
        vision, vision_seconds = timed(lambda: model.encode_vision(image))
        batches = chunks(group["languages"])
        encoded, text_seconds = timed(lambda: [language_batch(model, vision, batch) for batch in batches])
        decoded = 0.0
        if adapting:
            def minibatches():
                for batch in encoded:
                    size = len(batch["input_ids"])
                    yield (lambda batch=batch, size=size: model(vision, [0] * size, batch),
                           torch.arange(size, device=model.device, dtype=torch.float32) % 3 + 1,
                           torch.full((size,), 5., device=model.device))
            step, decoded = timed(lambda: train_step(model, optimizer, scaler, config, minibatches(), synthetic_mass))
        else:
            model.eval()
            with torch.no_grad():
                retained, decoded = timed(lambda: torch.cat([model.features(vision, [0] * len(batch["input_ids"]), batch)
                                                             for batch in encoded]))
            if not retained.isfinite().all():
                raise ValueError("Qwen hidden states contain non-finite values")
            step = {"feature_rms": float(retained.float().square().mean().sqrt())}
        return step | {"painting": group["painting"], "pairs": len(group["languages"]), "chunks": len(encoded),
                       "image_tokens": len(vision.merged[0]), "cpu_image_seconds": cpu_seconds,
                       "vision_transfer_forward_seconds": vision_seconds, "language_pack_transfer_seconds": text_seconds,
                       "decoder_seconds": decoded, "seconds": time.perf_counter() - start}

    try:
        if config["checkpoint_revision"] != base["revision"]:
            raise ValueError("Pinned Qwen revision changed")
        receipt["asset_sha256"] = verified_assets(args.checkpoint, base)
        torch, model, processor, receipt["hardware"] = load_model(args.checkpoint, base | {"attention": config["attention"]})
        save()
        frozen = CulturalQwen(model.model, config, adapted=False).eval()
        # Native-path parity uses a 2x2 cross of artwork and target language in the same padded batch.
        pixels = [prepare_image(group)[0] for group in groups[:2]]
        lang = ["English", "Vietnamese", "English", "Vietnamese"]
        row_images = [0, 0, 1, 1]
        texts = [language_prompt(processor, language, config["prompt"]) for language in lang]
        reference = processor(text=texts, images=[pixels[i] for i in row_images], return_tensors="pt",
                              padding=True, padding_side="left").to(frozen.device)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            vision = frozen.encode_vision(processor.image_processor(images=pixels, return_tensors="pt"))
            encoded = encode_text(processor, lang, [len(vision.merged[i]) for i in row_images], config["prompt"], frozen.device)
            for key in ("input_ids", "attention_mask"):
                if not torch.equal(encoded[key], reference[key]):
                    raise ValueError(f"Grouped {key} differs from the pinned native processor")
            expected = last_valid(model.model(**reference, use_cache=False).last_hidden_state, reference["attention_mask"])
            actual = frozen.features(vision, row_images, encoded)
            again = frozen.features(vision, row_images, encoded)
            initial_logits = frozen.logits(actual)
            native_diff = (actual.float() - expected.float()).abs().max()
            native_cosine = torch.nn.functional.cosine_similarity(actual.float(), expected.float()).min()
            native_vision = (tuple(vision.merged[i] for i in row_images),
                             [torch.cat([tap[i] for i in row_images]) for tap in vision.deepstack])
            with patch.object(model.model, "get_image_features", return_value=native_vision):
                same_vision = last_valid(model.model(**reference, use_cache=False).last_hidden_state,
                                         reference["attention_mask"])
            single_ref = processor(text=[texts[0]], images=[pixels[0]], return_tensors="pt").to(frozen.device)
            single_vision = frozen.encode_vision(processor.image_processor(images=[pixels[0]], return_tensors="pt"))
            single_text = encode_text(processor, lang[:1], [len(single_vision.merged[0])], config["prompt"], frozen.device)
            single_a = frozen.features(single_vision, [0], single_text)
            single_b = last_valid(model.model(**single_ref, use_cache=False).last_hidden_state, single_ref["attention_mask"])
            single_exact = torch.equal(single_a, single_b)
            shared_exact = torch.equal(actual, same_vision)
            receipt["parity"] = {"single_native_exact": single_exact, "same_vision_native_exact": shared_exact,
                                 "duplicate_vision_native_exact": torch.equal(actual, expected),
                                 "duplicate_vision_max_abs_error": float(native_diff),
                                 "duplicate_vision_min_cosine": float(native_cosine),
                                 "repeat_exact": torch.equal(actual, again), "token_ids_and_mask_exact": True}
            if not (single_exact and shared_exact and torch.equal(actual, again)):
                raise ValueError("Single-image native path, native decoder with shared vision, and repeat must match exactly")
            image = (actual[0].float() - actual[2].float()).norm()
            language = (actual[0].float() - actual[1].float()).norm()
            interaction = ((actual[0].float() - actual[1].float()) - (actual[2].float() - actual[3].float())).norm()
            if not (image > 0 and language > 0 and interaction > 0):
                raise ValueError("Artwork and target language must both affect the hidden representation")
            receipt["parity"].update(padded_tokens=int((encoded["attention_mask"] == 0).sum()),
                                     image_hidden_l2=float(image), language_hidden_l2=float(language),
                                     interaction_hidden_l2=float(interaction), initial_logits=initial_logits.tolist())
        del reference, expected, again, same_vision, native_vision, single_ref, single_vision, single_a, single_b
        torch.cuda.reset_peak_memory_stats()
        for group in groups:
            entry = measure_artwork(frozen, group)
            receipt["frozen"].append(entry)
            print(f"Frozen {len(receipt['frozen'])}/6: {entry['pairs']} pairs in {entry['seconds']:.2f}s", flush=True)
        receipt["frozen_memory"] = {"allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "reserved_bytes": torch.cuda.max_memory_reserved()}
        del frozen
        adapted = CulturalQwen(model.model, config, adapted=True).eval()
        with torch.no_grad():
            initial = adapted.features(vision, row_images, encoded)
            if not torch.equal(initial, actual) or not torch.equal(adapted.logits(initial), initial_logits):
                raise ValueError("Zero-init LoRA or task-head initialization changed the starting model")
        receipt["parity"]["matched_initial_logits_exact"] = True
        receipt["parity"]["zero_lora_hidden_exact"] = True
        del vision, encoded, pixels, actual, initial
        optimizer = optimizer_for(adapted, config)
        scaler = torch.amp.GradScaler("cuda", init_scale=128.)
        synthetic_mass = 5 * sum(len(row["languages"]) for row in groups) / len(groups)
        original = [p.detach().clone() for p in adapted.adapter_parameters()]
        frozen_versions = {name: p._version for name, p in adapted.named_parameters() if not p.requires_grad}
        receipt["trainable"] = {name: p.numel() for name, p in adapted.named_parameters() if p.requires_grad}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for group in groups:
            entry = measure_artwork(adapted, group, adapting=True)
            receipt["adapted"].append(entry)
            save()
            print(f"QLoRA {len(receipt['adapted'])}/6: {entry['pairs']} pairs in {entry['seconds']:.2f}s", flush=True)
        receipt["adapted_memory"] = {"allocated_bytes": torch.cuda.max_memory_allocated(),
                                     "reserved_bytes": torch.cuda.max_memory_reserved()}
        changed = [float((a.detach() - b).norm()) for a, b in zip(adapted.adapter_parameters(), original, strict=True)]
        if not all(n > 0 for n in changed) or not all(x["adapter_grad_norm"] > 0 for x in receipt["adapted"]):
            raise ValueError("Some QLoRA parameter matrices did not learn on the synthetic probe")
        if any(frozen_versions[name] != p._version or p.grad is not None
               for name, p in adapted.named_parameters() if not p.requires_grad):
            raise ValueError("A frozen parameter was updated or received gradients")
        receipt["adapter_updates_l2"] = changed
        receipt["frozen_parameters_unchanged"] = True
        receipt["training_peak_reserve_ok"] = (receipt["hardware"]["initial_free_bytes"] - receipt["adapted_memory"]["reserved_bytes"]
                                               >= config["memory_reserve_mib"] * 1024**2)
        for group in groups:
            receipt["adapted_inference"].append(measure_artwork(adapted, group))
        receipt["state"] = "PASS" if receipt["training_peak_reserve_ok"] else "INSUFFICIENT_MEMORY_MARGIN"
    except Exception as error:
        receipt.update(state="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/competition_direct_qwen/stage1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/qwen3-vl-4b-ebb281e")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/upstream")
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/competition_neighborhood_residual/results")
    parser.add_argument("--cuda-feasibility", action="store_true")
    parser.add_argument("--probe-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config, base = read_json(DIRECT), read_json(CONFIG)
    groups, proof = prepare(args, config, base)
    result = args.run_dir / "feasibility.json"
    if args.probe_worker:
        if not args.cuda_feasibility or result.exists():
            raise ValueError("The CUDA preflight requires explicit selection and a new output directory")
        preflight(args, config, base, groups, proof)
    elif args.cuda_feasibility:
        if result.exists():
            raise FileExistsError(result)
        try:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--probe-worker"],
                           check=True, timeout=config["feasibility_seconds"])
        except subprocess.TimeoutExpired as error:
            last = read_json(result) if result.is_file() else {"protocol": proof}
            write_atomic(result, last | {"state": "TIME_LIMIT", "error": str(error)})
            raise
    else:
        write_atomic(args.run_dir / "protocol.json", proof)
        print(f"Painting-disjoint fold {config['pilot_fold']}: {proof['fit']} fitting, {proof['held']} held; no votes read")


if __name__ == "__main__":
    main()
