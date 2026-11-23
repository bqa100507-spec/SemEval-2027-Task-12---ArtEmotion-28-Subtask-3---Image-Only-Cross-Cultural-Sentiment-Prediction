#!/usr/bin/env python3
"""Run the one EXP-036 fit, score saved predictions, or replay an image-only bundle."""

import argparse
import importlib.metadata
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3 import data, submission
from st3.contrast_study import new_output
from st3.exp036 import Candidate, choose_epoch, finish, inference_state, loss_parts, rounded
from st3.exp036_io import check, export, read, restore, sha, write
from st3.exp036_train import Paintings, amp, backbone, budget, model_assets, panels, text_targets
from st3.metrics import score

CONFIG = ROOT / "configs/exp036.json"
RUNTIME = ("torch", "torchvision", "transformers", "numpy", "pandas", "safetensors", "Pillow")


def predictions(model, panel, inputs, processor, micro, device, deadline):
    model.eval()
    values = np.zeros(len(panel.pairs))
    with torch.inference_mode():
        for i in range(0, len(panel.members), micro):
            budget(deadline)
            group = list(range(i, min(i + micro, len(panel.members))))
            ids, index = panel.groups(group)
            pixels = panel.pixels(group, inputs, processor, device)
            with amp(device):
                q = model.predict(pixels, torch.as_tensor(index, device=device),
                                  torch.as_tensor(panel.languages[ids], device=device))
            values[ids] = q.float().cpu().numpy()
    return rounded(panel.pairs[data.KEYS].assign(p_positive=values), panel.pairs)


def apply_state(model, state):
    wanted = {name for name, p in model.named_parameters() if p.requires_grad}
    if state.keys() != wanted:
        raise ValueError("Checkpoint trainable keys differ")
    model.load_state_dict(state, strict=False)


def rng_state(device):
    name, keys, pos, gaussian, cache = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "python": random.getstate(), "numpy": (name, keys.tolist(), pos, gaussian, cache)}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    random.setstate(state["python"])
    name, keys, pos, gaussian, cache = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cache))


def checkpoint(root, identity, model, opt, scheduler, scaler, boundary, device):
    index = root / "checkpoints"
    index.mkdir(exist_ok=True)
    name = f"{boundary['epoch']:02d}-{boundary['step']:05d}-{len(boundary['history']):02d}"
    dest = index / name
    if not dest.exists():
        with new_output(dest) as stage:
            state = {"model": {n: p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
                     "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "rng": rng_state(device), "boundary": boundary}
            torch.save(state, stage / "state.pt")
            write(stage / "seal.json", {"identity": identity, "sha256": sha(stage / "state.pt")})
    write(root / "latest.json", {"name": name})
    return dest


def resume_state(root, identity):
    if not (root / "latest.json").exists():
        return None
    latest = read(root / "latest.json")["name"]
    index = root / "checkpoints" / latest
    seal = read(index / "seal.json")
    if seal["identity"] != identity:
        raise ValueError("Checkpoint identity differs; restore on the same pinned runtime and accelerator type")
    check(index, {"state.pt": seal["sha256"]})
    return torch.load(index / "state.pt", weights_only=True, map_location="cpu")


def checkpoint_files(root):
    files = [p for p in root.iterdir() if p.is_file() and not p.name.endswith(".tmp")]
    for folder in ("epochs", "inference", "standalone", "ensemble"):
        files.extend(p for p in (root / folder).rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    if (root / "latest.json").exists():
        index = root / "checkpoints" / read(root / "latest.json")["name"]
        files.extend(index / name for name in ("seal.json", "state.pt"))
    if not (root / "targets.json").exists():
        for p in (root / "caption_cache").glob("*.json"):
            files.extend((p, p.with_suffix(".npy")))
    return files


def optimize(model, panel, order, microbatch, inputs, processor, cfg, device, scaler):
    ids, _ = panel.groups(order)
    vote_mass = float(panel.n[ids].sum())
    caption_mass = max(1., float((panel.mask[ids] * panel.n[ids]).sum()))
    terms = np.zeros(2)
    for i in range(0, len(order), microbatch):
        micro = order[i:i + microbatch]
        ids, groups = panel.groups(micro)
        pixels = panel.pixels(micro, inputs, processor, device)
        batch = panel.batch(ids, groups, device)
        with amp(device):
            bce, caption = loss_parts(model, pixels, batch, vote_mass, caption_mass, cfg["caption_epsilon"])
            value = bce + cfg["caption_lambda"] * caption
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite CS-CVA objective")
        scaler.scale(value).backward()
        terms += [bce.detach().item(), caption.detach().item()]
    return terms


def benchmark(model, panel, processor, inputs, cfg, device, deadline):
    model.train()
    timings = []
    order = np.random.default_rng(cfg["seed"]).permutation(len(panel.members))
    batch = cfg["batch_paintings"]
    for micro in cfg["microbatches"]:
        budget(deadline)
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        before = time.monotonic()
        failed = None
        try:
            scaler = torch.amp.GradScaler("cuda", init_scale=128.)
            optimize(model, panel, order[:batch], micro, inputs, processor, cfg, device, scaler)
            optimize(model, panel, order[batch:2*batch], micro, inputs, processor, cfg, device, scaler)
            trainable = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
                raise FloatingPointError("Invalid preflight gradients")
            if not any(m.right.grad.norm() > 0 for m in model.modules() if hasattr(m, "right")):
                raise RuntimeError("No LoRA update signal on real training paintings")
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError("A frozen parameter received a gradient")
        except torch.cuda.OutOfMemoryError:
            failed = "CUDA OOM"
        elapsed = time.monotonic() - before
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        if reserved + cfg["fp16_margin_mib"] * 2**20 >= total:
            failed = "insufficient VRAM margin"
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        timings.append({"microbatch": micro, "seconds": elapsed, "peak_allocated_bytes": allocated,
                        "peak_reserved_bytes": reserved, "failure": failed})
        if failed is None:
            return micro, timings, (2*batch / elapsed)
    raise RuntimeError(f"No feasible painting microbatch: {timings}")


def fitting(args, manifest, cfg, deadline, device, identity):
    from transformers import AutoImageProcessor

    root = args.output
    inputs = args.inputs
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    directory = model_assets(cfg, args.model)
    pairs, dev, captions = panels(inputs, cfg)
    langs = {lang: i for i, lang in enumerate(sorted(pairs.language.unique()))}
    values, mask = text_targets(root, directory, pairs, captions, cfg, device, deadline)
    del captions
    train = Paintings(pairs, langs, values=values, mask=mask)
    development = Paintings(dev, langs)
    processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True, use_fast=False)
    vision = backbone(directory, cfg, text=False)
    vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    previous = resume_state(root, identity)
    if previous is None:
        pixels = train.pixels([0], inputs, processor, device)
        vision.to(device)
        with torch.no_grad(), amp(device):
            reference = torch.nn.functional.normalize(vision.vision_model(pixels).pooler_output.float(), dim=-1)
    model = Candidate(vision.vision_model, len(langs), cfg).to(device)
    if previous:
        micro = previous["boundary"]["microbatch"]
        write(root / "resume_receipt.json", {"epoch": previous["boundary"]["epoch"],
                                            "completed_steps_in_epoch": previous["boundary"]["step"]})
    else:
        model.eval()
        with torch.no_grad(), amp(device):
            frozen = model.features(pixels)
        torch.testing.assert_close(reference, frozen, atol=2e-5, rtol=2e-4)
        difference = (reference - frozen).abs().max().item()
        del pixels, reference, frozen
        micro, trials, throughput = benchmark(model, train, processor, inputs, cfg, device, deadline)
        write(root / "preflight.json", {"gpu": identity["gpu"], "adapter_targets": model.targets,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "microbatch": micro, "attempts": trials, "zero_lora_max_abs_error": difference,
              "paintings_per_second": throughput,
              "estimated_training_seconds": len(train.members)*cfg["epochs"]/throughput})
        print(f"{throughput:.2f} train paintings/s, microbatch {micro}. Forecast recorded; using session progress.", flush=True)

    adapters = [p for n,p in model.named_parameters() if n.startswith("vision.") and p.requires_grad]
    heads = [p for n,p in model.named_parameters() if not n.startswith("vision.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": adapters, "lr": cfg["adapter_lr"]}, {"params": heads, "lr": cfg["head_lr"]}],
                            weight_decay=cfg["weight_decay"], foreach=False)
    steps = math.ceil(len(train.members) / cfg["batch_paintings"])
    full_steps = steps * cfg["epochs"]
    warmup = max(1, int(full_steps * cfg["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i+1)/warmup if i < warmup else
        0.1 + 0.9*(1+math.cos(math.pi*(i-warmup)/(full_steps-warmup)))/2)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    boundary = {"epoch": 1, "step": 0, "sums": [0., 0.], "history": [], "microbatch": micro}
    if previous:
        apply_state(model, previous["model"])
        opt.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        restore_rng(previous["rng"], device)
        boundary = previous["boundary"]
        del previous

    # Development counts are opened by the scorer, after prediction CSVs exist.
    def gold():
        return pd.read_csv(inputs / "scoring/dev_gold.csv")

    history = boundary["history"]
    window = cfg["batch_paintings"]
    while boundary["epoch"] <= cfg["epochs"]:
        epoch = boundary["epoch"]
        model.train()
        order = np.random.default_rng(cfg["seed"]+epoch).permutation(len(train.members))
        torch.cuda.reset_peak_memory_stats(device)
        began = time.monotonic()
        for batch in range(boundary["step"], steps):
            if time.time() >= deadline:
                checkpoint(root, identity, model, opt, scheduler, scaler, boundary, device)
                budget(deadline)
            group = order[batch*window:(batch+1)*window]
            for attempt in range(4):
                opt.zero_grad(set_to_none=True)
                losses = optimize(model, train, group, micro, inputs, processor, cfg, device, scaler)
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(adapters + heads, cfg["gradient_clip"])
                scaler.step(opt)
                scaler.update()
                if torch.isfinite(norm):
                    break
            else:
                raise FloatingPointError("Repeated nonfinite gradient in the same optimizer step")
            scheduler.step()
            boundary["sums"] = (np.asarray(boundary["sums"]) + losses).tolist()
            boundary["step"] = batch + 1
            if (batch+1) % cfg["checkpoint_steps"] == 0 or batch+1 == steps:
                checkpoint(root, identity, model, opt, scheduler, scaler, boundary, device)
                print(f"Epoch {epoch}/{cfg['epochs']} step {batch+1}/{steps}", flush=True)

        pred = predictions(model, development, inputs, processor, micro, device, deadline)
        folder = root / "epochs" / f"epoch_{epoch}"
        folder.mkdir(parents=True, exist_ok=True)
        pred.to_csv(folder / "predictions.csv", index=False, lineterminator="\n", float_format="%.6f")
        row = {"epoch": epoch, "bce": boundary["sums"][0]/steps, "caption_cosine": boundary["sums"][1]/steps,
               "metrics": score(gold(), pred), "seconds_this_session": time.monotonic()-began,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "predictions_sha256": sha(folder / "predictions.csv")}
        torch.save(inference_state(model), folder / "inference.pt.tmp")
        (folder / "inference.pt.tmp").replace(folder / "inference.pt")
        row["inference_sha256"] = sha(folder / "inference.pt")
        write(folder / "metrics.json", row)
        history.append(row)
        boundary = boundary | {"epoch": epoch+1, "step": 0, "sums": [0.,0.]}
        checkpoint(root, identity, model, opt, scheduler, scaler, boundary, device)
        write(root / "history.json", history)
        print(f"Epoch {epoch}: Contrast={row['metrics']['cultural_contrast']:.6f}, "
              f"log loss={row['metrics']['log_loss']:.6f}", flush=True)
        export(root.parent / "exp036_progress.zip", root, checkpoint_files(root), args.handoff)

    selected = choose_epoch(history)
    saved = root / "epochs" / f"epoch_{selected['epoch']}"
    check(saved, {"inference.pt": selected["inference_sha256"], "predictions.csv": selected["predictions_sha256"]})
    state = torch.load(saved / "inference.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=False)
    result = predictions(model, development, inputs, processor, micro, device, deadline)
    recorded = pd.read_csv(saved / "predictions.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(result, recorded, check_exact=True)
    ridge = submission.read_predictions(inputs / "ridge/predictions.csv", dev)
    files = finish(dev, gold(), result, ridge, root, cfg)
    inference = root / "inference"
    inference.mkdir(exist_ok=True)
    shutil.copyfile(saved / "inference.pt", inference / "weights.pt")
    processor.save_pretrained(inference)
    write(inference / "bundle.json", {"config": cfg, "languages": langs, "epoch": selected["epoch"],
          "source": manifest["source"], "weights_sha256": selected["inference_sha256"],
          "processor_sha256": sha(inference / "preprocessor_config.json"), "identity": identity})
    write(root / "metrics.json", files | {"selected_epoch": selected["epoch"],
          "standalone_sha256": sha(root / "exp036_standalone_submission.zip"),
          "ensemble_sha256": sha(root / "exp036_ensemble_submission.zip")})


def run(args):
    cfg = read(CONFIG)
    manifest = read(args.inputs / "manifest.json")
    if sha(args.inputs / "manifest.json") != args.handoff:
        raise ValueError("EXP-036 handoff manifest mismatch")
    check(args.inputs, manifest["files"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Enable Kaggle CUDA with one visible GPU")
    device = torch.device("cuda:0")
    # The same deterministic kernel policy is used for fitting, resume and inference replay.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    identity = {"handoff": args.handoff, "source": manifest["source"], "gpu": torch.cuda.get_device_name(0),
                "python": sys.version.split()[0], "cuda": torch.version.cuda,
                "runtime": {name: importlib.metadata.version(name) for name in RUNTIME}}
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        restore(args.resume, root, args.handoff)
    if (root / "run.json").exists() and read(root / "run.json") != identity:
        raise ValueError("Run identity changed")
    write(root / "run.json", identity)
    write(root / "status.json", {"state": "IN_PROGRESS"})
    shutil.copyfile(CONFIG, root / "exp036.json")
    try:
        fitting(args, manifest, cfg, args.deadline, device, identity)
        write(root / "status.json", {"state": "COMPLETE"})
    except (Exception, KeyboardInterrupt) as exc:
        write(root / "status.json", {"state": "RESUMABLE_STOP" if isinstance(exc, TimeoutError) else "FAILED",
                                     "reason": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if args.log.is_file():
            shutil.copyfile(args.log, root / "worker.log")
        try:
            export(root.parent / "exp036_results.zip", root, checkpoint_files(root), args.handoff)
        except Exception as exc:
            write(root / "status.json", {"state": "EXPORT_FAILED", "reason": str(exc)})
            raise


def infer(args):
    from transformers import AutoImageProcessor

    bundle = read(args.bundle / "bundle.json")
    check(args.bundle, {"weights.pt": bundle["weights_sha256"],
                        "preprocessor_config.json": bundle["processor_sha256"]})
    if bundle["config"] != read(CONFIG):
        raise ValueError("Bundle is incompatible with this EXP-036 source")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the pinned FP16 inference runtime")
    if {name: importlib.metadata.version(name) for name in RUNTIME} != bundle["identity"]["runtime"]:
        raise ValueError("Inference dependencies differ from the trained bundle")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    cfg = bundle["config"]
    directory = model_assets(cfg, args.model, text_assets=False)
    processor = AutoImageProcessor.from_pretrained(args.bundle, local_files_only=True, use_fast=False)
    model = Candidate(backbone(directory, cfg, text=False).vision_model,
                      len(bundle["languages"]), cfg, auxiliary=False).to(device)
    apply_state(model, torch.load(args.bundle / "weights.pt", weights_only=True, map_location="cpu"))
    with args.pairs.open(encoding="utf-8", newline="") as f:
        pairs = data.read_exact_csv(f, data.INPUT_COLUMNS)
    data.unique_pairs(pairs)
    frame = predictions(model, Paintings(pairs, bundle["languages"]), args.images, processor, 1, device, float("inf"))
    frame.to_csv(args.destination, index=False, float_format="%.6f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    for field in ("inputs", "output", "log"):
        train.add_argument(f"--{field}", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--handoff", required=True)
    train.add_argument("--deadline", type=float, required=True)
    predict = sub.add_parser("infer")
    for field in ("bundle", "pairs", "images", "destination"):
        predict.add_argument(f"--{field}", type=Path, required=True)
    for command in (train, predict):
        command.add_argument("--model", type=Path)
    args = parser.parse_args()
    (run if args.command == "train" else infer)(args)
