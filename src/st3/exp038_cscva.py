"""Additional EXP-036 CS-CVA seeds with the epoch frozen at 5 (no Development epoch selection).

The recipe is ``configs/exp036.json`` with only ``seed`` replaced. The learning-rate schedule keeps the
original six-epoch horizon, so epoch-5 weights see the same schedule as the original epoch-5 checkpoint.
Training stops after epoch 5; Development is predicted once, after training, and scored by the caller.
"""

import importlib.metadata
import math
import os
import random
import shutil
import time

import numpy as np

from .exp036_io import read, sha, write

RUNTIME = ("torch", "torchvision", "transformers", "numpy", "pandas", "safetensors", "Pillow")


def stage_inputs(base, manifest, destination):
    """EXP-036's loader layout with ``images/<logical name>`` linked to the verified artwork files."""
    for name in ("training/annotations.csv", "phase_inputs/dev/st3_test.csv"):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(base / name, target)
    for logical, physical in manifest["images"].items():
        target = destination / "images" / logical
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink((base / physical).resolve(), target)
        except OSError:
            shutil.copyfile(base / physical, target)
    return destination


def train_seed(*, root, base, manifest, seed, folder, snapshot, deadline, targets_sha, preflight_only=False,
               freeze_epoch=5, source=""):
    import torch
    from transformers import AutoImageProcessor

    import run_exp036
    from .exp036 import Candidate, inference_state
    from .exp036_train import Paintings, amp, backbone, budget, model_assets, panels

    original = read(root / "configs/exp036.json")
    cfg = original | {"seed": seed}
    if not 1 <= freeze_epoch <= cfg["epochs"]:
        raise ValueError("Frozen epoch must lie inside the original schedule")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CS-CVA seeds need one visible CUDA GPU")
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    folder.mkdir(parents=True, exist_ok=True)
    staged = stage_inputs(base, manifest, folder.parent / f"tmp_inputs_{seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    directory = model_assets(cfg, snapshot, text_assets=False)
    pairs, dev, _ = panels(staged, cfg)
    target_path = base / "training/targets.npz"
    if sha(target_path) != targets_sha:
        raise ValueError("Sealed EXP-036 caption targets differ")
    with np.load(target_path, allow_pickle=False) as arrays:
        values, mask = arrays["values"], arrays["mask"]
    if values.shape != (len(pairs), cfg["hidden_size"]) or mask.shape != (len(pairs),) or not np.isfinite(values).all():
        raise ValueError("Caption targets do not align with the training pairs")
    langs = {lang: i for i, lang in enumerate(sorted(pairs.language.unique()))}
    train = Paintings(pairs, langs, values=values, mask=mask)
    development = Paintings(dev, langs)
    processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True, use_fast=False)
    vision = backbone(directory, cfg, text=False)
    vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pixels = train.pixels([0], staged, processor, device)
    vision.to(device)
    with torch.no_grad(), amp(device):
        reference = torch.nn.functional.normalize(vision.vision_model(pixels).pooler_output.float(), dim=-1)
    model = Candidate(vision.vision_model, len(langs), cfg).to(device)
    model.eval()
    with torch.no_grad(), amp(device):
        frozen = model.features(pixels)
    torch.testing.assert_close(reference, frozen, atol=2e-5, rtol=2e-4)
    del pixels, reference, frozen
    started = time.monotonic()
    micro, trials, throughput = run_exp036.benchmark(model, train, processor, staged, cfg, device, deadline)
    preflight = {"seed": seed, "microbatch": micro, "attempts": trials, "paintings_per_second": throughput,
                 "benchmark_seconds": time.monotonic() - started, "gpu": torch.cuda.get_device_name(0),
                 "forecast_fit_seconds": len(train.members) * freeze_epoch / throughput,
                 "development_paintings": len(development.members)}
    if preflight_only:
        shutil.rmtree(staged, ignore_errors=True)
        return preflight
    adapters = [p for n, p in model.named_parameters() if n.startswith("vision.") and p.requires_grad]
    heads = [p for n, p in model.named_parameters() if not n.startswith("vision.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": adapters, "lr": cfg["adapter_lr"]}, {"params": heads, "lr": cfg["head_lr"]}],
                            weight_decay=cfg["weight_decay"], foreach=False)
    steps = math.ceil(len(train.members) / cfg["batch_paintings"])
    full_steps = steps * cfg["epochs"]
    warmup = max(1, int(full_steps * cfg["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warmup if i < warmup else
                                                  0.1 + 0.9 * (1 + math.cos(math.pi * (i - warmup) / (full_steps - warmup))) / 2)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    window = cfg["batch_paintings"]
    history = []
    for epoch in range(1, freeze_epoch + 1):
        model.train()
        order = np.random.default_rng(cfg["seed"] + epoch).permutation(len(train.members))
        sums, began = np.zeros(2), time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        for batch in range(steps):
            budget(deadline)
            group = order[batch * window:(batch + 1) * window]
            for _ in range(4):
                opt.zero_grad(set_to_none=True)
                losses = run_exp036.optimize(model, train, group, micro, staged, processor, cfg, device, scaler)
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(adapters + heads, cfg["gradient_clip"])
                if torch.isfinite(norm):
                    scaler.step(opt)
                    scaler.update()
                    break
                scaler.update(new_scale=scaler.get_scale() / 2)
            else:
                raise FloatingPointError("Repeated nonfinite gradient in the same optimizer step")
            scheduler.step()
            sums += losses
        history.append({"epoch": epoch, "bce": sums[0] / steps, "caption_cosine": sums[1] / steps,
                        "seconds": time.monotonic() - began, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        print(f"CS-CVA seed {seed}: epoch {epoch}/{freeze_epoch} trained in {history[-1]['seconds']:.0f}s", flush=True)
    tick = time.monotonic()
    prediction = run_exp036.predictions(model, development, staged, processor, micro, device, deadline)
    prediction.to_csv(folder / "predictions.csv", index=False, lineterminator="\n", float_format="%.6f")
    bundle = folder / "inference"
    bundle.mkdir(exist_ok=True)
    torch.save(inference_state(model), bundle / "weights.pt")
    processor.save_pretrained(bundle)
    runtime = {name: importlib.metadata.version(name) for name in RUNTIME}
    write(bundle / "bundle.json", {"config": original, "seed": seed, "languages": langs, "epoch": freeze_epoch,
                                   "source": source, "weights_sha256": sha(bundle / "weights.pt"),
                                   "processor_sha256": sha(bundle / "preprocessor_config.json"),
                                   "identity": {"runtime": runtime, "gpu": torch.cuda.get_device_name(0)}})
    write(folder / "train.json", {"seed": seed, "freeze_epoch": freeze_epoch, "schedule_epochs": cfg["epochs"],
                                  "steps_per_epoch": steps, "history": history, "preflight": preflight,
                                  "development_inference_seconds": time.monotonic() - tick})
    shutil.rmtree(staged, ignore_errors=True)
    return preflight

