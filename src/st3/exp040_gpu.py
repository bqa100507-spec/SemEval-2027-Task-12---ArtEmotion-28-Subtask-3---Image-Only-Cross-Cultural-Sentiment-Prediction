"""EXP-040 GPU path: Qwen3-VL-Embedding-2B (E3's exact formatting, pixels and last-token pooling) with LoRA
on every decoder layer's q/k/v/o and the ridge-initialized interaction head.

Heavy imports stay inside functions so CPU tests can import this module. One painting is forwarded once per
micro-step and scores all its eligible languages; 16 paintings form one optimizer step (gradient accumulation).
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import KEYS
from .exp036_io import sha, write
from .exp038_gpu import last_token, load_rgb, official_instruction, qwen_conversation
from .exp040 import (binomial_nll, choose_epoch, interaction_head, schedule, steps_per_epoch,
                     test_like_contrast)

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class TrainingTimeout(TimeoutError):
    """The session deadline arrived; the latest checkpoint is complete and resumable."""


def deterministic(seed):
    import random

    import torch

    from .exp038_gpu import deterministic_cuda

    deterministic_cuda()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lora_class():
    """EXP-037's LowRankUpdate (FP32 factors, zero right factor) plus LoRA input dropout."""
    from torch import nn

    from .competition_direct_qwen import LowRankUpdate

    class DropoutLowRank(LowRankUpdate):
        def __init__(self, base, rank, alpha, generator, dropout):
            super().__init__(base, rank, alpha, generator)
            self.dropout = nn.Dropout(dropout)

        def forward(self, states):
            import torch
            import torch.nn.functional as F

            base = self.base(states)
            with torch.autocast(device_type=states.device.type, enabled=False):
                delta = F.linear(F.linear(self.dropout(states.float()), self.left), self.right)
            return base + (delta * self.scale).to(base.dtype)

    return DropoutLowRank


def attach_lora(language_model, lora, seed):
    """Replace q/k/v/o of every decoder layer; returns the adapted module names. Nothing else trains."""
    import torch
    from torch import nn

    layers = language_model.layers
    if len(layers) != lora["decoder_layers"]:
        raise ValueError(f"Unexpected decoder depth {len(layers)}")
    if tuple(lora["targets"]) != TARGETS:
        raise ValueError("LoRA targets are fixed to q/k/v/o")
    cls = lora_class()
    generator = torch.Generator().manual_seed(seed)
    names = []
    for i, layer in enumerate(layers):
        for target in TARGETS:
            base = getattr(layer.self_attn, target)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Unexpected LoRA target {i}.{target}")
            setattr(layer.self_attn, target, cls(base, lora["rank"], lora["alpha"], generator, lora["dropout"]))
            names.append(f"language_model.layers.{i}.self_attn.{target}")
    return names


def lora_parameters(module):
    from .competition_direct_qwen import LowRankUpdate

    return [p for m in module.modules() if isinstance(m, LowRankUpdate) for p in (m.left, m.right)]


def single_gpu(module, device):
    """Assert single-GPU placement: every parameter and buffer on ``device``."""
    places = {str(t.device) for t in [*module.parameters(), *module.buffers()]}
    if places != {str(device)}:
        raise RuntimeError(f"E3L must sit on one GPU ({device}); found {sorted(places)}")
    return sorted(places)


def load_backbone(spec, directory, device):
    """The pinned 2B checkpoint, FP16, single GPU (device_map {'': 0}), frozen, gradient checkpointing."""
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    from .key_audit import forbid_head, loader_ignored

    if str(device) != "cuda:0":
        raise ValueError("E3L runs on cuda:0 only")
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(
        directory, dtype=torch.float16, attn_implementation=spec["attention"], local_files_only=True,
        output_loading_info=True, device_map={"": 0})
    ignored, missing = loader_ignored(info, model.config.to_dict())
    forbid_head(model)
    if model.config.text_config.hidden_size != spec["width"]:
        raise ValueError("Unexpected Qwen3-VL-Embedding width")
    model.requires_grad_(False).eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    processor = Qwen3VLProcessor.from_pretrained(directory, local_files_only=True, padding_side="right")
    return model, processor, {"loader_missing_keys": missing, "ignored_missing_keys": ignored}


def encode(processor, spec, images, device):
    """E3's official formatting for a batch of images with the E3 instruction (right padding)."""
    from qwen_vl_utils.vision_process import process_vision_info

    conversations = [qwen_conversation(image, spec["instruction"], spec) for image in images]
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    pictures, videos, video_kwargs = process_vision_info(conversations, image_patch_size=16,
                                                         return_video_metadata=True, return_video_kwargs=True)
    if videos is not None:
        raise ValueError("Unexpected video input")
    inputs = processor(text=text, images=pictures, videos=None, video_metadata=None, truncation=True,
                       max_length=spec["max_length"], padding=True, do_resize=False, return_tensors="pt",
                       **video_kwargs)
    return {key: value.to(device) for key, value in inputs.items()}


def pooled(backbone, inputs):
    """Unit-norm FP32 last-token embedding (differentiable)."""
    import torch.nn.functional as F

    hidden = backbone(**inputs, use_cache=False).last_hidden_state
    return F.normalize(last_token(hidden, inputs["attention_mask"]).float(), dim=-1)


class Paintings:
    """Training/evaluation rows grouped by painting: image path, head-language indices, k and n."""

    def __init__(self, pairs, images, languages):
        index = {language: i for i, language in enumerate(languages)}
        self.names, self.rows = [], []
        for painting, part in pairs.sort_values(KEYS).groupby("painting", sort=True):
            if part.language.map(index).isna().any():
                raise ValueError("A pair language is missing from the head")
            self.names.append(painting)
            self.rows.append((images[painting], part.language.map(index).to_numpy(dtype=np.int64),
                              part.k.to_numpy(dtype=np.float32), part.n.to_numpy(dtype=np.float32),
                              part.language.to_numpy()))

    def __len__(self):
        return len(self.names)


class E3L:
    """Backbone (frozen, LoRA on q/k/v/o) + interaction head; ``self.trainable`` lists LoRA then head params."""

    def __init__(self, model, processor, spec, head, lora_names, device):
        self.model, self.backbone, self.processor = model, model.model, processor
        self.spec, self.head, self.device = spec, head.to(device), device
        self.lora_names = lora_names
        self.lora = lora_parameters(self.backbone)
        for p in self.lora:
            p.requires_grad_(True)
        self.trainable = [*self.lora, *self.head.parameters()]
        trainable_ids = {id(p) for p in self.trainable}
        extra = [n for n, p in self.backbone.named_parameters() if p.requires_grad and id(p) not in trainable_ids]
        if extra:
            raise RuntimeError(f"Unexpected trainable backbone parameters: {extra[:3]}")

    def train(self, mode=True):
        self.backbone.eval()
        self.backbone.language_model.train(mode)  # Dropout on LoRA inputs and HF checkpointing need train mode.
        self.head.train(mode)

    def features(self, image_paths):
        """FP16 autocast for training and prediction alike (the step-0 check uses this same path)."""
        import torch

        images = [load_rgb(path) for path in image_paths]
        with torch.autocast("cuda", dtype=torch.float16):
            return pooled(self.backbone, encode(self.processor, self.spec, images, self.device))

    def state(self):
        names = {id(p): n for n, p in self.backbone.named_parameters()}
        tensors = {f"lora:{names[id(p)]}": p.detach().cpu().clone() for p in self.lora}
        tensors |= {f"head:{n}": p.detach().cpu().clone() for n, p in self.head.named_parameters()}
        return tensors

    def load_state(self, tensors):
        import torch

        current = self.state()
        if set(current) != set(tensors) or any(current[k].shape != tensors[k].shape for k in current):
            raise ValueError("Learned tensor identities or shapes differ")
        names = {n: p for n, p in self.backbone.named_parameters()}
        heads = dict(self.head.named_parameters())
        with torch.no_grad():
            for key, value in tensors.items():
                kind, name = key.split(":", 1)
                (names[name] if kind == "lora" else heads[name]).copy_(value)


def build(spec, lora, head_fields, languages, lam, directory, device, seed):
    from .competition_ridge import Head

    model, processor, receipt = load_backbone(spec, directory, device)
    names = attach_lora(model.model.language_model, lora, seed)
    head = interaction_head(Head(float(head_fields["intercept"]), head_fields["language_intercepts"],
                                 head_fields["shared"], head_fields["deviations"],
                                 tuple(head_fields["languages"].tolist())), languages, lam)
    e3l = E3L(model, processor, spec, head, names, device)
    receipt |= {"placement": single_gpu(model, device), "lora_modules": len(names),
                "lora_parameters": int(sum(p.numel() for p in e3l.lora)),
                "head_parameters": int(sum(p.numel() for p in e3l.head.parameters()))}
    return e3l, receipt


def predict(e3l, panel, log=None, label="predict"):
    """Eval-mode embeddings (float32) and per-pair probabilities for every painting of ``panel``."""
    import torch

    e3l.train(False)
    embeddings, rows = [], []
    tick = time.time()
    with torch.inference_mode():
        for i, (path, index, _, _, languages) in enumerate(panel.rows):
            features = e3l.features([path])
            logits = e3l.head(features)[0, torch.as_tensor(index, device=e3l.device)]
            embeddings.append(features[0].cpu().numpy().astype(np.float32))
            probabilities = torch.sigmoid(logits.double()).cpu().numpy()
            rows.extend(zip([panel.names[i]] * len(languages), languages, probabilities))
            if log and (i + 1) % 200 == 0:
                log(f"{label}: {i + 1}/{len(panel)} paintings, {(i + 1) / (time.time() - tick):.2f}/s")
    values = np.stack(embeddings) if embeddings else np.zeros((0, e3l.spec["width"]), dtype=np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite E3L embedding")
    frame = pd.DataFrame(rows, columns=["painting", "language", "p_positive"])
    return pd.DataFrame(values, index=panel.names), frame, len(panel) / max(time.time() - tick, 1e-9)


def optimizer(e3l, train_cfg, total_steps):
    import torch

    opt = torch.optim.AdamW([{"params": e3l.lora, "lr": train_cfg["lora_lr"], "weight_decay": train_cfg["lora_weight_decay"]},
                             {"params": list(e3l.head.parameters()), "lr": train_cfg["head_lr"],
                              "weight_decay": train_cfg["head_weight_decay"]}], foreach=False)
    plan = torch.optim.lr_scheduler.LambdaLR(opt, schedule(total_steps, train_cfg["warmup_fraction"]))
    scaler = torch.amp.GradScaler("cuda", init_scale=train_cfg["grad_scaler_init"])
    return opt, plan, scaler


def optimizer_step(e3l, panel, group, opt, scaler, train_cfg):
    """One optimizer step over ``group`` paintings (one forward each); retries a nonfinite step at half scale."""
    import torch

    votes = float(sum(panel.rows[i][3].sum() for i in group))
    tick = time.time()
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        nll_total = 0.
        for i in group:
            path, index, k, n, _ = panel.rows[i]
            features = e3l.features([path])
            logits = e3l.head(features)[0, torch.as_tensor(index, device=e3l.device)]
            k_t, n_t = torch.as_tensor(k, device=e3l.device), torch.as_tensor(n, device=e3l.device)
            part = binomial_nll(logits, k_t, n_t) * (float(n.sum()) / votes)
            if not torch.isfinite(part):
                raise FloatingPointError("Nonfinite binomial loss")
            scaler.scale(part).backward()
            nll_total += float(part.detach())
        anchor = e3l.head.lam * e3l.head.anchor()
        scaler.scale(anchor).backward()
        scaler.unscale_(opt)
        if any(p.grad is None for p in e3l.trainable):
            raise RuntimeError("Disconnected trainable parameters")
        norm = torch.nn.utils.clip_grad_norm_(e3l.trainable, train_cfg["gradient_clip"])
        if torch.isfinite(norm):
            scaler.step(opt)
            scaler.update()
            return {"nll": nll_total, "anchor": float(anchor.detach()), "grad_norm": float(norm),
                    "clipped": bool(norm > train_cfg["gradient_clip"]), "scale": float(scaler.get_scale()),
                    "seconds": time.time() - tick}
        scaler.update(new_scale=scaler.get_scale() / 2)
    raise FloatingPointError("Repeated nonfinite gradients at one optimizer boundary")


def rng_state():
    import random

    import torch

    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def set_rng(state):
    import random

    import torch

    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_atomic(obj, path):
    import torch

    scratch = path.with_name(path.name + ".tmp")
    torch.save(obj, scratch)
    scratch.replace(path)
    return sha(path)


def inner_contrast(e3l, inner, inner_gold, excluded):
    _, frame, rate = predict(e3l, inner)
    return test_like_contrast(inner_gold, frame, excluded), frame, rate


def train(e3l, panel, train_cfg, folder, deadline, seed, *, epochs_to_run, inner=None, inner_gold=None,
          excluded=(), identity=None, log=print, max_steps=None):
    """Resumable training. Saves ``epochs/epoch_{e}.pt`` (LoRA + head) after each epoch, starting with epoch 0.

    With ``inner`` the inner test-like Contrast is recorded per epoch (epoch 0 = the frozen initialization).
    The schedule always spans ``train_cfg['epochs']`` epochs; ``epochs_to_run`` may stop earlier (full refit).
    """
    import torch

    folder = Path(folder)
    (folder / "epochs").mkdir(parents=True, exist_ok=True)
    batch = train_cfg["batch_paintings"]
    steps = steps_per_epoch(len(panel), batch)
    total = steps * train_cfg["epochs"]
    opt, plan, scaler = optimizer(e3l, train_cfg, total)
    checkpoint = folder / "state.pt"
    boundary = {"epoch": 1, "step": 0, "history": [], "log": []}
    if checkpoint.is_file():
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        if saved["identity"] != identity:
            raise ValueError("Fold checkpoint identity differs; refusing to resume")
        e3l.load_state(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        plan.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        set_rng(saved["rng"])
        boundary = saved["boundary"]
        log(f"Resumed at epoch {boundary['epoch']} step {boundary['step']}")
    else:
        receipt = {"sha256": save_atomic(e3l.state(), folder / "epochs/epoch_0.pt")}
        if inner is not None:
            contrast, frame, rate = inner_contrast(e3l, inner, inner_gold, excluded)
            receipt |= {"inner_test_like_contrast": contrast, "inner_paintings_per_second": rate}
            log(f"Epoch 0 (frozen init): inner test-like Contrast {contrast:.6f}")
        boundary["history"].append({"epoch": 0, **receipt})

    def save_state():
        save_atomic({"identity": identity, "model": e3l.state(), "optimizer": opt.state_dict(),
                     "scheduler": plan.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                     "boundary": boundary}, checkpoint)

    executed = 0
    while boundary["epoch"] <= epochs_to_run:
        epoch = boundary["epoch"]
        e3l.train(True)
        order = np.random.default_rng(seed * 100 + epoch).permutation(len(panel))
        tick = time.time()
        torch.cuda.reset_peak_memory_stats(e3l.device)
        for step in range(boundary["step"], steps):
            if time.time() >= deadline:
                save_state()
                raise TrainingTimeout(f"Deadline at epoch {epoch} step {step}")
            group = order[step * batch:(step + 1) * batch]
            row = optimizer_step(e3l, panel, group, opt, scaler, train_cfg)
            plan.step()
            boundary["step"] = step + 1
            boundary["log"].append({"epoch": epoch, "step": step + 1, **row, "lr": plan.get_last_lr()})
            executed += 1
            if (step + 1) % train_cfg["log_steps"] == 0:
                log(f"Epoch {epoch} step {step + 1}/{steps}: nll {row['nll']:.4f}, anchor {row['anchor']:.2e}, "
                    f"{row['seconds']:.2f}s/step")
            if max_steps is not None and executed >= max_steps:
                return boundary
            if (step + 1) % train_cfg["checkpoint_steps"] == 0:
                save_state()
        epoch_nll = [r["nll"] for r in boundary["log"] if r["epoch"] == epoch]
        receipt = {"epoch": epoch, "train_nll": float(np.mean(epoch_nll)) if epoch_nll else None,
                   "seconds_this_session": time.time() - tick,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(e3l.device),
                   "sha256": save_atomic(e3l.state(), folder / f"epochs/epoch_{epoch}.pt")}
        if inner is not None:
            contrast, _, rate = inner_contrast(e3l, inner, inner_gold, excluded)
            receipt |= {"inner_test_like_contrast": contrast, "inner_paintings_per_second": rate}
            e3l.train(True)
            log(f"Epoch {epoch}: train nll {receipt['train_nll']:.4f}, inner test-like Contrast {contrast:.6f}")
        boundary["history"].append(receipt)
        boundary = boundary | {"epoch": epoch + 1, "step": 0}
        save_state()
    return boundary


def run_request(request, log=print):
    """GPU child entry: ``mode`` in fold, full, step0 or smoke. Everything it needs is in ``request``."""
    import torch

    from .exp036_io import read

    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("E3L needs exactly one visible GPU")
    deterministic(request["seed"])
    folder = Path(request["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    head_fields = dict(np.load(request["head"], allow_pickle=False))
    pairs = pd.read_csv(request["pairs"])
    images = json.loads(Path(request["images"]).read_text(encoding="utf-8"))
    languages = request["languages"]
    tick = time.time()
    e3l, receipt = build(request["spec"], request["lora"], head_fields, languages, request["train"]["lambda"],
                         request["snapshot"], device, request["seed"])
    receipt["load_seconds"] = time.time() - tick

    def panel(names):
        return Paintings(pairs[pairs.painting.isin(set(names))], images, languages)

    mode = request["mode"]
    done = folder / "result.json"
    if mode in {"fold", "full"} and done.is_file() and not (folder / "state.pt").is_file():
        previous = json.loads(done.read_text(encoding="utf-8"))
        if previous.get("identity") == request["identity"]:
            log("Result already complete for this identity; reusing")
            return previous
    if mode == "step0":
        held = panel(request["held"])
        embeddings, frame, rate = predict(e3l, held, log, "step-0")
        embeddings.to_csv(folder / "step0_embeddings.csv", float_format="%.9g")
        frame.to_csv(folder / "step0_probabilities.csv", index=False, float_format="%.12g")
        write(folder / "step0.json", receipt | {"paintings": len(held), "paintings_per_second": rate,
                                              "lora_right_zero": all(float(p.abs().max()) == 0 for p in e3l.lora[1::2])})
        return
    if mode == "smoke":
        return smoke(e3l, panel, request, folder, receipt, log)
    train_panel = panel(request["train_paintings"])
    inner = panel(request["inner"]) if request.get("inner") else None
    inner_gold = pairs[pairs.painting.isin(set(request.get("inner") or []))]
    boundary = train(e3l, train_panel, request["train"], folder, request["deadline"], request["seed"],
                     epochs_to_run=request["epochs_to_run"], inner=inner, inner_gold=inner_gold,
                     excluded=request["excluded"], identity=request["identity"], log=log)
    history = boundary["history"]
    selected = choose_epoch(history) if inner is not None else request["epochs_to_run"]
    result = receipt | {"identity": request["identity"], "history": history, "selected_epoch": selected, "train_paintings": len(train_panel),
                        "steps_per_epoch": steps_per_epoch(len(train_panel), request["train"]["batch_paintings"])}
    if selected > 0 or mode == "full":
        state = torch.load(folder / f"epochs/epoch_{selected}.pt", weights_only=True, map_location="cpu")
        e3l.load_state(state)
        embed = panel(request["embed"])
        embeddings, _, rate = predict(e3l, embed, log, "adapted embeddings")
        _, outer, _ = predict(e3l, panel(request["held"]), log, "held-out")
        np.savez(folder / "embeddings.npz", paintings=np.asarray(embeddings.index, dtype=str),
                 values=embeddings.to_numpy(dtype=np.float32))
        outer.to_csv(folder / "held_probabilities.csv", index=False, float_format="%.12g")
        result |= {"embedding_paintings_per_second": rate, "embedded": len(embed),
                   "embeddings_sha256": sha(folder / "embeddings.npz"),
                   "held_probabilities_sha256": sha(folder / "held_probabilities.csv"),
                   "weights_sha256": sha(folder / f"epochs/epoch_{selected}.pt")}
    write(folder / "result.json", result)
    (folder / "state.pt").unlink(missing_ok=True)
    return result


def smoke(e3l, panel, request, folder, receipt, log):
    """SMOKE_ONLY: a short real training run, one inner pass, devices, VRAM, finiteness and determinism."""
    import torch

    train_panel, inner = panel(request["train_paintings"]), panel(request["inner"])
    init = e3l.state()
    probe = [train_panel.rows[i][0] for i in range(min(2, len(train_panel)))]
    e3l.train(False)
    with torch.inference_mode():
        first = e3l.features(probe).cpu()
        second = e3l.features(probe).cpu()
        inputs = encode(e3l.processor, e3l.spec, [load_rgb(probe[0])], e3l.device)
        hidden = e3l.backbone(**inputs, use_cache=False).last_hidden_state
        devices = {"inputs": sorted({str(v.device) for v in inputs.values()}), "hidden": str(hidden.device),
                   "output": str(e3l.head(first.to(e3l.device)).device),
                   "parameters": single_gpu(e3l.model, e3l.device)}
    del inputs, hidden
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(e3l.device)
    steps = request["smoke_steps"]
    tick = time.time()
    boundary = train(e3l, train_panel, request["train"], folder / "smoke_train", request["deadline"], request["seed"],
                     epochs_to_run=1, identity=request["identity"], log=log, max_steps=steps)
    torch.cuda.synchronize()
    seconds = time.time() - tick
    peak = {"allocated_bytes": torch.cuda.max_memory_allocated(e3l.device),
            "reserved_bytes": torch.cuda.max_memory_reserved(e3l.device),
            "total_bytes": torch.cuda.get_device_properties(e3l.device).total_memory}
    rows = boundary["log"]
    _, inner_frame, inner_rate = predict(e3l, inner, log, "smoke inner pass")
    # Determinism: two identical 2-step runs from the same initialization and seed.
    repeats = []
    for attempt in range(2 if request.get("determinism", True) else 0):
        e3l.load_state(init)
        deterministic(request["seed"])
        trial = train(e3l, train_panel, request["train"], folder / f"smoke_repeat_{attempt}", request["deadline"],
                      request["seed"], epochs_to_run=1, identity=request["identity"], log=lambda *_: None, max_steps=2)
        repeats.append(([r["nll"] for r in trial["log"]], e3l.state()))
    if repeats:
        loss_difference = float(np.abs(np.asarray(repeats[0][0]) - np.asarray(repeats[1][0])).max())
        weight_difference = max(float((repeats[0][1][k] - repeats[1][1][k]).abs().max()) for k in repeats[0][1])
    else:
        loss_difference = weight_difference = None
    result = receipt | {
        "mode": "SMOKE_ONLY", "gpu": torch.cuda.get_device_name(0), "optimizer_steps": len(rows),
        "paintings_per_step": request["train"]["batch_paintings"], "training_seconds": seconds,
        "seconds_per_optimizer_step": seconds / max(len(rows), 1),
        "train_paintings_per_second": len(rows) * request["train"]["batch_paintings"] / max(seconds, 1e-9),
        "warm_seconds_per_optimizer_step": float(np.median([r["seconds"] for r in rows[1:]])) if len(rows) > 1 else None,
        "inner_paintings": len(inner), "inner_paintings_per_second": inner_rate,
        "inner_pairs_scored": len(inner_frame), "peak_memory": peak, "devices": devices,
        "losses": [r["nll"] for r in rows], "finite_losses": bool(np.isfinite([r["nll"] for r in rows]).all()),
        "grad_norms": [r["grad_norm"] for r in rows],
        "eval_forward_bitwise_repeatable": bool(torch.equal(first, second)),
        "two_pass_training": {"steps": 2 if repeats else 0, "max_loss_difference": loss_difference,
                              "max_weight_difference": weight_difference,
                              "bitwise": bool(repeats) and loss_difference == 0 and weight_difference == 0}}
    write(folder / "smoke.json", result)
    return result
