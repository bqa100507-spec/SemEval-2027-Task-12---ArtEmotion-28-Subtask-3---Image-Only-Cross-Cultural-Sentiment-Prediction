"""Train one resumable EXP-031 fold-3 model on A100 or feasible L4."""
import argparse
from collections import OrderedDict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# The existing restoration helpers belong to a CPU-only runner.
visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
from run_exp028_colab_cpu import MANIFEST_SHA256, ZIP_SHA256, restore_member
if visibility is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = visibility

import numpy as np
import pandas as pd
import torch
from PIL import Image
from st3.data import ANNOTATIONS, KEYS
from st3.exp028_cache import read_protocol
from st3.exp030 import EMOTIONS, best_epoch, count_votes, fold3, matched_reference
from st3.exp031 import binary_terms, conditional_ce, painting_batches, training_priors
from st3.exp031_model import CulturalEmotionModel
from st3.experiment_workflow import Budget, StopRun, fingerprint, publish_model, sha256, verify_model, write_json
from st3.metrics import score
from st3.vlm_supervised import adapter_parameters, attach_adapters

CFG = json.loads((ROOT / "configs/exp031.json").read_text())
RUNTIME = json.loads((ROOT / "configs/exp026.json").read_text())["runtime"] | {
    "scipy": "1.15.3", "psutil": "7.2.2",
}


def prepare(handoff, work):
    if sha256(handoff) != ZIP_SHA256:
        raise ValueError("Use the original EXP-028 exp023_handoff.zip with the pinned SHA-256")
    with zipfile.ZipFile(handoff) as archive:
        raw = archive.read("input_manifest.json")
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
            raise ValueError("Input manifest checksum differs")
        files = json.loads(raw)["files"]
        for name in ("training/pairs.csv", "training/protocol.csv", "history/exp022_R0.csv"):
            restore_member(archive, files, name, work)
        rows = read_protocol(ROOT, work, labels=True)
        for image, digest in rows[["image", "image_sha256"]].drop_duplicates().itertuples(index=False, name=None):
            name = f"artwork/images/{image}"
            if files.get(name) != digest:
                raise ValueError("P2 artwork checksum differs")
            restore_member(archive, files, name, work)
    annotation = work / ANNOTATIONS
    annotation.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/artemotion-28/artemotion-28/{CFG['organizer_revision']}/{ANNOTATIONS}"
    with urlopen(url, timeout=90) as response:
        annotation.write_bytes(response.read())
    if sha256(annotation) != CFG["annotation_sha256"]:
        raise ValueError("Pinned organizer annotation checksum differs")
    rows = count_votes(annotation, rows)
    if fingerprint(rows[list(EMOTIONS)].values.tolist()) != CFG["p2_emotion_counts_sha256"]:
        raise ValueError("Pinned eight-way counts differ")
    train, val = fold3(rows, json.loads((ROOT / "configs/exp028_fold3_manifest.json").read_text()))
    pred = matched_reference(work / "history/exp022_R0.csv", val)
    baseline = score(val[KEYS + ["k", "n"]], pred)
    if (round(baseline["cultural_contrast"], 6), round(baseline["log_loss"], 6)) != (.433536, .713192):
        raise ValueError("Fold-3 reference metrics differ")
    return train, val, baseline


def load_model(work, languages, prior_logits):
    from huggingface_hub import snapshot_download
    from bitsandbytes.nn import Linear4bit

    names = [*CFG["assets_git_sha1"], *CFG["assets_sha256"]]
    path = Path(snapshot_download(CFG["model"], revision=CFG["revision"], local_dir=work / "weights",
                                  allow_patterns=names))
    for name, digest in CFG["assets_sha256"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Model shard checksum differs: {name}")
    for name, digest in CFG["assets_git_sha1"].items():
        data = (path / name).read_bytes()
        if hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != digest:
            raise ValueError(f"Model asset checksum differs: {name}")
    if "license: apache-2.0" not in (path / "README.md").read_text(encoding="utf-8").split("---")[1]:
        raise ValueError("Pinned model card does not declare Apache-2.0")
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=["visual", "lm_head"])
    processor = AutoProcessor.from_pretrained(path, use_fast=False, min_pixels=CFG["min_pixels"],
                                              max_pixels=CFG["max_pixels"], local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True).eval()
    if (model.config.hidden_size, model.config.num_hidden_layers) != (3584, 28):
        raise ValueError("Unexpected pinned 7B architecture")
    targets = attach_adapters(model, {"decoder_layers": 28, "seed": CFG["seed"],
                              "lora": {"rank": CFG["lora_rank"], "alpha": CFG["lora_alpha"],
                                       "targets": ["q_proj", "v_proj"], "expected_modules": 56}})
    if any(not isinstance(model.get_submodule(name).base, Linear4bit) for name in targets):
        raise ValueError("Decoder Q/V base was not quantized")
    if any(isinstance(layer, Linear4bit) for layer in model.visual.modules()):
        raise ValueError("Native visual encoder must remain unquantized")
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return CulturalEmotionModel(model, processor, CFG, languages, prior_logits), targets


class VisionCache:
    def __init__(self, native, artwork, limit):
        self.native, self.artwork, self.limit = native, artwork, limit
        self.entries = OrderedDict()
        self.size = 0

    def __call__(self, name):
        if name in self.entries:
            self.entries.move_to_end(name)
            return self.entries[name][0]
        with Image.open(self.artwork / name) as image:
            values = self.native.vision(image.convert("RGB"))
        size = sum(t.numel() * t.element_size() for t in values.values())
        if size <= self.limit:
            while self.size + size > self.limit:
                _, (_, evicted) = self.entries.popitem(last=False)
                self.size -= evicted
            self.entries[name] = (values, size)
            self.size += size
        return values



def backward_painting(model, vision, languages, counts, votes, paintings, config, *, diagnose=False):
    """Each decoder graph is backpropagated before the next forward.

    Joint CE = binary BCE + the two conditional NLL sums. Ranking only
    needs the neutral state and never touches the language-conditioned graph.
    """
    binary, interaction = model.binary(vision, languages)
    bce, rank, reg = binary_terms(binary, interaction, counts, config)
    sentiment = (1 + config["binary_weight"]) * bce / votes
    ranking = config["ranking_weight"] * rank / paintings
    diagnostic = None
    if diagnose:
        # Compare the two signals at the exact point they meet, including
        # their actual optimizer-update denominators and weights.
        grad_bce = torch.autograd.grad(sentiment, binary, retain_graph=True)[0]
        grad_rank = torch.autograd.grad(ranking, binary, retain_graph=True)[0]
        diagnostic = {"weighted_binary_logit_norm": float(grad_bce.norm()),
                      "weighted_rank_logit_norm": float(grad_rank.norm()),
                      "rank_to_binary_ratio": float(grad_rank.norm() / grad_bce.norm().clamp_min(1e-12))}
    loss = sentiment + ranking + config["interaction_regularization"] * reg / paintings
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite binary/ranking objective")
    loss.backward()
    del loss, sentiment, ranking, binary, interaction
    conditional = 0.
    width = config["microbatch_pairs"]
    for start in range(0, len(languages), width):
        part = slice(start, start + width)
        term = conditional_ce(model.conditionals(vision, languages[part]), counts[part])
        if not torch.isfinite(term):
            raise FloatingPointError("Nonfinite conditional emotion objective")
        (term / votes).backward()
        conditional += float(term.detach())
        del term
    return conditional + float(bce.detach()), float(bce.detach()), float(rank.detach()), float(reg.detach()), diagnostic


def counts_for(frame):
    return torch.as_tensor(frame[list(EMOTIONS)].to_numpy(), device="cuda", dtype=torch.float32)


def update(model, cache, groups, optimizer, *, diagnose=False):
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(int(group.n.sum()) for group in groups)
    totals = np.zeros(4)
    diagnostics = []
    for group in groups:
        result = backward_painting(model, cache(group.iloc[0].image), group.language.tolist(),
                                   counts_for(group), denominator, len(groups), CFG, diagnose=diagnose)
        totals += result[:4]
        if result[4] is not None:
            diagnostics.append(result[4])
    grad = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], CFG["clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    return (*totals.tolist(), denominator, len(groups), float(grad), diagnostics)


@torch.no_grad()
def validate(model, cache, val, budget):
    model.eval()
    predictions, totals = [], np.zeros(4)
    paintings = 0
    for groups in painting_batches(val, seed=CFG["seed"], epoch=None, max_pairs=CFG["accumulation_pairs"]):
        for frame in groups:
            budget.check()
            vision = cache(frame.iloc[0].image)
            languages = frame.language.tolist()
            counts = counts_for(frame)
            binary, interaction = model.binary(vision, languages)
            bce, rank, reg = binary_terms(binary, interaction, counts, CFG)
            conditional = 0.
            for start in range(0, len(frame), CFG["microbatch_pairs"]):
                part = slice(start, start + CFG["microbatch_pairs"])
                conditional += float(conditional_ce(model.conditionals(vision, languages[part]), counts[part]))
            totals += [conditional + float(bce), float(bce), float(rank), float(reg)]
            paintings += 1
            predictions.append(frame[KEYS].assign(p_positive=binary.sigmoid().cpu().tolist()))
    pred = pd.concat(predictions, ignore_index=True)
    metrics = score(val[KEYS + ["k", "n"]], pred)
    if not np.isfinite(totals).all() or metrics["cultural_contrast"] is None or metrics["log_loss"] is None:
        raise FloatingPointError("Invalid held-out losses or coverage")
    return pred, metrics, *(totals / [val.n.sum(), val.n.sum(), paintings, paintings]).tolist()


@torch.no_grad()
def verify_prediction_replay(model, cache, val, expected, budget):
    model.eval()
    chunks = []
    for _, frame in val.groupby("painting", sort=True):
        budget.check()
        binary, _ = model.binary(cache(frame.iloc[0].image), frame.language.tolist())
        chunks.append(frame[KEYS].assign(p_positive=binary.sigmoid().cpu().tolist()))
    actual = pd.concat(chunks, ignore_index=True)
    matched = expected.merge(actual, on=KEYS, validate="one_to_one", suffixes=("_saved", "_actual"))
    if len(matched) != len(expected) or len(actual) != len(expected):
        raise ValueError("Resume checkpoint prediction coverage differs")
    np.testing.assert_allclose(matched.p_positive_actual, matched.p_positive_saved, atol=1e-6, rtol=0)


def preflight(model, cache, train, params, budget, optimizer=None):
    """Check native parity and the real sequential backward path at the image cap."""
    budget.check()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    # Use the largest painting group and the processor's maximum image grid.
    grouped = train.groupby("painting", sort=True)
    name = grouped.size().idxmax()
    group = grouped.get_group(name)
    with Image.open(cache.artwork / group.iloc[0].image) as image:
        parity = model.native.parity(image.convert("RGB"), cache(group.iloc[0].image))
    saved = {n: p.detach().cpu().clone() for n, p in params.items()} if optimizer is None else {}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    wide = model.native.vision(Image.new("RGB", (896, 896)))
    model.train()
    probe_optimizer = make_optimizer(model) if optimizer is None else None
    # Restored Adam buffers are resident during the resume probe; leave them untouched.
    steps = 2 if optimizer is None else 1
    for step in range(steps):
        budget.check()
        model.zero_grad(set_to_none=True)
        result = backward_painting(model, wide, group.language.tolist(), counts_for(group),
                                   int(group.n.sum()), 1, CFG, diagnose=True)
        norms = {name: float(sum(float(p.grad.detach().norm().square()) for p in values if p.grad is not None) ** .5)
                 for name, values in (("adapters", adapter_parameters(model.decoder)),
                                      ("interaction_embedding", model.head.language.parameters()),
                                      ("interaction_visual", model.head.visual.parameters()),
                                      ("shared", model.head.shared.parameters()),
                                      ("emotion", model.head.emotions.parameters()))}
        required = set(norms) - ({"interaction_visual"} if step == 0 and optimizer is None else set())
        if (any(not np.isfinite(norms[key]) or norms[key] <= 0 for key in required)
                or any(p.grad is not None for p in model.native.model.parameters() if not p.requires_grad)):
            raise RuntimeError(f"Gradient preflight failed trainable/frozen separation: {norms}")
        torch.nn.utils.clip_grad_norm_(list(params.values()), CFG["clip_norm"], error_if_nonfinite=True)
        if probe_optimizer is not None:
            probe_optimizer.step()
    if optimizer is None and not any(not torch.equal(p.detach().cpu(), saved[n]) for n, p in params.items()
                                     if n.startswith("decoder.")):
        raise RuntimeError("Adapters did not update in preflight")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    receipt = {"gpu": torch.cuda.get_device_name(), "total_vram_bytes": total,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "free_after_probe_bytes": free, "max_grid_vision_tokens_per_image": int(wide["tokens"].shape[0]),
               "probe_languages": len(group), "seconds": time.perf_counter() - started,
               "official_vision_path_parity": parity, "grad_norms": norms, "binary_rank_balance": result[4],
               "fresh_optimizer_probe_steps": steps if optimizer is None else 0}
    if min(free, total - receipt["peak_reserved_bytes"]) < 1024 ** 3:
        raise RuntimeError("Less than 1 GiB CUDA margin: " + str(receipt))
    with torch.no_grad():
        for key, value in saved.items():
            params[key].copy_(value)
    model.zero_grad(set_to_none=True)
    del probe_optimizer
    torch.cuda.empty_cache()
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    return receipt


def make_optimizer(model):
    return torch.optim.AdamW(
        [{"params": model.head.parameters(), "lr": CFG["head_lr"]},
         {"params": adapter_parameters(model.decoder), "lr": CFG["adapter_lr"]}],
        weight_decay=CFG["weight_decay"], fused=False, foreach=False)


def publish_epoch(directory, identity, model_state, predictions, summary):
    if directory.exists():
        raise FileExistsError(f"Epoch already published: {directory}")
    def writer(path):
        torch.save(model_state, path / "resume.pt")
        predictions.to_csv(path / "predictions.csv", index=False, float_format="%.17g")
        write_json(path / "epoch.json", summary)
        for name in ("resume.pt", "predictions.csv", "epoch.json"):
            with (path / name).open("r+b") as stream:
                os.fsync(stream.fileno())

    publish_model(directory, identity, writer)


def completed_epochs(root, identity, val):
    from pandas import read_csv

    history = []
    found = sorted(path.name for path in root.glob("epoch-*"))
    if found != [f"epoch-{epoch:02d}" for epoch in range(1, len(found) + 1)] or len(found) > CFG["epochs"]:
        raise ValueError("Unexpected epoch checkpoint sequence")
    for epoch, name in enumerate(found, 1):
        directory = root / name
        seal = verify_model(directory, identity | {"epoch": epoch})
        if set(seal["files"]) != {"resume.pt", "predictions.csv", "epoch.json"}:
            raise ValueError("Unexpected checkpoint contents")
        summary = json.loads((directory / "epoch.json").read_text())
        pred = read_csv(directory / "predictions.csv", float_precision="round_trip")
        metrics = score(val[KEYS + ["k", "n"]], pred)
        if summary["epoch"] != epoch or not np.allclose(
                [summary["contrast"], summary["macro_log_loss_bits"]],
                [metrics["cultural_contrast"], metrics["log_loss"]], atol=1e-12, rtol=0):
            raise ValueError("Stored epoch predictions and scores disagree")
        history.append(summary)
    return history


def restore(directory, params, optimizer, epoch, anchor):
    state = torch.load(directory / "resume.pt", weights_only=True, map_location="cpu")
    if state["weights"].keys() != params.keys() or state["epoch"] != epoch:
        raise ValueError("Checkpoint does not match trainable parameters or completed epoch")
    with torch.no_grad():
        for key, value in state["weights"].items():
            if value.shape != params[key].shape or value.dtype != params[key].dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint tensor {key}")
            params[key].copy_(value)
    if not torch.equal(state["anchor"], anchor.detach().cpu()):
        raise ValueError("Pretrained interaction anchor differs from sealed checkpoint")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng"])


def progress(output, history, baseline, identity, directory, status):
    best = best_epoch(history) if history else None
    change = None
    if len(history) >= 2:
        before, after = history[-2:]
        change = {key: after[key] - before[key] for key in
                  ("train_loss_nats", "validation_bce_nats", "contrast", "macro_log_loss_bits")}
    best_record = next((row for row in history if row["epoch"] == best), None)
    report = {"status": status, "completed_epochs": len(history), "best_epoch": best, "epochs": history,
              "reference": baseline, "last_epoch_change": change, "production": "E50+L", "checkpoint_root": str(directory),
              "identity": identity, "deltas_at_best": None if not best_record else {
                  "contrast": best_record["contrast"] - baseline["cultural_contrast"],
                  "macro_log_loss_bits": best_record["macro_log_loss_bits"] - baseline["log_loss"]}}
    write_json(output / "results.json", report)
    print(json.dumps({k: report[k] for k in ("status", "completed_epochs", "best_epoch", "deltas_at_best")}), flush=True)


def run(args):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if source != args.source or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("Use the clean, notebook-pinned source")
    actual = {"python": platform.python_version()} | {name: importlib.metadata.version(name)
              for name in RUNTIME if name != "python"}
    if actual != RUNTIME:
        raise RuntimeError("Pinned CUDA dependency lock does not match")
    budget = Budget(args.hours, args.hours, cuda=True, reserve=240, started=args.started)
    budget.check()
    print("Verifying P2 data and matched E50+L", flush=True)
    train, val, baseline = prepare(args.handoff, args.work)
    languages, priors = training_priors(train, pseudocount=CFG["prior_pseudocount"])
    if not set(val.language).issubset(set(languages)):
        raise ValueError("Val contains an unseen language")
    write_json(args.output / "reference.json", baseline)
    write_json(args.output / "fitting_priors.json",
               {"languages": languages, "prior_logits": priors.tolist()})
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("A CUDA GPU with native BF16 is required")
    device = torch.cuda.get_device_properties(0)
    if not ("A100" in device.name or "L4" in device.name):
        raise RuntimeError("Use an A100 or L4 Colab GPU")
    import psutil
    cache_bytes = min(CFG["visual_cache_max_bytes"], max(0, psutil.virtual_memory().available - 2 * 1024 ** 3))
    torch.manual_seed(CFG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    # Use one kernel profile and fail if an operation has no deterministic implementation.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     text=True).splitlines()[0].strip()
    identity = {"source": source, "train_config": CFG, "input_sha256": ZIP_SHA256,
                "fitting_priors": fingerprint([languages, priors.tolist()]),
                "fold_manifest_sha256": sha256(ROOT / "configs/exp028_fold3_manifest.json"),
                "protocol_input_sha256": MANIFEST_SHA256,
                "runtime": actual, "gpu": device.name, "driver": driver, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "lock_sha256": sha256(ROOT / "requirements-exp026.lock.txt")}
    directory = args.checkpoints
    directory.mkdir(parents=True, exist_ok=True)
    history = completed_epochs(directory, identity, val)
    write_json(args.output / "run_identity.json", identity)
    progress(args.output, history, baseline, identity, directory, "COMPLETE" if len(history) == CFG["epochs"] else "READY")
    if len(history) == CFG["epochs"]:
        return
    budget.check()
    print("Loading pinned native Qwen 7B NF4; download is about 16.6 GB", flush=True)
    import shutil
    if shutil.disk_usage(args.work).free < 18_000_000_000:
        raise RuntimeError("Need at least 18 GB free in local runtime storage for 7B shards")
    model, targets = load_model(args.work, languages, priors)
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    expected_ids = {id(p) for p in [*model.head.parameters(), *adapter_parameters(model.decoder)]}
    if {id(p) for p in params.values()} != expected_ids:
        raise ValueError("Unexpected trainable parameters")
    cache = VisionCache(model.native, args.work / "artwork/images", cache_bytes)
    model.eval()
    model.set_anchor(model.native.vision(Image.new("RGB", (448, 448), (128, 128, 128))))
    optimizer = make_optimizer(model)
    if history:
        last = directory / f"epoch-{len(history):02d}"
        restore(last, params, optimizer, len(history), model.head.anchor)
        expected = pd.read_csv(last / "predictions.csv", float_precision="round_trip")
        verify_prediction_replay(model, cache, val, expected, budget)
        write_json(args.output / "resume_replay.json", {"epoch": len(history), "pairs": len(expected),
                   "status": "PASS", "prediction_tolerance": 1e-6})
    print("Running CUDA parity and feasibility checks", flush=True)
    try:
        receipt = preflight(model, cache, train, params, budget, optimizer if history else None)
    except torch.OutOfMemoryError as exc:
        raise RuntimeError("Pretrained CUDA feasibility failed: use A100 or release other GPU memory") from exc
    write_json(args.output / "preflight.json", receipt | {"targets": targets,
               "trainable_parameters": sum(p.numel() for p in params.values()), "cpu_cache_limit_bytes": cache_bytes})
    print(json.dumps(receipt), flush=True)
    try:
        for epoch in range(len(history) + 1, CFG["epochs"] + 1):
            budget.check()
            # Prior complete epochs supply a planning estimate for both training and validation.
            if history:
                budget.check(history[-1]["seconds"] * 1.05)
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            model.train()

            ce = bce = ranking = regularizer = votes = seen = npaintings = 0
            grad_sum = clipped = 0
            diagnostics = []
            updates = painting_batches(train, seed=CFG["seed"], epoch=epoch,
                                       max_pairs=CFG["accumulation_pairs"])
            for step, groups in enumerate(updates, 1):
                budget.check()
                a, b, r, g, n, npain, grad, sample = update(
                    model, cache, groups, optimizer, diagnose=step in (1, 8, 32) or step % 128 == 0)
                grad_sum += grad
                clipped += grad > CFG["clip_norm"]
                if sample:
                    diagnostics.append({"step": step, "paintings": sample})
                ce += a
                bce += b
                ranking += r
                regularizer += g
                votes += n
                npaintings += npain
                seen += sum(len(group) for group in groups)
                if step in (8, 32) or step % 64 == 0 or seen == len(train):
                    rate = seen / (time.perf_counter() - tick)
                    print(f"Epoch {epoch} {seen}/{len(train)} at {rate:.2f} pairs/s. "
                          f"Train-only projection {len(train)/rate/3600:.2f}h; "
                          f"remaining budget {budget.remaining()/3600:.2f}h.", flush=True)
                    if step == 8:
                        write_json(args.output / "initial_throughput.json",
                                   {"epoch": epoch, "pairs": seen, "pairs_per_second": rate,
                                    "projected_training_only_epoch_hours": len(train)/rate/3600})
            train_seconds = time.perf_counter() - tick
            pred, metrics, val_ce, val_bce, val_ranking, val_reg = validate(model, cache, val, budget)
            row = {"epoch": epoch, "train_loss_nats": (ce + CFG["binary_weight"] * bce) / votes
                   + CFG["ranking_weight"] * ranking / npaintings
                   + CFG["interaction_regularization"] * regularizer / npaintings,
                   "train_emotion_ce_nats": ce / votes, "train_bce_nats": bce / votes,
                   "train_rank_nats": ranking / npaintings,
                   "train_interaction_regularizer": regularizer / npaintings,
                   "validation_emotion_ce_nats": val_ce, "validation_bce_nats": val_bce,
                   "validation_rank_nats": val_ranking, "validation_interaction_regularizer": val_reg,
                   "training_pairs_per_second": seen / train_seconds, "training_seconds": train_seconds,
                   "optimizer_steps": step, "mean_preclip_gradient_norm": grad_sum / step,
                   "clip_fraction": clipped / step, "sampled_binary_rank_gradients": diagnostics,
                   "contrast": metrics["cultural_contrast"], "macro_log_loss_bits": metrics["log_loss"],
                   "seconds": time.perf_counter() - tick, "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved()}
            state = {"epoch": epoch, "weights": {name: p.detach().cpu() for name, p in params.items()},
                     "anchor": model.head.anchor.detach().cpu(), "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state()}
            publication_started = time.perf_counter()
            publish_epoch(directory / f"epoch-{epoch:02d}", identity | {"epoch": epoch}, state, pred, row)
            print(f"Epoch {epoch} checkpoint published in {time.perf_counter() - publication_started:.1f}s",
                  flush=True)
            del state
            history.append(row)
            print(json.dumps(row), flush=True)
            progress(args.output, history, baseline, identity, directory, "COMPLETE" if epoch == CFG["epochs"] else "IN_PROGRESS")
    except StopRun:
        progress(args.output, history, baseline, identity, directory, "STOPPED_BUDGET")
    except Exception:
        progress(args.output, history, baseline, identity, directory, "BLOCKED")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("handoff", "work", "output", "checkpoints"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--hours", required=True, type=float)
    parser.add_argument("--started", required=True, type=float)
    parser.add_argument("--authorize-research", required=True, choices=["EXP031"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not 0 < args.hours <= 12 or not np.isfinite(args.started) or args.started > time.time():
        parser.error("Use a finite authorization up to 12 hours and the original notebook start time")
    # One writer for the fixed checkpoint directory, even across notebook sessions.
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    lock = args.checkpoints / ".active"
    try:
        lock.mkdir()
    except FileExistsError:
        parser.error("Checkpoint directory is locked; after checking the previous session is stopped, remove .active on Drive")
    def interrupt(_signal, _frame):
        raise StopRun("STOPPED_BUDGET", "Worker stopped by outer runtime deadline")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        run(args)
    except StopRun as error:
        write_json(args.output / "failure.json", {"status": error.status, "error": str(error)})
    except Exception as error:
        write_json(args.output / "failure.json", {"status": "BLOCKED",
                                                  "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        lock.rmdir()


if __name__ == "__main__":
    main()
