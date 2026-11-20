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
from st3.exp030 import EMOTIONS, best_epoch, count_votes, fold3, loss_sums, matched_reference
from st3.exp031 import grouped_objective, painting_batches, training_priors
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


def ordered_indices(frame, epoch):
    groups = list(frame.groupby("painting", sort=True).indices.values())
    rng = np.random.default_rng(CFG["seed"] + epoch)
    return np.concatenate([groups[i] for i in rng.permutation(len(groups))])


def logits_for(model, cache, frame):
    visions = [cache(image) for image in frame.image]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(visions, frame.language.tolist())


def counts_for(frame):
    return torch.as_tensor(frame[list(EMOTIONS)].to_numpy(), device="cuda", dtype=torch.float32)


def update(model, cache, frame, optimizer):
    optimizer.zero_grad(set_to_none=True)
    total_votes = int(frame.n.sum())
    ce_total = bce_total = 0.
    for start in range(0, len(frame), CFG["microbatch_pairs"]):
        batch = frame.iloc[start:start + CFG["microbatch_pairs"]]
        ce, bce = loss_sums(logits_for(model, cache, batch), counts_for(batch))
        loss = (ce + CFG["binary_weight"] * bce) / total_votes
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training objective")
        loss.backward()
        ce_total += float(ce.detach())
        bce_total += float(bce.detach())
    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           CFG["clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    return ce_total, bce_total, total_votes, float(grad)


@torch.no_grad()
def validate(model, cache, val, budget):
    model.eval()
    probs, ce, bce = [], 0., 0.
    for start in range(0, len(val), CFG["microbatch_pairs"]):
        budget.check()
        batch = val.iloc[start:start + CFG["microbatch_pairs"]]
        logits = logits_for(model, cache, batch)
        a, b = loss_sums(logits, counts_for(batch))
        ce += float(a)
        bce += float(b)
        probs.extend(positive_probability(logits).cpu().tolist())
    pred = val[KEYS].assign(p_positive=probs)
    metrics = score(val[KEYS + ["k", "n"]], pred)
    if not np.isfinite([ce, bce]).all():
        raise FloatingPointError("Nonfinite held-out loss")
    return pred, metrics, ce / val.n.sum(), bce / val.n.sum()

def preflight(model, cache, train, params, budget):
    """Exercise the longest allowed vision grid, the real processor and an optimizer step."""
    budget.check()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    # A large square image exercises the processor's pixel ceiling. Use real
    # artwork for official-path parity and the gradient/update probe.
    longest = train.language.drop_duplicates().tolist()
    tokenizer = model.native.processor.tokenizer
    longest.sort(key=lambda lang: len(tokenizer(model.native.text(lang)).input_ids), reverse=True)
    images = train.drop_duplicates("image").head(4)
    languages = longest[:CFG["microbatch_pairs"]]
    name = images.iloc[0].image
    with Image.open(cache.artwork / name) as image:
        parity = model.native.parity(image.convert("RGB"), cache(name))
    saved = {n: p.detach().cpu().clone() for n, p in params.items()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    model.eval()
    with torch.no_grad():
        wide = model.native.vision(Image.new("RGB", (896, 896)))
        probe = model([wide] * CFG["microbatch_pairs"], languages)
        if not torch.isfinite(probe).all():
            raise FloatingPointError("Nonfinite maximum-pixel forward")
    # Backpropagate the capped grid with actual train vote counts, then step the
    # same optimizer as training so its buffers are included in the VRAM probe.
    model.train()
    probe_optimizer = make_optimizer(model)
    batch = images.iloc[:CFG["microbatch_pairs"]]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model([wide] * len(batch), languages)
    loss = loss_sums(logits, counts_for(batch))
    if not torch.isfinite(torch.stack(loss)).all():
        raise FloatingPointError("Nonfinite CUDA preflight objective")
    ((loss[0] + CFG["binary_weight"] * loss[1]) / int(batch.n.sum())).backward()
    adapters = adapter_parameters(model.decoder)
    if (not any(p.grad is not None and p.grad.norm() > 0 for p in adapters)
            or not any(p.grad is not None and p.grad.norm() > 0 for p in model.head.parameters())
            or any(p.grad is not None for p in model.native.model.parameters() if not p.requires_grad)):
        raise RuntimeError("Gradient preflight failed trainable/frozen separation")
    torch.nn.utils.clip_grad_norm_(list(params.values()), CFG["clip_norm"], error_if_nonfinite=True)
    probe_optimizer.step()
    if not any(not torch.equal(p.detach().cpu(), saved[n]) for n, p in params.items() if n.startswith("decoder.")):
        raise RuntimeError("Adapters did not update in the feasibility step")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    result = {"gpu": torch.cuda.get_device_name(), "total_vram_bytes": total, "peak_allocated_bytes": allocated,
              "peak_reserved_bytes": reserved, "free_after_probe_bytes": free,
              "max_grid_vision_tokens_per_image": int(wide["tokens"].shape[0]),
              "seconds": time.perf_counter() - started, "official_vision_path_parity": parity,
              "adapter_gradient_and_update": True}
    if min(free, total - reserved) < 1024 ** 3:
        raise RuntimeError(f"Less than 1 GiB remaining CUDA margin after feasibility check: {result}")
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(saved[name])
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    return result


def make_optimizer(model):
    return torch.optim.AdamW(
        [{"params": model.head.parameters(), "lr": CFG["head_lr"]},
         {"params": adapter_parameters(model.decoder), "lr": CFG["adapter_lr"]}],
        weight_decay=CFG["weight_decay"], fused=False, foreach=False)


def publish_epoch(directory, identity, model_state, predictions, summary):
    def writer(path):
        torch.save(model_state, path / "resume.pt")
        predictions.to_csv(path / "predictions.csv", index=False, float_format="%.17g")
        write_json(path / "epoch.json", summary)

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


def restore(directory, params, optimizer, epoch):
    state = torch.load(directory / "resume.pt", weights_only=True, map_location="cpu")
    if state["weights"].keys() != params.keys() or state["epoch"] != epoch:
        raise ValueError("Checkpoint does not match trainable parameters or completed epoch")
    with torch.no_grad():
        for key, value in state["weights"].items():
            if value.shape != params[key].shape or value.dtype != params[key].dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint tensor {key}")
            params[key].copy_(value)
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
    write_json(args.output / "reference.json", baseline)
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
    model, targets = load_model(args.work)
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    expected_ids = {id(p) for p in [*model.head.parameters(), *adapter_parameters(model.decoder)]}
    if {id(p) for p in params.values()} != expected_ids:
        raise ValueError("Unexpected trainable parameters")
    cache = VisionCache(model.native, args.work / "artwork/images", cache_bytes)
    print("Running CUDA parity and feasibility checks", flush=True)
    try:
        receipt = preflight(model, cache, train, params, budget)
    except torch.OutOfMemoryError as exc:
        raise RuntimeError("Pretrained CUDA feasibility failed: use A100 or release other GPU memory") from exc
    write_json(args.output / "preflight.json", receipt | {"targets": targets,
               "trainable_parameters": sum(p.numel() for p in params.values()), "cpu_cache_limit_bytes": cache_bytes})
    print(json.dumps(receipt), flush=True)
    optimizer = make_optimizer(model)
    if history:
        restore(directory / f"epoch-{len(history):02d}", params, optimizer, len(history))
    try:
        for epoch in range(len(history) + 1, CFG["epochs"] + 1):
            budget.check()
            # Prior complete epochs supply a planning estimate for both training and validation.
            if history:
                budget.check(history[-1]["seconds"] * 1.05)
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            model.train()
            order = ordered_indices(train, epoch)
            ce = bce = votes = seen = 0
            for start in range(0, len(train), CFG["accumulation_pairs"]):
                budget.check()
                frame = train.iloc[order[start:start + CFG["accumulation_pairs"]]]
                a, b, n, _ = update(model, cache, frame, optimizer)
                ce += a
                bce += b
                votes += n
                seen += len(frame)
                if start in (CFG["accumulation_pairs"] * 7, CFG["accumulation_pairs"] * 31) or seen % 1024 == 0:
                    rate = seen / (time.perf_counter() - tick)
                    print(f"Epoch {epoch} {seen}/{len(train)} at {rate:.2f} pairs/s. "
                          f"Train-only projection {len(train)/rate/3600:.2f}h; "
                          f"remaining budget {budget.remaining()/3600:.2f}h.", flush=True)
                    if start == CFG["accumulation_pairs"] * 7:
                        write_json(args.output / "initial_throughput.json",
                                   {"epoch": epoch, "pairs": seen, "pairs_per_second": rate,
                                    "projected_training_only_epoch_hours": len(train)/rate/3600})
            pred, metrics, val_ce, val_bce = validate(model, cache, val, budget)
            row = {"epoch": epoch, "train_loss_nats": (ce + CFG["binary_weight"] * bce) / votes,
                   "train_emotion_ce_nats": ce / votes, "train_bce_nats": bce / votes,
                   "validation_emotion_ce_nats": val_ce, "validation_bce_nats": val_bce,
                   "contrast": metrics["cultural_contrast"], "macro_log_loss_bits": metrics["log_loss"],
                   "seconds": time.perf_counter() - tick, "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved()}
            state = {"epoch": epoch, "weights": {name: p.detach().cpu() for name, p in params.items()},
                     "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state()}
            publish_epoch(directory / f"epoch-{epoch:02d}", identity | {"epoch": epoch}, state, pred, row)
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
