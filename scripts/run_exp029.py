"""Run one bounded, training-only EXP-029 prototype. No submission path."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# The reused CPU-only helper sets a module-level CUDA guard. Preserve the
# caller's device visibility before importing or initializing Torch.
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
from run_exp028_colab_cpu import (MANIFEST_SHA256, ZIP_SHA256, rebuild_visual, restore_member)
if visible is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from st3.exp028_cache import KEYS, read_protocol, visual_blocks
from st3.exp028_stage1 import matched_reference, weighted_bce
from st3.exp029 import (VisualCorrection, baseline_frame, crossfit_partitions, pilot_partitions,
                        shuffled_images, visual_vectors)
from st3.experiment_workflow import Budget, StopRun, fingerprint, publish_model, sha256, verify_model, write_json
from st3.metrics import score
from st3.vlm_supervised import adapter_parameters, attach_adapters, vote_loss
from st3.vlm_semantics import load_vlm
from run_exp026 import check_assets

PARENT = json.loads((ROOT / "configs/exp026.json").read_text())
CONFIG = json.loads((ROOT / "configs/exp029.json").read_text())


def selected_votes(path, rows):
    keys = set(rows[KEYS].itertuples(index=False, name=None))
    with path.open(encoding="utf-8", newline="") as stream:
        records = [(r["painting"], r["language"], int(r["k"]), int(r["n"]))
                   for r in csv.DictReader(stream) if (r["painting"], r["language"]) in keys]
    votes = pd.DataFrame(records, columns=[*KEYS, "k", "n"])
    merged = rows.merge(votes, on=KEYS, validate="one_to_one")
    if len(merged) != len(rows) or ((merged.k < 0) | (merged.k > merged.n) | (merged.n < 5)).any():
        raise ValueError("Invalid or missing pilot vote counts")
    return merged


def inputs(handoff, local):
    if sha256(handoff) != ZIP_SHA256:
        raise ValueError("Use the original checksum-pinned MyDrive/exp023_handoff.zip from TASK-034")
    with zipfile.ZipFile(handoff) as archive:
        manifest = archive.read("input_manifest.json")
        if hashlib.sha256(manifest).hexdigest() != MANIFEST_SHA256:
            raise ValueError("Original input manifest checksum differs")
        pins = json.loads(manifest)["files"]
        for name in ("training/pairs.csv", "training/protocol.csv", "visual/features.npz",
                     "visual/provenance.json", "visual/image_hashes.json"):
            restore_member(archive, pins, name, local)
    rows = read_protocol(ROOT, local)
    train, val = pilot_partitions(rows, PARENT, CONFIG)
    inventory = json.loads((ROOT / "docs/exp028_feature_inventory.json").read_text())
    rebuild_visual(local, local, rows, inventory)
    selected = pd.concat([train, val])
    blocks = visual_blocks(ROOT, local, selected)
    return [selected_votes(local / "training/pairs.csv", x) for x in (train, val)], blocks


def reference(train, val, blocks, budget):
    chunks, receipt = [], []
    with threadpool_limits(limits=1):
        for fit, query in crossfit_partitions(train, val, CONFIG["crossfit_folds"], CONFIG["seed"]):
            budget.check()
            pred, fits = matched_reference(fit, query, blocks)
            chunks.append(pred)
            receipt.append({"fit_paintings": sorted(fit.painting.unique()),
                            "predicted_paintings": sorted(query.painting.unique()), "fits": fits})
        budget.check()
        pred, fits = matched_reference(train, val, blocks)
    receipt.append({"fit_paintings": sorted(train.painting.unique()),
                    "predicted_paintings": sorted(val.painting.unique()), "fits": fits})
    return baseline_frame(train, pd.concat(chunks)), baseline_frame(val, pred), receipt


def backend(cache, technical):
    runtime = PARENT["runtime"]
    if platform.python_version() != runtime["python"] or any(
        importlib.metadata.version(pkg) != ver for pkg, ver in runtime.items() if pkg != "python"
    ):
        raise RuntimeError("Pinned EXP-026 Python/CUDA dependency lock required")
    if not torch.cuda.is_available() or (not technical and "L4" not in torch.cuda.get_device_name()):
        raise RuntimeError("Use a CUDA runtime with NVIDIA L4 for the pilot")
    from huggingface_hub import snapshot_download
    pins = PARENT["vlm"]
    # Download the original pinned assets, regardless of whether HF has a newer release.
    model_dir = Path(snapshot_download(pins["model"], revision=pins["revision"], local_dir=cache / pins["revision"],
                                      allow_patterns=[*pins["assets_sha256"], *pins["assets_git_sha1"]]))
    check_assets(model_dir, PARENT)
    processor, model = load_vlm(PARENT, model_dir)
    if model.config.hidden_size != 2048 or model.config.num_hidden_layers != 36:
        raise ValueError("Pinned decoder architecture differs")
    rope = model.config.rope_scaling
    if not rope or 2 * sum(rope["mrope_section"]) != model.config.hidden_size // model.config.num_attention_heads:
        raise ValueError("Pinned three-axis rotary layout differs")
    names = attach_adapters(model, PARENT)
    from bitsandbytes.nn import Linear4bit
    if not all(isinstance(model.get_submodule(name).base, Linear4bit) for name in names):
        raise ValueError("Expected all adapted Q/V base weights to be 4-bit")
    processor.tokenizer.padding_side = "right"
    decoder = model.model
    decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return processor.tokenizer, decoder, names


def encode_prompts(tokenizer, languages):
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": CONFIG["prompt"].format(language=lang)}],
        add_generation_prompt=True, tokenize=False) for lang in languages]
    batch = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
    ids, mask = batch.input_ids.cuda(), batch.attention_mask.cuda()
    if mask[:, 0].min() != 1 or (mask[:, 1:] > mask[:, :-1]).any():
        raise ValueError("Expected contiguous right-padded language prompts")
    return ids, mask


def checked_gradients(groups, *, require_all=False):
    norms = {}
    for name, params in groups.items():
        grads = [p.grad for p in params if p.grad is not None]
        norm = float(torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads]))) if grads else 0.
        if not np.isfinite(norm) or (require_all and norm <= 0):
            raise ValueError(f"Disconnected or nonfinite gradient: {name} ({norm})")
        norms[name] = norm
    return norms


def new_correction():
    torch.manual_seed(CONFIG["seed"])
    return VisualCorrection(2048, CONFIG).cuda()


def state(parameters):
    return {name: p.detach().cpu().clone() for name, p in parameters.items()}


def smoke(decoder, tokenizer, train, vectors, local):
    # One row per painting so the intervention cannot merely swap language rows.
    batch = train.drop_duplicates("painting").head(6)
    if len(batch) != 6:
        raise ValueError("Smoke needs six training paintings")
    ids, mask = encode_prompts(tokenizer, batch.language)
    image = torch.as_tensor(np.stack([vectors[name] for name in batch.image]), device="cuda")
    base = torch.as_tensor(batch.p_positive.to_numpy(), device="cuda", dtype=torch.float32)
    module = new_correction()
    started = time.perf_counter()
    decoder.eval()
    adapters = adapter_parameters(decoder)
    with torch.no_grad():
        start = module(decoder, image, ids, mask, base)
        torch.testing.assert_close(start, torch.logit(base), rtol=0, atol=0)
        signal = (module.states(decoder, image, ids, mask)
                  - module.states(decoder, image.roll(1, dims=0), ids, mask)).abs().max().item()
        if not np.isfinite(signal) or signal <= 0:
            raise ValueError("Visual-prefix replacement did not change the final decoder state")
        # A zero head necessarily blocks upstream gradients on the first step.
        # Use a throwaway nonzero probe head; restore pristine adapters afterward.
        torch.nn.init.normal_(module.head.weight, std=.01)
    groups = {"projector": list(module.projector.parameters()), "head": list(module.head.parameters()),
              "adapter": adapters}
    optim = torch.optim.AdamW([p for group in groups.values() for p in group], lr=.001)
    decoder.train()
    before = {name: [p.detach().clone() for p in params] for name, params in groups.items()}
    logits = module(decoder, image, ids, mask, base)
    positive, votes = (torch.as_tensor(batch[c].to_numpy(), device="cuda") for c in ("k", "n"))
    vote_loss(logits, positive, votes).backward()
    gradients = checked_gradients(groups, require_all=True)
    if any(p.grad is not None for p in decoder.parameters() if not p.requires_grad):
        raise ValueError("Frozen base received gradients")
    optim.step()
    updates = {name: sum(float((p.detach() - old).norm()) for p, old in zip(params, before[name]))
               for name, params in groups.items()}
    if any(v <= 0 or not np.isfinite(v) for v in updates.values()):
        raise ValueError("Smoke optimizer step did not update all trainable groups")
    decoder.eval()
    with torch.no_grad():
        swapped = (module(decoder, image, ids, mask, base)
                   - module(decoder, image.roll(1, dims=0), ids, mask, base)).abs().max().item()
        if not np.isfinite(swapped) or swapped <= 0:
            raise ValueError("Visual-prefix replacement did not change the smoke logits")
        for p, old in zip(adapters, before["adapter"]):
            p.copy_(old)
    decoder.zero_grad(set_to_none=True)
    write_json(local / "smoke.json", {"paintings": len(batch), "soft_prefix_shape": [6, 4, 2048],
               "attention_mask_shape": [6, ids.shape[1] + 4],
               "position_ids_shape": [3, 6, ids.shape[1] + 4],
               "zero_correction": True, "swapped_state_max_abs": signal, "swapped_logit_max_abs": swapped, "gradient_norms": gradients,
               "update_norms": updates, "seconds": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()})


def batches(frame, order, vocab, features, names=None):
    image_names = frame.image.tolist() if names is None else names
    for start in range(0, len(order), CONFIG["batch_pairs"]):
        indices = order[start:start + CONFIG["batch_pairs"]]
        part = frame.iloc[indices]
        prompt = [vocab[lang] for lang in part.language]
        width = max(len(x) for x in prompt)
        ids = torch.zeros((len(prompt), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(prompt):
            ids[i, :len(sequence)] = sequence
            mask[i, :len(sequence)] = 1
        vectors = np.stack([features[image_names[i]] for i in indices])
        yield (torch.as_tensor(vectors, device="cuda"), ids.cuda(), mask.cuda(),
               *(torch.as_tensor(part[c].to_numpy(), device="cuda") for c in ("p_positive", "k", "n")))


@torch.no_grad()
def predict(module, decoder, frame, vocab, vectors, budget, names=None):
    decoder.eval()
    logits = []
    for x, ids, mask, base, _, _ in batches(frame, np.arange(len(frame)), vocab, vectors, names):
        budget.check()
        logits.extend(module(decoder, x, ids, mask, base).cpu().numpy().astype(np.float64))
    logits = np.asarray(logits)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite validation logits")
    return logits


def probabilities(frame, logits):
    from scipy.special import expit
    return frame[KEYS].assign(p_positive=expit(logits).round(6))


def fit_arm(name, decoder, train, val, vocab, vectors, budget, identity, model_dir, initial):
    start = time.perf_counter()
    module = new_correction()
    lora = name == "projector_qlora"
    adapters = {n: p for n, p in decoder.named_parameters() if n in initial}
    for key, param in adapters.items():
        with torch.no_grad():
            param.copy_(initial[key])
        param.requires_grad_(lora)
    params = {"projector": list(module.projector.parameters()), "head": list(module.head.parameters())}
    if lora:
        params["adapter"] = list(adapters.values())
    arm_identity = dict(identity, arm=name)
    torch.cuda.reset_peak_memory_stats()
    resumed = model_dir.exists()
    if resumed:
        verify_model(model_dir, arm_identity)
        saved = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
    else:
        optimizer = torch.optim.AdamW([{"params": p, "lr": CONFIG["adapter_lr" if key == "adapter" else "head_lr"]}
                                       for key, p in params.items()], weight_decay=.01)
        best, curves, gradients = None, [], None
        for epoch in range(CONFIG["epochs"]):
            decoder.train()
            order = np.random.default_rng(CONFIG["seed"] + epoch).permutation(len(train))
            total, count = 0., 0
            for step, (x, ids, mask, base, k, n) in enumerate(batches(train, order, vocab, vectors)):
                budget.check()
                optimizer.zero_grad(set_to_none=True)
                logits = module(decoder, x, ids, mask, base)
                loss = vote_loss(logits, k, n)
                loss.backward()
                if step == 1:
                    gradients = checked_gradients(params, require_all=True)
                torch.nn.utils.clip_grad_norm_([p for ps in params.values() for p in ps], 1.,
                                               error_if_nonfinite=True)
                optimizer.step()
                votes = int(n.sum())
                total += float(loss.detach()) * votes
                count += votes
            val_logits = predict(module, decoder, val, vocab, vectors, budget)
            value = weighted_bce(val_logits, val)
            curves.append({"epoch": epoch + 1, "train_BCE_nats": total / count, "validation_BCE_nats": value,
                           "gradient_norms_step_2": gradients})
            print(name, curves[-1], flush=True)
            if best is None or value < best:
                best = value
                saved = {"model": state(dict(module.named_parameters())),
                         "adapters": state(adapters) if lora else {}, "epoch": epoch + 1,
                         "validation_logits": val_logits.tolist()}
        saved["curves"] = curves
        saved["fit_seconds"] = time.perf_counter() - start
        saved["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        saved["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    module.load_state_dict(saved["model"])
    with torch.no_grad():
        for key, value in saved["adapters"].items():
            adapters[key].copy_(value)
    if lora and not any(not torch.equal(value, initial[key]) for key, value in saved["adapters"].items()):
        raise ValueError("Selected LoRA checkpoint has no adapter updates")
    logits = predict(module, decoder, val, vocab, vectors, budget)
    np.testing.assert_allclose(logits, saved["validation_logits"], atol=1e-6, rtol=0)
    preds = probabilities(val, logits)
    if resumed:
        original = pd.read_csv(model_dir / "validation_predictions.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(original, preds, check_exact=True)
    else:
        def writer(directory):
            torch.save(saved, directory / "best.pt")
            preds.to_csv(directory / "validation_predictions.csv", index=False, float_format="%.6f")
        publish_model(model_dir, arm_identity, writer)
    shuffled = (predict(module, decoder, val, vocab, vectors, budget, shuffled_images(val, CONFIG["seed"]))
                if lora else None)
    stats = {k: v for k, v in saved.items() if k not in ("model", "adapters", "validation_logits")}
    stats.update(resumed=resumed, attempt_seconds=time.perf_counter() - start)
    return preds, shuffled, stats


def run(args):
    start = time.perf_counter()
    if args.authorize != "RESEARCH_ONLY_L4":
        raise ValueError("Explicit research-only GPU authorization required")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if source != args.source or subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("Use the clean, pinned source checkout")
    budget = Budget(args.seconds / 3600, args.seconds / 3600, cuda=True, reserve=90)
    local = args.output
    local.mkdir(parents=True, exist_ok=True)
    print("Verifying cached pilot inputs", flush=True)
    (train, val), blocks = inputs(args.handoff, local.parent / "inputs")
    budget.check()
    if args.technical:
        train = train[train.painting.isin(train.painting.unique()[:6])].assign(p_positive=.5)
    else:
        print("Fitting fresh pilot-only E50+L component heads", flush=True)
        train, val, receipt = reference(train, val, blocks, budget)
        write_json(local / "crossfit.json", receipt)
    vectors = visual_vectors(blocks, pd.concat([train, val]))
    budget.check()
    print("Loading pinned Qwen NF4 decoder", flush=True)
    tokenizer, decoder, targets = backend(args.model_cache, args.technical)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(CONFIG["seed"])
    budget.check()
    torch.cuda.reset_peak_memory_stats()
    smoke(decoder, tokenizer, train, vectors, local)
    if args.technical:
        write_json(local / "status.json", {"status": "TECHNICAL_SMOKE_PASS", "scientific_metrics": False,
                                          "gpu": torch.cuda.get_device_name(), "wall_seconds": time.perf_counter()-start})
        return
    adapter_ids = {id(p) for p in adapter_parameters(decoder)}
    initial = state({name: p for name, p in decoder.named_parameters() if id(p) in adapter_ids})
    languages = sorted(set(train.language) | set(val.language))
    prompt_ids, prompt_mask = encode_prompts(tokenizer, languages)
    vocab = {lang: ids[mask.bool()].cpu() for lang, ids, mask in zip(languages, prompt_ids, prompt_mask)}
    identity = {"source": source, "config": CONFIG, "archive_sha256": ZIP_SHA256, "runtime": PARENT["runtime"],
                "gpu": torch.cuda.get_device_name(), "qwen_revision": PARENT["vlm"]["revision"],
                "train_crossfit": fingerprint(train.p_positive.tolist()), "validation_baseline": fingerprint(val.p_positive.tolist())}
    write_json(local / "identity.json", identity)
    baseline = val[KEYS + ["p_positive"]]
    models = {"E50+L": baseline}
    details = {}
    for arm in ("projector_only", "projector_qlora"):
        print(f"Starting {arm}; {budget.remaining():.0f}s remaining", flush=True)
        pred, shuffled, stats = fit_arm(arm, decoder, train, val, vocab, vectors, budget, identity,
                                      args.models / fingerprint(identity) / arm, initial)
        models[arm] = pred
        details[arm] = stats
        if shuffled is not None:
            models["projector_qlora_shuffled"] = probabilities(val, shuffled)
    metrics = {name: score(val, predictions) for name, predictions in models.items()}
    for name, pred in models.items():
        pred.to_csv(local / f"{name}.csv", index=False, float_format="%.6f")
    contrast = {k: v["cultural_contrast"] for k, v in metrics.items()}
    loss = {k: v["log_loss"] for k, v in metrics.items()}
    report = {"status": "PILOT_COMPLETE / EXPLORATORY", "metrics": metrics, "training": details,
              "comparisons": {
                  f"{a} vs {b}": {"contrast_delta": None if None in (contrast[a], contrast[b]) else contrast[a] - contrast[b],
                                  "macro_loss_delta_bits": None if None in (loss[a], loss[b]) else loss[a] - loss[b]}
                  for a, b in (("projector_only", "E50+L"), ("projector_qlora", "E50+L"),
                               ("projector_qlora", "projector_only"), ("projector_qlora", "projector_qlora_shuffled"))
              },
              "baseline_training_BCE_nats": weighted_bce(np.log(train.p_positive / (1-train.p_positive)), train),
              "baseline_validation_BCE_nats": weighted_bce(np.log(val.p_positive / (1-val.p_positive)), val),
              "shuffle_mean_abs_probability_change": float(np.abs(models["projector_qlora"].p_positive
                                                    - models["projector_qlora_shuffled"].p_positive).mean()),
              "gpu": torch.cuda.get_device_name(), "wall_seconds": time.perf_counter()-start,
              "lora_targets": targets, "production": "E50+L", "model_identity": fingerprint(identity)}
    write_json(local / "status.json", report)
    print(json.dumps({name: {k: s[k] for k in ("cultural_contrast", "log_loss")} for name, s in metrics.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "model-cache", "output", "models"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--authorize", choices=["RESEARCH_ONLY_L4"], required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--technical", action="store_true", help="Six paintings; stop after the isolated gradient probe")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        write_json(args.output / "status.json", {"status": exc.status if isinstance(exc, StopRun) else "BLOCKED",
                                                  "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
