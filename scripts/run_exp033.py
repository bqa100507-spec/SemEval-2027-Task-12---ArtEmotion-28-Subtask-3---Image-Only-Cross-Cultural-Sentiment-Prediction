#!/usr/bin/env python3
"""Pinned NaFlex CPU-testable runner. GPU work is explicit and owner-gated."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import monotonic

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.exp033 import suffix
from st3.exp033_io import (KEYS, atomic, canonical, fuse, load_population, publish_checkpoint, read_checkpoint,
                          read_json, restore_inputs, save_json, sha256)
from st3.exp033_train import Bank, Panel, Trainer, deterministic_order, inner_split, select_epoch
from st3.exp033_vision import build_cache, extract_batch, load_model
from st3.metrics import score

CONFIG = ROOT / "configs/exp033.json"
ARMS = ("A", "B", "C", "D")


def source_identity(manifest_sha, device):
    from importlib.metadata import version

    sources = [Path(__file__), CONFIG, ROOT / "requirements-exp033.lock.txt"]
    sources += sorted((ROOT / "src/st3").glob("exp033*.py"))
    sources += [ROOT / f"src/st3/{name}.py" for name in ("data", "submission", "metrics")]
    identity = {"input_manifest": manifest_sha, "files": {str(p.relative_to(ROOT).as_posix()): sha256(p) for p in sources},
                "python": sys.version.split()[0], "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device), "capability": list(torch.cuda.get_device_capability(device)),
                "versions": {n: version(n) for n in ("transformers", "huggingface-hub", "tokenizers",
                                                    "safetensors", "numpy", "scipy", "pandas", "Pillow")}}
    return identity


def enforce_lock():
    from importlib.metadata import version

    if sys.version.split()[0] != "3.12.10" or str(torch.__version__) != "2.7.1+cu126":
        raise ValueError("EXP-033 requires Python 3.12.10 and Torch 2.7.1+cu126")
    for line in (ROOT / "requirements-exp033.lock.txt").read_text(encoding="utf8").splitlines():
        if line and not line.startswith("#"):
            pkg, pin = line.split("==")
            if version(pkg) != pin:
                raise ValueError(f"Dependency mismatch: {pkg} must be {pin}")


def technical_preflight(out, identity, vision, processor, root, frame, images, config, device, deadline):
    pool = images.loc[images.fold.ne(1)].set_index("painting")
    keys = deterministic_order(pool.index, config["seed"])[:config["preflight_images"]]
    if len(keys) != config["preflight_images"]:
        raise ValueError("Preflight needs the full representative diagnostic sample")
    selected = pool.loc[keys]
    pairs = frame.loc[frame.painting.isin(keys)]
    panel = Panel(pairs, {key: i for i, key in enumerate(keys)})
    timings, errors = None, []
    for mb in config["microbatch_options"]:
        try:
            worker = frozen_worker = bank = batch = pieces = first = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            free, _ = torch.cuda.mem_get_info(device)
            base_reserved = torch.cuda.memory_reserved(device)
            started = monotonic()
            pieces, parity = [], {}
            decode = 0.
            for i in range(0, len(keys), mb):
                if monotonic() > deadline:
                    raise TimeoutError("Preflight wall budget reached")
                paths = [root / p for p in selected.image.iloc[i:i + mb]]
                data, deltas, seconds = extract_batch(vision, processor, paths, config, device, parity=i == 0)
                decode += seconds
                pieces.append(data)
                parity.update(deltas)
            bank = Bank({k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}, device)
            extraction = monotonic() - started
            worker = Trainer(vision, bank, panel, config)
            batch = bank.take(list(range(mb)), ("prefix", "mask", "tokens"))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                initial = suffix(vision, batch["prefix"], batch["mask"], config["tail_blocks"], worker.adapters)
            torch.testing.assert_close(initial[batch["mask"]], batch["tokens"][batch["mask"]], atol=2e-4, rtol=2e-4)
            parity["zero_adapter"] = float((initial - batch["tokens"]).abs()[batch["mask"]].max())
            epoch = config["warmup_epochs"] + 1
            diagnostics = worker.train_epoch("D", epoch, mb, deadline=deadline)
            save_json(out / "preflight_step_diagnostics.json", diagnostics)
            if diagnostics["clip_fraction"] > config["max_clip_fraction"]:
                raise FloatingPointError("Preflight excessive clipping. Inspect measured module norms.")
            if diagnostics["skip_fraction"] > config["max_scaler_skip_fraction"]:
                raise FloatingPointError("Preflight FP16 skipped too many optimizer steps")
            for adapter in worker.adapters:
                for weight in (adapter.up.weight, adapter.down.weight):
                    if weight.grad is None or not torch.isfinite(weight.grad).all() or weight.grad.norm().item() == 0:
                        raise RuntimeError("Adapter gradient audit failed")
            if sum(a.up.weight.norm().item() for a in worker.adapters) == 0:
                raise RuntimeError("No adapter parameter update")
            with tempfile.TemporaryDirectory(dir=out) as candidate:
                stage = Path(candidate)
                publish_checkpoint(stage, identity, worker.state(epoch, diagnostics))
                restored = read_checkpoint(stage, identity)
                worker.train_epoch("D", epoch + 1, mb, deadline=deadline)
                first = {k: p.detach().cpu().clone() for k, p in worker.adapters.state_dict().items()}
                worker.restore(restored)
                worker.train_epoch("D", epoch + 1, mb, deadline=deadline)
                for k, actual in worker.adapters.state_dict().items():
                    torch.testing.assert_close(actual.cpu(), first[k], atol=0, rtol=0)
            peak = torch.cuda.max_memory_reserved(device)
            if free - max(0, peak - base_reserved) < config["reserve_gib"] * 1024**3:
                errors.append({"microbatch": mb, "reason": "Measured reserved-memory headroom below 2 GiB"})
                del worker, bank, batch, pieces
                continue
            rate = diagnostics["paintings_per_second"]
            frozen_worker = Trainer(vision, bank, panel, config)
            frozen = frozen_worker.train_epoch("C", epoch, mb, deadline=deadline)
            started_prediction = monotonic()
            worker.predict(pairs[KEYS], {key: i for i, key in enumerate(keys)}, "D", epoch + 1, mb)
            prediction_seconds = monotonic() - started_prediction
            peak = torch.cuda.max_memory_reserved(device)
            if free - max(0, peak - base_reserved) < config["reserve_gib"] * 1024**3:
                errors.append({"microbatch": mb, "reason": "Combined measurements exceed the 2 GiB reserve"})
                del worker, frozen_worker, bank, batch, pieces
                continue
            projected = (
                100000 / rate + 307560 / frozen["paintings_per_second"]
                + 30000 * prediction_seconds / len(keys) + 1335 * extraction / len(keys)
            ) / 3600
            timings = {"status": "PASS", "mode": "TECHNICAL_ONLY", "identity": identity, "microbatch": mb,
                       "gpu": identity["gpu"], "training_paintings": len(keys), "parity_max_abs": parity,
                       "peak_allocated": torch.cuda.max_memory_allocated(device), "peak_reserved": peak,
                       "free_before_bytes": free, "extraction_seconds": extraction, "decode_seconds": decode,
                       "train": diagnostics, "frozen_train": frozen, "prediction_seconds": prediction_seconds,
                       "projected_measured_compute_gpu_hours": projected, "optimizer_rng_replay": "EXACT_ADAPTER_TENSORS",
                       "projected_suffix_gpu_hours": 100000 / rate / 3600, "prior_attempts": errors}
            publish_checkpoint(out / "persistence_canary", identity, worker.state(epoch + 1, diagnostics))
            timings["canary_sha256"] = sha256(out / "persistence_canary/state.pt")
            timings["publication_nonce"] = os.urandom(16).hex()
            save_json(out / "preflight.json", timings)
            break
        except torch.OutOfMemoryError as error:
            # Leave the exception scope before the next variant releases traceback-held tensors.
            errors.append({"microbatch": mb, "reason": f"CUDA OOM: {str(error).splitlines()[0]}"})
    if timings is None:
        raise RuntimeError(f"CUDA memory feasibility sequence exhausted: {errors}")
    return timings


def verify_preflight(folder, identity, expected_receipt):
    path = folder / "preflight.json"
    if sha256(path) != expected_receipt:
        raise ValueError("Local preflight receipt SHA-256 mismatch")
    receipt = read_json(path)
    if receipt["identity"] != identity or receipt["status"] != "PASS":
        raise ValueError("Local preflight must come from the same source, input, runtime, and GPU type")
    checkpoint = read_checkpoint(folder / "persistence_canary", identity)
    if (sha256(folder / "persistence_canary/state.pt") != receipt["canary_sha256"]
            or not checkpoint["optimizer"]["state"] or not checkpoint["scaler"]):
        raise ValueError("Local canary lost optimizer or scaler state")
    return receipt


def fit_scope(path, identity, factory, arm, epochs, microbatch, config, deadline,
              *, warmup=None, validation=None, indices=None, probe_state=None):
    worker = factory()
    key = {"run": identity, "scope": path.parent.name, "fold": path.parent.parent.name, "condition": arm}
    lower = 1
    if warmup is not None:
        worker.restore(read_checkpoint(warmup, key | {"condition": "C", "epoch": config["warmup_epochs"]}))
        lower = config["warmup_epochs"] + 1
    if probe_state is not None:
        # Freeze D token features for this fold's independent fixed-pooling readout.
        worker.adapters.load_state_dict(probe_state["adapters"])
        worker.probe = True
        with torch.no_grad():
            features = {}
            panel = Panel(validation[KEYS], indices)
            rows_all = sorted(set(worker.panel.rows) | set(panel.rows))
            for start in range(0, len(rows_all), microbatch):
                if monotonic() >= deadline:
                    raise TimeoutError("Session budget reached during fixed-probe extraction")
                rows = rows_all[start:start + microbatch]
                h, _ = worker.forward(rows, "B", 0)
                features.update(zip(rows, h.cpu()))
        worker.fixed = features
    records = {}
    if warmup is not None:
        # The C and D first three epochs are exactly the same training trajectory.
        for shared in range(1, lower):
            state = read_checkpoint(warmup.parent / f"epoch_{shared:02d}", key | {"condition": "C", "epoch": shared})
            records[shared] = state["receipt"]
    for epoch in range(lower, epochs + 1):
        directory = path / f"epoch_{epoch:02d}"
        checkpoint_identity = key | {"epoch": epoch}
        if (directory / "seal.json").exists():
            state = read_checkpoint(directory, checkpoint_identity)
            worker.restore(state)
            record = state["receipt"]
        else:
            started = monotonic()
            record = worker.train_epoch("B" if arm == "probe" else arm, epoch, microbatch, deadline=deadline)
            if validation is not None and arm != "probe":
                pred = worker.predict(validation, indices, arm, epoch, microbatch)
                scores = score(validation[KEYS + ["k", "n"]], pred)
                record["inner"] = {k: scores[k] for k in ("cultural_contrast", "log_loss", "pair_mean_log_loss")}
            record["epoch_wall_seconds"] = monotonic() - started
            publish_checkpoint(directory, checkpoint_identity, worker.state(epoch, record))
        records[epoch] = record
        save_json(path / "curves.json", {"epochs": records, "shared_warmup_epochs": lower - 1})
        print(f"{path}: epoch {epoch}/{epochs} clip={record['clip_fraction']:.3f} "
              f"skips={record['skip_fraction']:.3f}", flush=True)
        if (record["skip_fraction"] > config["max_scaler_skip_fraction"]
                or (epoch - 1 in records and min(record["clip_fraction"], records[epoch - 1]["clip_fraction"])
                    > config["max_clip_fraction"])):
            save_json(path / "numerical_block.json", {"epoch": epoch, "diagnostics": record})
            raise FloatingPointError("Excessive clipping or scaler skips. Published diagnostics require review.")
    return worker, records


def complete_fold(out, fold, identity, population, bank, indices, vision, config, mb, deadline):
    held = population.loc[population.fold.eq(fold)]
    train = population.loc[population.fold.ne(fold)]
    fit, inner = inner_split(train, config["inner_fraction"], config["seed"] + fold)
    inner_panel, outer_panel = Panel(fit, indices), Panel(train, indices)

    def factory(panel):
        return lambda: Trainer(vision, bank, panel, config)

    scope = out / f"fold_{fold}"
    selections = {}
    for arm in ("C", "D"):
        base = scope / "inner"
        warm = base / "C" / f"epoch_{config['warmup_epochs']:02d}" if arm == "D" else None
        _, receipts = fit_scope(base / arm, identity, factory(inner_panel), arm, max(config["checkpoints"]), mb,
                                config, deadline, warmup=warm, validation=inner, indices=indices)
        selections[arm] = {e: receipts[e]["inner"] for e in config["checkpoints"]}
    epoch = select_epoch(selections["C"], selections["D"], config["guardrail"])
    save_json(scope / "selection.json", {"epoch": epoch, "inner": selections})
    checks = {}
    for arm in (*ARMS, "probe"):
        base = scope / "outer"
        warm = base / "C" / f"epoch_{config['warmup_epochs']:02d}" if arm == "D" else None
        worker, receipts = fit_scope(base / arm, identity, factory(outer_panel), arm, epoch, mb, config,
                                     deadline, warmup=warm, indices=indices,
                                     validation=held if arm == "probe" else None,
                                     probe_state=adapted_state if arm == "probe" else None)
        if arm == "D":
            adapted_state = worker.state(epoch, receipts[epoch])
        preds = worker.predict(held[KEYS], indices, "B" if arm == "probe" else arm, epoch, mb)
        destination = scope / f"{arm}.csv"
        atomic(destination, lambda p: preds.to_csv(p, index=False, float_format="%.6f"))
        checks[arm] = sha256(destination)
    save_json(scope / "outer_seal.json", {"identity": identity, "epoch": epoch, "prediction_sha256": checks})


def evaluate_all(out, identity, frame, refs, side, config):
    from st3.exp033_metrics import report

    predictions = {}
    for fold in range(1, 6):
        path = out / f"fold_{fold}"
        if not (path / "outer_seal.json").exists():
            return {"status": "PARTIAL_FOLDS", "outer_scores": None}
        seal = read_json(path / "outer_seal.json")
        if seal["identity"] != identity or set(seal["prediction_sha256"]) != set(ARMS) | {"probe"}:
            raise ValueError("Outer predictions have incompatible identity or missing conditions")
        keys = frame.loc[frame.fold.eq(fold), KEYS]
        from st3.submission import validate_predictions
        for name, digest in seal["prediction_sha256"].items():
            csv = path / f"{name}.csv"
            if sha256(csv) != digest:
                raise ValueError(f"Corrupt outer prediction: {csv}")
            prediction = pd.read_csv(csv)
            validate_predictions(prediction, keys)
            predictions.setdefault(name, []).append(prediction)
    solo = {name: pd.concat(parts, ignore_index=True) for name, parts in predictions.items()}
    ensemble = {}
    for name, single in solo.items():
        s = side.merge(single, on=KEYS, validate="one_to_one")
        ensemble[name] = fuse(s, s["p_positive"].to_numpy())
        atomic(out / f"standalone_{name}.csv", lambda p: single.to_csv(p, index=False, float_format="%.6f"))
        atomic(out / f"fusion_{name}.csv", lambda p: ensemble[name].to_csv(p, index=False, float_format="%.6f"))
    return report(frame, solo, ensemble | refs, config)


def run_five_folds(out, identity, population, images, refs, side, vision, processor,
                   inputs, config, device, receipt, deadline, cache_dir):
    mb = receipt["microbatch"]
    indices = {name: i for i, name in enumerate(images.painting)}
    # The frozen bank is scoped to input, vision, source and runtime, never to an adapted fold.
    cache_root = cache_dir / ("frozen_" + hashlib.sha256(canonical(identity).encode()).hexdigest()[:16])
    arrays = build_cache(cache_root, identity, [inputs / p for p in images.image],
                         vision, processor, config, device, mb, deadline)
    started = monotonic()
    for fold in range(1, 6):
        folder = out / f"fold_{fold}"
        if (folder / "outer_seal.json").exists():
            seal = read_json(folder / "outer_seal.json")
            if seal["identity"] != identity or set(seal["prediction_sha256"]) != set(ARMS) | {"probe"}:
                raise ValueError(f"Incompatible outer fold seal: {fold}")
            for name, digest in seal["prediction_sha256"].items():
                if sha256(folder / f"{name}.csv") != digest:
                    raise ValueError(f"Corrupt sealed outer prediction: fold {fold}, arm {name}")
            print(f"EXP-033 P2 fold {fold} reused verified predictions", flush=True)
            continue
        print(f"EXP-033 P2 fold {fold} started", flush=True)
        complete_fold(out, fold, identity, population, Bank(arrays, device), indices, vision, config, mb, deadline)
    result = evaluate_all(out, identity, population, refs, side, config)
    result["attempt_wall_seconds"] = monotonic() - started
    result["peak_allocated"] = torch.cuda.max_memory_allocated(device)
    result["peak_reserved"] = torch.cuda.max_memory_reserved(device)
    return result


def resume_plan(out, identity, preflight_microbatch):
    path = out / "run_plan.json"
    if path.exists():
        stored = read_json(path)
        if stored["identity"] != identity or stored["microbatch"] > preflight_microbatch:
            raise ValueError("Resume requires the original run identity and a safe original microbatch")
        return stored["microbatch"]
    save_json(path, {"identity": identity, "microbatch": preflight_microbatch})
    return preflight_microbatch


def run_session(out, identity, population, images, refs, side, inputs, config, device,
                model_dir, deadline, cache_dir, preflight_only=False):
    attempt = Path(tempfile.mkdtemp(prefix="preflight-", dir=out))
    save_json(out / "decision.json", {"status": "RUNNING", "stage": "preflight", "attempt": attempt.name})
    stage = "preflight"
    try:
        vision, processor = load_model(config, model_dir, device)
        technical_preflight(attempt, identity, vision, processor, inputs, population, images, config, device, deadline)
        receipt = verify_preflight(attempt, identity, sha256(attempt / "preflight.json"))
        print(json.dumps(receipt, indent=2), flush=True)
        if preflight_only:
            result = receipt
        else:
            stage = "cv"
            receipt = receipt | {"microbatch": resume_plan(out, identity, receipt["microbatch"])}
            save_json(out / "decision.json", {"status": "RUNNING", "stage": stage, "attempt": attempt.name})
            result = run_five_folds(out, identity, population, images, refs, side, vision, processor,
                                    inputs, config, device, receipt, deadline, cache_dir)
        save_json(out / "decision.json", result)
        print(json.dumps({"status": result["status"]}), flush=True)
        return result
    except (Exception, KeyboardInterrupt) as error:
        save_json(out / "decision.json", {"status": "STOPPED", "stage": stage, "reason": str(error),
                                         "attempt": attempt.name})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "preflight", "run"))
    for arg in ("input-zip", "inputs", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--zip-sha", required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    config = read_json(CONFIG)
    if args.stage == "run" and not args.authorize:
        parser.error("The complete run requires owner authorization")
    if args.max_seconds <= 0:
        parser.error("A positive finite budget is required")
    deadline = monotonic() + args.max_seconds
    out.mkdir(parents=True, exist_ok=True)
    args.inputs.parent.mkdir(parents=True, exist_ok=True)
    restore_inputs(args.input_zip, args.inputs, args.zip_sha, args.manifest_sha)
    population, images, refs, side = load_population(args.inputs, config)
    if args.stage == "validate":
        print(f"Validated {len(population)} pairs, {len(images)} P2 paintings and exact ridge replay")
        return
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise ValueError("Enable a compatible CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    enforce_lock()
    identity = source_identity(args.manifest_sha, device)
    cache_dir = (args.cache_dir or out.parent).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_session(out, identity, population, images, refs, side, args.inputs, config, device,
                args.model_dir, deadline, cache_dir, preflight_only=args.stage == "preflight")


if __name__ == "__main__":
    main()
