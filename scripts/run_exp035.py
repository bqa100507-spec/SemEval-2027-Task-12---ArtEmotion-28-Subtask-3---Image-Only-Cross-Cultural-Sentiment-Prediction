#!/usr/bin/env python3
"""Prepare EXP-035 on CPU, then run one bounded real-image fold pilot."""

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_competition_direct_qwen_pilot import load_sealed, restore, save_sealed, trainables
from st3.baseline import file_sha256
from st3.competition_challenger import check_hash
from st3.competition_neighborhood import CONFIG as CORRECTION
from st3.competition_semantic_evaluation import write_atomic
from st3.competition_semantic_generation import CONFIG as BASE, TRACK1, verified_assets, verified_rgb
from st3.competition_track1 import load_p2, load_protocol
from st3.contrast_study import new_output
from st3.data import KEYS
from st3.e50_heads import read_json
from st3.e50_inference import safe_image_path
from st3.exp035_protocol import held_metrics, paired_epoch_curves, prepare_frames, scores, select_epoch

RULE = ROOT / "configs/exp035.json"
PILOT = ROOT / "runs/exp035_fold3_pilot"
HISTORICAL = ROOT / "runs/exp035_cultural_residual"
HISTORICAL_BUDGET_SHA = "d4a1b62879e051dc04405b21836aa4e2a5ce75a0c9e28f114f89ddc30d42a5fa"
KAGGLE_PREPARED_SHA = "baa3c118856a4867af80f734a093024baccb8ff110913e5b2e6c02e9e0753e07"
KAGGLE_RUN = Path("/kaggle/working/exp035_fold3")
KAGGLE_BUDGET = 5 * 60 * 60
PARTITIONS = ("fit", "inner", "outer")
SOURCE_FILES = (RULE, Path(__file__), ROOT / "src/st3/exp035.py", ROOT / "src/st3/exp035_protocol.py",
                BASE, CORRECTION, TRACK1, ROOT / "src/st3/competition_challenger.py",
                ROOT / "src/st3/competition_neighborhood.py", ROOT / "src/st3/competition_direct_qwen.py",
                ROOT / "src/st3/competition_semantic_recovery.py", ROOT / "scripts/run_competition_direct_qwen_pilot.py")


def prepare(args):
    track, base, config = map(read_json, (TRACK1, BASE, RULE))
    identity = {"source": {str(p.relative_to(ROOT)): file_sha256(p) for p in SOURCE_FILES},
                "inputs": track["p2_input_sha256"], "reference": base["incumbent_oof_sha256"]}
    path = args.run_dir / "prepared"
    if args.kaggle:
        # Preserve the original Windows preparation receipt verbatim. The sealed
        # CPU offsets are inputs to the Linux worker and are never refitted here.
        check_hash(path / "manifest.json", KAGGLE_PREPARED_SHA)
    if not path.exists():
        gold, features, _ = load_p2(args.inputs, track)
        frames, provenance = prepare_frames(gold, features, args.reference, track, read_json(CORRECTION), base, config)
        with new_output(path) as stage:
            files = {}
            for name, frame in frames.items():
                file = stage / (name + ".csv")
                frame.to_csv(file, index=False, lineterminator="\n", float_format="%.6f")
                files[file.name] = file_sha256(file)
            write_atomic(stage / "manifest.json", {"identity": identity, "files": files, "offsets": provenance})
    manifest = read_json(path / "manifest.json")
    expected_identity = identity
    if args.kaggle:
        expected_identity = identity | {"source": manifest["identity"]["source"]}
        check_hash(args.reference / "metrics.json", base["incumbent_metrics_sha256"])
    if manifest["identity"] != expected_identity:
        raise ValueError("EXP-035 prepared data use a different frozen identity")
    for key, digest in track["p2_input_sha256"].items():
        check_hash(args.inputs / key, digest)
    check_hash(args.reference / "oof.csv", base["incumbent_oof_sha256"])
    frames = {}
    for name in PARTITIONS:
        file = path / (name + ".csv")
        check_hash(file, manifest["files"][file.name])
        frames[name] = pd.read_csv(file, float_precision="round_trip")
    groups = [set(frames[n].painting) for n in PARTITIONS]
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError("EXP-035 partitions overlap")
    if not frames["outer"].fold.eq(config["fold"]).all() or any(
            frames[name].fold.eq(config["fold"]).any() for name in ("fit", "inner")):
        raise ValueError("EXP-035 partitions disagree with the pinned outer fold")
    print({name: (len(groups[i]), len(frames[name])) for i, name in enumerate(PARTITIONS)}, flush=True)
    return frames, manifest


def worker(args, deadline):
    import torch
    from st3.competition_semantic_recovery import load_model
    from st3.exp035 import (ResidualHead, add_merger_lora, frozen_feature, image_hidden, objective,
                           odds_update, select_languages, update)

    config, base, track = map(read_json, (RULE, BASE, TRACK1))
    frames, _ = prepare(args)
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-035 requires a CUDA GPU")
    gpu = torch.cuda.get_device_name()
    if args.kaggle:
        if torch.cuda.device_count() != 1 or not ("L4" in gpu or "T4" in gpu):
            raise RuntimeError(f"Choose Kaggle L4 or T4 x2 with only GPU 0 visible, found {gpu}")
        dtype_probe = torch.ones((16, 16), device="cuda", dtype=torch.float16)
        if not torch.equal(dtype_probe @ dtype_probe, torch.full_like(dtype_probe, 16)):
            raise RuntimeError("CUDA FP16 compatibility probe failed")
    elif "RTX 4060 Laptop" not in gpu:
        raise RuntimeError("This bounded pilot is configured for the local RTX 4060 Laptop")
    assets = verified_assets(args.checkpoint, base)
    _, model, processor, hw = load_model(args.checkpoint, base | {"attention": "sdpa"})
    model = model.model
    inventory = load_protocol(args.inputs, track).set_index("painting")
    lists = {n: dict(tuple(f.groupby("painting", sort=True))) for n, f in frames.items()}
    language_map = {name: i for i, name in enumerate(sorted(frames["fit"].language.unique()))}
    votes = float(frames["fit"].n.sum() / len(lists["fit"]))
    identity = {"prepared": file_sha256(args.run_dir / "prepared/manifest.json"), "assets": assets,
                "runtime": hw["runtime"], "gpu": hw["gpu"]}
    if args.kaggle:
        identity.update(source=read_json(args.run_dir / "provenance.json")["source_manifest_sha256"],
                        environment=read_json(args.run_dir / "provenance.json")["runtime"],
                        cuda=hw["cuda_runtime"], capability=list(torch.cuda.get_device_capability()))
    prompt = processor.apply_chat_template([{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": config["prompt"]}]}], tokenize=False, add_generation_prompt=True)

    def check_time(limit=deadline):
        if time.time() + 10 > min(limit, deadline):
            raise TimeoutError("Stopping within the GPU budget. Sealed epochs are reusable.")

    def inputs(name):
        row = inventory.loc[name]
        rgb = verified_rgb(safe_image_path(args.data_dir, row.image), row.image_sha256)
        return processor(text=[prompt], images=[rgb], return_tensors="pt").to(device)

    def values(panel, column):
        return torch.tensor(panel[column].to_numpy(), device=device, dtype=torch.float64)

    def offsets(panel):
        return values(panel, "prior"), torch.tensor([language_map.get(lang, -1) for lang in panel.language], device=device)

    def rows(panel):
        return *offsets(panel), values(panel, "k"), values(panel, "n")

    def new_head():
        return ResidualHead(model.language_model.config.hidden_size, len(language_map), config).to(device)

    def setup(head, adapter):
        params = [{"params": head.parameters(), "lr": config["head_lr"]}]
        if adapter is not None:
            params.append({"params": [adapter.left, adapter.right], "lr": config["adapter_lr"]})
        opt = torch.optim.AdamW(params, foreach=False, weight_decay=config["weight_decay"])
        return opt, torch.amp.GradScaler("cuda", init_scale=128.)

    completed = {}
    for arm in ("frozen", "merger"):
        path = args.run_dir / arm
        epochs = sorted(int(p.name.removeprefix("epoch_")) for p in path.glob("epoch_*"))
        if epochs != list(range(1, len(epochs) + 1)) or len(epochs) > config["epochs"]:
            raise ValueError(f"Incomplete checkpoint sequence for {arm}")
        for epoch in epochs:
            load_sealed(path / f"epoch_{epoch}", identity | {"arm": arm, "epoch": epoch}, torch)
        if (path / "outer").exists() and len(epochs) != config["epochs"]:
            raise ValueError(f"Outer predictions are present before the last {arm} epoch")
        completed[arm] = len(epochs)

    with new_output(args.run_dir / f"preflight_{time.time_ns()}") as preflight:
        names = sorted(lists["fit"])
        example = [names[i] for i in np.linspace(0, len(names)-1, config["preflight_artworks"], dtype=int)]
        pre_deadline = time.time() + config["preflight_seconds"]
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        frozen_example = {}
        for name in example:
            check_time(pre_deadline)
            frozen_feature(frozen_example, name, lambda: image_hidden(model, inputs(name)), device)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - t
        parity_sample = [example[0], example[len(example) // 2], example[-1]] if args.kaggle else example
        with torch.no_grad():
            for name in parity_sample:
                check_time(pre_deadline)
                live = image_hidden(model, inputs(name))
                cached = frozen_example[name].to(device)
                if live.dtype != cached.dtype or not torch.isfinite(live).all() or not torch.equal(live, cached):
                    raise RuntimeError(f"Cached Qwen state differs from an independent live frozen forward for {name}")
        head = new_head()
        opt, scale = setup(head, None)
        t = time.perf_counter()
        for name in example:
            check_time(pre_deadline)
            update(head, None, opt, scale, frozen_example[name].to(device), *rows(lists["fit"][name]), votes, config)
        torch.cuda.synchronize()
        cached_seconds = time.perf_counter() - t
        del opt, scale
        head = new_head()
        first = example[0]
        prior, ids, k, n = rows(lists["fit"][first])
        adapter = add_merger_lora(model, config)
        frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]

        def isolation():
            if any(p.grad is not None or p._version != version for p, version in frozen):
                raise RuntimeError("A frozen Qwen weight received a gradient or changed")

        with torch.no_grad():
            zero = image_hidden(model, inputs(first))
            if not torch.equal(zero, frozen_example[first].to(device)) or not torch.equal(
                    odds_update(prior, select_languages(head(zero).squeeze(0), ids)), prior):
                raise RuntimeError("Zero-initialized merger or residual did not preserve exact predictions")
        optimizer, scaler = setup(head, adapter)
        initial = trainables(adapter)
        t = time.perf_counter()
        probe = []
        for name in example:
            check_time(pre_deadline)
            state = image_hidden(model, inputs(name))
            probe.append(update(head, adapter, optimizer, scaler, state, *rows(lists["fit"][name]), votes, config))
            del state
        torch.cuda.synchronize()
        back_seconds = time.perf_counter() - t
        changes = {k: float((v-initial[k]).norm()) for k, v in trainables(adapter).items()}
        isolation()
        if any(not v > 0 for v in changes.values()) or not max(x["adapter_gradient"] for x in probe[1:]) > 1e-8:
            raise RuntimeError(f"Merger adapter failed gradient/update verification: {changes}")
        peak = torch.cuda.max_memory_reserved()
        fit_size, inner_size, outer_size = (len(lists[name]) for name in PARTITIONS)
        b_left, c_left = (config["epochs"] - completed[arm] for arm in ("frozen", "merger"))
        b_outer, c_outer = (outer_size if not (args.run_dir / arm / "outer").exists() else 0
                            for arm in ("frozen", "merger"))
        b_pairs = fit_size + inner_size
        b_seconds = (forward_seconds * ((b_pairs if b_left else 0) + b_outer)
                     + cached_seconds * (b_left * b_pairs + b_outer)) / len(example)
        c_seconds = (back_seconds * fit_size * c_left
                     + forward_seconds * (c_left * inner_size + c_outer)) / len(example)
        estimate = b_seconds + c_seconds
        needed = estimate * config["runtime_margin"]
        fit_budget = peak + config["memory_margin_mib"] * 2**20 < hw["initial_free_bytes"]
        receipt = {"state": "PASS" if fit_budget else "MEMORY_MARGIN_FAIL", "hardware": hw, "targets": ["visual.merger.linear_fc2"],
                   "matrix_updates": changes, "zero_parity": True, "frozen_weights_verified": True,
                   "frozen_cache_exact_parity": True, "parity_sample": parity_sample, "completed_epochs": completed,
                   "peak_reserved_bytes": peak, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "adapter_parameters": adapter.left.numel() + adapter.right.numel(),
                   "head_parameters": sum(p.numel() for p in head.parameters()),
                   "sample": example, "forward_seconds": forward_seconds, "adapter_step_seconds": back_seconds,
                   "frozen_cached_step_seconds": cached_seconds,
                   "estimated_frozen_seconds": b_seconds, "estimated_merger_seconds": c_seconds,
                   "estimated_pilot_seconds": estimate, "estimated_with_reserve_seconds": needed, "steps": probe}
        write_atomic(preflight / "result.json", receipt)
    print({k: receipt[k] for k in ("state", "estimated_with_reserve_seconds", "peak_reserved_bytes")}, flush=True)
    remaining = deadline - time.time()
    if args.preflight_only or not fit_budget or needed + 30 > remaining:
        reason = "PREFLIGHT_ONLY" if args.preflight_only else (
            "PILOT_EXCEEDS_MEASURED_BUDGET" if fit_budget else "MEMORY_MARGIN_FAIL")
        write_atomic(args.run_dir / "status.json", {"state": reason, "remaining_seconds": remaining,
                                                   "required_seconds": needed + 30})
        return

    # Discard diagnostic updates before either scientific arm starts.
    restore(adapter, initial)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scaler, frozen_example
    frozen_cache = {}
    for arm in ("frozen", "merger"):
        path = args.run_dir / arm
        path.mkdir(exist_ok=True)
        current = new_head()
        adapted = adapter if arm == "merger" else None
        opt, scale = setup(current, adapted)
        wrapping = torch.nn.ModuleDict({"head": current, **({"adapter": adapted} if adapted is not None else {})})
        arm_id = identity | {"arm": arm}

        def feature(name):
            if adapted is not None:
                return image_hidden(model, inputs(name))
            return frozen_feature(frozen_cache, name, lambda: image_hidden(model, inputs(name)), device)

        @torch.no_grad()
        def predict(partition):
            inner = partition == "inner"
            out, losses = [], []
            for name, panel in lists[partition].items():
                check_time()
                prior, ids = offsets(panel)
                delta = select_languages(current(feature(name)).squeeze(0), ids)
                predictions = odds_update(prior, delta)
                if inner:
                    loss, terms = objective(prior, delta, values(panel, "k"), values(panel, "n"), votes, config)
                    diagnostics = [loss, *terms, delta.square().mean().sqrt(), (predictions - prior).abs().mean()]
                    losses.append(torch.stack(diagnostics).cpu().numpy())
                out.append(panel[KEYS].assign(candidate=predictions.cpu().numpy()))
            return pd.concat(out, ignore_index=True), dict(zip(
                ("loss", "bce", "centered_mse", "residual_penalty", "delta_rms", "mean_abs_probability_change"),
                map(float, np.mean(losses, axis=0)))) if inner else {}

        history = []
        for epoch in range(1, config["epochs"] + 1):
            check_time()
            directory = path / f"epoch_{epoch}"
            eid = arm_id | {"epoch": epoch}
            if directory.exists():
                state = load_sealed(directory, eid, torch)
                restore(wrapping, state["trainable"])
                opt.load_state_dict(state["optimizer"])
                scale.load_state_dict(state["scaler"])
                if args.kaggle:
                    rng = state["rng"]
                    torch.set_rng_state(rng["torch"])
                    torch.cuda.set_rng_state(rng["cuda"])
                    random.setstate(rng["python"])
                    name, keys, pos, has_gauss, cached_gauss = rng["numpy"]
                    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached_gauss))
            else:
                begin = time.perf_counter()
                old = trainables(adapted) if adapted is not None else {}
                log = []
                torch.cuda.reset_peak_memory_stats()
                order = np.random.default_rng(config["seed"] + epoch).permutation(sorted(lists["fit"]))
                for name in order:
                    check_time()
                    log.append(update(current, adapted, opt, scale, feature(name), *rows(lists["fit"][name]), votes, config))
                isolation()
                train_seconds = time.perf_counter() - begin
                inner_predictions, losses = predict("inner")
                pred = frames["inner"].merge(inner_predictions, on=KEYS, validate="one_to_one")
                inner = scores(pred, ("prior", "candidate"), track["test_like_excluded"])
                curve = {"epoch": epoch, "train": {k: float(np.mean([v[k] for v in log])) for k in log[0]},
                         "inner": inner, "inner_losses": losses, "train_seconds": train_seconds,
                         "total_seconds": time.perf_counter()-begin,
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "adapter_updates": {k: float((v-old[k]).norm()) for k, v in trainables(adapted).items()}
                         if adapted is not None else {}}
                state = {"trainable": trainables(wrapping), "optimizer": opt.state_dict(), "scaler": scale.state_dict(),
                         "curve": curve}
                if args.kaggle:
                    name, keys, pos, has_gauss, cached_gauss = np.random.get_state()
                    state["rng"] = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                                    "python": random.getstate(), "order_seed": config["seed"] + epoch,
                                    "numpy": (name, keys.tolist(), pos, has_gauss, cached_gauss)}
                save_sealed(directory, eid, state, torch)
            history.append(state["curve"])
            print(arm, epoch, history[-1], flush=True)
        winner, guardrail = select_epoch(history, config["inner_log_loss_margin"])
        selected = path / f"epoch_{winner['epoch']}"
        restore(wrapping, load_sealed(selected, arm_id | {"epoch": winner["epoch"]}, torch)["trainable"])
        final = path / "outer"
        checkpoint_sha = read_json(selected / "seal.json")["sha256"]
        if final.exists():
            seal = read_json(final / "seal.json")
            if seal["selected_epoch"] != winner["epoch"] or seal["checkpoint"] != checkpoint_sha:
                raise ValueError("Sealed outer predictions do not match the inner-selected checkpoint")
        else:
            pred, _ = predict("outer")
            pred = pred.rename(columns={"candidate": arm})
            with new_output(final) as stage:
                pred.to_csv(stage / "predictions.csv", index=False, float_format="%.17g")
                write_atomic(stage / "seal.json", {"identity": arm_id, "selected_epoch": winner["epoch"],
                    "checkpoint": checkpoint_sha,
                    "history": history, "inner_log_loss_guardrail": guardrail,
                    "sha256": file_sha256(stage / "predictions.csv")})

    result = frames["outer"].copy()
    arms = {}
    for arm in ("frozen", "merger"):
        final = args.run_dir / arm / "outer"
        seal = read_json(final / "seal.json")
        if seal["identity"] != identity | {"arm": arm}:
            raise ValueError("Outer predictions are from another EXP-035 run")
        check_hash(final / "predictions.csv", seal["sha256"])
        result = result.merge(pd.read_csv(final / "predictions.csv", float_precision="round_trip"), on=KEYS,
                              validate="one_to_one")
        if len(result) != len(frames["outer"]):
            raise ValueError("Outer prediction coverage differs")
        arms[arm] = seal
    summary = held_metrics(result, track)
    with new_output(args.run_dir / "results") as stage:
        result[KEYS + ["prior", "frozen", "merger"]].to_csv(stage / "predictions.csv", index=False, float_format="%.17g")
        write_atomic(stage / "metrics.json", summary | {"arms": arms, "preflight": receipt,
                                                          "paired_inner_curves": paired_epoch_curves(arms)})
        write_atomic(stage / "seal.json", {p.name: file_sha256(p) for p in stage.iterdir()})
    write_atomic(args.run_dir / "status.json", {"state": "FOLD_PILOT_COMPLETE"})


def supervise(args):
    budget = read_json(RULE)["cumulative_gpu_seconds"]
    root = args.run_dir
    ledger = root / "gpu_budget.json"
    with FileLock(str(ledger) + ".lock", timeout=0):
        historical = HISTORICAL / "gpu_budget.json"
        check_hash(historical, HISTORICAL_BUDGET_SHA)
        used = float(read_json(historical)["charged_seconds"])
        if ledger.exists():
            last = read_json(ledger)
            used = float(last["charged_seconds"])
            if last["active_since"] is not None:
                raise RuntimeError("Previous GPU worker did not close its budget ledger. Confirm exit before resuming.")
        if used >= budget - 30:
            raise TimeoutError("The EXP-035 GPU budget is exhausted")
        start = time.time()
        deadline = start + budget - used
        if args.preflight_only:
            deadline = min(deadline, start + read_json(RULE)["preflight_seconds"])
        write_atomic(ledger, {"charged_seconds": used, "active_since": start, "deadline": deadline})
        try:
            subprocess.run([sys.executable, "-u", str(Path(__file__)), *sys.argv[1:], "--worker"],
                           check=True, timeout=max(0, deadline - time.time() - 2))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            write_atomic(root / "status.json", {"state": "FAILED_OR_BUDGET_STOP", "sealed_epochs_retained": True})
            raise
        finally:
            write_atomic(ledger, {"charged_seconds": used + time.time() - start, "active_since": None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/upstream")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/competition_neighborhood_residual/results")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/qwen3-vl-4b-ebb281e")
    parser.add_argument("--run-dir", type=Path, default=PILOT)
    parser.add_argument("--run-bounded-gpu", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--kaggle", action="store_true", help="Use the pinned Kaggle handoff through its launcher")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.kaggle and (not sys.platform.startswith("linux") or not Path("/kaggle/input").is_dir()
                        or args.run_dir.resolve() != KAGGLE_RUN or not args.worker):
        parser.error("Kaggle execution requires Linux, the Kaggle mounts and the notebook launcher")
    if not args.kaggle and args.run_bounded_gpu and args.run_dir.resolve() != PILOT.resolve():
        parser.error("GPU runs must use the canonical EXP-035 pilot directory to share the cumulative budget")
    if args.worker:
        lease = read_json(args.run_dir / "gpu_budget.json")
        if not args.run_bounded_gpu or lease["active_since"] is None:
            raise PermissionError("GPU worker requires an active budget ledger")
        if args.kaggle and (lease.get("platform") != "kaggle" or
                            lease["deadline"] - lease["active_since"] + lease["charged_seconds"] > KAGGLE_BUDGET):
            raise PermissionError("Kaggle GPU worker requires the five-hour notebook lease")
        worker(args, lease["deadline"])
    else:
        prepare(args)
        if args.run_bounded_gpu and not (args.run_dir / "results").exists():
            supervise(args)


if __name__ == "__main__":
    main()
