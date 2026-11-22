#!/usr/bin/env python3
"""Prepare EXP-035 on CPU, then run one bounded real-image fold pilot."""

import argparse
import json
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
from st3.exp035_protocol import held_metrics, prepare_frames, scores, select_epoch

RULE = ROOT / "configs/exp035.json"
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
    if manifest["identity"] != identity:
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
    from st3.exp035 import ResidualHead, add_merger_lora, image_hidden, odds_update, select_languages, update

    config, base, track = map(read_json, (RULE, BASE, TRACK1))
    frames, prepared = prepare(args)
    device = "cuda"
    if not torch.cuda.is_available() or "RTX 4060 Laptop" not in torch.cuda.get_device_name():
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
    prompt = processor.apply_chat_template([{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": config["prompt"]}]}], tokenize=False, add_generation_prompt=True)

    def check_time(limit=deadline):
        if time.time() + 10 > min(limit, deadline):
            raise TimeoutError("Stopping within the GPU budget. Sealed epochs are reusable.")

    def inputs(name):
        row = inventory.loc[name]
        rgb = verified_rgb(safe_image_path(args.data_dir, row.image), row.image_sha256)
        return processor(text=[prompt], images=[rgb], return_tensors="pt").to(device)

    def rows(panel):
        prior, k, n = [torch.tensor(panel[z].to_numpy(), device=device, dtype=torch.float64) for z in ("prior", "k", "n")]
        return prior, torch.tensor([language_map.get(lang, -1) for lang in panel.language], device=device), k, n

    def new_head():
        return ResidualHead(model.language_model.config.hidden_size, len(language_map), config).to(device)

    def setup(head, adapter):
        params = [{"params": head.parameters(), "lr": config["head_lr"]}]
        if adapter is not None:
            params.append({"params": [adapter.left, adapter.right], "lr": config["adapter_lr"]})
        opt = torch.optim.AdamW(params, foreach=False, weight_decay=config["weight_decay"])
        return opt, torch.amp.GradScaler("cuda", init_scale=128.)

    with new_output(args.run_dir / f"preflight_{time.time_ns()}") as preflight:
        names = sorted(lists["fit"])
        example = [names[i] for i in np.linspace(0, len(names)-1, config["preflight_artworks"], dtype=int)]
        pre_deadline = time.time() + config["preflight_seconds"]
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        with torch.no_grad():
            frozen_example = {}
            for name in example:
                check_time(pre_deadline)
                frozen_example[name] = image_hidden(model, inputs(name))
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - t
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
            if not torch.equal(zero, frozen_example[first]) or not torch.equal(
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
        estimate = (back_seconds * len(lists["fit"]) * config["epochs"] + forward_seconds * (
            len(lists["fit"]) + (1 + config["epochs"]) * len(lists["inner"]) + 2 * len(lists["outer"]))) / len(example)
        needed = estimate * config["runtime_margin"]
        fit_budget = peak + config["memory_margin_mib"] * 2**20 < hw["initial_free_bytes"]
        receipt = {"state": "PASS" if fit_budget else "MEMORY_MARGIN_FAIL", "hardware": hw, "targets": ["visual.merger.linear_fc2"],
                   "matrix_updates": changes, "zero_parity": True, "frozen_weights_verified": True,
                   "peak_reserved_bytes": peak, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "adapter_parameters": adapter.left.numel() + adapter.right.numel(),
                   "head_parameters": sum(p.numel() for p in head.parameters()),
                   "sample": example, "forward_seconds": forward_seconds, "adapter_step_seconds": back_seconds,
                   "estimated_pilot_seconds": estimate, "estimated_with_reserve_seconds": needed, "steps": probe}
        write_atomic(preflight / "result.json", receipt)
    print({k: receipt[k] for k in ("state", "estimated_with_reserve_seconds", "peak_reserved_bytes")}, flush=True)
    if args.preflight_only or not fit_budget or needed + 30 > deadline - time.time():
        reason = "PREFLIGHT_ONLY" if args.preflight_only else "PILOT_EXCEEDS_MEASURED_BUDGET"
        write_atomic(args.run_dir / "status.json", {"state": reason})
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
            if name not in frozen_cache:
                with torch.no_grad():
                    frozen_cache[name] = image_hidden(model, inputs(name)).cpu()
            return frozen_cache[name].to(device)

        @torch.no_grad()
        def predict(partition):
            out = []
            for name, panel in lists[partition].items():
                check_time()
                prior, ids, _, _ = rows(panel)
                delta = select_languages(current(feature(name)).squeeze(0), ids)
                out.append(panel.assign(candidate=odds_update(prior, delta).cpu().numpy()))
            return pd.concat(out, ignore_index=True)

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
                pred = predict("inner")
                inner = scores(pred, ("prior", "candidate"), track["test_like_excluded"])
                curve = {"epoch": epoch, "train": {k: float(np.mean([v[k] for v in log])) for k in log[0]},
                         "inner": inner, "train_seconds": train_seconds, "total_seconds": time.perf_counter()-begin,
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "adapter_updates": {k: float((v-old[k]).norm()) for k, v in trainables(adapted).items()}
                         if adapted is not None else {}}
                state = {"trainable": trainables(wrapping), "optimizer": opt.state_dict(), "scaler": scale.state_dict(),
                         "curve": curve}
                save_sealed(directory, eid, state, torch)
            history.append(state["curve"])
            print(arm, epoch, history[-1], flush=True)
        winner, guardrail = select_epoch(history, config["inner_log_loss_margin"])
        selected = path / f"epoch_{winner['epoch']}"
        restore(wrapping, load_sealed(selected, arm_id | {"epoch": winner["epoch"]}, torch)["trainable"])
        final = path / "outer"
        if not final.exists():
            pred = predict("outer")[KEYS + ["candidate"]].rename(columns={"candidate": arm})
            with new_output(final) as stage:
                pred.to_csv(stage / "predictions.csv", index=False, float_format="%.17g")
                write_atomic(stage / "seal.json", {"identity": arm_id, "selected_epoch": winner["epoch"],
                    "checkpoint": read_json(selected / "seal.json")["sha256"],
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
        write_atomic(stage / "metrics.json", summary | {"arms": arms, "preflight": receipt})
        write_atomic(stage / "seal.json", {p.name: file_sha256(p) for p in stage.iterdir()})
    write_atomic(args.run_dir / "status.json", {"state": "FOLD_PILOT_COMPLETE"})


def supervise(args):
    budget = read_json(RULE)["cumulative_gpu_seconds"]
    root = args.run_dir
    ledger = root / "gpu_budget.json"
    with FileLock(str(ledger) + ".lock", timeout=0):
        used = 0.
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
                           check=True, timeout=deadline-start)
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
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/exp035_cultural_residual")
    parser.add_argument("--run-bounded-gpu", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        lease = read_json(args.run_dir / "gpu_budget.json")
        if not args.run_bounded_gpu or lease["active_since"] is None:
            raise PermissionError("GPU worker requires an active budget ledger")
        worker(args, lease["deadline"])
    else:
        prepare(args)
        if args.run_bounded_gpu and not (args.run_dir / "results").exists():
            supervise(args)


if __name__ == "__main__":
    main()
