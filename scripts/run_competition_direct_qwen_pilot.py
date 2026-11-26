#!/usr/bin/env python3
"""Run or verify the owner-authorized, single-fold direct-Qwen pilot."""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.special import expit

from run_competition_direct_qwen import DIRECT, ROOT, prepare

from st3.baseline import file_sha256
from st3.competition_challenger import check_hash
from st3.competition_semantic_evaluation import charge, json_digest, write_atomic
from st3.competition_semantic_generation import CONFIG, TRACK1, verified_assets, verified_rgb
from st3.competition_semantic_recovery import load_model
from st3.competition_stack import compare
from st3.competition_track1 import load_protocol
from st3.contrast_study import new_output
from st3.data import KEYS, require_same_pairs
from st3.e50_heads import read_json
from st3.e50_inference import safe_image_path
from st3.metrics import painting_scores, score

CAP_SECONDS = 10800
PROGRESS_EVERY = 128
ARMS = ("frozen", "qlora")
STAGE1_SHA256 = "c84582d199dc2ebb624691ec5a8a58bca931818ff38f64eb06dd57901ec69a34"
VOTES = [f"count_{emotion}" for emotion in ("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness")]


def plan(args):
    config, base, track = read_json(DIRECT), read_json(CONFIG), read_json(TRACK1)
    check_hash(args.feasibility, STAGE1_SHA256)
    feasibility = read_json(args.feasibility)
    core = ROOT / "src/st3/competition_direct_qwen.py"
    check_hash(core, feasibility["source_sha256"])
    check_hash(DIRECT, feasibility["config_sha256"])
    check_hash(CONFIG, feasibility["base_config_sha256"])
    if feasibility["state"] != "PASS" or config["pilot_fold"] != 3 or config["pilot_epochs"] != 2:
        raise ValueError("Only the verified two-epoch fold-3 recipe is authorized")
    _, proof = prepare(args, config, base)
    if proof != feasibility["protocol"]:
        raise ValueError("The preregistered fold and Stage 1 protocol differ")
    check_hash(args.data_dir / "annotations/artelingo28_train_val.csv", track["official_training"]["annotations_sha256"])
    pairs = pd.read_csv(args.inputs / "training/pairs.csv", usecols=[*KEYS, "fold", "k", "n", *VOTES])
    counts = pairs[VOTES].to_numpy()
    if (not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any()
            or not np.array_equal(counts[:, :4].sum(1), pairs.k) or not np.array_equal(counts.sum(1), pairs.n)
            or not pairs.n.ge(5).all()):
        raise ValueError("Pinned vote aggregation does not match the ST3 positive and total counts")
    fit, held = [scope.sort_values(KEYS, ignore_index=True) for scope in
                 (pairs.loc[pairs.fold.ne(3)], pairs.loc[pairs.fold.eq(3)])]
    if set(fit.painting) & set(held.painting) or (len(fit), len(held)) != (10879, 2721):
        raise ValueError("Unexpected fold-3 vote population")
    images = load_protocol(args.inputs, track).set_index("painting")
    keys = pairs.sort_values(KEYS, ignore_index=True)
    groups = [{"painting": p, "image": images.at[p, "image"], "image_sha256": images.at[p, "image_sha256"],
               "languages": block.language.tolist()} for p, block in keys.groupby("painting", sort=True)]
    if len(groups) != 1335:
        raise ValueError("Unexpected P2 artwork coverage")
    identity = {"head_and_adapter": file_sha256(core), "config": file_sha256(DIRECT), "base": file_sha256(CONFIG),
                "runner": file_sha256(Path(__file__)), "stage1": STAGE1_SHA256, "inputs": track["p2_input_sha256"],
                "baseline": base["incumbent_oof_sha256"], "groups": json_digest(groups), "fold": 3,
                "max_gpu_seconds": CAP_SECONDS, "progress_every": PROGRESS_EVERY}
    return config, base, track, proof, fit, held, groups, identity


def verdict_path(root, arm, epoch):
    return root / arm / f"epoch_{epoch}" / "validation"


def tensor_save(path, value, torch):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_sealed(path, identity, value, torch):
    with new_output(path) as stage:
        tensor_save(stage / "state.pt", value, torch)
        check_hash(stage / "state.pt", digest := file_sha256(stage / "state.pt"))
        wire = torch.load(stage / "state.pt", map_location="cpu", weights_only=True)
        if wire.keys() != value.keys():
            raise ValueError("Checkpoint failed CPU readback")
        write_atomic(stage / "seal.json", {"identity": identity, "sha256": digest})


def load_sealed(path, identity, torch):
    seal = read_json(path / "seal.json")
    if seal["identity"] != identity:
        raise ValueError(f"Checkpoint has a different frozen identity: {path}")
    check_hash(path / "state.pt", seal["sha256"])
    return torch.load(path / "state.pt", map_location="cpu", weights_only=True)


def trainables(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore(model, parameters):
    current = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if current.keys() != parameters.keys() or any(p.shape != parameters[name].shape or p.dtype != parameters[name].dtype
                                                 for name, p in current.items()):
        raise ValueError("Saved head/adapter set differs from the pinned model")
    for name, p in current.items():
        p.data.copy_(parameters[name])


def epoch_order(names, seed, epoch):
    return np.random.default_rng(seed + epoch).permutation(sorted(names)).tolist()


def exact_bce(gold, frame):
    require_same_pairs(gold, frame)
    merged = gold.merge(frame, on=KEYS, validate="one_to_one")
    if not np.isfinite(merged.logit).all():
        raise ValueError("Invalid logit in saved held-out predictions")
    expected = expit(merged.logit)
    np.testing.assert_allclose(merged.p_positive, expected, atol=2e-7, rtol=0)
    return float((merged.k * np.logaddexp(0, -merged.logit) + (merged.n - merged.k) * np.logaddexp(0, merged.logit)).sum() / merged.n.sum())


def validation_seal(path, identity, frame, gold):
    nll = exact_bce(gold, frame)
    with new_output(path) as stage:
        frame.to_csv(stage / "predictions.csv", index=False, float_format="%.17g", lineterminator="\n")
        write_atomic(stage / "seal.json", {"identity": identity, "sha256": file_sha256(stage / "predictions.csv"),
                                          "vote_weighted_bce_nats": nll, "held_pairs": len(frame)})


def read_validation(path, identity, gold):
    seal = read_json(path / "seal.json")
    if seal["identity"] != identity or seal["held_pairs"] != len(gold):
        raise ValueError("Validation checkpoint does not match expected stage and coverage")
    csv = path / "predictions.csv"
    check_hash(csv, seal["sha256"])
    frame = pd.read_csv(csv, float_precision="round_trip")
    if abs(exact_bce(gold, frame) - seal["vote_weighted_bce_nats"]) > 1e-12:
        raise ValueError("Validation BCE does not replay")
    return frame


def run_worker(args, deadline):
    import torch
    from st3.competition_direct_qwen import CulturalQwen, encode_text, optimizer_for, train_step

    config, base, _, proof, fit, held, groups, identity = plan(args)
    root = args.run_dir
    check_hash(root / "plan.json", read_json(root / "plan_receipt.json")["sha256"])
    if read_json(root / "plan.json")["identity"] != identity:
        raise ValueError("Worker protocol and source pins changed")
    if not torch.cuda.is_available() or "RTX 4060 Laptop" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This run is authorized on the local RTX 4060 Laptop GPU only")
    assets = verified_assets(args.checkpoint, base)
    torch, pretrained, processor, hw = load_model(args.checkpoint, base | {"attention": config["attention"]})
    write_atomic(root / "hardware" / f"{time.time_ns()}.json", hw | {"assets": assets})
    by_name = {g["painting"]: (i, g) for i, g in enumerate(groups)}
    fit_panels = dict(tuple(fit.groupby("painting", sort=True)))
    held_keys = held[KEYS + ["k", "n"]]
    vote_mass = float(fit.n.sum() / len(fit_panels))
    width = config["pair_microbatch"]

    def check_time():
        if time.time() >= deadline - 5:
            raise TimeoutError("Stopping at the owner-authorized GPU limit with sealed progress retained")

    def chunks(size):
        return [slice(i, i + width) for i in range(0, size, width)]

    def vision_input(model, group):
        pixels = verified_rgb(safe_image_path(args.data_dir, group["image"]), group["image_sha256"])
        return model.encode_vision(processor.image_processor(images=[pixels], return_tensors="pt"))

    def live_features(model, vision, languages, sl):
        subset = languages[sl]
        text = encode_text(processor, subset, [len(vision.merged[0])] * len(subset), config["prompt"], model.device)
        return model.features(vision, [0] * len(subset), text)

    def cache_path(i):
        return root / "frozen_features" / f"{i:04}"

    def cache_identity(group):
        return {"run": identity, "artwork": json_digest(group)}

    def fixed_features(i, group, model):
        value = load_sealed(cache_path(i), cache_identity(group), torch)
        features = value["features"]
        if features.shape != (len(group["languages"]), 2560) or features.dtype != torch.float16 or not features.isfinite().all():
            raise ValueError("Frozen representation cache is malformed")
        return features.to(model.device)

    for arm in ARMS:
        for epoch in range(1, config["pilot_epochs"] + 1):
            target = verdict_path(root, arm, epoch)
            if target.exists():
                arm_id = {"run": identity, "arm": arm, "epoch": epoch}
                ckpt = target.parent / "fit"
                load_sealed(ckpt, arm_id, torch)
                read_validation(target, arm_id | {"checkpoint_sha256": read_json(ckpt / "seal.json")["sha256"]}, held_keys)
        if all(verdict_path(root, arm, epoch).exists() for epoch in range(1, 3)):
            continue
        is_b = arm == "qlora"
        model = CulturalQwen(pretrained.model, config, adapted=is_b)
        optimizer = optimizer_for(model, config)
        scaler = torch.amp.GradScaler("cuda", init_scale=128.)
        frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
        if not is_b:
            model.eval()
            for i, group in enumerate(groups):
                check_time()
                path = cache_path(i)
                if path.exists():
                    fixed_features(i, group, model)
                else:
                    with torch.no_grad():
                        vision = vision_input(model, group)
                        features = torch.cat([live_features(model, vision, group["languages"], sl)
                                              for sl in chunks(len(group["languages"]))]).cpu()
                    if features.dtype != torch.float16 or not features.isfinite().all():
                        raise ValueError("Frozen Qwen extraction returned invalid features")
                    save_sealed(path, cache_identity(group), {"features": features}, torch)
                if (i + 1) % 50 == 0 or i + 1 == len(groups):
                    print(f"Frozen cache {i+1}/{len(groups)}; GPU time remaining {(deadline-time.time())/3600:.2f} h", flush=True)
        for epoch in range(1, config["pilot_epochs"] + 1):
            check_time()
            arm_id = {"run": identity, "arm": arm, "epoch": epoch}
            epoch_path = root / arm / f"epoch_{epoch}"
            checkpoint = epoch_path / "fit"
            order = epoch_order(fit_panels, config["seed"], epoch)
            if checkpoint.exists():
                fitted = load_sealed(checkpoint, arm_id, torch)
                restore(model, fitted["trainable"])
                optimizer.load_state_dict(fitted["optimizer"])
                scaler.load_state_dict(fitted["scaler"])
            else:
                start = trainables(model)
                steps, elapsed = [], 0.
                progress = sorted(epoch_path.glob("progress_[0-9][0-9][0-9][0-9]"))
                if progress:
                    partial = load_sealed(progress[-1], arm_id, torch)
                    restore(model, partial["trainable"])
                    optimizer.load_state_dict(partial["optimizer"])
                    scaler.load_state_dict(partial["scaler"])
                    steps, start, elapsed = partial["steps"], partial["epoch_start"], partial["seconds"]
                    if partial["order"] != order or len(steps) != int(progress[-1].name.rsplit("_", 1)[1]):
                        raise ValueError("Saved optimizer steps differ from the frozen painting order")
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                for i, name in enumerate(order[len(steps):], start=len(steps) + 1):
                    check_time()
                    group_idx, group = by_name[name]
                    panel = fit_panels[name]
                    if group["languages"] != panel.language.tolist():
                        raise ValueError("Fit languages differ from the pinned image keys")
                    k, n = [torch.tensor(panel[v].to_numpy(), device=model.device, dtype=torch.float32) for v in ("k", "n")]
                    if is_b:
                        vision = vision_input(model, group)
                    else:
                        features = fixed_features(group_idx, group, model)

                    def logits(sl):
                        h = live_features(model, vision, group["languages"], sl) if is_b else features[sl]
                        return model.logits(h)

                    batch = ((lambda sl=sl: logits(sl), k[sl], n[sl]) for sl in chunks(len(k)))
                    steps.append(train_step(model, optimizer, scaler, config, batch, vote_mass))
                    if i % PROGRESS_EVERY == 0:
                        state = {"trainable": trainables(model), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                                 "order": order, "steps": steps, "epoch_start": start,
                                 "seconds": elapsed + time.perf_counter() - started}
                        save_sealed(epoch_path / f"progress_{i:04}", arm_id, state, torch)
                        print(f"{arm} epoch {epoch} {i}/{len(order)}; BCE {np.mean([s['bce'] for s in steps]):.5f}; "
                              f"GPU time remaining {(deadline-time.time())/3600:.2f} h", flush=True)
                end = trainables(model)
                if any(p.grad is not None or p._version != frozen_versions[name]
                       for name, p in model.named_parameters() if not p.requires_grad):
                    raise ValueError("Pretrained parameter was modified during fitting")
                drift = {name: float((tensor - start[name]).norm()) for name, tensor in end.items() if name.startswith("backbone.")}
                if is_b and (not drift or not all(value > 0 for value in drift.values())):
                    raise RuntimeError("All targeted LoRA matrices must update on official votes")
                fitted = {"trainable": end, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                          "order": order, "steps": steps, "epoch_start": start, "adapter_update_l2_by_matrix": drift,
                          "vote_normalizer": vote_mass, "seconds": elapsed + time.perf_counter() - started,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                save_sealed(checkpoint, arm_id, fitted, torch)
                print(f"{arm} epoch {epoch} fitting sealed; {fitted['seconds']:.1f}s", flush=True)
            validation = verdict_path(root, arm, epoch)
            validation_id = arm_id | {"checkpoint_sha256": read_json(checkpoint / "seal.json")["sha256"]}
            if validation.exists():
                read_validation(validation, validation_id, held_keys)
                continue
            model.eval()
            predicted = []
            with torch.no_grad():
                for name, panel in held.groupby("painting", sort=True):
                    check_time()
                    group_idx, group = by_name[name]
                    if group["languages"] != panel.language.tolist():
                        raise ValueError("Held-out keys differ from the recorded image languages")
                    if is_b:
                        vision = vision_input(model, group)
                        features = torch.cat([live_features(model, vision, group["languages"], sl)
                                              for sl in chunks(len(group["languages"]))])
                    else:
                        features = fixed_features(group_idx, group, model)
                    scores = torch.cat([model.logits(features[sl]) for sl in chunks(len(features))])
                    predicted.append(panel[KEYS].assign(p_positive=scores.sigmoid().cpu().numpy().astype(float),
                                                        logit=scores.cpu().numpy().astype(float)))
            validation_seal(validation, validation_id, pd.concat(predicted, ignore_index=True), held_keys)
            print(f"{arm} epoch {epoch} held-out predictions sealed", flush=True)


def checkpoint_summary(value):
    steps = value["steps"]
    names = ("preclip_grad_norm", "head_grad_norm", "adapter_grad_norm", "anchor")
    return {"training_bce_nats": float(np.mean([s["bce"] for s in steps])), "seconds": value["seconds"],
            "updates": len(steps), "clipping_fraction": float(np.mean([s["clipped"] for s in steps])),
            "gradients": {n: {"mean": float(np.mean([s[n] for s in steps])), "max": max(s[n] for s in steps),
                              "p95": float(np.quantile([s[n] for s in steps], .95))} for n in names},
            "adapter_update_l2": float(np.linalg.norm(list(value["adapter_update_l2_by_matrix"].values()))),
            "adapter_update_l2_by_matrix": value["adapter_update_l2_by_matrix"],
            "peak_allocated_bytes": value["peak_allocated_bytes"], "peak_reserved_bytes": value["peak_reserved_bytes"]}


def evaluate(args, prepared):
    import torch

    config, _, track, proof, fit, held, _, identity = prepared
    comparator = pd.read_csv(args.reference / "oof.csv", float_precision="round_trip")
    reference = comparator.loc[comparator.fold.eq(3), KEYS + ["corrected"]].rename(columns={"corrected": "p_positive"})
    gold = held[KEYS + ["k", "n"]]
    require_same_pairs(gold, reference)
    curves, predictions = {arm: {} for arm in ARMS}, {"corrected_ridge": reference}
    for arm in ARMS:
        for epoch in range(1, config["pilot_epochs"] + 1):
            arm_id = {"run": identity, "arm": arm, "epoch": epoch}
            target = verdict_path(args.run_dir, arm, epoch)
            checkpoint = target.parent / "fit"
            if not target.exists():
                continue
            fitted = load_sealed(checkpoint, arm_id, torch)
            if fitted["order"] != epoch_order(fit.painting.unique(), config["seed"], epoch) or len(fitted["steps"]) != fit.painting.nunique():
                raise ValueError("Completed fit does not cover every fitting painting exactly once")
            frame = read_validation(target, arm_id | {"checkpoint_sha256": read_json(checkpoint / "seal.json")["sha256"]}, gold)
            curves[arm][epoch] = checkpoint_summary(fitted) | {"held_bce_nats": exact_bce(gold, frame),
                                                                 "predictions_sha256": file_sha256(target / "predictions.csv")}
            predictions[f"{arm}_{epoch}"] = frame[KEYS + ["p_positive"]]
    report = {"complete": all(len(stages) == config["pilot_epochs"] for stages in curves.values()),
              "protocol": proof, "curves": curves, "preregistered_endpoint": config["pilot_epochs"], "identity": identity}
    order = track["decision"]
    for name, excluded in (("full", []), ("test_like", track["test_like_excluded"])):
        local_gold = gold.loc[~gold.language.isin(excluded)]
        scores, per_painting = {}, {}
        for system, raw in predictions.items():
            aligned = raw.loc[~raw.language.isin(excluded)]
            scores[system] = score(local_gold, aligned)
            frame = local_gold.merge(aligned, on=KEYS, validate="one_to_one").assign(p=lambda df: df.k / df.n)
            per_painting[system] = painting_scores(frame).set_index("painting").contrast
        paired = {}
        for epoch in range(1, config["pilot_epochs"] + 1):
            for a, b in ((f"frozen_{epoch}", "corrected_ridge"), (f"qlora_{epoch}", "corrected_ridge"),
                         (f"qlora_{epoch}", f"frozen_{epoch}")):
                if a in scores and b in scores:
                    delta = (per_painting[a] - per_painting[b]).dropna()
                    paired[f"{a}_minus_{b}"] = compare(delta, order) | {"log_loss_delta": scores[a]["log_loss"] - scores[b]["log_loss"]}
        report[name] = {"scores": scores, "paired": paired, "pairs": len(local_gold)}
    write_atomic(args.run_dir / "results.json", report)
    print(f"Fold-3 verification complete={report['complete']}; results in {args.run_dir / 'results.json'}", flush=True)
    return report


def supervise(args, identity):
    root = args.run_dir
    path = root / "gpu_budget.json"
    with FileLock(str(root / "worker.lock"), timeout=0):
        ledger = read_json(path) if path.exists() else {"identity": identity, "limit_seconds": CAP_SECONDS, "charged_seconds": 0.0}
        if ledger["identity"] != identity or ledger["limit_seconds"] != CAP_SECONDS:
            raise ValueError("Existing GPU charge belongs to a different recipe or authorization")
        start = time.time()
        ledger.update(charged_seconds=charge(ledger, start), active_since=None)
        remaining = CAP_SECONDS - ledger["charged_seconds"]
        if remaining <= 5:
            write_atomic(path, ledger)
            raise TimeoutError("Three-hour cumulative GPU authorization exhausted")
        ledger.update(active_since=start, last_exit=None)
        write_atomic(path, ledger)
    process = None
    deadline = start + remaining
    try:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker-deadline", str(deadline)]
        process = subprocess.Popen(command)
        code = process.wait(timeout=max(0, deadline - time.time()))
        if code:
            raise RuntimeError(f"GPU worker exited with code {code}; completed checkpoints retained")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        ledger.update(charged_seconds=charge(ledger, time.time()), active_since=None,
                      last_exit=process.returncode if process is not None else None)
        write_atomic(path, ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/competition_direct_qwen/fold3_pilot")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/qwen3-vl-4b-ebb281e")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/upstream")
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/exp023_budget_delivery/release/restore_check")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/competition_neighborhood_residual/results")
    parser.add_argument("--feasibility", type=Path, default=ROOT / "runs/competition_direct_qwen/stage1_parity/feasibility.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--authorize-fold3-3h", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--worker-deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.run_dir
    if args.worker_deadline is not None:
        ledger = read_json(root / "gpu_budget.json")
        if (not args.authorize_fold3_3h or ledger["active_since"] is None or
                args.worker_deadline != ledger["active_since"] + (CAP_SECONDS - ledger["charged_seconds"])):
            raise PermissionError("The GPU worker requires the active owner-authorized budget")
        stop = threading.Event()

        def watchdog():
            if not stop.wait(max(0, args.worker_deadline - time.time())):
                os._exit(124)

        threading.Thread(target=watchdog, daemon=True).start()
        try:
            with FileLock(str(root / "worker.lock"), timeout=0):
                run_worker(args, args.worker_deadline)
        finally:
            stop.set()
        return
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "run.lock"), timeout=0):
        prepared = plan(args)
        *_, identity = prepared
        spec = {"identity": identity, "protocol": prepared[3], "held_selection": "fixed_epoch_2",
                "systems": [*ARMS, "corrected_ridge"], "held_fold": 3}
        path = root / "plan.json"
        if path.exists() and read_json(path) != spec:
            raise ValueError("Cannot resume a different frozen pilot")
        write_atomic(path, spec)
        write_atomic(root / "plan_receipt.json", {"sha256": file_sha256(path)})
        if args.verify_only:
            evaluate(args, prepared)
        elif args.authorize_fold3_3h:
            write_atomic(root / "status.json", {"state": "RUNNING"})
            try:
                if not all(verdict_path(root, arm, ep).exists() for arm in ARMS for ep in (1, 2)):
                    supervise(args, identity)
                report = evaluate(args, prepared)
                write_atomic(root / "status.json", {"state": "COMPLETE" if report["complete"] else "INCOMPLETE"})
            except BaseException as exc:
                write_atomic(root / "status.json", {"state": "STOPPED", "error": f"{type(exc).__name__}: {exc}"})
                raise
        else:
            print("Verified the frozen fold-3 cohort, real vote counts, Stage 1 receipt and three-hour budget; CUDA was not started.")


if __name__ == "__main__":
    main()
