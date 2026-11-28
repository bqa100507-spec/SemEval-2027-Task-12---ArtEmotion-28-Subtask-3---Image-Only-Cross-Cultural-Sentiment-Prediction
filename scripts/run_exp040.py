#!/usr/bin/env python3
"""EXP-040 worker: E3-LoRA (Qwen3-VL-Embedding-2B adapted from a ridge-initialized interaction head).

Full mode: Setup -> Validate (collect-all) -> Benchmark -> P2 LoRA folds (sealed per fold, resumable) ->
P2 evaluation -> Full refit and Development -> Submission validation -> Export -> Release.
SMOKE_ONLY: Setup -> Validate (replays may be skipped) -> one fold per GPU x N optimizer steps + inner passes, all
GPUs at once -> exp040_smoke.json with per-GPU throughput, VRAM, placement, finiteness, determinism and the parallel
full-run forecast; no submission.

GPU work runs as child processes, one per GPU, each seeing only its GPU (CUDA_VISIBLE_DEVICES) with the whole model
on it. Outer folds are independent children: each GPU takes the next unsealed fold when it is free. The full refit
runs after every fold, on one GPU.

The EXP-038/039 workers are the library (context, population, sealed caches, R0/C2 replays, bundles, export).
C2 is recomputed on-platform from the sealed EXP-038 features and is the gate reference. If E3L fails
technically or the forecast exceeds the session, the run still exports a NO_GO report with the reason and the
byte-identical incumbent as the best ZIP.
"""

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd

import run_exp038 as r38
import run_exp039 as r39
from st3 import data, submission
from st3 import exp038 as science
from st3 import exp040 as plan
from st3.competition_ridge import fit_head as track1_fit_head
from st3.competition_ridge import make_panel
from st3.exp036_io import read, restore, sha, write
from st3.exp038_io import (BudgetStop, Clock, check_config_pins, completed, copy_verified, exportable, publish,
                           restore_overlay, seal)

STAGES = ("Setup", "Validate", "Benchmark", "P2 LoRA folds", "P2 evaluation", "Full refit and Development",
          "Submission validation", "Export", "Release")
SMOKE_STAGES = ("Setup", "Validate", "Smoke")
MANIFEST = "exp040_manifest.json"
KEY_ARMS = ("E3",)


def configs():
    return (read(ROOT / "configs/exp040.json"), read(ROOT / "configs/competition_track1.json"),
            read(ROOT / "configs/competition_neighborhood_residual.json"))


class Context(r39.Context):
    name = "exp040"
    manifest_name = MANIFEST
    config_file = "configs/exp040.json"
    script = Path(__file__)

    def __init__(self, **kwargs):
        default = configs()
        kwargs.setdefault("config", default[0])
        kwargs.setdefault("track1", default[1])
        kwargs.setdefault("neighborhood", default[2])
        super().__init__(**kwargs)
        self.e3l = {"available": True, "reason": None}
        self.vision_cache = {"enabled": False, "reason": "not decided yet"}


class ValidationFailed(r39.ValidationFailed):
    pass


# Population helpers.

def base_features(ctx):
    return r39.features_for(ctx, plan.BASE_ARMS)


def image_paths(ctx):
    pop = r38.population(ctx)
    path = dict(r38.image_items(ctx))
    return {p: path[d] for p, d in zip(pop["paintings"].painting, pop["paintings"].image_sha256)}


def languages(ctx):
    pop = r38.population(ctx)
    return sorted(set(pop["train"].language) | set(pop["dev"].language))


def ridge(ctx, train, e3):
    """The frozen E3 Track 1 head on ``train`` (the init of the E3L head)."""
    return track1_fit_head(make_panel(train), np.ascontiguousarray(
        e3.loc[sorted(train.painting.unique())].to_numpy(dtype=np.float64)), ctx.track1["ridge"])


def sealed_stage(folder, identity, required):
    """True for a sealed stage; False while absent or still in progress (a GPU child's resumable folder)."""
    if not (folder / "seal.json").is_file():
        return False
    return completed(folder, identity, required)


def save_head(head, path):
    np.savez(path, intercept=np.asarray(head.intercept), language_intercepts=head.language_intercepts,
             shared=head.shared, deviations=head.deviations, languages=np.asarray(head.languages, dtype=str))
    return sha(path)


def fold_split(ctx, fold):
    gold = r38.population(ctx)["gold"]
    outer = gold[gold.fold.ne(fold)]
    held = gold[gold.fold.eq(fold)]
    fit, inner = plan.inner_split(outer.painting, ctx.config["selection"]["inner_fraction"],
                                  ctx.config["selection"]["inner_seed"] + int(fold))
    if set(inner) & set(fit) or set(held.painting) & set(outer.painting):
        raise ValueError("Fold painting sets overlap")
    return outer, held, fit, inner


def request(ctx, name, mode, *, pairs, train_paintings, inner, held, embed, head, epochs_to_run, deadline,
            identity, seed, extra=None):
    folder = ctx.root / "tmp_requests" / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    pairs[data.KEYS + ["k", "n"]].to_csv(folder / "pairs.csv", index=False, lineterminator="\n")
    paths = image_paths(ctx)
    needed = sorted(set(pairs.painting))
    (folder / "images.json").write_text(json.dumps({p: paths[p] for p in needed}), encoding="utf-8")
    save_head(head, folder / "head.npz")
    body = {"mode": mode, "spec": ctx.config["arms"]["E3"], "lora": ctx.config["lora"], "train": ctx.config["train"],
            "languages": languages(ctx), "head": str(folder / "head.npz"), "pairs": str(folder / "pairs.csv"),
            "images": str(folder / "images.json"), "train_paintings": list(train_paintings), "inner": list(inner),
            "held": list(held), "embed": list(embed), "epochs_to_run": int(epochs_to_run), "deadline": deadline,
            "identity": identity, "seed": int(seed), "excluded": ctx.config["test_like_excluded"],
            "snapshot": str(ctx.models.get("E3")), "folder": None,
            "vision_cache": {"enabled": bool(ctx.vision_cache.get("enabled")),
                             "max_bytes": ctx.config["vision_cache"]["max_bytes_per_child"]}} | (extra or {})
    return folder, body


def dispatch(ctx, backend, body, folder, deadline, gpu=0):
    path = ctx.root / "tmp_requests" / f"{Path(body['folder']).name}_request.json"
    body = body | {"deadline": deadline}
    path.write_text(json.dumps(body), encoding="utf-8")
    backend.e3l(path, deadline, gpu=gpu)


# GPU children in parallel: one child per GPU at a time.

def run_jobs(ctx, backend, jobs, gpus, on_done=None):
    """Run ``jobs`` ({label, body, request, gpu: pinned GPU or None}) with one worker thread per GPU.

    Each worker takes, in order, the first pending job pinned to its GPU or unpinned. ``on_done(job, gpu)`` runs
    in the calling thread (sealing, status and progress stay single-threaded). A technical failure stops new work
    and kills the other children (their checkpoints stay resumable) and is returned as (job, error). A session
    timeout stops new work and is re-raised once every child has stopped.
    """
    pending, lock, events, stop = list(jobs), threading.Lock(), queue.Queue(), threading.Event()

    def take(gpu):
        with lock:
            for i, job in enumerate(pending):
                if job.get("gpu") in (None, gpu):
                    return pending.pop(i)
        return None

    def worker(gpu):
        while not stop.is_set():
            job = take(gpu)
            if job is None:
                return
            events.put(("start", job, gpu, None))
            try:
                dispatch(ctx, backend, job["body"], job["request"], ctx.clock.work_deadline(), gpu=gpu)
            except BaseException as error:  # Reported to the calling thread, which decides.
                events.put(("error", job, gpu, error))
                return
            events.put(("done", job, gpu, None))

    threads = [threading.Thread(target=worker, args=(gpu,), name=f"exp040-gpu{gpu}", daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    failure = timeout = None
    try:
        while any(t.is_alive() for t in threads) or not events.empty():
            try:
                kind, job, gpu, error = events.get(timeout=.5)
            except queue.Empty:
                continue
            if kind == "start":
                ctx.log(f"{job['label']}: started on GPU {gpu}")
            elif kind == "done":
                ctx.log(f"{job['label']}: finished on GPU {gpu}")
                if on_done is not None:
                    on_done(job, gpu)
            elif isinstance(error, TimeoutError):
                timeout = timeout or error
                stop.set()
            elif failure is None:
                failure = (job, error)
                ctx.log(f"{job['label']}: failed on GPU {gpu}: {type(error).__name__}: {error}")
                stop.set()
                backend.cancel()
            else:
                ctx.log(f"{job['label']}: stopped after {failure[0]['label']} failed ({type(error).__name__})")
    except BaseException:
        stop.set()
        backend.cancel()
        raise
    finally:
        for thread in threads:
            thread.join()
    if failure is None and timeout is not None:
        raise timeout
    return failure


def progress_sealed(ctx):
    """Progress export while children run: in-flight fold folders (no seal yet) are left out of the snapshot."""
    def stable(path):
        rel = path.relative_to(ctx.root).parts
        return not (len(rel) > 2 and rel[0] == "p2_folds" and not (ctx.root / rel[0] / rel[1] / "seal.json").is_file())

    try:
        publish(ctx.root.parent / f"{ctx.name}_progress.zip", ctx.root,
                [p for p in exportable(ctx.root) if stable(p)], ctx.overlay_sha)
    except Exception as error:  # A failed periodic transfer never stops the run; it is recorded.
        ctx.log(f"Progress export failed: {error}")


# Validate.

def key_audit(ctx):
    from st3 import key_audit as ka
    from st3.exp038_gpu import git_blob_sha1

    out = {}
    for arm in KEY_ARMS:
        arm_spec = ctx.config["arms"][arm]
        config_path = ctx.overlay / f"keys/{arm}_config.json"
        if git_blob_sha1(config_path) != arm_spec["metadata_git_blobs"]["config.json"]:
            raise ValueError(f"{arm} audit config.json differs from its pinned git blob")
        config = read(config_path)
        keys = read(ctx.overlay / f"keys/{arm}_checkpoint_keys.json")
        if arm in ctx.models and ka.snapshot_keys(ctx.models[arm]) != sorted(keys):
            raise ValueError(f"{arm} downloaded checkpoint names differ from the audited list")
        class_keys, tied_keys = ka.qwen_class_keys(config)
        out[arm] = ka.require(ka.audit(keys, class_keys, config, tied_keys), arm) | {
            "model": arm_spec["model"], "revision": arm_spec["revision"], "snapshot_cross_checked": arm in ctx.models}
    return out


def head_cpu(ctx):
    """CPU half of the step-0 invariant: the linear head built from the ridge reproduces its probabilities."""
    import torch

    fold = ctx.config["smoke"]["fold"]
    outer, held, fit, _ = fold_split(ctx, fold)
    e3 = base_features(ctx)["E3"]
    head, _ = ridge(ctx, outer[outer.painting.isin(fit)], e3)
    module = plan.interaction_head(head, languages(ctx), ctx.config["train"]["lambda"])
    keys = held[data.KEYS].sort_values(data.KEYS, ignore_index=True)
    expected = plan.ridge_probabilities(head, e3, keys)
    x = torch.as_tensor(e3.loc[keys.painting].to_numpy(dtype=np.float32))
    index = torch.as_tensor(keys.language.map({l: i for i, l in enumerate(languages(ctx))}).to_numpy())
    with torch.no_grad():
        got = torch.sigmoid(module(x).double()[torch.arange(len(keys)), index]).numpy()
    difference = float(np.abs(got - expected).max())
    tolerance = ctx.config["head"]["step0_tolerance"]
    if difference > tolerance:
        raise ValueError(f"Ridge-initialized head differs from the ridge by {difference:.3g} > {tolerance}")
    return {"fold": fold, "pairs": len(keys), "max_abs_probability_difference": difference, "tolerance": tolerance,
            "anchor_at_init": float(module.anchor())}


def step0(ctx, backend):
    """GPU step-0 invariant on one fold's held-out paintings, through the exact training forward path."""
    fold = ctx.config["smoke"]["fold"]
    outer, held, fit, _ = fold_split(ctx, fold)
    e3 = base_features(ctx)["E3"]
    head, _ = ridge(ctx, outer[outer.painting.isin(fit)], e3)
    folder, body = request(ctx, f"step0_fold{fold}", "step0", pairs=held, train_paintings=[], inner=[],
                           held=sorted(held.painting.unique()), embed=[], head=head, epochs_to_run=0,
                           deadline=ctx.clock.work_deadline(), identity=ctx.identity, seed=ctx.config["train"]["seed"])
    cache = ctx.config["vision_cache"]
    body["vision_cache"] = {"enabled": False, "check": True, "tolerance": cache["tolerance"],
                            "max_bytes": cache["max_bytes_per_child"]}
    out = ctx.root / "validate_step0"
    shutil.rmtree(out, ignore_errors=True)
    body["folder"] = str(out)
    dispatch(ctx, backend, body, folder, ctx.clock.work_deadline(), gpu=ctx.config["parallel"]["step0_gpu"])
    receipt = read(out / "step0.json")
    embeddings = pd.read_csv(out / "step0_embeddings.csv", index_col=0)
    model = pd.read_csv(out / "step0_probabilities.csv")
    keys = model[data.KEYS]
    same_path = plan.ridge_probabilities(head, embeddings, keys)
    sealed = plan.ridge_probabilities(head, e3, keys)
    difference = float(np.abs(model.p_positive.to_numpy() - same_path).max())
    tolerance = ctx.config["head"]["step0_tolerance"]
    a, b = embeddings.to_numpy(dtype=np.float64), e3.loc[embeddings.index].to_numpy(dtype=np.float64)
    result = receipt | {
        "fold": fold, "pairs": len(keys), "max_abs_probability_difference": difference, "tolerance": tolerance,
        "definition": "E3L at step 0 (LoRA right factor zero, head = ridge) vs the frozen ridge in float64 on the "
                      "embeddings of the same forward",
        "diagnostic_vs_sealed_E3": {
            "max_abs_probability_difference": float(np.abs(model.p_positive.to_numpy() - sealed).max()),
            "max_abs_embedding_difference": float(np.abs(a - b).max()),
            "min_cosine": float((a * b).sum(1).min() / 1.0),
            "note": "sealed E3 used batch-8 right-padded forwards without autocast; E3L uses batch 1 under FP16 "
                    "autocast, so this is reported, not gated"}}
    if not receipt["lora_right_zero"] or difference > tolerance:
        raise ValueError(f"Step-0 equality failed: max |dp| {difference:.3g}, lora_right_zero "
                         f"{receipt['lora_right_zero']}")
    if result["diagnostic_vs_sealed_E3"]["min_cosine"] < .99:
        raise ValueError("E3L step-0 embeddings disagree with sealed E3 (cosine < 0.99): formatting differs")
    return result


def available_memory():
    """MemAvailable in bytes on Linux; None where /proc/meminfo is absent."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def vision_cache_decision(ctx, step0_result, gpus):
    """Enable the vision cache only if step 0 matched the uncached path and the projected cache fits."""
    cfg = ctx.config["vision_cache"]
    check = (step0_result or {}).get("vision_cache")
    if not check:
        return {"enabled": False, "reason": "no step-0 cache check (step 0 did not run)"}
    summary = {k: check[k] for k in ("max_abs_probability_difference", "tolerance", "passed")} | {
        "bitwise": all(p["bitwise"] for p in check["passes"].values()), "cache": check["cache"]}
    if not check["passed"]:
        return {"enabled": False, "check": summary, "reason": (
            f"cached step-0 predictions differ from the uncached path by {check['max_abs_probability_difference']:.3g}"
            f" > {check['tolerance']}; the cache is skipped")}
    n = counts(ctx)
    paintings = max([c["embed"] for c in n["folds"].values()] + [n["train"] + n["dev"]])
    per_entry = check["cache"].get("max_entry_bytes") or check["cache"]["bytes_per_entry"]
    projected = int(per_entry * paintings)
    memory = available_memory()
    summary |= {"projected_bytes_per_child": projected, "paintings_per_child": paintings, "gpus": len(gpus),
                "available_memory_bytes": memory}
    if projected > cfg["max_bytes_per_child"]:
        return {"enabled": False, "check": summary, "reason": "projected cache exceeds max_bytes_per_child"}
    if memory is not None and len(gpus) * projected > cfg["memory_fraction"] * memory:
        return {"enabled": False, "check": summary, "reason": "projected caches exceed the memory fraction"}
    return {"enabled": True, "check": summary, "reason": "step-0 cached path matched within tolerance"}


def stage_validate(ctx, backend, status, smoke=False):
    folder = ctx.root / "validate"
    if completed(folder, ctx.identity, ["validate.json"]):
        backend.fetch("E3")
        ctx.log("Validate: sealed receipts verified; reusing")
        return read(folder / "validate.json")
    state = {}

    def inputs():
        pop = r38.population(ctx)
        return {"train_pairs": len(pop["train"]), "dev_pairs": len(pop["dev"]), "p2_pairs": len(pop["gold"]),
                "overlay_files": len(pop["overlay_manifest"]["files"])}

    def sealed():
        return {arm: {"rows": len(r39.sealed_feature(ctx, arm)[0])} for arm in plan.BASE_ARMS} | {
            "components": [r39.carried(ctx, n).name for n in ("p2_oof.csv", "development_predictions.csv",
                                                              "CSCVA_3_submission.zip", "incumbent_submission.zip")]}

    def r0():
        return {"p2": r38.r0_p2(ctx), "development": r38.r0_development(ctx), "platform": r38.platform_receipt(ctx)}

    def c2():
        tick = time.time()
        result = r39._c2_only(ctx)
        state["c2_seconds"] = time.time() - tick
        return result

    def skipped():
        return {"status": "SKIPPED_SMOKE"}

    receipts, failures = r39.collect([
        ("pin_formats", lambda: check_config_pins(ctx.config), ()),
        ("inputs", inputs, ()),
        ("snapshot_2B", lambda: backend.fetch("E3"), ()),
        ("sealed_exp038", sealed, ("inputs",)),
        ("key_audit", lambda: key_audit(ctx), ("inputs",)),
        ("head_cpu", lambda: head_cpu(ctx), ("inputs", "sealed_exp038")),
        ("step0_gpu", lambda: step0(ctx, backend), ("inputs", "sealed_exp038", "snapshot_2B", "head_cpu")),
        ("r0_replay", skipped if smoke else r0, ("inputs",)),
        ("c2_replay", skipped if smoke else c2, ("inputs", "sealed_exp038")),
    ])
    if failures:
        write(ctx.root / "validate_failures.json", receipts)
        raise ValidationFailed(failures)
    receipts["c2_seconds"] = state.get("c2_seconds")
    receipts["gpus"] = backend.gpus()
    receipts["vision_cache"] = vision_cache_decision(ctx, receipts["step0_gpu"]["result"], receipts["gpus"])
    if smoke:
        write(ctx.root / "validate_smoke.json", receipts)
        return receipts
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    write(folder / "validate.json", receipts)
    seal(folder, ctx.identity)
    return receipts


# Benchmark, smoke and forecast.

def smoke_request(ctx, name, steps, inner_limit, determinism, fold=None):
    fold = ctx.config["smoke"]["fold"] if fold is None else fold
    outer, held, fit, inner = fold_split(ctx, fold)
    e3 = base_features(ctx)["E3"]
    head, _ = ridge(ctx, outer[outer.painting.isin(fit)], e3)
    inner = inner[:inner_limit] if inner_limit else inner
    folder, body = request(ctx, name, "smoke", pairs=outer[outer.painting.isin(set(fit) | set(inner))],
                           train_paintings=fit, inner=inner, held=[], embed=[], head=head, epochs_to_run=1,
                           deadline=ctx.clock.work_deadline(), identity=ctx.identity | {"smoke": name, "fold": fold},
                           seed=ctx.config["train"]["seed"], extra={"smoke_steps": steps, "determinism": determinism})
    out = ctx.root / name
    shutil.rmtree(out, ignore_errors=True)
    body["folder"] = str(out)
    return folder, body, out


def counts(ctx):
    pop = r38.population(ctx)
    gold = pop["gold"]
    folds = {}
    for fold in sorted(gold.fold.unique()):
        outer, held, fit, inner = fold_split(ctx, fold)
        folds[int(fold)] = {"fit": len(fit), "inner": len(inner), "held": held.painting.nunique(),
                            "embed": outer.painting.nunique() + held.painting.nunique()}
    return {"folds": folds, "train": int(pop["train"].painting.nunique()), "dev": int(pop["dev"].painting.nunique())}


def rates(ctx, measured):
    """Conservative per-piece rates over the GPUs measured concurrently: the slowest step, the slowest forward.

    First visits pay the vision forward; revisits (cache hits) do not. Without the cache both are the first-visit
    rates.
    """
    cached = bool(ctx.vision_cache.get("enabled"))
    rows = list(measured.values())

    def worst(key, fallback=None, rate=False):
        values = [m.get(key) or (m.get(fallback) if fallback else None) for m in rows]
        values = [v for v in values if v]
        return (1 / min(values)) if rate else max(values)

    step = worst("warm_seconds_per_optimizer_step", "seconds_per_optimizer_step")
    forward = worst("inner_paintings_per_second", rate=True)
    step_hit = worst("revisit_seconds_per_optimizer_step", "warm_seconds_per_optimizer_step") if cached else step
    forward_hit = worst("revisit_inner_paintings_per_second", "inner_paintings_per_second", rate=True) if cached \
        else forward
    return {"step": step, "step_revisit": min(step_hit, step) if cached else step, "forward": forward,
            "forward_revisit": min(forward_hit, forward) if cached else forward,
            "load": max(m.get("load_seconds", 60) for m in rows) + 30, "vision_cache": cached}


def queue_plan(seconds, gpus):
    """The execution order: folds in order, each to the GPU that frees first (ties to the lower index)."""
    free, plan_ = {g: 0. for g in gpus}, {g: [] for g in gpus}
    for fold in sorted(seconds):
        gpu = min(gpus, key=lambda g: (free[g], g))
        free[gpu] += seconds[fold]
        plan_[gpu].append(fold)
    return {g: {"folds": plan_[g], "seconds": free[g]} for g in gpus}


def forecast(ctx, measured, c2_seconds, gpus):
    """Seconds per piece from measured warm throughput; the full refit assumes the worst case (4 epochs).

    ``measured`` maps GPU -> smoke/benchmark receipt. The fold phase takes the longest GPU queue.
    """
    train_cfg = ctx.config["train"]
    epochs, batch = train_cfg["epochs"], train_cfg["batch_paintings"]
    r = rates(ctx, measured)
    n = counts(ctx)

    def training(paintings):
        return plan.steps_per_epoch(paintings, batch) * (r["step"] + (epochs - 1) * r["step_revisit"])

    folds = {}
    for fold, c in n["folds"].items():
        folds[fold] = (r["load"] + training(c["fit"]) + c["inner"] * (r["forward"] + epochs * r["forward_revisit"])
                       + (c["fit"] + c["inner"]) * r["forward_revisit"] + c["held"] * (r["forward"] + r["forward_revisit"]))
    pieces = {f"fold_{fold}": seconds for fold, seconds in folds.items()}
    queues = queue_plan(folds, gpus)
    makespan = max(q["seconds"] for q in queues.values())
    pieces["full_refit"] = (r["load"] + training(n["train"]) + n["train"] * r["forward_revisit"]
                            + n["dev"] * (r["forward"] + r["forward_revisit"]))
    cpu = c2_seconds or 900
    pieces["cpu_p2_and_development"] = 2.0 * cpu + 600
    pieces["export"] = 600
    after = pieces["full_refit"] + pieces["cpu_p2_and_development"] + pieces["export"]
    return {"pieces": pieces, "queues": queues, "fold_phase_seconds": makespan, "total": makespan + after,
            "serial_total": sum(pieces.values()), "gpus": list(gpus), "inputs": {
                "rates": r, "c2_cpu_seconds": cpu, "counts": n, "method": (
                    "fold = load + ceil(fit/16) x (first-visit step + (epochs-1) x revisit step) + inner x "
                    "(first-visit forward + epochs x revisit forward) + (fit+inner) x revisit forward + held x "
                    "(first-visit + revisit forward); folds run as GPU queues (in order, to the first free GPU) and "
                    "the fold phase is the longest queue; the full refit (after the folds, one GPU) assumes the "
                    "worst-case 4 epochs plus train and Development embeddings; CPU = 2 x the measured on-platform "
                    "C2 replay + 600 s; export 600 s; rates are the slowest GPU measured while all run at once; "
                    "revisit rates equal first-visit rates unless the vision cache is on")}}


def concurrent_smoke(ctx, backend, gpus, prefix, steps, inner_limit, determinism):
    """One smoke child per GPU at once (fold ``smoke_folds[i]`` on GPU i); returns GPU -> receipt."""
    folds = ctx.config["parallel"]["smoke_folds"]
    jobs, outs = [], {}
    for i, gpu in enumerate(gpus):
        req, body, out = smoke_request(ctx, f"{prefix}_gpu{gpu}", steps, inner_limit, determinism, fold=folds[i])
        jobs.append({"label": f"{prefix} GPU {gpu} (fold {folds[i]})", "body": body, "request": req, "gpu": gpu})
        outs[gpu] = (out, folds[i])
    failure = run_jobs(ctx, backend, jobs, gpus)
    if failure is not None:
        raise failure[1]
    return {gpu: read(out / "smoke.json") | {"fold": fold} for gpu, (out, fold) in outs.items()}


def budget_plan(ctx, measured, validate, gpus, available):
    estimate = forecast(ctx, measured, validate.get("c2_seconds"), gpus)
    margin = ctx.config["budget"]["forecast_margin"]
    return estimate | {"margin": margin, "planned_seconds": margin * estimate["total"],
                       "serial_planned_seconds": margin * estimate["serial_total"], "available_seconds": available,
                       "fits": margin * estimate["total"] <= available}


def stage_benchmark(ctx, backend, status, validate):
    folder = ctx.root / "benchmark"
    gpus = validate["gpus"]
    if not completed(folder, ctx.identity, ["benchmark.json"]):
        cfg = ctx.config["smoke"]
        measured = concurrent_smoke(ctx, backend, gpus, "tmp_benchmark", cfg["benchmark_steps"],
                                    cfg["benchmark_inner_paintings"], False)
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        write(folder / "benchmark.json", {"per_gpu": {str(g): m for g, m in measured.items()}})
        seal(folder, ctx.identity)
    measured = {int(g): m for g, m in read(folder / "benchmark.json")["per_gpu"].items()}
    budget = budget_plan(ctx, measured, validate, sorted(measured), ctx.clock.remaining())
    budget["remaining_seconds"] = budget.pop("available_seconds")
    write(ctx.root / "budget_plan.json", budget)
    return budget


def smoke_summary(receipt):
    keys = ("fold", "gpu", "devices", "peak_memory", "optimizer_steps", "warm_seconds_per_optimizer_step",
            "train_paintings_per_second", "revisit_seconds_per_optimizer_step", "inner_paintings_per_second",
            "revisit_inner_paintings_per_second", "load_seconds", "finite_losses", "eval_forward_bitwise_repeatable",
            "two_pass_training", "vision_cache", "placement")
    return {k: receipt.get(k) for k in keys}


def stage_smoke(ctx, backend, status, validate):
    cfg = ctx.config["smoke"]
    gpus = validate["gpus"]
    measured = concurrent_smoke(ctx, backend, gpus, "tmp_smoke", cfg["optimizer_steps"], None, True)
    session = ctx.config["budget"]["session_hours"] * 3600 - ctx.config["budget"]["persistence_reserve_seconds"]
    budget = budget_plan(ctx, measured, validate, gpus, session)
    budget["session_seconds"] = budget.pop("available_seconds")
    budget["fits_session"] = budget.pop("fits")
    result = {"mode": "SMOKE_ONLY", "submission_produced": False, "gpus": gpus, "concurrent": len(gpus) > 1,
              "validate": {k: (v.get("status") if isinstance(v, dict) and "status" in v else v)
                           for k, v in validate.items() if k not in {"vision_cache", "gpus"}},
              "vision_cache": validate.get("vision_cache"),
              "step0": (validate.get("step0_gpu") or {}).get("result"),
              "per_gpu": {str(g): smoke_summary(m) for g, m in measured.items()},
              "smoke": {str(g): m for g, m in measured.items()}, "forecast": budget}
    write(ctx.output / "exp040_smoke.json", result)
    return result


# P2 LoRA folds.

def stage_folds(ctx, backend, status, budget):
    if not budget["fits"]:
        ctx.e3l = {"available": False, "reason": (
            f"FORECAST_EXCEEDS_ALLOWANCE: planned {budget['planned_seconds']:.0f}s > remaining "
            f"{budget['remaining_seconds']:.0f}s; stopped before training (protocol not shrunk)")}
        status.piece("E3L", "NOT_RUN", ctx.e3l["reason"])
        return {}
    pop = r38.population(ctx)
    e3 = base_features(ctx)["E3"]
    results, jobs, receipts = {}, [], {}
    for fold in sorted(pop["gold"].fold.unique()):
        fold = int(fold)
        folder = ctx.root / "p2_folds" / f"fold_{fold}"
        if sealed_stage(folder, ctx.identity, ["result.json"]):
            results[fold] = read(folder / "result.json")
            continue
        outer, held, fit, inner = fold_split(ctx, fold)
        head, receipts[fold] = ridge(ctx, outer[outer.painting.isin(fit)], e3)
        req, body = request(ctx, f"fold_{fold}", "fold", pairs=pd.concat([outer, held]), train_paintings=fit,
                            inner=inner, held=sorted(held.painting.unique()),
                            embed=sorted(set(outer.painting) | set(held.painting)), head=head,
                            epochs_to_run=ctx.config["train"]["epochs"], deadline=ctx.clock.work_deadline(),
                            identity=ctx.identity | {"fold": fold, "vision_cache": bool(ctx.vision_cache["enabled"])},
                            seed=ctx.config["train"]["seed"] + fold)
        body["folder"] = str(folder)
        jobs.append({"label": f"fold {fold}", "fold": fold, "body": body, "request": req, "gpu": None})
    status.item(f"folds {[j['fold'] for j in jobs]} on GPUs {budget['gpus']}")

    def done(job, gpu):
        fold = job["fold"]
        folder = ctx.root / "p2_folds" / f"fold_{fold}"
        result = read(folder / "result.json") | {"init_ridge": receipts[fold], "gpu_index": gpu}
        write(folder / "result.json", result)
        seal(folder, ctx.identity)
        results[fold] = result
        status.piece(f"E3L_fold_{fold}", "COMPLETE", f"selected epoch {result['selected_epoch']} on GPU {gpu}")
        progress_sealed(ctx)

    failure = run_jobs(ctx, backend, jobs, budget["gpus"], done)
    if failure is not None:  # E3L is the experiment: a technical failure becomes a reported NO_GO.
        job, error = failure
        ctx.e3l = {"available": False, "reason": f"TECHNICAL_FAILURE fold {job['fold']}: {type(error).__name__}: {error}"}
        status.piece("E3L", "FAILED", ctx.e3l["reason"])
        return results
    status.piece("E3L", "COMPLETE")
    return results


def fold_inputs(ctx, fold, result, features):
    """(features with E3L, E3L held-out probabilities or None for epoch 0 = frozen E3)."""
    if result["selected_epoch"] == 0:
        return plan.with_e3l(features, features["E3"]), None
    folder = ctx.root / "p2_folds" / f"fold_{fold}"
    with np.load(folder / "embeddings.npz", allow_pickle=False) as saved:
        frame = pd.DataFrame(saved["values"].astype(np.float64), index=saved["paintings"].tolist())
    return plan.with_e3l(features, frame), pd.read_csv(folder / "held_probabilities.csv")


# P2 evaluation.

def evaluate(ctx, folds):
    from threadpoolctl import threadpool_limits

    pop = r38.population(ctx)
    gold, features = pop["gold"], base_features(ctx)
    frames, parts, r0s, receipts = [], [], [], {}
    with threadpool_limits(limits=ctx.config["runtime"]["ridge_threads"]):
        for fold, held in gold.groupby("fold"):
            tick = time.time()
            fold = int(fold)
            train, keys = gold.loc[gold.fold.ne(fold)], held[data.KEYS]
            fx, probabilities = fold_inputs(ctx, fold, folds[fold], features)
            frame, outer, r0, receipt, _ = plan.fit_and_predict(train, keys, fx, probabilities, ctx.track1,
                                                                 ctx.neighborhood, fold)
            frames.append(frame)
            parts.append(outer)
            r0s.append(r0)
            receipts[fold] = receipt | {"selected_epoch": folds[fold]["selected_epoch"], "seconds": time.time() - tick}
            ctx.log(f"P2 fold {fold}: E3L epoch {folds[fold]['selected_epoch']}, scored {len(keys)} pairs")
    frame = pd.concat(frames, ignore_index=True).sort_values(data.KEYS, ignore_index=True)
    frame = frame.merge(gold[data.KEYS + ["fold"]], on=data.KEYS, validate="one_to_one")
    r0 = pd.concat(r0s, ignore_index=True).sort_values(data.KEYS, ignore_index=True)
    r0 = r0.merge(gold[data.KEYS + ["fold"]], on=data.KEYS, validate="one_to_one")
    components = pd.concat(parts, ignore_index=True).sort_values(data.KEYS, ignore_index=True)
    return frame, components, r0, receipts


def stage_p2(ctx, status, folds):
    folder = ctx.root / "p2"
    if completed(folder, ctx.identity, ["oof.csv", "metrics.json", "gates.json"]):
        return read(folder / "gates.json")
    pop = r38.population(ctx)
    tick = time.time()
    available = ctx.e3l["available"] and len(folds) == pop["gold"].fold.nunique()
    if available:
        frame, components, r0, receipts = evaluate(ctx, folds)
        replay = r38.p2_replay(ctx, components, r0)
    else:
        frame, components, r0, receipts, _, _ = r39.c2_p2(ctx)
        replay = r38.p2_replay(ctx, components, r0)
    c2_checks = r39.c2_p2_checks(ctx, frame)
    report, paintings, gates = plan.gates(pop["gold"], frame, ctx.config, available, ctx.e3l["reason"])
    selected = plan.select(gates, ctx.config["gate"]["tie_tolerance"]) if available else None
    diagnostics = {"selected_epochs": {f: r["selected_epoch"] for f, r in folds.items()},
                   "curves": {f: r["history"] for f, r in folds.items()}}
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    if available:
        diagnostics |= {"vs_E3_and_fixed": plan.diagnostics(pop["gold"], frame, ctx.config)}
        plan.per_language(pop["gold"], frame, ctx.config).to_csv(folder / "language_deltas.csv", index=False,
                                                                lineterminator="\n")
    for name, table in (("oof.csv", frame), ("components.csv", components)):
        table.to_csv(folder / name, index=False, lineterminator="\n", float_format="%.6f")
    paintings.to_csv(folder / "painting_contrast.csv", index=False, lineterminator="\n")
    write(folder / "metrics.json", {"report": report, "replay": replay, "c2_replay": c2_checks,
                                    "fold_receipts": receipts, "e3l": ctx.e3l, "seconds": time.time() - tick,
                                    "gate_reference": "on-platform C2 recomputed in this run from the sealed "
                                                      "EXP-038 features; the archived CSV is replay-only"})
    write(folder / "diagnostics.json", diagnostics)
    write(folder / "gates.json", {"gates": gates, "selected": selected, "decision": "GO" if selected else "NO_GO",
                                  "e3l": ctx.e3l, "selected_epochs": diagnostics["selected_epochs"]})
    seal(folder, ctx.identity)
    return read(folder / "gates.json")


# Full refit and Development.

def stage_full(ctx, backend, status, folds):
    """The full-train E3L at the median selected epoch (or the frozen E3 when that epoch is 0)."""
    pop = r38.population(ctx)
    epochs = [r["selected_epoch"] for r in folds.values()]
    epoch = plan.median_epoch(epochs)
    folder = ctx.root / "full_refit"
    if epoch == 0:
        return {"epoch": 0, "selected_epochs": epochs, "note": "median epoch 0: E3L is the frozen E3"}
    if sealed_stage(folder, ctx.identity, ["result.json"]):
        return read(folder / "result.json") | {"epoch": epoch}
    e3 = base_features(ctx)["E3"]
    head, receipt = ridge(ctx, pop["train"], e3)
    dev = pop["dev"][data.KEYS].assign(k=0, n=0)
    paintings = sorted(pop["train"].painting.unique())
    req, body = request(ctx, "full", "full", pairs=pd.concat([pop["train"][data.KEYS + ["k", "n"]], dev]),
                        train_paintings=paintings, inner=[], held=sorted(dev.painting.unique()),
                        embed=sorted(set(paintings) | set(dev.painting)), head=head, epochs_to_run=epoch,
                        deadline=ctx.clock.work_deadline(),
                        identity=ctx.identity | {"full": epoch, "vision_cache": bool(ctx.vision_cache["enabled"])},
                        seed=ctx.config["train"]["seed"] + 100)
    body["folder"] = str(folder)
    gpu = ctx.config["parallel"]["full_refit_gpu"]
    dispatch(ctx, backend, body, req, ctx.clock.work_deadline(), gpu=gpu)
    result = read(folder / "result.json") | {"init_ridge": receipt, "selected_epochs": epochs, "gpu_index": gpu}
    write(folder / "result.json", result)
    seal(folder, ctx.identity)
    return result | {"epoch": epoch}


def development_frame(ctx, full):
    from threadpoolctl import threadpool_limits

    pop = r38.population(ctx)
    features = base_features(ctx)
    if full["epoch"] == 0:
        fx, probabilities = plan.with_e3l(features, features["E3"]), None
    else:
        folder = ctx.root / "full_refit"
        with np.load(folder / "embeddings.npz", allow_pickle=False) as saved:
            frame = pd.DataFrame(saved["values"].astype(np.float64), index=saved["paintings"].tolist())
        fx, probabilities = plan.with_e3l(features, frame), pd.read_csv(folder / "held_probabilities.csv")
    with threadpool_limits(limits=ctx.config["runtime"]["ridge_threads"]):
        frame, _, _, receipt, (heads, banks, visual) = plan.fit_and_predict(
            pop["train"], pop["dev"][data.KEYS], fx, probabilities, ctx.track1, ctx.neighborhood, 0)
    return pop["dev"][data.KEYS].merge(frame, on=data.KEYS, validate="one_to_one"), heads, banks, fx, receipt


def stage_development(ctx, status, gates, full):
    folder = ctx.root / "development"
    if completed(folder, ctx.identity, ["scores.json", "predictions.csv"]):
        return read(folder / "scores.json")
    pop = r38.population(ctx)
    dev = pop["dev"]
    available = ctx.e3l["available"] and full is not None
    if available:
        frame, heads, banks, features, info = development_frame(ctx, full)
        candidates = dict(plan.CANDIDATES)
        arms = [*plan.BASE_ARMS, plan.ARM]
    else:
        features = base_features(ctx)
        frame, heads, banks, info = r39.c2_development(ctx, features)
        candidates = {"C2": plan.CANDIDATES["C2"]}
        arms = list(plan.BASE_ARMS)
    replay = r38.replay_development(ctx, frame)
    c2_checks = r39.c2_development_checks(ctx, frame)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    frame.to_csv(folder / "predictions.csv", index=False, lineterminator="\n", float_format="%.6f")
    cscva3, c6 = r39.components(ctx)
    standalone = {name: science.system_frame(frame, name) for name in
                  ["R0_fixed", "R0_corrected", *arms, *[f"{c}_mix" for c in candidates], *candidates]}
    composed = science.blends({name: standalone[name] for name in candidates}, {"cscva3": cscva3}, c6, dev, ctx.config)
    archives = {}
    for name, prediction in (standalone | composed).items():
        archive = r38.write_submission(science.canonical(prediction, dev), dev, folder / "submissions",
                                       f"{ctx.name}_{name}_submission")
        archives[name] = {"zip": archive.relative_to(ctx.root).as_posix(), "sha256": sha(archive),
                          "eligibility": science.eligibility(name, ctx.config)}
    gold = pd.read_csv(ctx.base / "scoring/dev_gold.csv")
    canonical = {name: submission.validate_archive(ctx.root / item["zip"], dev) for name, item in archives.items()}
    scores = science.development_scores(gold, canonical, ctx.config["test_like_excluded"])
    intervals = science.development_intervals(gold, {k: v for k, v in canonical.items() if k.startswith("B2_F")},
                                              {"B2_C2_cscva3": canonical["B2_C2_cscva3"]},
                                              ctx.config["test_like_excluded"], ctx.config["diagnostic_seed"])
    # heads.npz keeps the E3L ridge only as a diagnostic; E3L itself is the LoRA head in bundle/e3l.
    r38.save_bundle(folder / "bundle", heads, banks, features, sorted(pop["train"].painting.unique()),
                    candidates, ctx)
    if available:
        e3l_bundle(ctx, folder / "bundle", full)
    write(folder / "scores.json", {"scores": scores, "intervals": intervals, "archives": archives, "replay": replay,
                                   "c2_replay": c2_checks, "fit": {k: v for k, v in info.items() if k != "outer_fits"}
                                   if isinstance(info, dict) else None, "full_refit": full,
                                   "development_selection": "none; Development is reporting only"})
    seal(folder, ctx.identity)
    return read(folder / "scores.json")


def e3l_bundle(ctx, bundle, full):
    """Deployment pieces for E3L: the LoRA + head weights at the deployment epoch (or 'frozen E3')."""
    target = bundle / "e3l"
    target.mkdir()
    record = {"epoch": full["epoch"], "languages": languages(ctx), "lora": ctx.config["lora"],
              "spec": ctx.config["arms"]["E3"], "lambda": ctx.config["train"]["lambda"]}
    if full["epoch"] > 0:
        source = ctx.root / "full_refit" / "epochs" / f"epoch_{full['epoch']}.pt"
        shutil.copyfile(source, target / "weights.pt")
        record["weights_sha256"] = sha(target / "weights.pt")
        shutil.copyfile(ctx.root / "tmp_requests" / "full" / "head.npz", target / "init_head.npz")
        record["init_head_sha256"] = sha(target / "init_head.npz")
    else:
        record["note"] = "deployment epoch 0: E3L is the frozen E3 and its Track 1 head (heads.npz:E3)"
    write(target / "e3l.json", record)


def stage_submissions(ctx, gates, scores):
    dev = r38.population(ctx)["dev"]
    for name, item in scores["archives"].items():
        if sha(ctx.root / item["zip"]) != item["sha256"]:
            raise ValueError(f"Submission archive changed: {name}")
        frame = submission.validate_archive(ctx.root / item["zip"], dev)
        if not frame[data.KEYS].equals(dev[data.KEYS]):
            raise ValueError(f"{name} key order differs")
    selected = gates["selected"]
    best = ctx.root / f"{ctx.name}_best_submission.zip"
    if selected:
        name = f"B2_{selected}_cscva3"
        source = ctx.root / scores["archives"][name]["zip"]
        recipe = {"decision": "GO", "selected_candidate": selected, "submission": name,
                  "formula": f"round6(0.80 * round6(0.75 * {selected} + 0.25 * CS-CVA-3) + 0.20 * C6)"}
    else:
        source = r39.carried(ctx, "incumbent_submission.zip")
        recipe = {"decision": "NO_GO", "selected_candidate": None, "submission": "EXP038_incumbent",
                  "formula": "byte-identical EXP-038 incumbent B2_C2_cscva3",
                  "reason": ctx.e3l["reason"] or "No EXP-040 candidate passed the frozen P2 gate against on-platform C2."}
    copy_verified(source, best)
    submission.validate_archive(best, dev)
    if not selected and sha(best) != ctx.refs["exp038_carried_sha256"]["incumbent_zip"]:
        raise ValueError("NO_GO best ZIP is not byte-identical to the EXP-038 incumbent")
    selection = recipe | {"coefficients": ctx.config["blends"] | ctx.config["mix"], "best_sha256": sha(best),
                          "components_sha256": {"CSCVA_3": ctx.refs["exp038_carried_sha256"]["cscva3_zip"],
                                                "C6": ctx.refs["exp037_c6_standalone_csv_sha256"]},
                          "eligibility": ctx.config["eligibility"]["unconfirmed_label"],
                          "development_used_for_selection": False}
    decision = {"decision": recipe["decision"], "selected": selected, "gates": gates["gates"], "e3l": ctx.e3l,
                "selected_epochs": gates.get("selected_epochs"),
                "selection_rule": "highest filtered P2 Contrast among passing; within 0.001 prefer F1",
                "pieces": read(ctx.root / "status.json")["pieces"]}
    write(ctx.root / "decision.json", decision)
    write(ctx.root / "selection.json", selection)
    return decision, selection


def run_pipeline(ctx, backend, smoke=False):
    ctx.root.mkdir(parents=True, exist_ok=True)
    success = ctx.output / f"{ctx.name}_SUCCESS.json"
    if success.exists():
        success.unlink()
    status = r38.Status(ctx)
    outcome = None
    stages = SMOKE_STAGES if smoke else STAGES
    try:
        status.begin(stages[0])
        r38.stage_setup(ctx, backend, status)
        ctx.identity |= {"mode": "smoke" if smoke else "full"}
        status.done(stages[0])
        status.begin(stages[1])
        validate = stage_validate(ctx, backend, status, smoke)
        ctx.vision_cache = validate["vision_cache"]
        ctx.log(f"Vision cache: {'ON' if ctx.vision_cache['enabled'] else 'OFF'} ({ctx.vision_cache['reason']}); "
                f"GPUs {validate['gpus']}")
        status.done(stages[1])
        if smoke:
            status.begin(stages[2])
            result = stage_smoke(ctx, backend, status, validate)
            status.done(stages[2])
            status.state |= {"state": "SMOKE_COMPLETE", "submission_produced": False}
            status.save()
            outcome = "SMOKE"
            return result
        r38.progress(ctx)
        status.begin(stages[2])
        budget = stage_benchmark(ctx, backend, status, validate)
        status.done(stages[2])
        status.begin(stages[3])
        folds = stage_folds(ctx, backend, status, budget)
        status.done(stages[3])
        status.begin(stages[4])
        gates = stage_p2(ctx, status, folds)
        if not gates["e3l"]["available"]:
            ctx.e3l = gates["e3l"]
        status.done(stages[4])
        r38.progress(ctx)
        status.begin(stages[5])
        full = None
        if ctx.e3l["available"]:
            try:
                full = stage_full(ctx, backend, status, folds)
            except TimeoutError:
                raise
            except Exception as error:
                ctx.e3l = {"available": False, "reason": f"TECHNICAL_FAILURE full refit: {type(error).__name__}: {error}"}
                status.piece("E3L_full", "FAILED", ctx.e3l["reason"])
                gates = gates | {"selected": None, "decision": "NO_GO"}
        scores = stage_development(ctx, status, gates, full)
        status.done(stages[5])
        status.begin(stages[6])
        decision, selection = stage_submissions(ctx, gates, scores)
        status.done(stages[6])
        status.begin(stages[7])
        status.state["state"] = "EXPORTING"
        status.save()
        digests = r38.stage_export(ctx, scores, decision)
        status.done(stages[7])
        status.begin(stages[8])
        status.state |= {"state": "COMPLETE", "decision": decision["decision"], "selected": decision["selected"]}
        status.save()
        write(success, {"decision": decision["decision"], "selected": decision["selected"],
                        "best_sha256": selection["best_sha256"], "exports": digests})
        status.done(stages[8])
        outcome = "COMPLETE"
        return decision
    except BaseException as error:
        resumable = isinstance(error, TimeoutError) and not isinstance(error, BudgetStop)
        status.state |= {"state": "RESUMABLE_STOP" if resumable else "FAILED",
                         "reason": f"{type(error).__name__}: {error}", "failures": getattr(error, "failures", None)}
        status.save()
        try:
            r38.progress(ctx)
        finally:
            r38.diagnostic(ctx, status, error)
        raise
    finally:
        if outcome != "COMPLETE" and success.exists():
            success.unlink()


# GPU backend and child.

class Backend(r38.SubprocessBackend):
    def runtime(self):
        return r39.Backend.runtime(self)

    def fetch(self, arm):
        from st3.exp038_gpu import fetch_snapshot

        if arm != "E3":
            raise ValueError("EXP-040 uses only the pinned 2B snapshot")
        path, receipt = fetch_snapshot(self.ctx.config["arms"]["E3"], self.ctx.snapshots.get("E3"))
        self.ctx.models["E3"] = path
        return {k: receipt[k] for k in ("model", "revision")}

    def __init__(self, ctx):
        super().__init__(ctx)
        self.children, self.lock = {}, threading.Lock()

    def gpus(self):
        """GPU indices in PCI bus order (as nvidia-smi lists them), capped at ``parallel.max_gpus``."""
        probe = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                               text=True, check=True)
        found = [int(line) for line in probe.stdout.split() if line.strip()]
        return sorted(found)[:self.ctx.config["parallel"]["max_gpus"]]

    def e3l(self, request_path, deadline, gpu=0):
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(int(gpu)), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        run_child(["gpu-e3l", "--request", request_path], deadline, self.ctx, env=env, label=f"gpu{gpu}",
                  registry=self)

    def cancel(self):
        with self.lock:
            children = list(self.children.values())
        for child in children:
            kill(child)


LOG_LOCK = threading.Lock()


def kill(child):
    """Kill a child started with start_new_session=True (its whole process group on POSIX)."""
    if child.poll() is not None:
        return
    if os.name == "posix":
        import signal

        try:
            os.killpg(child.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    child.kill()


def run_child(arguments, deadline, ctx, env=None, label="gpu0", registry=None, executable=None):
    """One GPU child with live output, each line relayed once with a ``[label]`` prefix; killed 60 s past its
    deadline or on ``registry.cancel()``. Exit codes follow the EXP-038 children."""
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Session allowance reached before a GPU stage")
    command = executable or [sys.executable, "-u", str(ctx.script)]
    log = ctx.root / "gpu_children.log"
    with subprocess.Popen([*command, *map(str, arguments)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", start_new_session=True, env=env) as child:
        if registry is not None:
            with registry.lock:
                registry.children[child.pid] = child

        def relay():
            for line in child.stdout:
                line = f"[{label}] {line}"
                with LOG_LOCK:
                    with log.open("a", encoding="utf-8") as stream:
                        stream.write(line)
                    print(line, end="", flush=True)

        reader = threading.Thread(target=relay, daemon=True)
        reader.start()
        try:
            code = child.wait(timeout=max(1., remaining + 60))
        except subprocess.TimeoutExpired:
            code = None
        finally:
            if child.poll() is None:
                kill(child)
                child.wait()
            reader.join(timeout=30)
            if registry is not None:
                with registry.lock:
                    registry.children.pop(child.pid, None)
    if code == r38.EXIT_BUDGET:
        raise BudgetStop(f"{arguments[0]} reached its cap")
    if code == r38.EXIT_TIMEOUT or code is None:
        raise TimeoutError(f"{arguments[0]} stopped at the session allowance")
    if code:
        raise RuntimeError(f"{arguments[0]} ({label}) exited with status {code}; see gpu_children.log")


def gpu_e3l(args):
    from st3 import exp040_gpu as gpu

    body = json.loads(args.request.read_text(encoding="utf-8"))
    try:
        gpu.run_request(body, log=lambda message: print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True))
    except gpu.TrainingTimeout:
        sys.exit(r38.EXIT_TIMEOUT)


# CPU preflight and learning curve.

def cpu_validate(args):
    """Every CPU integrity check of Validate (no downloads, no GPU), for Windows and Linux preflight."""
    args.output.mkdir(parents=True, exist_ok=True)
    overlay = restore_overlay(args.overlay, args.overlay_sha, args.base_sha, args.output / "tmp_overlay_restored",
                              manifest_name=MANIFEST)
    ctx = Context(base=args.base, overlay=overlay, root=args.output, output=args.output, base_sha=args.base_sha,
                  overlay_sha=args.overlay_sha, clock=Clock(time.time(), 10, 0))
    ctx.identity = {"cpu_validate": True}
    receipts, failures = r39.collect([
        ("pin_formats", lambda: check_config_pins(ctx.config), ()),
        ("inputs", lambda: len(r38.population(ctx)["gold"]), ()),
        ("sealed_exp038", lambda: {arm: len(r39.sealed_feature(ctx, arm)[0]) for arm in plan.BASE_ARMS}, ("inputs",)),
        ("key_audit", lambda: key_audit(ctx), ("inputs",)),
        ("head_cpu", lambda: head_cpu(ctx), ("inputs", "sealed_exp038")),
        ("r0_replay", lambda: {"p2": r38.r0_p2(ctx), "development": r38.r0_development(ctx)}, ("inputs",)),
        ("c2_replay", lambda: r39._c2_only(ctx), ("inputs", "sealed_exp038")),
    ])
    checks = []
    for name in ("r0_replay", "c2_replay"):
        result = receipts.get(name, {}).get("result") or {}
        for part in (result.get("p2", {}), result.get("development", {}), result):
            checks += part.get("checks", []) if isinstance(part, dict) else []
    summary = {"status": "FAIL" if failures else "PASS", "failures": failures, "checks": len(checks),
               "key_audit": (receipts.get("key_audit") or {}).get("result"),
               "head_cpu": (receipts.get("head_cpu") or {}).get("result"),
               "exact": bool(checks) and all(c["exact"] for c in checks),
               "max_ratio_to_threshold": max((c["max_abs_difference"] / c["threshold"] for c in checks), default=None),
               "platform": r38.platform_receipt(ctx)}
    write(args.output / "cpu_validate.json", {"summary": summary, "receipts": receipts})
    print(json.dumps(summary, indent=1))
    return 1 if failures else 0


def learning_curve(args):
    """CPU-only diagnostic: the frozen C2 recipe refitted on subsamples of each outer training fold."""
    from threadpoolctl import threadpool_limits

    args.output.mkdir(parents=True, exist_ok=True)
    overlay = restore_overlay(args.overlay, args.overlay_sha, args.base_sha, args.output / "tmp_overlay_restored",
                              manifest_name=MANIFEST)
    ctx = Context(base=args.base, overlay=overlay, root=args.output, output=args.output, base_sha=args.base_sha,
                  overlay_sha=args.overlay_sha, clock=Clock(time.time(), 10, 0))
    ctx.identity = {"learning_curve": True}
    cfg = ctx.config["learning_curve"]
    tick = time.time()
    with threadpool_limits(limits=ctx.config["runtime"]["ridge_threads"]):
        curve = plan.learning_curve(r38.population(ctx)["gold"], base_features(ctx), ctx.track1, ctx.neighborhood,
                                    ctx.config["test_like_excluded"], cfg["fractions"], cfg["seeds"], log=ctx.log)
    curve.to_csv(args.output / "learning_curve.csv", index=False, lineterminator="\n")
    result = {"config": cfg, "rows": curve.to_dict(orient="records"),
              "slope": plan.curve_slope(curve, seed=ctx.config["diagnostic_seed"]),
              "seconds": time.time() - tick, "platform": r38.platform_receipt(ctx),
              "inputs": {"overlay_manifest_sha256": args.overlay_sha, "base_manifest_sha256": args.base_sha}}
    write(args.output / "learning_curve.json", result)
    print(json.dumps(result["slope"], indent=1))


def run(args):
    clock = Clock(args.started, args.hours, configs()[0]["budget"]["persistence_reserve_seconds"])
    snapshots = {"E3": args.e3_snapshot} if args.e3_snapshot else {}
    ctx = Context(base=args.base, overlay=args.overlay, root=args.output / "exp040_run", output=args.output,
                  base_sha=args.base_sha, overlay_sha=args.overlay_sha, clock=clock, snapshots=snapshots)
    if args.resume:
        ctx.root.mkdir(parents=True, exist_ok=True)
        if any(ctx.root.iterdir()):
            raise ValueError("Resume needs a fresh /kaggle/working/exp040_run")
        restore(args.resume, ctx.root, args.overlay_sha)
    run_pipeline(ctx, Backend(ctx), smoke=args.smoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    v, p, c = sub.add_parser("cpu-validate"), sub.add_parser("run"), sub.add_parser("learning-curve")
    for command in (v, p, c):
        for name in ("base", "overlay", "output"):
            command.add_argument(f"--{name}", type=Path, required=True)
        for name in ("base-sha", "overlay-sha"):
            command.add_argument(f"--{name}", required=True)
    p.add_argument("--started", type=float, required=True)
    p.add_argument("--hours", type=float, default=10.5)
    p.add_argument("--resume", type=Path)
    p.add_argument("--e3-snapshot", type=Path)
    p.add_argument("--smoke", action="store_true", help="SMOKE_ONLY: no training run, no submission")
    g = sub.add_parser("gpu-e3l")
    g.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    result = {"run": run, "cpu-validate": cpu_validate, "learning-curve": learning_curve,
              "gpu-e3l": gpu_e3l}[args.command](args)
    if isinstance(result, int):
        sys.exit(result)


if __name__ == "__main__":
    main()
