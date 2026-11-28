#!/usr/bin/env python3
"""EXP-040 worker: E3-LoRA (Qwen3-VL-Embedding-2B adapted from a ridge-initialized interaction head).

Full mode: Setup -> Validate (collect-all) -> Benchmark -> P2 LoRA folds (sealed per fold, resumable) ->
P2 evaluation -> Full refit and Development -> Submission validation -> Export -> Release.
SMOKE_ONLY: Setup -> Validate (replays may be skipped) -> one fold x N optimizer steps + one inner pass ->
exp040_smoke.json with throughput, VRAM, devices, finiteness, determinism and a full-run forecast; no submission.

The EXP-038/039 workers are the library (context, population, sealed caches, R0/C2 replays, bundles, export).
C2 is recomputed on-platform from the sealed EXP-038 features and is the gate reference. If E3L fails
technically or the forecast exceeds the session, the run still exports a NO_GO report with the reason and the
byte-identical incumbent as the best ZIP.
"""

import argparse
import json
import os
import shutil
import sys
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
from st3.exp038_io import BudgetStop, Clock, check_config_pins, completed, copy_verified, restore_overlay, seal

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
            "snapshot": str(ctx.models.get("E3")), "folder": None} | (extra or {})
    return folder, body


def dispatch(ctx, backend, body, folder, deadline):
    path = ctx.root / "tmp_requests" / f"{Path(body['folder']).name}_request.json"
    body = body | {"deadline": deadline}
    path.write_text(json.dumps(body), encoding="utf-8")
    backend.e3l(path, deadline)


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
    out = ctx.root / "validate_step0"
    shutil.rmtree(out, ignore_errors=True)
    body["folder"] = str(out)
    dispatch(ctx, backend, body, folder, ctx.clock.work_deadline())
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
    if smoke:
        write(ctx.root / "validate_smoke.json", receipts)
        return receipts
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    write(folder / "validate.json", receipts)
    seal(folder, ctx.identity)
    return receipts


# Benchmark, smoke and forecast.

def smoke_request(ctx, name, steps, inner_limit, determinism):
    fold = ctx.config["smoke"]["fold"]
    outer, held, fit, inner = fold_split(ctx, fold)
    e3 = base_features(ctx)["E3"]
    head, _ = ridge(ctx, outer[outer.painting.isin(fit)], e3)
    inner = inner[:inner_limit] if inner_limit else inner
    folder, body = request(ctx, name, "smoke", pairs=outer[outer.painting.isin(set(fit) | set(inner))],
                           train_paintings=fit, inner=inner, held=[], embed=[], head=head, epochs_to_run=1,
                           deadline=ctx.clock.work_deadline(), identity=ctx.identity | {"smoke": name},
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


def forecast(ctx, measured, c2_seconds):
    """Seconds per piece from measured warm throughput; the full refit assumes the worst case (4 epochs)."""
    train_cfg = ctx.config["train"]
    epochs, batch = train_cfg["epochs"], train_cfg["batch_paintings"]
    step = measured["warm_seconds_per_optimizer_step"] or measured["seconds_per_optimizer_step"]
    forward = 1 / measured["inner_paintings_per_second"]
    load = measured.get("load_seconds", 60) + 30
    n = counts(ctx)
    pieces = {}
    for fold, c in n["folds"].items():
        pieces[f"fold_{fold}"] = (load + epochs * plan.steps_per_epoch(c["fit"], batch) * step
                                  + (epochs + 1) * c["inner"] * forward + (c["embed"] + c["held"]) * forward)
    pieces["full_refit"] = (load + epochs * plan.steps_per_epoch(n["train"], batch) * step
                            + (n["train"] + 2 * n["dev"]) * forward)
    cpu = c2_seconds or 900
    pieces["cpu_p2_and_development"] = 2.0 * cpu + 600
    pieces["export"] = 600
    return {"pieces": pieces, "total": sum(pieces.values()), "inputs": {
        "seconds_per_optimizer_step": step, "forward_seconds_per_painting": forward, "load_seconds": load,
        "c2_cpu_seconds": cpu, "counts": n, "method": (
            "fold = load + epochs x ceil(fit/16) x warm step seconds + (epochs+1) inner passes + embedding of the "
            "outer-train and held-out paintings + held-out probabilities; full refit assumes the worst-case 4 "
            "epochs over all training paintings plus train and Development embeddings; CPU = 2 x the measured "
            "on-platform C2 replay + 600 s; export 600 s")}}


def stage_benchmark(ctx, backend, status, validate):
    folder = ctx.root / "benchmark"
    if not completed(folder, ctx.identity, ["benchmark.json"]):
        steps = ctx.config["smoke"]["benchmark_steps"]
        req, body, out = smoke_request(ctx, "tmp_benchmark", steps, 32, False)
        dispatch(ctx, backend, body, req, ctx.clock.work_deadline())
        measured = read(out / "smoke.json")
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        write(folder / "benchmark.json", measured)
        seal(folder, ctx.identity)
    measured = read(folder / "benchmark.json")
    estimate = forecast(ctx, measured, validate.get("c2_seconds"))
    margin = ctx.config["budget"]["forecast_margin"]
    remaining = ctx.clock.remaining()
    budget = estimate | {"margin": margin, "planned_seconds": margin * estimate["total"], "remaining_seconds": remaining,
                         "fits": margin * estimate["total"] <= remaining}
    write(ctx.root / "budget_plan.json", budget)
    return budget


def stage_smoke(ctx, backend, status, validate):
    cfg = ctx.config["smoke"]
    req, body, out = smoke_request(ctx, "tmp_smoke", cfg["optimizer_steps"], None, True)
    dispatch(ctx, backend, body, req, ctx.clock.work_deadline())
    measured = read(out / "smoke.json")
    estimate = forecast(ctx, measured, validate.get("c2_seconds"))
    margin = ctx.config["budget"]["forecast_margin"]
    session = ctx.config["budget"]["session_hours"] * 3600 - ctx.config["budget"]["persistence_reserve_seconds"]
    result = {"mode": "SMOKE_ONLY", "submission_produced": False, "validate": {
        k: (v.get("status") if isinstance(v, dict) else v) for k, v in validate.items()},
        "step0": (validate.get("step0_gpu") or {}).get("result"), "smoke": measured,
        "forecast": estimate | {"margin": margin, "planned_seconds": margin * estimate["total"],
                                "session_seconds": session, "fits_session": margin * estimate["total"] <= session}}
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
    results = {}
    for fold in sorted(pop["gold"].fold.unique()):
        fold = int(fold)
        status.item(f"fold {fold}")
        folder = ctx.root / "p2_folds" / f"fold_{fold}"
        if sealed_stage(folder, ctx.identity, ["result.json"]):
            results[fold] = read(folder / "result.json")
            continue
        outer, held, fit, inner = fold_split(ctx, fold)
        head, fit_receipt = ridge(ctx, outer[outer.painting.isin(fit)], e3)
        req, body = request(ctx, f"fold_{fold}", "fold", pairs=pd.concat([outer, held]), train_paintings=fit,
                            inner=inner, held=sorted(held.painting.unique()),
                            embed=sorted(set(outer.painting) | set(held.painting)), head=head,
                            epochs_to_run=ctx.config["train"]["epochs"], deadline=ctx.clock.work_deadline(),
                            identity=ctx.identity | {"fold": fold}, seed=ctx.config["train"]["seed"] + fold)
        body["folder"] = str(folder)
        try:
            dispatch(ctx, backend, body, req, ctx.clock.work_deadline())
        except TimeoutError:
            raise
        except Exception as error:  # E3L is the experiment: a technical failure becomes a reported NO_GO.
            ctx.e3l = {"available": False, "reason": f"TECHNICAL_FAILURE fold {fold}: {type(error).__name__}: {error}"}
            status.piece("E3L", "FAILED", ctx.e3l["reason"])
            return results
        result = read(folder / "result.json") | {"init_ridge": fit_receipt}
        write(folder / "result.json", result)
        seal(folder, ctx.identity)
        results[fold] = result
        status.piece(f"E3L_fold_{fold}", "COMPLETE", f"selected epoch {result['selected_epoch']}")
        r38.progress(ctx)
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
                        deadline=ctx.clock.work_deadline(), identity=ctx.identity | {"full": epoch},
                        seed=ctx.config["train"]["seed"] + 100)
    body["folder"] = str(folder)
    dispatch(ctx, backend, body, req, ctx.clock.work_deadline())
    result = read(folder / "result.json") | {"init_ridge": receipt, "selected_epochs": epochs}
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

    def e3l(self, request_path, deadline):
        env = os.environ | {"CUDA_VISIBLE_DEVICES": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        r38.run_child(["gpu-e3l", "--request", request_path], deadline, self.ctx, env=env)


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
