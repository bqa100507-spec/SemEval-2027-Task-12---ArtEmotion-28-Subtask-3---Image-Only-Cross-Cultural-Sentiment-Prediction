#!/usr/bin/env python3
"""EXP-039 worker: scale the VLM-embedding axis on top of the EXP-038 C2 recipe.

Stages: Setup -> Validate -> Benchmark -> Feature extraction (E5, E6, E7) -> P2 evaluation ->
Full refit and Development -> Submission validation -> Export -> Release.

The EXP-038 worker is the library: context, status, population, R0 tolerance replay, sealed caches,
budget planning, export and diagnostics come from ``run_exp038``. E1-E3 come from the sealed EXP-038
run-4 caches in the overlay, so C2 is recomputed on-platform without GPU work.
"""

import argparse
import json
import os
import shutil
import subprocess
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
from st3 import data, submission
from st3 import exp038 as science
from st3 import exp039 as plan39
from st3.exp036_io import check, read, restore, sha, write
from st3.exp038_io import (BudgetStop, Clock, check_config_pins, completed, copy_verified, load_features,
                           plan_budget, restore_overlay, seal)

STAGES = ("Setup", "Validate", "Benchmark", "Feature extraction", "P2 evaluation", "Full refit and Development",
          "Submission validation", "Export", "Release")
MANIFEST = "exp039_manifest.json"


def configs():
    return (read(ROOT / "configs/exp039.json"), read(ROOT / "configs/competition_track1.json"),
            read(ROOT / "configs/competition_neighborhood_residual.json"))


class Context(r38.Context):
    name = "exp039"
    manifest_name = MANIFEST
    config_file = "configs/exp039.json"
    script = Path(__file__)

    def __init__(self, **kwargs):
        default = configs()
        kwargs.setdefault("config", default[0])
        kwargs.setdefault("track1", default[1])
        kwargs.setdefault("neighborhood", default[2])
        super().__init__(**kwargs)
        self.e5 = "E5"
        self.gpu_seconds = {}


class ValidationFailed(RuntimeError):
    """Every independent integrity failure found by Validate, reported together."""

    def __init__(self, failures):
        self.failures = failures
        super().__init__(f"{len(failures)} integrity failure(s): " + " | ".join(failures))


def spec(ctx, arm):
    return plan39.arm_spec(ctx.config, arm)


def identity_for(ctx, arm):
    return ctx.identity | {"arm": arm, "spec": spec(ctx, arm), "chunk_items": ctx.config["chunk_items"]}


def items_for(ctx, arm):
    images = r38.image_items(ctx)
    if plan39.config_arm(arm) == "E6":
        return plan39.pool_items(images, len(ctx.config["arms"]["E6"]["instructions"]))
    return images


def new_labels(ctx):
    return tuple(plan39.label(a, ctx.e5) for a in plan39.NEW)


# Sealed EXP-038 caches.

def sealed_feature(ctx, arm):
    folder = ctx.overlay / "exp038/features" / arm
    meta = read(folder / "seal.json")
    run38 = read(ctx.overlay / "exp038/run.json")
    identity = {k: v for k, v in meta["identity"].items() if k not in ("arm", "spec", "chunk_items")}
    if identity != run38 or meta["identity"]["arm"] != arm:
        raise ValueError(f"Sealed EXP-038 {arm} cache does not belong to the EXP-038 run identity")
    if sha(folder / "seal.json") != ctx.refs["exp038_carried_sha256"][f"feature_{arm}_seal.json"]:
        raise ValueError(f"Sealed EXP-038 {arm} seal differs from its pinned hash")
    check(folder, meta["files"])
    with np.load(folder / "features.npz", allow_pickle=False) as cache:
        keys, values = cache["keys"].tolist(), cache["values"]
    items = r38.image_items(ctx)
    if keys != [item[0] for item in items] or not np.isfinite(values).all():
        raise ValueError(f"Sealed EXP-038 {arm} cache coverage differs from the population")
    if values.shape[1] != ctx.config["arms"][arm]["width"]:
        raise ValueError(f"Sealed EXP-038 {arm} width differs")
    return values, items


def features_for(ctx, arms):
    pop = r38.population(ctx)
    features = dict(pop["features"])
    sealed = {}
    for arm in plan39.SEALED:
        sealed[arm], items = sealed_feature(ctx, arm)
        features[arm] = r38.arm_frame(ctx, sealed[arm], items)
    images = r38.image_items(ctx)
    for arm in arms:
        if arm in plan39.SEALED:
            continue
        values = load_features(ctx.root / "features" / arm, identity_for(ctx, arm), [i[0] for i in items_for(ctx, arm)])
        if plan39.config_arm(arm) == "E6":
            values = plan39.pooled_instructions(sealed["E3"], values, len(ctx.config["arms"]["E6"]["instructions"]))
        features[arm] = r38.arm_frame(ctx, values, images)
    return features


def carried(ctx, name):
    path = ctx.overlay / "exp038" / name
    key = {"p2_oof.csv": "p2_oof", "development_predictions.csv": "development_predictions",
           "CSCVA_3_submission.zip": "cscva3_zip", "incumbent_submission.zip": "incumbent_zip",
           "run.json": "run"}[name]
    if sha(path) != ctx.refs["exp038_carried_sha256"][key]:
        raise ValueError(f"Carried EXP-038 artifact differs: {name}")
    return path


def components(ctx):
    dev = r38.population(ctx)["dev"]
    cscva3 = submission.validate_archive(carried(ctx, "CSCVA_3_submission.zip"), dev)
    c6_path = ctx.overlay / "reference/exp037_c6_standalone/predictions.csv"
    if sha(c6_path) != ctx.refs["exp037_c6_standalone_csv_sha256"]:
        raise ValueError("Saved EXP-037 C6 Development predictions differ")
    return cscva3, submission.read_predictions(c6_path, dev)


# Validate: collect every independent integrity failure, then stop.

def c2_p2(ctx, arms=plan39.SEALED, candidates=None):
    from threadpoolctl import threadpool_limits

    pop = r38.population(ctx)
    features = features_for(ctx, arms)
    candidates = candidates or {"C2": ("E1", "E2", "E3")}
    config = ctx.config | {"candidates": candidates}
    with threadpool_limits(limits=ctx.config["runtime"]["ridge_threads"]):
        frame, parts, r0, receipts, used = science.evaluate_p2(
            pop["gold"], features, None, arms, config, ctx.track1, ctx.neighborhood,
            ctx.refs["track1_inner_oof_sha256"], log=ctx.log)
    return frame, parts, r0, receipts, used, features


def c2_p2_checks(ctx, frame):
    archived = pd.read_csv(carried(ctx, "p2_oof.csv"))
    return plan39.replay_columns(r38.population(ctx)["gold"], frame, archived,
                                 ("E1", "E2", "E3", "C2_mix", "C2", "R0_fixed", "R0_corrected"),
                                 ctx.config["replay_tolerance"]["probability"], "C2 replay P2")


def c2_development(ctx, features, arms=plan39.SEALED, candidates=None):
    from threadpoolctl import threadpool_limits

    pop = r38.population(ctx)
    candidates = candidates or {"C2": ("E1", "E2", "E3")}
    with threadpool_limits(limits=ctx.config["runtime"]["ridge_threads"]):
        heads, banks, info, visual = science.deployment_fit(pop["train"], features, None, arms, ctx.track1,
                                                            ctx.neighborhood, candidates)
        frame, _ = science.development_predictions(heads, banks, visual, pop["dev"][data.KEYS], features, None,
                                                   candidates, ctx.neighborhood)
    return frame, heads, banks, info


def c2_development_checks(ctx, frame):
    pop = r38.population(ctx)
    dev, threshold = pop["dev"], ctx.config["replay_tolerance"]["probability"]
    archived = pd.read_csv(carried(ctx, "development_predictions.csv"))
    checks = plan39.replay_columns(dev, frame, archived, ("E1", "E2", "E3", "C2", "R0_corrected"), threshold,
                                   "C2 replay Development")
    cscva3, c6 = components(ctx)
    b2 = science.blends({"C2": science.system_frame(frame, "C2")}, {"cscva3": cscva3}, c6, dev,
                        ctx.config)["B2_C2_cscva3"]
    incumbent = submission.validate_archive(carried(ctx, "incumbent_submission.zip"), dev)
    checks.append(science.measure("C2 replay B2_C2_cscva3 vs EXP-038 incumbent",
                                  science.canonical(b2, dev).p_positive.to_numpy(),
                                  incumbent.p_positive.to_numpy(), threshold))
    return checks


KEY_ARMS = ("E3", "E5")


def key_audit(ctx):
    """Checkpoint weight names vs the exact model class on the meta device (checklist item 8)."""
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
        if arm in ctx.models:  # On Kaggle the downloaded snapshot must hold exactly the audited names.
            if ka.snapshot_keys(ctx.models[arm]) != sorted(keys):
                raise ValueError(f"{arm} downloaded checkpoint names differ from the audited list")
        class_keys, tied_keys = ka.qwen_class_keys(config)
        out[arm] = ka.require(ka.audit(keys, class_keys, config, tied_keys), arm) | {
            "model": arm_spec["model"], "revision": arm_spec["revision"],
            "snapshot_cross_checked": arm in ctx.models}
    return out


def collect(checks):
    """Run independent checks, skip dependants of failures, and report every failure together."""
    receipts, failures, failed = {}, [], set()
    for name, function, needs in checks:
        missing = [n for n in needs if n in failed]
        if missing:
            receipts[name] = {"status": "SKIPPED_DEPENDENCY", "needs": missing}
            failed.add(name)
            continue
        tick = time.time()
        try:
            receipts[name] = {"status": "PASS", "result": function(), "seconds": time.time() - tick}
        except Exception as error:
            receipts[name] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
            failures.append(f"{name}: {type(error).__name__}: {error}")
            failed.add(name)
    return receipts, failures


def stage_validate(ctx, backend, status):
    folder = ctx.root / "validate"
    if completed(folder, ctx.identity, ["validate.json"]):
        for arm in ("E3", "E5"):
            backend.fetch(arm)
        ctx.log("Validate: sealed receipts verified; reusing")
        return read(folder / "validate.json")
    state = {}

    def inputs():
        pop = r38.population(ctx)
        return {"train_pairs": len(pop["train"]), "dev_pairs": len(pop["dev"]), "p2_pairs": len(pop["gold"]),
                "overlay_files": len(pop["overlay_manifest"]["files"])}

    def sealed():
        return {arm: {"rows": len(sealed_feature(ctx, arm)[0])} for arm in plan39.SEALED} | {
            "components": [carried(ctx, n).name for n in ("p2_oof.csv", "development_predictions.csv",
                                                          "CSCVA_3_submission.zip", "incumbent_submission.zip")]}

    def r0():
        return {"p2": r38.r0_p2(ctx), "development": r38.r0_development(ctx), "platform": r38.platform_receipt(ctx)}

    def c2():
        frame, parts, r0_frame, receipts, used, features = c2_p2(ctx)
        checks = c2_p2_checks(ctx, frame)
        dev_frame, *_ = c2_development(ctx, features)
        checks += c2_development_checks(ctx, dev_frame)
        state["c2_seconds"] = sum(r["seconds"] for r in receipts.values())
        return {"checks": checks, "exact": all(c["exact"] for c in checks)}

    receipts, failures = collect([
        ("pin_formats", lambda: check_config_pins(ctx.config), ()),
        ("inputs", inputs, ()),
        ("snapshot_2B", lambda: backend.fetch("E3"), ()),
        ("snapshot_8B", lambda: backend.fetch("E5"), ()),
        ("sealed_exp038", sealed, ("inputs",)),
        ("key_audit", lambda: key_audit(ctx), ("inputs",)),
        ("r0_replay", r0, ("inputs",)),
        ("c2_replay", c2, ("inputs", "sealed_exp038")),
    ])
    if failures:
        write(ctx.root / "validate_failures.json", receipts)
        raise ValidationFailed(failures)
    receipts["c2_seconds"] = state["c2_seconds"]
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    write(folder / "validate.json", receipts)
    seal(folder, ctx.identity)
    return receipts


# Benchmark, forecast, extraction.

def stage_benchmark(ctx, backend, status):
    folder = ctx.root / "benchmark"
    if not completed(folder, ctx.identity, ["benchmark.json"]):
        result = {"arms": {}, "method": (
            "Two passes over 32 seeded real items per new arm on its GPU placement; the warm second pass gives "
            "items/s. Forecast = items / rate + load + 120 s download allowance. E6 runs 3 extra instructions per "
            "image (the 4th is the sealed E3). CPU forecast = 2.5 x the measured on-platform C2 replay time.")}
        for arm in new_labels(ctx):
            status.item(arm)
            sample = r38.benchmark_items(items_for(ctx, arm), 32, ctx.config["diagnostic_seed"])
            try:
                result["arms"][arm] = backend.benchmark(arm, sample)
                if not result["arms"][arm]["deterministic_within_1e-3"]:
                    raise RuntimeError(f"{arm} failed the two-pass determinism check")
            except Exception as error:
                result["arms"][arm] = {"failed": f"{type(error).__name__}: {error}"}
                ctx.log(f"Benchmark {arm}: FAILED_OPTIONAL ({result['arms'][arm]['failed']})")
        result["counts"] = {arm: len(items_for(ctx, arm)) for arm in new_labels(ctx)}
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        write(folder / "benchmark.json", result)
        seal(folder, ctx.identity)
    return read(folder / "benchmark.json")


def forecast(ctx, bench, validate):
    pieces = {}
    for arm in new_labels(ctx):
        measured = bench["arms"].get(arm, {})
        if (ctx.root / "features" / arm / "seal.json").is_file() or "items_per_second" not in measured:
            continue
        pieces[plan39.config_arm(arm)] = (bench["counts"][arm] / measured["items_per_second"]
                                          + measured.get("load_seconds", 60) + 120)
    pieces["cpu_p2_and_development"] = 2.5 * validate["c2_seconds"] + 600
    pieces["export"] = 600
    return pieces


def stage_features(ctx, backend, status, plan, bench):
    later = sum(v for k, v in plan["forecast_seconds"].items() if k.startswith(("cpu", "export")))
    for arm in new_labels(ctx):
        status.item(arm)
        folder = ctx.root / "features" / arm
        items = items_for(ctx, arm)
        if (folder / "seal.json").is_file():
            load_features(folder, identity_for(ctx, arm), [i[0] for i in items])
            status.piece(arm, "COMPLETE", "sealed cache reused")
            continue
        key = plan39.config_arm(arm)
        if key in plan["skipped"]:
            status.piece(arm, "BUDGET_SKIPPED", plan["skipped"][key])
            continue
        if "failed" in bench["arms"].get(arm, {}):
            status.piece(arm, "FAILED_OPTIONAL", f"benchmark: {bench['arms'][arm]['failed']}")
            continue
        tick = time.time()
        try:
            backend.extract(arm, items, folder, identity_for(ctx, arm),
                            ctx.clock.work_deadline() - plan["margin"] * later)
            load_features(folder, identity_for(ctx, arm), [i[0] for i in items])
        except (BudgetStop, TimeoutError) as error:
            status.piece(arm, "BUDGET_STOP", str(error))
        except Exception as error:  # Every new arm is optional: C2 and the other D candidates survive.
            status.piece(arm, "FAILED_OPTIONAL", f"{type(error).__name__}: {error}")
        else:
            ctx.gpu_seconds[arm] = time.time() - tick
            write(folder.parent / f"{arm}_gpu_seconds.json", {"seconds": ctx.gpu_seconds[arm]})
            status.piece(arm, "COMPLETE")
        r38.progress(ctx)
    for arm in new_labels(ctx):
        receipt = ctx.root / "features" / f"{arm}_gpu_seconds.json"
        if receipt.is_file():
            ctx.gpu_seconds[arm] = read(receipt)["seconds"]


def available(ctx):
    return (*plan39.SEALED, *[a for a in new_labels(ctx) if (ctx.root / "features" / a / "seal.json").is_file()])


# P2, Development, submissions.

def stage_p2(ctx, status):
    folder = ctx.root / "p2"
    if completed(folder, ctx.identity, ["oof.csv", "metrics.json", "gates.json"]):
        return read(folder / "gates.json")
    pop = r38.population(ctx)
    arms = available(ctx)
    candidates = plan39.candidate_set(ctx.config, arms, ctx.e5)
    tick = time.time()
    frame, parts, r0, receipts, used, features = c2_p2(ctx, arms, candidates)
    replay = r38.p2_replay(ctx, parts, r0)
    c2_checks = c2_p2_checks(ctx, frame)
    # Gate integrity: the reference is the C2 recomputed in this run, never the archived EXP-038 CSV.
    gated = {name: parts_ for name, parts_ in candidates.items() if name != "C2"}
    systems = ["R0_fixed", "R0_corrected", *arms, *[f"{c}_mix" for c in candidates], *candidates]
    report, paintings, gates = science.p2_report(pop["gold"], frame, systems, gated, ctx.config)
    pieces = read(ctx.root / "status.json")["pieces"]
    gates |= plan39.unavailable(ctx.config, candidates, pieces, ctx.e5)
    selected = plan39.select({k: v for k, v in gates.items() if k in gated}, ctx.config, ctx.gpu_seconds)
    diagnostics, languages, painting_deltas = science.p2_diagnostics(pop["gold"], frame, arms, candidates, ctx.config)
    diagnostics["vs_E3"] = versus_e3(pop["gold"], frame, arms, ctx.config)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for name, table in (("oof.csv", frame), ("components.csv", parts)):
        table.to_csv(folder / name, index=False, lineterminator="\n", float_format="%.6f")
    paintings.to_csv(folder / "painting_contrast.csv", index=False, lineterminator="\n")
    painting_deltas.to_csv(folder / "painting_deltas.csv", index=False, lineterminator="\n")
    languages.to_csv(folder / "language_deltas.csv", index=False, lineterminator="\n")
    write(folder / "metrics.json", {"report": report, "replay": replay, "c2_replay": c2_checks,
                                    "fold_receipts": receipts, "arms": list(arms),
                                    "candidates": {k: list(v) for k, v in candidates.items()},
                                    "gate_reference": "on-platform C2 recomputed in this run from the sealed "
                                                      "EXP-038 features; the archived CSV is replay-only",
                                    "seconds": time.time() - tick})
    write(folder / "diagnostics.json", diagnostics)
    write(folder / "gates.json", {"gates": gates, "selected": selected, "decision": "GO" if selected else "NO_GO",
                                  "gpu_seconds": ctx.gpu_seconds})
    seal(folder, ctx.identity)
    return read(folder / "gates.json")


def versus_e3(gold, frame, arms, config):
    from st3.exp037_diagnostics import pair_diversity, paired_bootstrap
    from st3.metrics import painting_scores

    joined = gold[data.KEYS + ["fold", "k", "n"]].merge(frame.drop(columns="fold"), on=data.KEYS, validate="one_to_one")
    joined["p"] = joined.k / joined.n
    out = {}
    for scope, block in science.scopes(joined, config["test_like_excluded"]):
        scored = painting_scores(block, prediction="E3")
        eligible = set(scored.loc[scored.status.eq("scored"), "painting"])
        for arm in arms:
            if arm in plan39.SEALED:
                continue
            table = science.painting_contrasts(block, [arm, "E3"])
            pair = pair_diversity(block, ["E3", arm], eligible)[0]
            out.setdefault(scope, {})[arm] = {
                "minus_E3": paired_bootstrap(table[arm], table["E3"], seed=config["diagnostic_seed"]),
                "centered_residual_r_vs_E3": pair["centered_residual_r"]}
    return out


def stage_development(ctx, status, gates):
    folder = ctx.root / "development"
    if completed(folder, ctx.identity, ["scores.json", "predictions.csv"]):
        return read(folder / "scores.json")
    pop = r38.population(ctx)
    dev, arms = pop["dev"], available(ctx)
    candidates = plan39.candidate_set(ctx.config, arms, ctx.e5)
    features = features_for(ctx, arms)
    frame, heads, banks, info = c2_development(ctx, features, arms, candidates)
    replay = r38.replay_development(ctx, frame)
    c2_checks = c2_development_checks(ctx, frame)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    frame.to_csv(folder / "predictions.csv", index=False, lineterminator="\n", float_format="%.6f")
    cscva3, c6 = components(ctx)
    standalone = {name: science.system_frame(frame, name) for name in
                  ["R0_fixed", "R0_corrected", *arms, *[f"{c}_mix" for c in candidates], *candidates]}
    composed = science.blends({name: standalone[name] for name in candidates}, {"cscva3": cscva3}, c6, dev, ctx.config)
    diagnostic = {f"{name}_plus_C6": science.mix_pair(standalone[name], c6, ctx.config["blends"]["B2_c6_weight"], dev)
                  for name in candidates}
    archives = {}
    for name, prediction in (standalone | composed).items():
        archive = r38.write_submission(science.canonical(prediction, dev), dev, folder / "submissions",
                                       f"{ctx.name}_{name}_submission")
        archives[name] = {"zip": archive.relative_to(ctx.root).as_posix(), "sha256": sha(archive),
                          "eligibility": science.eligibility(name, ctx.config)}
    gold = pd.read_csv(ctx.base / "scoring/dev_gold.csv")
    canonical = {name: submission.validate_archive(ctx.root / item["zip"], dev) for name, item in archives.items()}
    scores = science.development_scores(gold, canonical, ctx.config["test_like_excluded"])
    diagnostic_scores = science.development_scores(gold, {k: science.canonical(v, dev) for k, v in diagnostic.items()},
                                                   ctx.config["test_like_excluded"])
    intervals = science.development_intervals(gold, {k: v for k, v in canonical.items()
                                                     if k.startswith("B2_D")},
                                              {"B2_C2_cscva3": canonical["B2_C2_cscva3"]},
                                              ctx.config["test_like_excluded"], ctx.config["diagnostic_seed"])
    r38.save_bundle(folder / "bundle", heads, banks, features, sorted(pop["train"].painting.unique()), candidates, ctx)
    write(folder / "scores.json", {"scores": scores, "diagnostic_not_selectable": diagnostic_scores,
                                   "intervals": intervals, "archives": archives, "replay": replay,
                                   "c2_replay": c2_checks, "fit": info,
                                   "development_selection": "none; Development is reporting only"})
    seal(folder, ctx.identity)
    return read(folder / "scores.json")


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
        name = "EXP038_incumbent"
        source = carried(ctx, "incumbent_submission.zip")
        recipe = {"decision": "NO_GO", "selected_candidate": None, "submission": name,
                  "formula": "byte-identical EXP-038 incumbent B2_C2_cscva3",
                  "note": "No EXP-039 candidate passed the frozen P2 gate against on-platform C2."}
    copy_verified(source, best)
    submission.validate_archive(best, dev)
    if not selected and sha(best) != ctx.refs["exp038_carried_sha256"]["incumbent_zip"]:
        raise ValueError("NO_GO best ZIP is not byte-identical to the EXP-038 incumbent")
    unavailable = {k: g["reason"] for k, g in gates["gates"].items() if g.get("status") == "NOT_AVAILABLE"}
    selection = recipe | {"coefficients": ctx.config["blends"] | ctx.config["mix"], "best_sha256": sha(best),
                          "components_sha256": {"CSCVA_3": ctx.refs["exp038_carried_sha256"]["cscva3_zip"],
                                                "C6": ctx.refs["exp037_c6_standalone_csv_sha256"]},
                          "eligibility": ctx.config["eligibility"]["unconfirmed_label"],
                          "development_used_for_selection": False, "unavailable_candidates": unavailable,
                          "e5_label": ctx.e5}
    decision = {"decision": recipe["decision"], "selected": selected, "gates": gates["gates"],
                "unavailable_candidates": unavailable, "gpu_seconds": gates.get("gpu_seconds"),
                "selection_rule": "highest filtered P2 Contrast among passing; within 0.001 the simpler "
                                  "(D2 = D3 < D1 < D4), same rank by lower GPU cost",
                "pieces": read(ctx.root / "status.json")["pieces"]}
    write(ctx.root / "decision.json", decision)
    write(ctx.root / "selection.json", selection)
    return decision, selection


def run_pipeline(ctx, backend):
    ctx.root.mkdir(parents=True, exist_ok=True)
    if (ctx.output / f"{ctx.name}_SUCCESS.json").exists():
        (ctx.output / f"{ctx.name}_SUCCESS.json").unlink()
    status = r38.Status(ctx)
    outcome = None
    try:
        status.begin(STAGES[0])
        runtime = backend.runtime()
        ctx.e5 = "E5" if runtime.get("gpu_count", 1) >= 2 else "E5_nf4"
        r38.stage_setup(ctx, backend, status)
        extra = {"e5_label": ctx.e5, "gpu_count": runtime.get("gpu_count", 1)}
        if (ctx.root / "run39.json").is_file() and read(ctx.root / "run39.json") != extra:
            raise r38.BlockedCache("Resume requires the same GPU count and E5 placement")
        write(ctx.root / "run39.json", extra)
        ctx.identity |= extra
        status.done(STAGES[0])
        status.begin(STAGES[1])
        validate = stage_validate(ctx, backend, status)
        status.done(STAGES[1])
        r38.progress(ctx)
        status.begin(STAGES[2])
        bench = stage_benchmark(ctx, backend, status)
        plan = plan_budget(forecast(ctx, bench, validate), ctx.clock.remaining(), ctx.config["budget"])
        write(ctx.root / "budget_plan.json", plan)
        status.done(STAGES[2])
        status.begin(STAGES[3])
        stage_features(ctx, backend, status, plan, bench)
        status.done(STAGES[3])
        status.begin(STAGES[4])
        gates = stage_p2(ctx, status)
        status.done(STAGES[4])
        r38.progress(ctx)
        status.begin(STAGES[5])
        scores = stage_development(ctx, status, gates)
        status.done(STAGES[5])
        status.begin(STAGES[6])
        decision, selection = stage_submissions(ctx, gates, scores)
        status.done(STAGES[6])
        status.begin(STAGES[7])
        status.state["state"] = "EXPORTING"
        status.save()
        digests = r38.stage_export(ctx, scores, decision)
        status.done(STAGES[7])
        status.begin(STAGES[8])
        status.state |= {"state": "COMPLETE", "decision": decision["decision"], "selected": decision["selected"]}
        status.save()
        write(ctx.output / f"{ctx.name}_SUCCESS.json", {"decision": decision["decision"], "selected": decision["selected"],
                                                        "best_sha256": selection["best_sha256"], "exports": digests})
        status.done(STAGES[8])
        outcome = "COMPLETE"
        return decision
    except BaseException as error:
        resumable = isinstance(error, TimeoutError) and not isinstance(error, BudgetStop)
        status.state |= {"state": "RESUMABLE_STOP" if resumable else "FAILED",
                         "reason": f"{type(error).__name__}: {error}",
                         "failures": getattr(error, "failures", None)}
        status.save()
        try:
            r38.progress(ctx)
        finally:
            r38.diagnostic(ctx, status, error)
        raise
    finally:
        if outcome != "COMPLETE" and (ctx.output / f"{ctx.name}_SUCCESS.json").exists():
            (ctx.output / f"{ctx.name}_SUCCESS.json").unlink()


# GPU backend and children.

class Backend(r38.SubprocessBackend):
    def runtime(self):
        result = super().runtime()
        probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        return result | {"gpu_count": len([line for line in probe.stdout.splitlines() if line.startswith("GPU ")])}

    def fetch(self, arm):
        from st3.exp038_gpu import fetch_snapshot

        key = {"E3": "E3", "E5": "E5"}[arm]
        path, receipt = fetch_snapshot(self.ctx.config["arms"][key], self.ctx.snapshots.get(key))
        self.ctx.models[key] = path
        return {k: receipt[k] for k in ("model", "revision")}

    def _env(self, arm):
        two = spec(self.ctx, arm).get("placement") == "two_gpu"
        return os.environ | {"CUDA_VISIBLE_DEVICES": "0,1" if two else "0"}

    def _files(self, arm, items):
        folder = self.ctx.root / "tmp_requests"
        folder.mkdir(exist_ok=True)
        (folder / f"{arm}_items.json").write_text(json.dumps(items), encoding="utf-8")
        (folder / f"{arm}_spec.json").write_text(json.dumps(spec(self.ctx, arm)), encoding="utf-8")
        return folder / f"{arm}_items.json", folder / f"{arm}_spec.json"

    def snapshot(self, arm):
        return self.ctx.models["E5" if plan39.config_arm(arm) == "E5" else "E3"]

    def benchmark(self, arm, items):
        item_file, spec_file = self._files(arm, items)
        out = self.ctx.root / "tmp_requests" / f"bench_{arm}.json"
        r38.run_child(["gpu-bench", "--arm", arm, "--items", item_file, "--spec", spec_file, "--snapshot",
                       self.snapshot(arm), "--out", out], self.ctx.clock.work_deadline(), self.ctx, env=self._env(arm))
        return read(out)

    def extract(self, arm, items, folder, identity, deadline, cap_deadline=None):
        item_file, spec_file = self._files(arm, items)
        ident = self.ctx.root / "tmp_requests" / f"{arm}_identity.json"
        ident.write_text(json.dumps(identity), encoding="utf-8")
        r38.run_child(["gpu-extract", "--arm", arm, "--items", item_file, "--spec", spec_file, "--snapshot",
                       self.snapshot(arm), "--folder", folder, "--identity", ident, "--deadline", deadline,
                       "--progress-root", self.ctx.root, "--handoff", self.ctx.overlay_sha,
                       "--interval", self.ctx.config["budget"]["progress_interval_seconds"],
                       "--chunk", self.ctx.config["chunk_items"]], deadline, self.ctx, env=self._env(arm))


def child_device(arm_spec):
    import torch

    count = torch.cuda.device_count()
    if arm_spec.get("placement") == "two_gpu" and count < 2:
        raise RuntimeError("E5 two-GPU placement needs two visible GPUs")
    if arm_spec.get("placement") != "two_gpu" and count != 1:
        raise RuntimeError("Single-GPU arms need exactly one visible GPU")
    return torch.device("cuda:0")


def gpu_bench(args):
    from st3 import exp038_gpu as gpu

    arm_spec = json.loads(args.spec.read_text(encoding="utf-8"))
    device = child_device(arm_spec)
    gpu.deterministic_cuda()
    tick = time.time()
    embed = gpu.encoder(args.arm, arm_spec, args.snapshot, device)
    loaded = time.time() - tick
    items = [tuple(i) for i in json.loads(args.items.read_text(encoding="utf-8"))]
    write(args.out, gpu.benchmark(args.arm, arm_spec, embed, items, device) | {
        "load_seconds": loaded, "placement": arm_spec.get("placement"),
        "loader_missing_keys": getattr(embed, "loader_missing_keys", None),
        "ignored_missing_keys": getattr(embed, "ignored_missing_keys", None)})


def gpu_extract(args):
    from st3 import exp038_gpu as gpu
    from st3.exp038_io import exportable, publish

    arm_spec = json.loads(args.spec.read_text(encoding="utf-8"))
    device = child_device(arm_spec)
    gpu.deterministic_cuda()
    embed = gpu.encoder(args.arm, arm_spec, args.snapshot, device)
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    items = [tuple(i) for i in json.loads(args.items.read_text(encoding="utf-8"))]
    last = [time.time()]

    def on_chunk(arm, done, total, seconds):
        print(f"Feature extraction {arm}: {done}/{total} items, {seconds:.0f}s elapsed", flush=True)
        if time.time() - last[0] >= args.interval:
            last[0] = time.time()
            try:
                publish(args.progress_root.parent / "exp039_progress.zip", args.progress_root,
                        list(exportable(args.progress_root)), args.handoff)
            except Exception as error:
                print(f"Progress export failed: {error}", flush=True)

    try:
        gpu.extract(args.arm, arm_spec, embed, items, args.folder, identity, args.chunk, args.deadline, on_chunk)
    except BudgetStop:
        sys.exit(r38.EXIT_BUDGET)
    except TimeoutError:
        sys.exit(r38.EXIT_TIMEOUT)


def cpu_validate(args):
    """Every CPU integrity check of Validate (no downloads, no GPU), for Windows and Linux preflight."""
    args.output.mkdir(parents=True, exist_ok=True)
    overlay = restore_overlay(args.overlay, args.overlay_sha, args.base_sha, args.output / "tmp_overlay_restored",
                              manifest_name=MANIFEST)
    ctx = Context(base=args.base, overlay=overlay, root=args.output, output=args.output, base_sha=args.base_sha,
                  overlay_sha=args.overlay_sha, clock=Clock(time.time(), 10, 0))
    ctx.identity = {"cpu_validate": True}

    class Offline:
        def fetch(self, arm):
            return {"skipped": "cpu-validate does not download snapshots"}

    receipts, failures = collect([
        ("pin_formats", lambda: check_config_pins(ctx.config), ()),
        ("inputs", lambda: len(r38.population(ctx)["gold"]), ()),
        ("sealed_exp038", lambda: {arm: len(sealed_feature(ctx, arm)[0]) for arm in plan39.SEALED}, ("inputs",)),
        ("key_audit", lambda: key_audit(ctx), ("inputs",)),
        ("r0_replay", lambda: {"p2": r38.r0_p2(ctx), "development": r38.r0_development(ctx)}, ("inputs",)),
        ("c2_replay", lambda: _c2_only(ctx), ("inputs", "sealed_exp038")),
    ])
    checks = []
    for name in ("r0_replay", "c2_replay"):
        result = receipts.get(name, {}).get("result") or {}
        for part in (result.get("p2", {}), result.get("development", {}), result):
            checks += part.get("checks", []) if isinstance(part, dict) else []
    summary = {"status": "FAIL" if failures else "PASS", "failures": failures, "checks": len(checks),
               "key_audit": (receipts.get("key_audit") or {}).get("result"),
               "exact": bool(checks) and all(c["exact"] for c in checks),
               "max_ratio_to_threshold": max((c["max_abs_difference"] / c["threshold"] for c in checks), default=None),
               "platform": r38.platform_receipt(ctx)}
    write(args.output / "cpu_validate.json", {"summary": summary, "receipts": receipts})
    print(json.dumps(summary, indent=1))
    return 1 if failures else 0


def _c2_only(ctx):
    frame, parts, r0, receipts, used, features = c2_p2(ctx)
    checks = c2_p2_checks(ctx, frame)
    dev_frame, *_ = c2_development(ctx, features)
    checks += c2_development_checks(ctx, dev_frame)
    return {"checks": checks, "exact": all(c["exact"] for c in checks)}


def run(args):
    clock = Clock(args.started, args.hours, configs()[0]["budget"]["persistence_reserve_seconds"])
    snapshots = {arm: getattr(args, f"{arm.lower()}_snapshot") for arm in ("E3", "E5")
                 if getattr(args, f"{arm.lower()}_snapshot")}
    ctx = Context(base=args.base, overlay=args.overlay, root=args.output / "exp039_run", output=args.output,
                  base_sha=args.base_sha, overlay_sha=args.overlay_sha, clock=clock, snapshots=snapshots)
    if args.resume:
        ctx.root.mkdir(parents=True, exist_ok=True)
        if any(ctx.root.iterdir()):
            raise ValueError("Resume needs a fresh /kaggle/working/exp039_run")
        restore(args.resume, ctx.root, args.overlay_sha)
    run_pipeline(ctx, Backend(ctx))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("cpu-validate")
    p = sub.add_parser("run")
    for command in (v, p):
        for name in ("base", "overlay", "output"):
            command.add_argument(f"--{name}", type=Path, required=True)
        for name in ("base-sha", "overlay-sha"):
            command.add_argument(f"--{name}", required=True)
    p.add_argument("--started", type=float, required=True)
    p.add_argument("--hours", type=float, default=10.5)
    p.add_argument("--resume", type=Path)
    for arm in ("e3", "e5"):
        p.add_argument(f"--{arm}-snapshot", type=Path)
    b, e = sub.add_parser("gpu-bench"), sub.add_parser("gpu-extract")
    for command in (b, e):
        command.add_argument("--arm", required=True)
        for name in ("items", "spec", "snapshot"):
            command.add_argument(f"--{name}", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    for name in ("folder", "identity", "progress-root"):
        e.add_argument(f"--{name}", type=Path, required=True)
    e.add_argument("--deadline", type=float, required=True)
    e.add_argument("--handoff", required=True)
    e.add_argument("--interval", type=float, default=900)
    e.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()
    result = {"run": run, "cpu-validate": cpu_validate, "gpu-bench": gpu_bench, "gpu-extract": gpu_extract}[args.command](args)
    if isinstance(result, int):
        sys.exit(result)


if __name__ == "__main__":
    main()
