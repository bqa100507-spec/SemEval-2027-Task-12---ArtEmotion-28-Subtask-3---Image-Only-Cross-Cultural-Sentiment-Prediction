#!/usr/bin/env python3
"""EXP-038 worker.

Stages: Setup -> Validate -> Benchmark -> Feature extraction -> CS-CVA seeds -> P2 evaluation ->
Full refit and Development scoring -> Submission validation -> Export -> Release.

The orchestrator itself is CPU-only. GPU work runs in child processes of this script (``gpu-*``
subcommands) through a backend object, which tests replace with a deterministic mock.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd

from st3 import data, submission
from st3 import exp038 as science
from st3.competition_stack import ENCODERS
from st3.exp036_io import read, restore, sha, write
from st3.exp038_io import (BlockedCache, BudgetStop, Clock, artwork, base_features, completed, copy_verified,
                           exportable, load_features, p2_population, painting_images, plan_budget, publish, seal,
                           training_population, verify_both, verify_seal, write_submission, zip_member)

STAGES = ("Setup", "Validate", "Benchmark", "Feature extraction", "CS-CVA seeds", "P2 evaluation",
          "Full refit and Development scoring", "Submission validation", "Export", "Release")
EXIT_BUDGET, EXIT_TIMEOUT = 75, 76
OPTIONAL_ARMS = ("E3", "E4")  # A Qwen failure must not discard the VLM-free C1.
PACKAGES = ("torch", "torchvision", "transformers", "numpy", "scipy", "pandas", "safetensors", "Pillow",
            "huggingface-hub", "qwen-vl-utils", "threadpoolctl")


def configs():
    return (read(ROOT / "configs/exp038.json"), read(ROOT / "configs/competition_track1.json"),
            read(ROOT / "configs/competition_neighborhood_residual.json"))


class Context:
    def __init__(self, *, base, overlay, root, output, base_sha, overlay_sha, clock, snapshots=None,
                 config=None, track1=None, neighborhood=None, echo=True):
        default = configs()
        self.config, self.track1, self.neighborhood = (config or default[0], track1 or default[1],
                                                       neighborhood or default[2])
        self.refs = self.config["references"]
        self.base, self.overlay, self.root, self.output = Path(base), Path(overlay), Path(root), Path(output)
        self.base_sha, self.overlay_sha, self.clock = base_sha, overlay_sha, clock
        self.snapshots = snapshots or {}
        self.echo = echo
        self.identity = None
        self.models = {}
        self.cache = {}

    def log(self, message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        if self.echo:
            print(line, flush=True)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "events.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


class Status:
    def __init__(self, ctx):
        self.ctx = ctx
        self.path = ctx.root / "status.json"
        self.state = read(self.path) if self.path.is_file() else {}
        self.state |= {"state": "IN_PROGRESS", "reason": None}
        self.state.setdefault("completed_stages", [])
        self.state.setdefault("pieces", {})
        self.save()

    def save(self):
        clock = self.ctx.clock
        self.state |= {"elapsed_seconds": round(time.time() - clock.started, 1),
                       "remaining_seconds": round(clock.deadline - time.time(), 1)}
        write(self.path, self.state)

    def begin(self, stage, item=None):
        self.state |= {"stage": stage, "item": item}
        self.save()
        self.ctx.log(f"{stage}{' / ' + item if item else ''}: elapsed {self.state['elapsed_seconds']:.0f}s, "
                     f"remaining {self.state['remaining_seconds']:.0f}s")

    def item(self, item):
        self.begin(self.state["stage"], item)

    def done(self, stage):
        if stage not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage)
        self.save()

    def piece(self, name, state, reason=None):
        self.state["pieces"][name] = {"state": state, "reason": reason}
        self.save()
        self.ctx.log(f"{name}: {state}{' (' + reason + ')' if reason else ''}")


# Data.

def population(ctx):
    if "population" not in ctx.cache:
        base_manifest, overlay_manifest = verify_both(ctx.base, ctx.overlay, ctx.base_sha, ctx.overlay_sha)
        if overlay_manifest["references"] != ctx.refs:
            raise ValueError("Overlay reference pins differ from configs/exp038.json")
        train, dev = training_population(ctx.base, ctx.refs["training_pair_sha256"])
        paintings = painting_images(ctx.base, base_manifest, train, dev)
        gold, protocol = p2_population(ctx.overlay, train, ctx.refs)
        features = base_features(ctx.overlay, paintings, ctx.track1["feature_widths"])
        ctx.cache["population"] = {"base_manifest": base_manifest, "overlay_manifest": overlay_manifest,
                                   "train": train, "dev": dev, "paintings": paintings, "gold": gold,
                                   "protocol": protocol, "features": features}
    return ctx.cache["population"]


def image_items(ctx):
    pop = population(ctx)
    rows = pop["paintings"].drop_duplicates("image_sha256").sort_values("image_sha256")
    return [(digest, str(artwork(ctx.base, pop["base_manifest"], image)[0]))
            for digest, image in zip(rows.image_sha256, rows.image)]


def pair_items(ctx):
    pop = population(ctx)
    keys = pd.concat([pop["train"][data.KEYS], pop["dev"][data.KEYS]], ignore_index=True)
    digest = pop["paintings"].set_index("painting").image_sha256
    path = pop["paintings"].drop_duplicates("image_sha256").set_index("image_sha256").image
    frame = keys.assign(image_sha256=keys.painting.map(digest))[["image_sha256", "language"]]
    frame = frame.drop_duplicates().sort_values(["image_sha256", "language"], ignore_index=True)
    return [(f"{d}|{language}", str(artwork(ctx.base, pop["base_manifest"], path[d])[0]), language)
            for d, language in zip(frame.image_sha256, frame.language)]


def arm_frame(ctx, values, items):
    pop = population(ctx)
    row = {item[0]: i for i, item in enumerate(items)}
    positions = pop["paintings"].image_sha256.map(row).to_numpy()
    return pd.DataFrame(values[positions], index=pop["paintings"].painting.to_numpy())


def pair_delta(ctx, values, items, neutral):
    pop = population(ctx)
    keys = pd.concat([pop["train"][data.KEYS], pop["dev"][data.KEYS]], ignore_index=True)
    digest = pop["paintings"].set_index("painting").image_sha256
    row = {item[0]: i for i, item in enumerate(items)}
    positions = [row[f"{digest[p]}|{language}"] for p, language in keys.itertuples(index=False, name=None)]
    return science.PairDelta.build(keys, values[positions], neutral)


def arm_identity(ctx, arm):
    spec = ctx.config["arms"][arm]
    if arm == "E4":
        spec = ctx.config["arms"]["E3"] | spec
    return ctx.identity | {"arm": arm, "spec": spec, "chunk_items": ctx.config["chunk_items"]}


def features_with_arms(ctx, arms):
    pop = population(ctx)
    features = dict(pop["features"])
    delta = None
    for arm in arms:
        if arm == "E4":
            continue
        items = image_items(ctx)
        values = load_features(ctx.root / "features" / arm, arm_identity(ctx, arm), [i[0] for i in items])
        features[arm] = arm_frame(ctx, values, items)
    if "E4" in arms:
        items = pair_items(ctx)
        values = load_features(ctx.root / "features/E4", arm_identity(ctx, "E4"), [i[0] for i in items])
        delta = pair_delta(ctx, values, items, features["E3"])
    return features, delta


# GPU backends.

def run_child(arguments, deadline, ctx):
    """Run one GPU child with live output; kill it 60 s after its deadline."""
    from threading import Thread

    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Session allowance reached before a GPU stage")
    log = ctx.root / "gpu_children.log"
    with log.open("a", encoding="utf-8") as stream, subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__)), *map(str, arguments)], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", start_new_session=True) as child:
        def relay():
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)

        reader = Thread(target=relay, daemon=True)
        reader.start()
        try:
            code = child.wait(timeout=max(1., remaining + 60))
        except subprocess.TimeoutExpired:
            code = None
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            reader.join(timeout=30)
    if code == EXIT_BUDGET:
        raise BudgetStop(f"{arguments[0]} reached its cap")
    if code == EXIT_TIMEOUT or code is None:
        raise TimeoutError(f"{arguments[0]} stopped at the session allowance")
    if code:
        raise RuntimeError(f"{arguments[0]} exited with status {code}; see gpu_children.log")


class SubprocessBackend:
    def __init__(self, ctx):
        self.ctx = ctx

    def runtime(self):
        from importlib.metadata import version

        probe = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                               text=True, check=True)
        names = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        return {"python": platform.python_version(), "gpu": names[0],
                "packages": {name: version(name) for name in PACKAGES}}

    def fetch_models(self):
        from st3.exp038_gpu import fetch_snapshot

        arms = self.ctx.config["arms"]
        receipts = {}
        for arm in ("E1", "E2", "E3"):
            self.ctx.log(f"Fetching and hashing pinned {arms[arm]['model']}@{arms[arm]['revision'][:12]}")
            path, receipts[arm] = fetch_snapshot(arms[arm], self.ctx.snapshots.get(arm))
            self.ctx.models[arm] = path
        self.ctx.models["E4"] = self.ctx.models["E3"]
        return receipts

    def _items_file(self, name, items):
        path = self.ctx.root / "tmp_requests" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(items), encoding="utf-8")
        return path

    def benchmark(self, arm, items):
        out = self.ctx.root / "tmp_requests" / f"bench_{arm}.json"
        run_child(["gpu-bench", "--arm", arm, "--items", self._items_file(f"bench_{arm}", items),
                   "--snapshot", self.ctx.models[arm], "--out", out], self.ctx.clock.work_deadline(), self.ctx)
        return read(out)

    def extract(self, arm, items, folder, identity, deadline, cap_deadline=None):
        ident = self.ctx.root / "tmp_requests" / f"identity_{arm}.json"
        ident.write_text(json.dumps(identity), encoding="utf-8")
        arguments = ["gpu-extract", "--arm", arm, "--items", self._items_file(f"extract_{arm}", items),
                     "--snapshot", self.ctx.models[arm], "--folder", folder, "--identity", ident,
                     "--deadline", deadline, "--progress-root", self.ctx.root, "--handoff", self.ctx.overlay_sha,
                     "--interval", self.ctx.config["budget"]["progress_interval_seconds"]]
        if cap_deadline is not None:
            arguments += ["--cap", cap_deadline]
        run_child(arguments, min(deadline, cap_deadline or deadline), self.ctx)

    def cscva(self, seed, folder, deadline, preflight=False):
        out = self.ctx.root / "tmp_requests" / f"cscva_{seed}_{'preflight' if preflight else 'fit'}.json"
        out.parent.mkdir(exist_ok=True)
        arguments = ["gpu-cscva", "--seed", seed, "--folder", folder, "--snapshot", self.ctx.models["E1"],
                     "--base", self.ctx.base, "--base-sha", self.ctx.base_sha, "--deadline", deadline,
                     "--out", out, "--source", self.ctx.identity["source"]]
        if preflight:
            arguments.append("--preflight")
        run_child(arguments, deadline, self.ctx)
        return read(out)


# Stages.

def stage_setup(ctx, backend, status):
    runtime = backend.runtime()
    manifest = read(ctx.overlay / "exp038_manifest.json")
    identity = {"overlay_manifest_sha256": ctx.overlay_sha, "base_manifest_sha256": ctx.base_sha,
                "source": manifest["source"], "python": runtime["python"], "gpu": runtime["gpu"],
                "packages": runtime["packages"], "config_sha256": sha(ROOT / "configs/exp038.json")}
    run = ctx.root / "run.json"
    if run.is_file() and read(run) != identity:
        raise BlockedCache("Resume requires the original source, inputs, runtime and GPU type")
    write(run, identity)
    ctx.identity = identity
    expected = ctx.config["runtime"]
    if runtime["python"] != expected["python"] or runtime["packages"]["torch"] != expected["torch"] \
            or runtime["packages"]["transformers"] != expected["transformers"] \
            or runtime["packages"]["qwen-vl-utils"] != expected["qwen_vl_utils"]:
        raise RuntimeError(f"Pinned runtime differs: {runtime}")
    write(ctx.root / "configs_used.json", {"exp038": ctx.config, "track1": ctx.track1,
                                          "neighborhood": ctx.neighborhood})


def stage_validate(ctx, backend, status):
    folder = ctx.root / "validate"
    pop = population(ctx)
    receipts = {"models": backend.fetch_models()}
    if completed(folder, ctx.identity, ["validate.json"]):
        ctx.log("Validate: sealed replay receipts verified; reusing")
        return read(folder / "validate.json")
    started = time.time()
    status.item("inputs")
    receipts |= {"inputs": {"base_files": len(pop["base_manifest"]["files"]),
                            "overlay_files": len(pop["overlay_manifest"]["files"]),
                            "train_pairs": len(pop["train"]), "train_paintings": int(pop["train"].painting.nunique()),
                            "dev_pairs": len(pop["dev"]), "p2_pairs": len(pop["gold"]),
                            "p2_paintings": int(pop["gold"].painting.nunique()),
                            "images_present": int(pop["paintings"].image_sha256.nunique())}}
    status.item("R0 P2 replay")
    tick = time.time()
    r0 = r0_p2(ctx)
    receipts["r0_p2"] = r0 | {"seconds": time.time() - tick}
    status.item("R0 Development replay")
    receipts["r0_development"] = r0_development(ctx)
    receipts["seconds"] = time.time() - started
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    write(folder / "validate.json", receipts)
    seal(folder, ctx.identity)
    return receipts


def reference_frame(ctx, name):
    return pd.read_csv(ctx.overlay / "reference" / name)


def r0_p2(ctx):
    pop = population(ctx)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=ctx.config["runtime"]["ridge_threads"]):
        frame, parts, r0, receipts, _ = science.evaluate_p2(
            pop["gold"], pop["features"], None, (), ctx.config, ctx.track1, ctx.neighborhood,
            ctx.refs["track1_inner_oof_sha256"], log=ctx.log)
    metrics = (read(ctx.overlay / "reference/track1_metrics.json"),
               read(ctx.overlay / "reference/corrected_metrics.json"))
    return science.replay_r0_p2(pop["gold"], parts, r0, reference_frame(ctx, "track1_oof.csv"),
                                ctx.refs["corrected_oof_sha256"], metrics, ctx.config["test_like_excluded"])


def compare_bank(bank, path):
    with np.load(path, allow_pickle=False) as saved:
        same = (saved["paintings"].tolist() == bank.visual.index.tolist()
                and saved["languages"].tolist() == bank.votes.columns.tolist()
                and np.array_equal(saved["numerator"], bank.numerator.to_numpy())
                and np.array_equal(saved["votes"], bank.votes.to_numpy())
                and np.array_equal(saved["visual"], bank.visual.to_numpy()))
    return same


def compare_heads(heads, path):
    with np.load(path, allow_pickle=False) as saved:
        out = {}
        for encoder in ENCODERS:
            head = heads[encoder]
            pairs = {field: (np.asarray(getattr(head, field), dtype=float), saved[f"{encoder}:{field}"])
                     for field in ("intercept", "language_intercepts", "shared", "deviations")}
            out[encoder] = {"bitwise": all(np.array_equal(a, b) for a, b in pairs.values()),
                            "max_abs_difference": max(float(np.abs(a - b).max()) for a, b in pairs.values())}
    return out


def r0_development(ctx):
    pop = population(ctx)
    from threadpoolctl import threadpool_limits

    dev = pop["dev"]
    with threadpool_limits(limits=ctx.config["runtime"]["ridge_threads"]):
        heads, bank_set, info, visual = science.deployment_fit(pop["train"], pop["features"], None, (), ctx.track1,
                                                               ctx.neighborhood, {})
        frame, _ = science.development_predictions(heads, bank_set, visual, dev[data.KEYS], pop["features"], None,
                                                   {}, ctx.neighborhood)
    check = replay_development(ctx, frame)
    check["bank_identical"] = compare_bank(bank_set["R0"], ctx.overlay / "reference/corrected_bundle/bank.npz")
    if not check["bank_identical"]:
        raise science.ReplayError("The refitted Corrected Ridge deployment bank differs from the saved bank")
    check["heads"] = compare_heads(heads, ctx.overlay / "reference/fixed_bundle/heads.npz")
    return check


def replay_development(ctx, frame):
    """Refitted R0 Development predictions and the two published compositions, byte for byte."""
    pop = population(ctx)
    dev = pop["dev"]
    reference = ctx.overlay / "reference"
    corrected = science.system_frame(frame, "R0_corrected")
    if science.csv_text(corrected).encode() != zip_member(reference / "corrected_ridge_submission.zip"):
        raise science.ReplayError("Refitted Corrected Ridge Development predictions differ from the archived ZIP")
    scratch = ctx.root / "tmp_replay"
    shutil.rmtree(scratch, ignore_errors=True)
    fixed = write_submission(science.system_frame(frame, "R0_fixed"), dev, scratch, "fixed")
    if zip_member(fixed) != zip_member(reference / "fixed_ridge_submission.zip"):
        raise science.ReplayError("Refitted Fixed Ridge Development predictions differ from the archived ZIP")
    cscva = submission.validate_archive(reference / "exp036_standalone_submission.zip", dev)
    c6 = submission.read_predictions(reference / "exp037_c6_standalone" / "predictions.csv", dev)
    composed = science.blends({"R0": corrected}, {"original": cscva}, c6, dev, ctx.config)
    for name, target in (("B1_R0_original", "exp036_ensemble_submission.zip"),
                         ("B2_R0_original", "exp037_best_submission.zip")):
        archive = write_submission(composed[name], dev, scratch, name)
        if zip_member(archive) != zip_member(reference / target):
            raise science.ReplayError(f"{name} does not reproduce {target}")
    shutil.rmtree(scratch, ignore_errors=True)
    return {"corrected_member_identical": True, "fixed_member_identical": True,
            "exp036_ensemble_identical": True, "exp037_original_blend_identical": True}


def benchmark_items(items, count, seed):
    if len(items) <= count:
        return list(items)
    chosen = np.sort(np.random.default_rng(seed).choice(len(items), size=count, replace=False))
    return [items[i] for i in chosen]


def stage_benchmark(ctx, backend, status):
    folder = ctx.root / "benchmark"
    if not completed(folder, ctx.identity, ["benchmark.json"]):
        result = {"arms": {}, "method": (
            "Two passes over 32 seeded real items per encoder on the first GPU; the warm second pass gives "
            "items/s. Extraction forecast = items / rate + measured model load. CS-CVA forecast = EXP-036's "
            "own two-batch preflight rate over 5 epochs of 1,571 paintings plus Development inference. CPU "
            "forecast scales a timed real ridge fit by encoder width and fit count.")}
        images, pairs = image_items(ctx), pair_items(ctx)
        for arm in ("E1", "E2", "E3", "E4"):
            status.item(arm)
            sample = benchmark_items(pairs if arm == "E4" else images, 32, ctx.config["diagnostic_seed"])
            if arm == "E4" and "failed" in result["arms"].get("E3", {}):
                result["arms"][arm] = {"skipped_dependency": "E3 benchmark failed"}
                continue
            try:
                result["arms"][arm] = backend.benchmark(arm, sample)
                if not result["arms"][arm]["deterministic_within_1e-3"]:
                    raise RuntimeError(f"{arm} failed the two-pass determinism check: {result['arms'][arm]}")
            except Exception as error:
                if arm not in OPTIONAL_ARMS:
                    raise
                result["arms"][arm] = {"failed": f"{type(error).__name__}: {error}"}
                ctx.log(f"Benchmark {arm}: FAILED_OPTIONAL ({result['arms'][arm]['failed']})")
        status.item("CS-CVA preflight")
        result["cscva"] = backend.cscva(ctx.config["cscva"]["new_seeds"][0], ctx.root / "tmp_cscva_preflight",
                                        ctx.clock.work_deadline(), preflight=True)
        shutil.rmtree(ctx.root / "tmp_cscva_preflight", ignore_errors=True)
        status.item("CPU ridge timing")
        result["cpu"] = cpu_timing(ctx)
        result["counts"] = {"images": len(images), "pairs": len(pairs)}
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        write(folder / "benchmark.json", result)
        seal(folder, ctx.identity)
    return read(folder / "benchmark.json")


def cpu_timing(ctx):
    from threadpoolctl import threadpool_limits

    from st3.competition_ridge import fit_head, make_panel

    gold = population(ctx)["gold"]
    train = gold.loc[gold.fold.ne(1)]
    panel = make_panel(train)
    width = max(ctx.config["arms"][a]["width"] for a in ("E1", "E2", "E3"))
    rng = np.random.default_rng(ctx.config["diagnostic_seed"])
    x = rng.standard_normal((len(panel.paintings), width))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    tick = time.time()
    with threadpool_limits(limits=ctx.config["runtime"]["ridge_threads"]):
        fit_head(panel, x, ctx.track1["ridge"])
    return {"width": width, "fit_seconds": time.time() - tick}


def forecast(ctx, bench, validate):
    counts, arms = bench["counts"], bench["arms"]
    load = 240.
    pieces = {}
    for arm in ("E1", "E2", "E3", "E4"):
        if (ctx.root / "features" / arm / "seal.json").is_file():
            continue
        if "items_per_second" not in arms[arm]:
            continue  # Failed or dependency-skipped at benchmark; never forecast.
        if arm == "E4" and "E3" not in pieces and not (ctx.root / "features/E3/seal.json").is_file():
            continue
        items = counts["pairs"] if arm == "E4" else counts["images"]
        pieces[arm] = items / arms[arm]["items_per_second"] + load
    rate = bench["cscva"]["paintings_per_second"]
    per_seed = bench["cscva"]["forecast_fit_seconds"] + 2 * 85 / rate + load
    for label, seed in zip(("cscva_seed_2", "cscva_seed_3"), ctx.config["cscva"]["new_seeds"]):
        if not (ctx.root / "cscva" / f"seed_{seed}" / "seal.json").is_file():
            pieces[label] = per_seed
    widths = {a: ctx.config["arms"][a]["width"] for a in ("E1", "E2", "E3")}
    per_width = bench["cpu"]["fit_seconds"] / bench["cpu"]["width"]
    new_fits = 24 * (sum(widths.values()) + 1.5 * widths["E3"]) * per_width
    base = 2 * validate["r0_p2"]["seconds"]
    pieces["cpu_p2_and_development"] = new_fits + base + 600
    pieces["export"] = 600
    return pieces


def stage_features(ctx, backend, status, plan, bench=None):
    results = {}
    budget = ctx.config["budget"]
    bench_arms = (bench or {}).get("arms", {})
    for arm in ("E1", "E2", "E3", "E4"):
        status.item(arm)
        folder = ctx.root / "features" / arm
        identity = arm_identity(ctx, arm)
        items = pair_items(ctx) if arm == "E4" else image_items(ctx)
        if (folder / "seal.json").is_file():
            load_features(folder, identity, [i[0] for i in items])
            status.piece(arm, "COMPLETE", "sealed cache reused")
            results[arm] = "COMPLETE"
            continue
        if arm == "E4" and results.get("E3") != "COMPLETE":
            reason = f"E4 needs E3 (E3 {results.get('E3')}); its budget is not spent"
            status.piece(arm, "SKIPPED_DEPENDENCY", reason)
            results[arm] = "SKIPPED_DEPENDENCY"
            continue
        if arm in plan["skipped"]:
            status.piece(arm, "BUDGET_SKIPPED", plan["skipped"][arm])
            results[arm] = "BUDGET_SKIPPED"
            continue
        if "failed" in bench_arms.get(arm, {}):
            status.piece(arm, "FAILED_OPTIONAL", f"benchmark: {bench_arms[arm]['failed']}")
            results[arm] = "FAILED_OPTIONAL"
            progress(ctx)
            continue
        if arm not in OPTIONAL_ARMS:
            backend.extract(arm, items, folder, identity, ctx.clock.work_deadline())
        else:
            later = sum(v for k, v in plan["forecast_seconds"].items()
                        if k not in plan["skipped"] and k.startswith(("cscva", "cpu", "export")))
            deadline = ctx.clock.work_deadline() - plan["margin"] * later
            cap = None
            if arm == "E4":
                deadline = min(deadline, time.time() + budget["e4_cap_seconds"])
                cap = min(time.time() + budget["e4_cap_seconds"], deadline)
            try:
                backend.extract(arm, items, folder, identity, deadline, cap_deadline=cap)
            except (BudgetStop, TimeoutError) as error:
                status.piece(arm, "BUDGET_STOP", str(error))
                results[arm] = "BUDGET_STOP"
                progress(ctx)
                continue
            except Exception as error:  # Optional arm: C2/C3 become unavailable instead of stopping C1.
                status.piece(arm, "FAILED_OPTIONAL", f"{type(error).__name__}: {error}")
                results[arm] = "FAILED_OPTIONAL"
                progress(ctx)
                continue
        load_features(folder, identity, [i[0] for i in items])
        status.piece(arm, "COMPLETE")
        results[arm] = "COMPLETE"
        progress(ctx)
    return results


def stage_cscva(ctx, backend, status, plan):
    results = {}
    later = sum(v for k, v in plan["forecast_seconds"].items() if k.startswith(("cpu", "export")))
    for label, seed in zip(("cscva_seed_2", "cscva_seed_3"), ctx.config["cscva"]["new_seeds"]):
        status.item(f"{label} ({seed})")
        folder = ctx.root / "cscva" / f"seed_{seed}"
        required = ["predictions.csv", "inference/weights.pt", "inference/bundle.json", "train.json"]
        if completed(folder, ctx.identity, required):
            status.piece(label, "COMPLETE", "sealed seed reused")
            results[label] = "COMPLETE"
            continue
        if label in plan["skipped"]:
            status.piece(label, "BUDGET_SKIPPED", plan["skipped"][label])
            results[label] = "BUDGET_SKIPPED"
            continue
        try:
            backend.cscva(seed, folder, ctx.clock.work_deadline() - plan["margin"] * later)
            submission.read_predictions(folder / "predictions.csv", population(ctx)["dev"])
            seal(folder, ctx.identity, {"seed": seed, "freeze_epoch": ctx.config["cscva"]["freeze_epoch"]})
            status.piece(label, "COMPLETE")
            results[label] = "COMPLETE"
        except TimeoutError as error:
            status.piece(label, "BUDGET_STOP", str(error))
            results[label] = "BUDGET_STOP"
        except Exception as error:
            status.piece(label, "FAILED_OPTIONAL", f"{type(error).__name__}: {error}")
            results[label] = "FAILED_OPTIONAL"
        finally:
            if results.get(label) != "COMPLETE" and folder.exists():
                shutil.rmtree(folder)
        progress(ctx)
    return results


def available_arms(ctx):
    return tuple(arm for arm in science.ARMS if (ctx.root / "features" / arm / "seal.json").is_file())


def stage_p2(ctx, status):
    folder = ctx.root / "p2"
    required = ["oof.csv", "metrics.json", "gates.json"]
    if completed(folder, ctx.identity, required):
        ctx.log("P2 evaluation: sealed results verified; reusing")
        return read(folder / "gates.json")
    from threadpoolctl import threadpool_limits

    pop = population(ctx)
    arms = available_arms(ctx)
    features, delta = features_with_arms(ctx, arms)
    tick = time.time()
    with threadpool_limits(limits=ctx.config["runtime"]["ridge_threads"]):
        frame, parts, r0, receipts, candidates = science.evaluate_p2(
            pop["gold"], features, delta, arms, ctx.config, ctx.track1, ctx.neighborhood,
            ctx.refs["track1_inner_oof_sha256"], log=ctx.log)
    metrics = (read(ctx.overlay / "reference/track1_metrics.json"),
               read(ctx.overlay / "reference/corrected_metrics.json"))
    replay = science.replay_r0_p2(pop["gold"], parts, r0, reference_frame(ctx, "track1_oof.csv"),
                                  ctx.refs["corrected_oof_sha256"], metrics, ctx.config["test_like_excluded"])
    systems = ["R0_fixed", "R0_corrected", *arms, *[f"{c}_mix" for c in candidates], *candidates]
    report, paintings, gates = science.p2_report(pop["gold"], frame, systems, candidates, ctx.config)
    for name, reason in unavailable_candidates(ctx, arms).items():
        gates[name] = reason
    diagnostics, languages, painting_deltas = science.p2_diagnostics(pop["gold"], frame, arms, candidates, ctx.config)
    selected = science.select(gates, ctx.config["complexity_order"], ctx.config["gate"]["tie_tolerance"])
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    frame.to_csv(folder / "oof.csv", index=False, lineterminator="\n", float_format="%.6f")
    parts.to_csv(folder / "components.csv", index=False, lineterminator="\n", float_format="%.6f")
    paintings.to_csv(folder / "painting_contrast.csv", index=False, lineterminator="\n")
    painting_deltas.to_csv(folder / "painting_deltas.csv", index=False, lineterminator="\n")
    languages.to_csv(folder / "language_deltas.csv", index=False, lineterminator="\n")
    write(folder / "metrics.json", {"report": report, "replay": replay, "fold_receipts": receipts,
                                    "arms": list(arms), "candidates": {k: list(v) for k, v in candidates.items()},
                                    "seconds": time.time() - tick})
    write(folder / "diagnostics.json", diagnostics)
    write(folder / "gates.json", {"gates": gates, "selected": selected,
                                  "decision": "GO" if selected else "NO_GO"})
    seal(folder, ctx.identity)
    return read(folder / "gates.json")


def unavailable_candidates(ctx, arms):
    """Every candidate whose arms are missing, with the arm state that removed it."""
    pieces = read(ctx.root / "status.json")["pieces"]
    out = {}
    for name, parts in ctx.config["candidates"].items():
        missing = [arm for arm in parts if arm not in arms]
        if not missing:
            continue
        if "E3" in arms:
            # E3 present and only E4 missing: the earlier E4 behavior is unchanged.
            out[name] = {"status": "BUDGET_SKIPPED", "passed": False,
                         "reason": f"arm(s) {missing} unavailable: {pieces.get(missing[0], {})}"}
            continue
        e3 = pieces.get("E3", {}).get("state")
        if "E4" in parts:
            reason = f"NOT_AVAILABLE (E3/E4 {e3}/{pieces.get('E4', {}).get('state')})"
        else:
            reason = f"NOT_AVAILABLE (E3 {e3})"
        out[name] = {"status": "NOT_AVAILABLE", "passed": False, "reason": reason,
                     "arm_states": {arm: pieces.get(arm) for arm in missing}}
    return out


def cscva_frames(ctx):
    dev = population(ctx)["dev"]
    reference = ctx.overlay / "reference"
    frames = {"original": submission.validate_archive(reference / "exp036_standalone_submission.zip", dev)}
    seeds = {}
    for seed in ctx.config["cscva"]["new_seeds"]:
        folder = ctx.root / "cscva" / f"seed_{seed}"
        if (folder / "seal.json").is_file():
            verify_seal(folder, ctx.identity)
            seeds[seed] = submission.read_predictions(folder / "predictions.csv", dev)
    if len(seeds) == len(ctx.config["cscva"]["new_seeds"]):
        frames["cscva3"] = science.mean_of([frames["original"], *seeds.values()], dev)
    return frames, seeds


def save_bundle(folder, heads, bank_set, features, train_paintings, candidates, ctx):
    folder.mkdir(parents=True)
    arrays = {}
    for name, head in heads.items():
        base = head.head if isinstance(head, science.PairHead) else head
        for field in ("intercept", "language_intercepts", "shared", "deviations", "languages"):
            arrays[f"{name}:{field}"] = np.asarray(getattr(base, field))
        if isinstance(head, science.PairHead):
            arrays[f"{name}:gamma"] = head.gamma
    np.savez_compressed(folder / "heads.npz", **arrays)
    banks = {}
    for system, bank in bank_set.items():
        banks[f"{system}:paintings"] = bank.numerator.index.to_numpy(dtype=str)
        banks[f"{system}:languages"] = bank.numerator.columns.to_numpy(dtype=str)
        banks[f"{system}:numerator"] = bank.numerator.to_numpy()
        banks[f"{system}:votes"] = bank.votes.to_numpy()
    np.savez_compressed(folder / "banks.npz", **banks)
    names = sorted({*ENCODERS, *[science.image_encoder(a) for a in heads if a not in ENCODERS]})
    np.savez_compressed(folder / "train_features.npz", paintings=np.asarray(train_paintings, dtype=str),
                        **{name: features[name].loc[train_paintings].to_numpy() for name in names})
    write(folder / "manifest.json", {"arms": [a for a in heads if a not in ENCODERS],
                                     "candidates": {k: list(v) for k, v in candidates.items()},
                                     "config": ctx.config, "track1_ridge": ctx.track1["ridge"],
                                     "neighborhood": ctx.neighborhood, "identity": ctx.identity,
                                     "files": {p.name: sha(p) for p in folder.iterdir()}})


def stage_development(ctx, status, gates):
    folder = ctx.root / "development"
    if completed(folder, ctx.identity, ["scores.json", "predictions.csv"]):
        ctx.log("Development: sealed predictions verified; reusing")
        return read(folder / "scores.json")
    from threadpoolctl import threadpool_limits

    pop = population(ctx)
    dev, arms = pop["dev"], available_arms(ctx)
    candidates = science.available_candidates(ctx.config["candidates"], arms)
    features, delta = features_with_arms(ctx, arms)
    with threadpool_limits(limits=ctx.config["runtime"]["ridge_threads"]):
        heads, bank_set, info, visual = science.deployment_fit(pop["train"], features, delta, arms, ctx.track1,
                                                               ctx.neighborhood, candidates)
        frame, _ = science.development_predictions(heads, bank_set, visual, dev[data.KEYS], features, delta,
                                                   candidates, ctx.neighborhood)
    replay = replay_development(ctx, frame)
    if not compare_bank(bank_set["R0"], ctx.overlay / "reference/corrected_bundle/bank.npz"):
        raise science.ReplayError("The refitted Corrected Ridge deployment bank differs from the saved bank")
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    frame.to_csv(folder / "predictions.csv", index=False, lineterminator="\n", float_format="%.6f")
    standalone = {name: science.system_frame(frame, name) for name in
                  ["R0_fixed", "R0_corrected", *arms, *[f"{c}_mix" for c in candidates], *candidates]}
    cscva, seeds = cscva_frames(ctx)
    c6 = submission.read_predictions(ctx.overlay / "reference/exp037_c6_standalone/predictions.csv", dev)
    bases = {"R0": standalone["R0_corrected"]} | {name: standalone[name] for name in candidates}
    composed = science.blends(bases, cscva, c6, dev, ctx.config)
    predictions = standalone | {f"CSCVA_seed_{s}": f for s, f in seeds.items()}
    if "cscva3" in cscva:
        predictions["CSCVA_3"] = cscva["cscva3"]
    predictions |= composed
    archives = {}
    zips = folder / "submissions"
    for name, prediction in predictions.items():
        archive = write_submission(science.canonical(prediction, dev), dev, zips, f"exp038_{name}_submission")
        archives[name] = {"zip": archive.relative_to(ctx.root).as_posix(), "sha256": sha(archive),
                          "eligibility": science.eligibility(name, ctx.config)}
    gold = pd.read_csv(ctx.base / "scoring/dev_gold.csv")
    if list(gold.columns) != [*data.KEYS, "k", "n"]:
        raise ValueError("Unexpected Development scoring columns")
    canonical = {name: submission.validate_archive(ctx.root / item["zip"], dev) for name, item in archives.items()}
    reference = ctx.overlay / "reference"
    references = {"EXP036_ensemble": submission.validate_archive(reference / "exp036_ensemble_submission.zip", dev),
                  "EXP037_original_blend": submission.validate_archive(reference / "exp037_best_submission.zip", dev)}
    scores = science.development_scores(gold, canonical | references, ctx.config["test_like_excluded"])
    intervals = science.development_intervals(gold, {k: v for k, v in canonical.items() if k.startswith("B")},
                                              references, ctx.config["test_like_excluded"],
                                              ctx.config["diagnostic_seed"])
    save_bundle(folder / "bundle", heads, bank_set, features, sorted(pop["train"].painting.unique()), candidates, ctx)
    write(folder / "scores.json", {"scores": scores, "intervals": intervals, "archives": archives,
                                   "replay": replay, "fit": info, "cscva": sorted(cscva),
                                   "development_selection": "none; Development is reporting only"})
    seal(folder, ctx.identity)
    return read(folder / "scores.json")


def stage_submissions(ctx, status, gates, scores, pieces):
    dev = population(ctx)["dev"]
    for name, item in scores["archives"].items():
        path = ctx.root / item["zip"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"Submission archive changed: {name}")
        frame = submission.validate_archive(path, dev)
        if not frame[data.KEYS].equals(dev[data.KEYS]) or not np.array_equal(frame.p_positive,
                                                                              frame.p_positive.round(6)):
            raise ValueError(f"{name} failed key order or six-decimal validation")
    selected = gates["selected"]
    best = ctx.root / "exp038_best_submission.zip"
    if selected:
        cscva = "cscva3" if "cscva3" in scores["cscva"] else "original"
        name = f"B2_{selected}_{cscva}"
        source = ctx.root / scores["archives"][name]["zip"]
        recipe = {"decision": "GO", "selected_candidate": selected, "submission": name,
                  "formula": f"round6(0.80 * round6(0.75 * {selected} + 0.25 * {cscva}) + 0.20 * C6)",
                  "cscva": cscva, "cscva_note": None if cscva == "cscva3" else
                  "A new CS-CVA seed was unavailable, so the original single epoch-5 CS-CVA is used."}
    else:
        name = "EXP037_original_blend"
        source = ctx.overlay / "reference/exp037_best_submission.zip"
        recipe = {"decision": "NO_GO", "selected_candidate": None, "submission": name,
                  "formula": "byte-identical EXP-037 original blend round6(0.80 * EXP036 + 0.20 * C6)",
                  "note": "No EXP-038 candidate passed the frozen P2 gate. The best ZIP is the unchanged "
                          "EXP-037 original blend."}
    copy_verified(source, best)
    frame = submission.validate_archive(best, dev)
    if not frame[data.KEYS].equals(dev[data.KEYS]):
        raise ValueError("Best ZIP key order differs")
    if not selected and sha(best) != ctx.refs["exp037_original_blend_zip_sha256"]:
        raise ValueError("NO_GO best ZIP is not byte-identical to the EXP-037 original blend")
    components = {"C6": ctx.refs["exp037_c6_standalone_csv_sha256"],
                  "CSCVA_original": ctx.refs["exp036_standalone_zip_sha256"]}
    for key in ("CSCVA_3", "R0_corrected", *(selected and [selected] or [])):
        if key in scores["archives"]:
            components[key] = scores["archives"][key]["sha256"]
    eligibility = (ctx.config["eligibility"]["unconfirmed_label"] if selected
                   else "UNCONFIRMED_QWEN_ST3 (EXP-037 C6 component)")
    selection = recipe | {"coefficients": ctx.config["blends"] | ctx.config["mix"], "components_sha256": components,
                          "best_sha256": sha(best), "eligibility": eligibility,
                          "development_used_for_selection": False}
    unavailable = {name: g.get("reason") for name, g in gates["gates"].items()
                   if g.get("status") in {"NOT_AVAILABLE", "BUDGET_SKIPPED"}}
    expected = [*science.ARMS, *ctx.config["candidates"],
                *[f"{kind}_{base}_{cs}" for kind in ("B1", "B2") for base in ("R0", *ctx.config["candidates"])
                  for cs in ("original", "cscva3")]]
    missing_zips = {name: "components unavailable" for name in expected if name not in scores["archives"]}
    selection |= {"unavailable_candidates": unavailable, "unavailable_zips": missing_zips}
    decision = {"decision": recipe["decision"], "selected": selected, "gates": gates["gates"],
                "unavailable_candidates": unavailable, "available_zips": sorted(scores["archives"]),
                "unavailable_zips": missing_zips,
                "selection_rule": "highest filtered P2 Contrast among passing candidates; within 0.001 the simpler "
                                  "(C1 < C2 < C3)", "pieces": pieces,
                "reasons": {name: ([k for k, ok in g.get("checks", {}).items() if not ok] or g.get("reason") or "pass")
                            for name, g in gates["gates"].items()}}
    write(ctx.root / "decision.json", decision)
    write(ctx.root / "selection.json", selection)
    return decision, selection


def progress(ctx):
    try:
        publish(ctx.root.parent / "exp038_progress.zip", ctx.root, list(exportable(ctx.root)), ctx.overlay_sha)
    except Exception as error:  # A failed periodic transfer never stops the run; it is recorded.
        ctx.log(f"Progress export failed: {error}")


def stage_export(ctx, scores, decision):
    output = ctx.output
    result = output / "exp038_results.zip"
    digests = {"exp038_results.zip": publish(result, ctx.root, list(exportable(ctx.root)), ctx.overlay_sha)}
    digests["exp038_best_submission.zip"] = copy_verified(ctx.root / "exp038_best_submission.zip",
                                                          output / "exp038_best_submission.zip")
    for name, item in scores["archives"].items():
        target = output / f"exp038_{name}_submission.zip"
        digests[target.name] = copy_verified(ctx.root / item["zip"], target)
    for name, digest in digests.items():
        receipt = output / f"{name}.sha256"
        if sha(output / name) != digest or receipt.read_text(encoding="ascii").split()[0] != digest:
            raise ValueError(f"Read-back failed for {name}")
    return digests


def diagnostic(ctx, status, error):
    path = ctx.output / "exp038_diagnostic.json"
    write(path, {"failed_stage": status.state.get("stage"), "item": status.state.get("item"),
                 "state": status.state.get("state"), "reason": status.state.get("reason"),
                 "completed_stages": status.state.get("completed_stages"), "pieces": status.state.get("pieces"),
                 "resume_scope": "Attach exp038_progress.zip (or results) as RESUME_INPUT with the same inputs, "
                                 "runtime and GPU type; sealed stages and caches are reused.",
                 "traceback": "".join(traceback.format_exception(error))[-20000:],
                 "submission_claimed": False})
    files = [path, *[p for p in (ctx.root / "status.json", ctx.root / "events.log", ctx.root / "gpu_children.log")
                     if p.is_file()]]
    from st3.exp036_io import export

    scratch = ctx.output / "diagnostic_files"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir()
    for file in files:
        shutil.copyfile(file, scratch / file.name)
    export(ctx.output / "exp038_diagnostic.zip", scratch, list(scratch.iterdir()), ctx.overlay_sha)
    shutil.rmtree(scratch, ignore_errors=True)


def run_pipeline(ctx, backend):
    ctx.root.mkdir(parents=True, exist_ok=True)
    for stale in ("exp038_SUCCESS.json",):
        if (ctx.output / stale).exists():
            (ctx.output / stale).unlink()  # Revoke stale success evidence before a new attempt.
    status = Status(ctx)
    outcome = None
    try:
        status.begin(STAGES[0])
        stage_setup(ctx, backend, status)
        status.done(STAGES[0])
        status.begin(STAGES[1])
        validate = stage_validate(ctx, backend, status)
        status.done(STAGES[1])
        progress(ctx)
        status.begin(STAGES[2])
        bench = stage_benchmark(ctx, backend, status)
        plan = plan_budget(forecast(ctx, bench, validate), ctx.clock.remaining(), ctx.config["budget"])
        write(ctx.root / "budget_plan.json", plan)
        ctx.log(f"Budget plan: {json.dumps(plan)}")
        status.done(STAGES[2])
        status.begin(STAGES[3])
        stage_features(ctx, backend, status, plan, bench)
        for arm in [a for a in ctx.config["budget"]["mandatory"] if a in science.ARMS]:
            if status.state["pieces"].get(arm, {}).get("state") != "COMPLETE":
                raise RuntimeError(f"Mandatory arm {arm} is unavailable")
        status.done(STAGES[3])
        status.begin(STAGES[4])
        stage_cscva(ctx, backend, status, plan)
        status.done(STAGES[4])
        status.begin(STAGES[5])
        gates = stage_p2(ctx, status)
        status.done(STAGES[5])
        progress(ctx)
        status.begin(STAGES[6])
        scores = stage_development(ctx, status, gates)
        status.done(STAGES[6])
        status.begin(STAGES[7])
        decision, selection = stage_submissions(ctx, status, gates, scores, status.state["pieces"])
        status.done(STAGES[7])
        status.begin(STAGES[8])
        status.state["state"] = "EXPORTING"
        status.save()
        digests = stage_export(ctx, scores, decision)
        status.done(STAGES[8])
        status.begin(STAGES[9])
        status.state |= {"state": "COMPLETE", "decision": decision["decision"], "selected": decision["selected"]}
        status.save()
        write(ctx.output / "exp038_SUCCESS.json", {"decision": decision["decision"], "selected": decision["selected"],
                                                   "best_sha256": selection["best_sha256"], "exports": digests,
                                                   "note": "Kaggle keeps these only in a saved version or download."})
        status.done(STAGES[9])
        outcome = "COMPLETE"
        return decision
    except BaseException as error:
        resumable = isinstance(error, TimeoutError) and not isinstance(error, BudgetStop)
        status.state |= {"state": "RESUMABLE_STOP" if resumable else "FAILED",
                         "reason": f"{type(error).__name__}: {error}"}
        status.save()
        ctx.log(f"Stopped in {status.state.get('stage')}: {status.state['reason']}")
        try:
            progress(ctx)
        finally:
            diagnostic(ctx, status, error)
        raise
    finally:
        if outcome != "COMPLETE" and (ctx.output / "exp038_SUCCESS.json").exists():
            (ctx.output / "exp038_SUCCESS.json").unlink()


# GPU child entry points.

def gpu_device():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("EXP-038 GPU stages need exactly one visible CUDA device")
    return torch.device("cuda:0")


def child_spec(arm):
    config = configs()[0]
    spec = config["arms"][arm]
    return config["arms"]["E3"] | spec if arm == "E4" else spec


def child_items(path):
    return [tuple(item) for item in json.loads(Path(path).read_text(encoding="utf-8"))]


def gpu_bench(args):
    from st3 import exp038_gpu as gpu

    device = gpu_device()
    gpu.deterministic_cuda()
    spec = child_spec(args.arm)
    tick = time.time()
    embed = gpu.encoder(args.arm, spec, args.snapshot, device)
    loaded = time.time() - tick
    result = gpu.benchmark(args.arm, spec, embed, child_items(args.items), device) | {"load_seconds": loaded}
    write(args.out, result)


def gpu_extract(args):
    from st3 import exp038_gpu as gpu

    device = gpu_device()
    gpu.deterministic_cuda()
    spec = child_spec(args.arm)
    identity = json.loads(Path(args.identity).read_text(encoding="utf-8"))
    embed = gpu.encoder(args.arm, spec, args.snapshot, device)
    last = [time.time()]

    def on_chunk(arm, done, total, seconds):
        rate = done / max(seconds, 1e-9)
        print(f"Feature extraction {arm}: {done}/{total} items, {seconds:.0f}s elapsed, "
              f"about {(total - done) / max(rate, 1e-9):.0f}s remaining", flush=True)
        if time.time() - last[0] >= args.interval:
            last[0] = time.time()
            try:
                publish(args.progress_root.parent / "exp038_progress.zip", args.progress_root,
                        list(exportable(args.progress_root)), args.handoff)
            except Exception as error:
                print(f"Progress export failed: {error}", flush=True)

    try:
        gpu.extract(args.arm, spec, embed, child_items(args.items), args.folder, identity, configs()[0]["chunk_items"],
                    args.deadline, on_chunk, cap_deadline=args.cap)
    except BudgetStop as error:
        print(f"BUDGET_STOP: {error}", flush=True)
        sys.exit(EXIT_BUDGET)
    except TimeoutError as error:
        print(f"TIMEOUT: {error}", flush=True)
        sys.exit(EXIT_TIMEOUT)


def gpu_cscva(args):
    from st3.exp037_io import verify_inputs
    from st3.exp038_cscva import train_seed

    manifest = verify_inputs(args.base, args.base_sha)
    config = configs()[0]
    try:
        result = train_seed(root=ROOT, base=args.base, manifest=manifest, seed=args.seed, folder=args.folder,
                            snapshot=args.snapshot, deadline=args.deadline,
                            targets_sha=config["references"]["caption_targets_sha256"],
                            preflight_only=args.preflight, freeze_epoch=config["cscva"]["freeze_epoch"],
                            source=args.source)
    except TimeoutError as error:
        print(f"TIMEOUT: {error}", flush=True)
        sys.exit(EXIT_TIMEOUT)
    write(args.out, result)


def run(args):
    clock = Clock(args.started, args.hours, configs()[0]["budget"]["persistence_reserve_seconds"])
    snapshots = {arm: getattr(args, f"{arm.lower()}_snapshot") for arm in ("E1", "E2", "E3")
                 if getattr(args, f"{arm.lower()}_snapshot")}
    ctx = Context(base=args.base, overlay=args.overlay, root=args.output / "exp038_run", output=args.output,
                  base_sha=args.base_sha, overlay_sha=args.overlay_sha, clock=clock, snapshots=snapshots)
    if args.resume:
        ctx.root.mkdir(parents=True, exist_ok=True)
        if any(ctx.root.iterdir()):
            raise ValueError("Resume needs a fresh /kaggle/working/exp038_run")
        restore(args.resume, ctx.root, args.overlay_sha)
        for marker in ("status.json",):
            if (ctx.root / marker).exists():
                state = read(ctx.root / marker)
                ctx.log(f"Restored run in state {state.get('state')} at {state.get('stage')}")
    run_pipeline(ctx, SubprocessBackend(ctx))


def cpu_validate(args):
    """Local or audit replay of every R0 check in Validate, without GPU work or model downloads."""
    clock = Clock(time.time(), 10, 0)
    args.output.mkdir(parents=True, exist_ok=True)
    ctx = Context(base=args.base, overlay=args.overlay, root=args.output, output=args.output,
                  base_sha=args.base_sha, overlay_sha=args.overlay_sha, clock=clock)
    ctx.identity = {"cpu_validate": True}
    tick = time.time()
    receipts = {"r0_p2": r0_p2(ctx)}
    receipts["r0_p2"]["seconds"] = time.time() - tick
    receipts["r0_development"] = r0_development(ctx)
    write(args.output / "cpu_validate.json", receipts)
    print(json.dumps(receipts, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("cpu-validate")
    for name in ("base", "overlay", "output"):
        v.add_argument(f"--{name}", type=Path, required=True)
    for name in ("base-sha", "overlay-sha"):
        v.add_argument(f"--{name}", required=True)
    p = sub.add_parser("run")
    for name in ("base", "overlay", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    for name in ("base-sha", "overlay-sha"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--started", type=float, required=True)
    p.add_argument("--hours", type=float, default=10.5)
    p.add_argument("--resume", type=Path)
    for arm in ("e1", "e2", "e3"):
        p.add_argument(f"--{arm}-snapshot", type=Path)
    b = sub.add_parser("gpu-bench")
    e = sub.add_parser("gpu-extract")
    for command in (b, e):
        command.add_argument("--arm", required=True, choices=("E1", "E2", "E3", "E4"))
        command.add_argument("--items", type=Path, required=True)
        command.add_argument("--snapshot", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    e.add_argument("--folder", type=Path, required=True)
    e.add_argument("--identity", type=Path, required=True)
    e.add_argument("--deadline", type=float, required=True)
    e.add_argument("--cap", type=float)
    e.add_argument("--progress-root", type=Path, required=True)
    e.add_argument("--handoff", required=True)
    e.add_argument("--interval", type=float, default=900)
    c = sub.add_parser("gpu-cscva")
    c.add_argument("--seed", type=int, required=True)
    for name in ("folder", "snapshot", "base", "out"):
        c.add_argument(f"--{name}", type=Path, required=True)
    c.add_argument("--base-sha", required=True)
    c.add_argument("--deadline", type=float, required=True)
    c.add_argument("--source", default="")
    c.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    {"run": run, "cpu-validate": cpu_validate, "gpu-bench": gpu_bench, "gpu-extract": gpu_extract, "gpu-cscva": gpu_cscva}[args.command](args)


if __name__ == "__main__":
    main()
