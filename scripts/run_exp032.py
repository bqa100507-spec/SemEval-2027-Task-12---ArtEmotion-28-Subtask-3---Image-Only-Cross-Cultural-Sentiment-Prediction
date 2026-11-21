#!/usr/bin/env python3
"""Run the fixed EXP-032 grouped CPU experiment on the pinned EXP-023 inputs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3.baseline import file_sha256
from st3.data import KEYS
from st3.exp032 import crossfit_residuals, estimate_graph, fit_head, graph_stability, make_panel, painting_splits
from st3.exp032_evaluation import CANDIDATE, CONTROL, REFERENCE, choose_inner, core_score, evaluate
from st3.submission import validate_predictions

CONFIG = ROOT / "configs/exp032.json"
SOURCE = (
    "scripts/run_exp032.py", "src/st3/exp032.py", "src/st3/exp032_evaluation.py",
    "src/st3/metrics.py", "src/st3/data.py", "src/st3/submission.py",
)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def write_json(path, value):
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        handle.write(json_bytes(value))
    temp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def structured_sha(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def load_inputs(root, config, run_dir):
    root, run_dir = root.resolve(), run_dir.resolve()
    if root == run_dir or root in run_dir.parents:
        raise ValueError("Run directory must be separate from the historical handoff")
    for file, pin in config["input_sha256"].items():
        source = root / file
        if not source.is_file() or file_sha256(source) != pin:
            raise ValueError(f"Missing or SHA-256 mismatch for {file}. Use the pinned EXP-023 handoff.")
    columns = KEYS + ["k", "n", "fold", "role", "split", "image", "image_sha256"]
    gold = pd.read_csv(root / "training/pairs.csv", usecols=columns)
    fold_spec = config["p2"]
    folds = list(range(1, len(fold_spec["fold_pair_counts"]) + 1))
    if (gold[columns].isna().any().any() or gold.duplicated(KEYS).any()
            or not gold["split"].eq("train").all() or not gold["role"].eq("development").all()
            or sorted(gold["fold"].unique()) != folds or not gold["n"].ge(5).all()
            or not (gold["k"].between(0, gold["n"])).all()
            or (gold[["k", "n"]].to_numpy() % 1 != 0).any()
            or gold["language"].nunique() != fold_spec["languages"]):
        raise ValueError("Historical training pairs fail identity, fold, or vote validation")
    identity = ["painting", "image", "image_sha256", "role", "fold"]
    paints = gold[identity].drop_duplicates().sort_values("painting").reset_index(drop=True)
    protocol = pd.read_csv(root / "training/protocol.csv")[identity]
    allowed = protocol.loc[protocol["role"].eq("development")].sort_values("painting").reset_index(drop=True)
    locked = protocol.loc[protocol["role"].eq("holdout")]
    if (protocol["painting"].duplicated().any() or protocol["image"].duplicated().any()
            or not paints.equals(allowed) or len(locked) != fold_spec["sealed_paintings"]
            or len(protocol) != len(allowed) + len(locked) or not locked["fold"].eq(0).all()
            or set(paints["image_sha256"]) & set(locked["image_sha256"])
            or not paints.groupby("fold").size().eq(fold_spec["paintings_per_fold"]).all()
            or gold.groupby("fold").size().tolist() != fold_spec["fold_pair_counts"]):
        raise ValueError("P2 painting isolation or exactly matched fold inventory failed")
    references = pd.read_csv(root / "history/exp022_R0.csv")
    validate_predictions(references[KEYS + ["p_positive"]], gold)
    mapped = gold[KEYS + ["fold"]].merge(references, on=KEYS, validate="one_to_one", suffixes=("_gold", "_r0"))
    if not mapped["fold_gold"].eq(mapped["fold_r0"]).all():
        raise ValueError("Frozen E50+L predictions have a different fold assignment")
    image_hashes = read_json(root / "visual/image_hashes.json")
    if paints["image"].map(image_hashes).ne(paints["image_sha256"]).any():
        raise ValueError("Pinned cache image identity mismatch")
    with np.load(root / "visual/features.npz", allow_pickle=False) as cache:
        names = cache["names"].tolist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate images in feature cache")
        indices = paints["image"].map({name: i for i, name in enumerate(names)})
        if indices.isna().any():
            raise ValueError("A P2 artwork is missing from the pinned feature cache")
        features = {}
        for encoder, width in config["features"].items():
            subset = np.ascontiguousarray(cache[encoder][indices.to_numpy(dtype=int)], dtype=np.float64)
            if (subset.shape != (len(paints), width) or not np.isfinite(subset).all()
                    or not np.allclose(np.linalg.norm(subset, axis=1), 1, atol=3e-6, rtol=0)):
                raise ValueError(f"Unexpected shape, normalization or values for {encoder}")
            features[encoder] = subset
    return (
        gold[KEYS + ["k", "n", "fold"]].sort_values(KEYS).reset_index(drop=True),
        references[KEYS + ["p_positive"]], features,
        dict(zip(paints["painting"], range(len(paints)))),
    )


def panel_features(matrix, panel, paint_indices):
    return matrix[[paint_indices[name] for name in panel.paintings]]


def serialize_graph(graph):
    result = {k: graph[k] for k in ("languages", "unsupported", "scale", "isotropic_distance", "overlap_only_distance")}
    result["edges"] = [dict(edge, languages=key) for key, edge in graph["edges"].items()]
    for key in ("laplacian", "projection"):
        result[key] = graph[key].tolist()
    return result


def deserialize_graph(graph):
    return graph | {key: np.asarray(graph[key]) for key in ("laplacian", "projection")}


def graph_scope(panel, features, config, seed):
    residuals, shared, receipts = crossfit_residuals(panel, features, config["graph"], config["optimizer"], seed)
    return estimate_graph(panel, residuals, shared, config["graph"]), receipts


def prepare_fold_graphs(train, combined_features, indices, config, fold):
    panel = make_panel(train)
    features = panel_features(combined_features, panel, indices)
    seed = config["seed"] + 100 * fold
    splits = painting_splits(panel.paintings, seed, 2)
    full, full_fits = graph_scope(panel, features, config, seed + 1)
    halves, half_fits, held_keys = [], [], []
    for split in (0, 1):
        membership = splits == split
        half_panel = panel.subset(membership)
        graph, fits = graph_scope(half_panel, features[membership], config, seed + 2 + split)
        halves.append(graph)
        half_fits.append(fits)
        held_keys.append(list(half_panel.paintings))
    stability = graph_stability(full, *halves, config["graph"])
    return {
        "fold": fold, "stability": stability, "outer_fitting_graph": serialize_graph(full),
        "inner_fitting_graphs": [serialize_graph(g) for g in halves],
        "fitting_halves": held_keys, "nuisance_optimizer_receipts": [full_fits, *half_fits],
    }


def predict(fit_panel, take_panel, eval_keys, features, indices, config, deviation_ratio, fraction, graph):
    components, receipts, states = {}, {}, {}
    if graph is not None and tuple(graph["languages"]) != fit_panel.languages:
        raise ValueError("Graph language order differs from fitting")
    for name, matrix in features.items():
        fit_x = panel_features(matrix, fit_panel, indices)
        eval_x = panel_features(matrix, take_panel, indices)
        head, receipt = fit_head(fit_panel, fit_x, config["optimizer"], deviation_ratio=deviation_ratio,
                                 fraction=fraction, graph=graph)
        row = eval_keys["painting"].map({key: i for i, key in enumerate(take_panel.paintings)})
        col = eval_keys["language"].map({key: i for i, key in enumerate(take_panel.languages)})
        components[name] = expit(head.logits(eval_x, take_panel.languages)[row, col]).round(6)
        receipts[name] = receipt
        states[name] = head
    weights = config["fusion_weights"]
    # Production rounds each component, the E50 midpoint, and the E50+L midpoint.
    intermediate = np.round(
        (weights["clip"] * components["clip"] + weights["siglip"] * components["siglip"])
        / (weights["clip"] + weights["siglip"]), 6,
    )
    probability = np.round((1 - weights["large"]) * intermediate + weights["large"] * components["large"], 6)
    return eval_keys[KEYS].assign(p_positive=probability), receipts, states


def select_parameters(train, graph_data, features, indices, config):
    selection = config["selection"]
    halves = [set(keys) for keys in graph_data["fitting_halves"]]
    if halves[0] & halves[1] or (halves[0] | halves[1]) != set(train["painting"]):
        raise ValueError("Inner split does not cover disjoint outer fitting paintings")
    tables = [train.loc[train["painting"].isin(half)] for half in halves]
    panels = [make_panel(table) for table in tables]
    inner_graphs = [deserialize_graph(graph) for graph in graph_data["inner_fitting_graphs"]]

    def trial(ratio, fraction, value):
        predictions, receipt = [], []
        for i in (0, 1):
            predicted, receipts, _ = predict(
                panels[i], panels[1 - i], tables[1 - i][KEYS], features, indices, config, ratio, fraction,
                inner_graphs[i] if fraction else None,
            )
            predictions.append(predicted)
            receipt.append(receipts)
        result = core_score(train, pd.concat(predictions, ignore_index=True))
        return {"value": value, "contrast": result["cultural_contrast"],
                "log_loss": result["log_loss"], "fit_receipts": receipt}

    ridge_trials = [trial(ratio, 0, ratio) for ratio in selection["deviation_ridge_ratios"]]
    tolerance = selection["log_loss_guardrail_bits"]
    ridge, _ = choose_inner(ridge_trials, tolerance)
    graph_trials = [trial(ridge["value"], fraction, fraction) for fraction in selection["trace_neutral_graph_fractions"]]
    graph, within = choose_inner(graph_trials, tolerance, baseline_loss=ridge["log_loss"])
    return {
        "ridge_ratio": ridge["value"], "graph_fraction": graph["value"],
        "graph_selection_within_inner_guardrail": within,
        "ridge_trials": ridge_trials, "graph_trials": graph_trials,
    }


def model_payload(states):
    return {
        f"{system}:{encoder}:{field}": np.asarray(getattr(head, field))
        for system, model in states.items() for encoder, head in model.items()
        for field in ("intercept", "language_intercepts", "shared", "deviations", "languages")
    }


def seal_outer_fold(destination, identity, train, held, graph_data, selected, features, indices, config):
    outer_fit, outer_held = make_panel(train), make_panel(held)
    graph = deserialize_graph(graph_data["outer_fitting_graph"])
    result, all_receipts, all_states = {}, {}, {}
    for system, fraction in ((CONTROL, 0.0), (CANDIDATE, selected["graph_fraction"])):
        preds, receipts, states = predict(
            outer_fit, outer_held, held[KEYS], features, indices, config, selected["ridge_ratio"], fraction,
            graph if fraction else None,
        )
        result[system], all_receipts[system], all_states[system] = preds, receipts, states
    with TemporaryDirectory(dir=destination.parent) as tmp_dir:
        published = Path(tmp_dir) / "complete"
        published.mkdir()
        for system, preds in result.items():
            preds.to_csv(published / f"{system}.csv", index=False, float_format="%.6f")
        np.savez_compressed(published / "coefficients.npz", **model_payload(all_states))
        write_json(published / "fit.json", {"selection": selected, "refit_receipts": all_receipts})
        hashes = {f.name: file_sha256(f) for f in published.iterdir()}
        write_json(published / "seal.json", {"identity": identity, "files": hashes})
        published.rename(destination)


def verified_fold(path, identity, held):
    seal = read_json(path / "seal.json")
    expected = {"coefficients.npz", "fit.json", f"{CONTROL}.csv", f"{CANDIDATE}.csv"}
    if (seal["identity"] != identity or set(seal["files"]) != expected
            or set(f.name for f in path.iterdir()) != expected | {"seal.json"}):
        raise ValueError(f"Incompatible or incomplete sealed fold: {path}")
    for name, fingerprint in seal["files"].items():
        if file_sha256(path / name) != fingerprint:
            raise ValueError(f"Corrupted sealed fold: {path / name}")
    frames = {name: pd.read_csv(path / f"{name}.csv") for name in (CONTROL, CANDIDATE)}
    for frame in frames.values():
        validate_predictions(frame, held)
    return frames


def process_peak_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ("peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "page", "peak_page")
            ]

        handle_fn = ctypes.windll.kernel32.GetCurrentProcess
        handle_fn.restype = wintypes.HANDLE
        counters = Counters()
        counters.size = ctypes.sizeof(counters)
        read_counters = ctypes.windll.psapi.GetProcessMemoryInfo
        read_counters.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        read_counters.restype = wintypes.BOOL
        ok = read_counters(handle_fn(), ctypes.byref(counters), counters.size)
        return int(counters.peak) if ok else None

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def run(inputs, run_dir, config):
    gold, reference, features, indices = inputs
    started = perf_counter()
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "platform": sys.platform, "threadpools": threadpool_info()}
    source = {file: file_sha256(ROOT / file) for file in SOURCE}
    receipt = {"input_sha256": config["input_sha256"], "config_sha256": file_sha256(CONFIG), "source": source,
               "strategy_commit": config["strategy_commit"], "versions": versions}
    run_identity = structured_sha(receipt)
    manifest = run_dir / "run.json"
    if manifest.exists():
        if read_json(manifest)["identity"] != run_identity:
            raise ValueError("Existing run has a different source, input, configuration, or runtime identity")
    else:
        if any(run_dir.iterdir()):
            raise ValueError("New run directory must be empty")
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        write_json(manifest, receipt | {"identity": run_identity, "git_commit": git_commit.stdout.strip() or None})
    # The one common nuisance graph uses all three normalized blocks with equal L2 contribution.
    combined = np.ascontiguousarray(np.concatenate(list(features.values()), axis=1) / np.sqrt(len(features)))
    graph_results = {}
    statusfile = run_dir / "decision.json"
    write_json(statusfile, {"status": "IN_PROGRESS", "identity": run_identity})
    stage = "fitting_only_graph"
    try:
        for fold in sorted(gold["fold"].unique()):
            fold = int(fold)
            scope = gold.loc[gold["fold"] != fold]
            path = run_dir / f"graph_fold_{fold}.json"
            if path.exists():
                sealed = read_json(path)
                graph = sealed["graph"]
                if (sealed["identity"] != run_identity or int(graph["fold"]) != fold
                        or sealed["sha256"] != structured_sha(graph)):
                    raise ValueError(f"Incompatible or corrupt fitting graph {path}")
            else:
                before = perf_counter()
                graph = prepare_fold_graphs(scope, combined, indices, config, fold)
                graph["seconds"] = perf_counter() - before
                write_json(path, {"identity": run_identity, "graph": graph, "sha256": structured_sha(graph)})
            graph_results[fold] = graph
            print(f"Fold {fold} fitting-only graph: {graph['stability']}", flush=True)

        all_stable = all(g["stability"]["pass"] for g in graph_results.values())
        if not all_stable:
            outcome = {
                "status": "NO_GO_GRAPH_STABILITY", "identity": run_identity, "outer_scored": False,
                "reason": "At least one required outer fitting scope failed the frozen graph evidence checks",
                "fit_scope_checks": {fold: g["stability"] for fold, g in graph_results.items()},
                "elapsed_seconds": perf_counter() - started, "peak_rss_bytes": process_peak_bytes(),
            }
            write_json(statusfile, outcome)
            return outcome

        stage = "nested_selection_and_outer_fits"
        predictions = {CONTROL: [], CANDIDATE: []}
        support = []
        for fold, graph in graph_results.items():
            train, held = gold.loc[gold["fold"] != fold], gold.loc[gold["fold"] == fold]
            language_support = train.groupby("language")["painting"].nunique()
            support.append(held.assign(fitting_paintings=held["language"].map(language_support).fillna(0).astype(int)))
            path = run_dir / f"fold_{fold}"
            identity = {"run": run_identity, "fold": fold, "graph_sha256": structured_sha(graph)}
            if not path.exists():
                selected = select_parameters(train, graph, features, indices, config)
                seal_outer_fold(path, identity, train, held, graph, selected, features, indices, config)
            for name, frame in verified_fold(path, identity, held).items():
                predictions[name].append(frame)
            print(f"Fold {fold}: both heads sealed and verified", flush=True)

        stage = "outer_scoring"
        complete = {name: pd.concat(frames, ignore_index=True) for name, frames in predictions.items()}
        oof = complete | {REFERENCE: reference}
        metrics = evaluate(pd.concat(support, ignore_index=True), oof, config["decision"])
        outcome = metrics["decision"] | {"identity": run_identity, "outer_scored": True,
                                       "elapsed_seconds": perf_counter() - started,
                                       "peak_rss_bytes": process_peak_bytes()}
        for name, frame in oof.items():
            frame.sort_values(KEYS).to_csv(run_dir / f"oof_{name.replace('+', '_')}.csv", index=False, float_format="%.6f")
        write_json(run_dir / "metrics.json", metrics)
        outcome["metrics_sha256"] = file_sha256(run_dir / "metrics.json")
        write_json(statusfile, outcome)
        return outcome
    except (Exception, KeyboardInterrupt) as error:
        write_json(statusfile, {"status": "FAILED", "stage": stage, "reason": str(error), "identity": run_identity})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Unpacked pinned EXP-023 input directory")
    parser.add_argument("--run-dir", type=Path, required=True, help="New or compatible existing EXP-032 results directory")
    args = parser.parse_args()
    config = read_json(CONFIG)
    inputs = load_inputs(args.inputs, config, args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=config["threads"]):
        result = run(inputs, args.run_dir, config)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
