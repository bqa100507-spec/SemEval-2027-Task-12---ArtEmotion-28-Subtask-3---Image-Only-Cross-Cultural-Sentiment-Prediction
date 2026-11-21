"""Checksum-verified nested CV and a separate official-training refit."""

import json
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import data
from .baseline import file_sha256
from .cli import json_output
from .competition_challenger import (check_hash, component_predictions, crossfit, fit_heads, frame_sha,
                                    load_bundle, metadata_identity, official_features, official_pairs,
                                    write_bundle)
from .competition_ridge import Head
from .competition_stack import ENCODERS, aligned_values, blend, evaluate
from .contrast_study import new_output
from .e50_heads import read_json
from .incremental import valid_vectors
from .metrics import score


def peak_memory_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak", "working", "peak_paged", "paged",
                                                     "peak_nonpaged", "nonpaged", "page", "peak_page")]

        counts = Counters()
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        if not ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(ctypes.windll.kernel32.GetCurrentProcess()),
                                                      ctypes.byref(counts), ctypes.sizeof(counts)):
            raise OSError("Could not read peak process memory")
        return int(counts.peak)
    import resource

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def load_protocol(root, config):
    filename = "training/protocol.csv"
    check_hash(root / filename, config["p2_input_sha256"][filename])
    protocol = pd.read_csv(root / filename)
    fields = ["painting", "image", "image_sha256", "role", "fold"]
    if (list(protocol.columns) != fields or protocol[fields].isna().any().any()
            or any(protocol[name].duplicated().any() for name in fields[:3])
            or set(protocol.role) != {"development", "holdout"}
            or len(protocol.loc[protocol.role.eq("holdout") & protocol.fold.eq(0)]) != config["p2"]["sealed_paintings"]):
        raise ValueError("Historical P2 protocol is inconsistent")
    return protocol


def load_p2(root, config):
    for name, digest in config["p2_input_sha256"].items():
        check_hash(root / name, digest)
    fields = ["painting", "image", "image_sha256", "role", "fold"]
    protocol = load_protocol(root, config)
    gold = pd.read_csv(root / "training/pairs.csv").sort_values(data.KEYS, ignore_index=True)
    data.unique_pairs(gold)
    paintings = gold[fields].drop_duplicates().sort_values("painting", ignore_index=True)
    expected = protocol.loc[protocol.role.eq("development")].sort_values("painting", ignore_index=True)
    if (not paintings.equals(expected) or not gold.split.eq("train").all()
            or not gold.n.ge(data.MIN_ANNOTATORS).all() or not gold.k.between(0, gold.n).all()
            or (gold[["n", "k"]].to_numpy() % 1 != 0).any()
            or gold.language.nunique() != config["p2"]["languages"]
            or gold.groupby("fold").size().tolist() != config["p2"]["fold_pair_counts"]
            or not paintings.groupby("fold").size().eq(config["p2"]["paintings_per_fold"]).all()):
        raise ValueError("Historical P2 painting groups or vote eligibility changed")
    hashes = read_json(root / "visual/image_hashes.json")
    if not paintings.image.map(hashes).eq(paintings.image_sha256).all():
        raise ValueError("P2 feature image digests differ")
    features = {}
    with np.load(root / "visual/features.npz", allow_pickle=False) as cache:
        names = cache["names"].tolist()
        index = paintings.image.map({name: i for i, name in enumerate(names)})
        if len(set(names)) != len(names) or index.isna().any():
            raise ValueError("Historical feature coverage changed")
        for native, active in zip(("clip", "siglip", "large"), ENCODERS):
            values = cache[native][index.to_numpy(dtype=int)]
            valid_vectors(values, len(paintings), config["feature_widths"][active])
            features[active] = pd.DataFrame(values, index=paintings.painting)
    ref = pd.read_csv(root / "history/exp022_R0.csv")
    data.require_same_pairs(gold, ref)
    if not gold[data.KEYS + ["fold"]].merge(ref, on=data.KEYS, validate="one_to_one").eval("fold_x == fold_y").all():
        raise ValueError("Historical E50+L reference uses different folds")
    return gold[data.KEYS + ["fold", "k", "n"]], features, ref


def archive_heads(path, arm):
    with np.load(path, allow_pickle=False) as saved:
        return {encoder: Head(float(saved[f"{arm}:{key}:intercept"]),
                              saved[f"{arm}:{key}:language_intercepts"], saved[f"{arm}:{key}:shared"],
                              saved[f"{arm}:{key}:deviations"], tuple(saved[f"{arm}:{key}:languages"]))
                for key, encoder in zip(("clip", "siglip", "large"), ENCODERS)}


def verify_archive(archive, gold, features, reference, config):
    for file, digest in config["archive_sha256"].items():
        check_hash(archive / file, digest)
    run = read_json(archive / "run.json")
    if run["git_commit"] != config["exp032_source"]:
        raise ValueError("EXP-032 training source pin differs")
    expected = read_json(archive / "metrics.json")["systems"]
    scores = {}
    for arm, file in (("E50+L", "oof_E50_L.csv"), ("no_graph", "oof_no_graph.csv"),
                      ("learned_graph", "oof_learned_graph.csv")):
        saved = pd.read_csv(archive / file)
        full = aligned_values(gold, saved, ["p_positive"]).ravel()
        if arm == "E50+L":
            np.testing.assert_array_equal(full, aligned_values(gold, reference, ["p_positive"]).ravel())
        else:
            for fold, held in gold.groupby("fold"):
                path = archive / f"fold_{fold}"
                seal = read_json(path / "seal.json")
                if seal["identity"]["run"] != run["identity"] or seal["identity"]["fold"] != fold:
                    raise ValueError(f"Sealed archive fold {fold} differs from EXP-032")
                for name, digest in seal["files"].items():
                    check_hash(path / name, digest)
                receipt = read_json(path / "fit.json")
                if receipt["selection"]["ridge_ratio"] != config["ridge"]["deviation_ratio"]:
                    raise ValueError("Archived fixed ridge ratio differs")
                parts = component_predictions(archive_heads(path / "coefficients.npz", arm), held[data.KEYS], features)
                pred = blend(aligned_values(held, parts, ENCODERS), config["fixed_weights"], fixed=True)
                np.testing.assert_array_equal(pred, full[held.index])
                np.testing.assert_array_equal(pred, aligned_values(held, pd.read_csv(path / f"{arm}.csv"),
                                                                   ["p_positive"]).ravel())
        scores[arm] = {}
        for tag, select in (("full", gold.index), ("test_like", gold.index[~gold.language.isin(config["test_like_excluded"])])):
            metrics = score(gold.loc[select, data.KEYS + ["k", "n"]],
                            gold.loc[select, data.KEYS].assign(p_positive=full[select]))
            keys = ("cultural_contrast", "log_loss")
            scores[arm][tag] = {key: metrics[key] for key in keys}
            if tag == "full":
                np.testing.assert_allclose([metrics[k] for k in keys], [expected[arm]["pooled"][k] for k in keys],
                                           rtol=0, atol=1e-12)
        for fold, held in gold.groupby("fold"):
            metrics = score(held[data.KEYS + ["k", "n"]], held[data.KEYS].assign(p_positive=full[held.index]))
            np.testing.assert_allclose([metrics[k] for k in keys],
                                       [expected[arm]["folds"][str(fold)][k] for k in keys], rtol=0, atol=1e-12)
    return scores


def save_csv(frame, path):
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.6f")


def fold_fit(path, train, held, features, config, fold, identity):
    proof = {"fold": fold, "source": identity["source_sha256"], "fit_sha256": frame_sha(train),
             "held_sha256": frame_sha(held)}
    if not path.exists():
        tick = time.perf_counter()
        stack = crossfit(train, features, config, config["inner_seed"] + fold)
        heads, receipts = fit_heads(train, features, config)
        components = component_predictions(heads, held, features)
        values = aligned_values(held, components, ENCODERS)
        with new_output(path) as stage:
            write_bundle(stage / "outer_heads", heads, mode="learned", weights=stack["weights"],
                         provenance={"scope": "outer_cv_only", **proof})
            predictions = components.assign(fixed=blend(values, config["fixed_weights"], fixed=True),
                                            learned=blend(values, np.asarray(stack["weights"]), fixed=False))
            save_csv(predictions, stage / "predictions.csv")
            json_output({"fit": receipts, "stack": stack, "seconds": time.perf_counter() - tick}, stage / "fit.json")
            files = ["predictions.csv", "fit.json", "outer_heads/manifest.json"]
            json_output({"proof": proof, "files": {name: file_sha256(stage / name) for name in files}}, stage / "seal.json")
    seal = read_json(path / "seal.json")
    if seal["proof"] != proof:
        raise ValueError(f"Completed fold provenance differs in {path}")
    for name, digest in seal["files"].items():
        check_hash(path / name, digest)
    heads, manifest = load_bundle(path / "outer_heads")
    predictions = pd.read_csv(path / "predictions.csv")
    reconstructed = component_predictions(heads, held, features)
    values = aligned_values(held, reconstructed, ENCODERS)
    np.testing.assert_array_equal(values, aligned_values(held, predictions, ENCODERS))
    np.testing.assert_array_equal(blend(values, config["fixed_weights"], fixed=True), predictions.fixed)
    np.testing.assert_array_equal(blend(values, np.asarray(manifest["weights"]), fixed=False), predictions.learned)
    return predictions, read_json(path / "fit.json")


def run_cv(args, config):
    start = time.perf_counter()
    identity = metadata_identity(config)
    with threadpool_limits(limits=config["threads"]):
        gold, features, ref = load_p2(args.inputs, config)
        baseline = verify_archive(args.archive, gold, features, ref, config)
        print("Verified all archived EXP-032 OOF keys, coefficients, pooled and fold metrics", flush=True)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        path = args.run_dir / "identity.json"
        if path.exists():
            if read_json(path) != identity:
                raise ValueError("Run fingerprints changed. Select a new run directory.")
        else:
            json_output(identity, path)
        results = args.run_dir / "results"
        if results.exists():
            verified_decision(args.run_dir, config)
            print(f"Completed CV output already verified: {results}", flush=True)
            return
        predictions, receipts = [], {}
        for fold, chunk in gold.groupby("fold"):
            train, held = gold.loc[gold.fold.ne(fold)], chunk[data.KEYS]
            if set(train.painting) & set(held.painting):
                raise ValueError("Outer fold painting overlap")
            preds, details = fold_fit(args.run_dir / f"fold_{fold}", train, held, features, config, int(fold), identity)
            predictions.append(preds)
            receipts[int(fold)] = details
            print(f"Verified fold {fold} in {details['seconds']:.1f}s, weights {details['stack']['weights']}", flush=True)
        outputs = gold[data.KEYS].merge(pd.concat(predictions, ignore_index=True), on=data.KEYS, validate="one_to_one")
        outputs["reference"] = aligned_values(outputs, ref, ["p_positive"]).ravel()
        archived = pd.read_csv(args.archive / "oof_no_graph.csv")
        np.testing.assert_array_equal(outputs.fixed, aligned_values(outputs, archived, ["p_positive"]).ravel())
        metrics, painted = evaluate(gold, outputs, config)
        weights = np.asarray([item["stack"]["weights"] for item in receipts.values()])
        metrics |= {"archive_replay": baseline, "weights_by_fold": weights.tolist(), "weights_sd": weights.std(axis=0).tolist(),
                    "weights_mean": weights.mean(axis=0).tolist(), "weights_min": weights.min(axis=0).tolist(),
                    "weights_max": weights.max(axis=0).tolist(), "threads": config["threads"],
                    "peak_rss_bytes": peak_memory_bytes(), "wall_seconds": time.perf_counter() - start,
                    "completed_fold_seconds": sum(item["seconds"] for item in receipts.values())}
        with new_output(results) as stage:
            save_csv(outputs.merge(gold[data.KEYS + ["fold"]], on=data.KEYS, validate="one_to_one"), stage / "oof.csv")
            save_csv(painted, stage / "painting_contrast.csv")
            json_output(metrics, stage / "metrics.json")
            json_output({"identity_sha256": file_sha256(path), "files": {p.name: file_sha256(p) for p in stage.iterdir()}},
                        stage / "seal.json")
        print(json.dumps(metrics["decision"]), flush=True)


def verified_decision(run_dir, config):
    directory = run_dir / "results"
    seal = read_json(directory / "seal.json")
    check_hash(run_dir / "identity.json", seal["identity_sha256"])
    if read_json(run_dir / "identity.json") != metadata_identity(config):
        raise ValueError("CV identity differs from the refit source or runtime")
    if set(seal["files"]) != {"oof.csv", "painting_contrast.csv", "metrics.json"}:
        raise ValueError("Incomplete CV result seal")
    for file, digest in seal["files"].items():
        check_hash(directory / file, digest)
    decision = read_json(directory / "metrics.json")["decision"]
    if decision["selected"] not in {"fixed", "learned"}:
        raise ValueError("Frozen CV choice is invalid")
    return decision


def refit(args, config):
    start = time.perf_counter()
    decision = verified_decision(args.run_dir, config)
    if args.output.exists():
        raise ValueError(f"Candidate bundle already exists: {args.output}")
    protocol = load_protocol(args.inputs, config)
    train = official_pairs(args.data_dir, config, protocol)
    features, sources = official_features(args, train, protocol, config)
    with threadpool_limits(limits=config["threads"]):
        # The CV decision enters only as the already frozen choice of fusion algorithm.
        learned = decision["selected"] == "learned"
        meta = crossfit(train, features, config, config["inner_seed"]) if learned else None
        heads, receipts = fit_heads(train, features, config)
    where = {"scope": "official_split_train_final_refit", "training_pairs": len(train),
             "training_paintings": train.painting.nunique(), "training_votes": int(train.n.sum()),
             "training_pair_sha256": frame_sha(train[data.KEYS + ["image", "k", "n"]]),
             "official_training": config["official_training"], "training_features": sources,
             "frozen_cv_metrics_sha256": file_sha256(args.run_dir / "results" / "metrics.json"),
             "ridge_config": config["ridge"], "fit_receipts": receipts, "crossfit_receipt": meta,
             "runtime": metadata_identity(config), "peak_rss_bytes": peak_memory_bytes(),
             "wall_seconds": time.perf_counter() - start}
    write_bundle(args.output, heads, mode=decision["selected"], weights=meta["weights"] if learned else config["fixed_weights"],
                 provenance=where)
    print(f"Exported {decision['selected']} heads fitted on {len(train)} official training pairs to {args.output}", flush=True)
