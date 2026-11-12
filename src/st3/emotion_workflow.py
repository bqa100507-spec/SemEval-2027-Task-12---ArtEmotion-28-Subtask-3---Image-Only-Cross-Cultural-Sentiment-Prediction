"""Pinned parent evidence, bounded memory target caching and descriptive audits."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from . import data
from .baseline import file_sha256
from .caption_comparison import image_banks
from .caption_data import allowed_captions, inner_partition, training_captions
from .caption_text import load_teacher
from .cli import ROOT, json_output
from .contrast_study import new_output
from .emotion_targets import (EMOTIONS, POLARITY, emotion_means, geometry_statistics, normalize,
                              project_targets)
from .gradient_experiment import choose as choose019
from .gradient_workflow import identity019, validate_parent
from .polarity_targets import average_pairwise_cosine
from .polarity_workflow import (ExecutionBudget, StageBlocked, config018, fingerprint, load_geometry,
                                load_raw, old_inputs, read_sealed, seal)


def config020():
    return json.loads((ROOT/"configs/exp020.json").read_text())


def old_args(args):
    return SimpleNamespace(**{**vars(args), "run_dir":args.exp018_dir})


def validate_inputs(args):
    parent_identity, train, predictions, _, count = validate_parent(args)
    cfg = config020()
    if cfg["emotions"]!=list(EMOTIONS) or cfg["polarity_indices"]!=POLARITY.tolist() or cfg["text_weight"]!=.1:
        raise ValueError("Frozen emotion order, polarity mapping or lambda changed")
    if file_sha256(ROOT/"configs/exp019.json") != cfg["exp019_config_sha256"]:
        raise ValueError("Pinned EXP-019 config changed")
    if not (args.exp019_dir/"decision/run.json").is_file():
        raise StageBlocked("Complete EXP-019 on its original branch with run_exp019.ps1 -Stage CV then -Stage Verify")
    identity = json.loads((args.exp019_dir/"decision/run.json").read_text())["identity"]
    current = identity019(args, parent_identity)
    expected_sources = identity["runtime_and_source"]["source_sha256"]
    if any(current["runtime_and_source"]["source_sha256"].get(k) != v for k, v in expected_sources.items()):
        raise ValueError("Historical EXP-019 source changed")
    current["runtime_and_source"]["source_sha256"] = expected_sources
    if current != identity:
        raise ValueError("EXP-019 config/runtime/source identity changed")
    for name, digest in cfg["exp019_manifests"].items():
        path = args.exp019_dir/name
        if file_sha256(path) != digest:
            raise ValueError(f"Pinned EXP-019 evidence changed: {name}")
        read_sealed(path.parent, identity)
    cv = read_sealed(args.exp019_dir/"cv", identity)
    verification = read_sealed(args.exp019_dir/"verification", identity)
    decision = read_sealed(args.exp019_dir/"decision", identity)
    attempts = sorted((args.exp019_dir/"verification_attempts").glob("*.json"))
    latest = json.loads(attempts[-1].read_text()) if attempts else {}
    if latest.get("status") != "PASS" or latest.get("cv_manifest_sha256") != file_sha256(args.exp019_dir/"cv/run.json"):
        raise StageBlocked("Latest EXP-019 replay not successful; rerun Verify on original branch")
    summary = pd.read_csv(args.exp019_dir/"cv/summary.csv", index_col="model", float_precision="round_trip")
    expected = choose019(summary, config018(), True)
    if any(decision[k] != v for k, v in expected.items()) or verification["checkpoints_replayed"] != 15:
        raise ValueError("EXP-019 decision/replay coverage differs")
    for phase in ("cv", "verification"):
        if decision[phase+"_manifest_sha256"] != file_sha256(args.exp019_dir/phase/"run.json"):
            raise ValueError("EXP-019 decision evidence differs")
    for key, digest in cv["seed_manifests"].items():
        fold, name = key.split("/")
        path = args.exp019_dir/"work"/f"fold_{fold}"/name
        read_sealed(path, identity)
        if file_sha256(path/"run.json") != digest:
            raise ValueError("EXP-019 seed manifest changed")
    if any((args.exp019_dir/phase).exists() for phase in ("confirm", "dev")):
        raise ValueError("Expected locked confirmation/public Development to remain sealed")
    return parent_identity, train, predictions, decision, count


def context020(args, *, require_preflight=True):
    parent_identity, train, predictions, decision, count = validate_inputs(args)
    identity = {"runtime_and_source":fingerprint(old_args(args)), "config_sha256":file_sha256(ROOT/"configs/exp020.json"),
                "runners":{p.name:file_sha256(p) for p in (ROOT/"scripts/run_exp020.py", ROOT/"scripts/run_exp020.ps1")}}
    if require_preflight:
        read_sealed(args.run_dir/"preflight", identity)
    return SimpleNamespace(config=config018(), protocol=config020(), identity=identity, parent_identity=parent_identity,
                           train=train, predictions=predictions, previous_decision=decision, history_count=count,
                           banks=image_banks(args, train))


def run_preflight(args):
    started = time.perf_counter()
    ctx = context020(args, require_preflight=False)
    if (args.run_dir/"preflight").exists():
        read_sealed(args.run_dir/"preflight", ctx.identity); return
    history = {str(p.resolve()):file_sha256(p) for root in (args.exp018_dir, args.exp019_dir)
               for p in root.rglob("*") if p.is_file()}
    with new_output(args.run_dir/"preflight") as stage:
        json_output(history, stage/"protected_file_sha256.json")
        json_output({"EXP019":ctx.previous_decision, "cv_models":30, "training_stages":60,
                     "reference_source":"EXP-018 M0/M3/B1/B3; no historical refits",
                     "locked_paintings_sealed":236}, stage/"provenance.json")
        seal(stage, ctx.identity, {"status":"INPUTS_VERIFIED", "protected_files":len(history),
              "historical_files":ctx.history_count, "seconds":time.perf_counter()-started})
    print(f"Verified EXP-018/019 and preserved {len(history)} files; confirmation sealed", flush=True)


def preserve(args, ctx):
    read_sealed(args.run_dir/"preflight", ctx.identity)
    history = json.loads((args.run_dir/"preflight/protected_file_sha256.json").read_text())
    actual = {str(p.resolve()) for root in (args.exp018_dir, args.exp019_dir) for p in root.rglob("*") if p.is_file()}
    if actual != set(history):
        raise ValueError("Historical files added or removed")
    for name, digest in history.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"Historical artifact changed: {name}")
    return len(history)


def load_audit(args, ctx):
    read_sealed(args.run_dir/"audit", ctx.identity)
    root = args.run_dir/"audit"
    return (pd.read_csv(root/"pairs.csv"), np.load(root/"emotion_means.npy", mmap_mode="r", allow_pickle=False),
            np.load(root/"counts.npy", allow_pickle=False))


def geometry_directory(args, fold, phase):
    return args.run_dir/"targets"/f"fold_{fold}"/phase


def read_targets(args, ctx, fold, phase, partition):
    directory = geometry_directory(args, fold, phase)
    metadata = read_sealed(directory, ctx.identity)
    parent = args.exp018_dir/"work"/f"fold_{fold}"/"geometry"/phase
    load_geometry(parent, ctx.parent_identity, partition.painting)
    if metadata["parent_pca_manifest_sha256"] != file_sha256(parent/"run.json"):
        raise ValueError("Reused PCA transform changed")
    records = pd.read_csv(directory/"pairs.csv")
    data.require_same_pairs(partition, records)
    if metadata["fit_paintings"] != sorted(partition.painting.unique()):
        raise ValueError("Emotion targets cross the training boundary")
    with np.load(directory/"targets.npz", allow_pickle=False) as archive:
        bundle = {name:archive[name] for name in archive.files}
    valid = bundle["valid"]
    for condition in ("A8", "E8"):
        value = bundle[condition]
        if (value.shape[:2] != valid.shape or not np.isfinite(value).all() or
                not np.allclose(np.linalg.norm(value[valid], axis=-1), 1., atol=2e-6) or np.count_nonzero(value[~valid])):
            raise ValueError("Invalid eight-channel geometry/masks")
    return {"records":records, **bundle}


def write_geometry_audit(stage, prefix, records, values, valid, polarity, polarity_valid, threshold):
    summary, matrix, counts, variation, similarities = geometry_statistics(records, values, valid, polarity, polarity_valid, threshold)
    pd.DataFrame(matrix, index=EMOTIONS, columns=EMOTIONS).to_csv(stage/f"{prefix}_cosine.csv")
    pd.DataFrame(counts, index=EMOTIONS, columns=EMOTIONS).to_csv(stage/f"{prefix}_coobserved.csv")
    variation.to_csv(stage/f"{prefix}_variation.csv", index=False)
    return summary, similarities


def projected_dispersion(selected, vectors, pca_path):
    with np.load(pca_path, allow_pickle=False) as archive:
        components, mean = archive["components"], archive["mean"]
    keys = sorted(selected.caption_id.unique())
    projected = {}
    with threadpool_limits(limits=4):
        for start in range(0, len(keys), 512):
            batch = keys[start:start+512]
            values = (np.stack([vectors[key] for key in batch])-mean) @ components.T
            unit, valid = normalize(values, np.ones(len(values), dtype=bool))
            projected.update({key:value for key, value, keep in zip(batch, unit, valid) if keep})
    records = []
    for key, group in selected.groupby(data.KEYS+["emotion"]):
        values = [projected[k] for k in group.caption_id if k in projected]
        if len(values) >= 2:
            records.append({"painting":key[0], "language":key[1], "emotion":key[2], "captions":len(values),
                            "caption_mean_pairwise_cosine":average_pairwise_cosine(np.stack(values))})
    return pd.DataFrame(records)


def run_audit(args):
    started = time.perf_counter(); ctx = context020(args)
    budget = ExecutionBudget(args.max_seconds or 600.)
    if (args.run_dir/"audit").exists():
        read_sealed(args.run_dir/"audit", ctx.identity)
        # A partially completed audit may still need projected partition caches.
        raw = load_audit(args, ctx)
        vectors = captions = None
    else:
        _, _, audit = old_inputs(old_args(args))
        vectors, _ = load_teacher(args.exp017_dir/"text/clip", ctx.config["teacher"], audit, ctx.train.painting.unique())
        captions = training_captions(args.data_dir, allowed_paintings=ctx.train.painting.unique())
        selected = allowed_captions(captions, ctx.train)
        pr, polarity, pmask = load_raw(old_args(args), ctx.parent_identity)
        with new_output(args.run_dir/"audit") as stage:
            records, means, counts, dispersion = emotion_means(captions, ctx.train, vectors, destination=stage/"emotion_means.npy")
            records.to_csv(stage/"pairs.csv", index=False)
            np.save(stage/"counts.npy", counts, allow_pickle=False)
            dispersion.to_csv(stage/"raw_caption_dispersion.csv", index=False)
            pi = pd.MultiIndex.from_frame(pr[data.KEYS]).get_indexer(pd.MultiIndex.from_frame(records[data.KEYS]))
            rawsummary, similarities = write_geometry_audit(stage, "raw", records, means, counts>0,
                polarity[pi, 1:], pmask[pi, 1:], ctx.protocol["near_identical_cosine"])
            coverage = []
            for channel, emotion in enumerate(EMOTIONS):
                present = counts[:, channel] > 0
                coverage.append({"emotion":emotion, "polarity":"positive" if channel<4 else "negative",
                    "annotation_rows":int(counts[:, channel].sum()), "pairs":int(present.sum()),
                    "pair_coverage":float(present.mean()), "one_caption_targets":int((counts[:,channel]==1).sum()),
                    "one_caption_fraction":float((counts[present,channel]==1).mean()) if present.any() else None,
                    "two_plus_caption_targets":int((counts[:,channel]>=2).sum())})
            pd.DataFrame(coverage).to_csv(stage/"emotion_coverage.csv", index=False)
            language = pd.concat([records.assign(emotion=emotion, count=counts[:,channel]) for channel,emotion in enumerate(EMOTIONS)])
            language.groupby(["language","emotion"]).agg(annotation_rows=("count","sum"),eligible_pairs=("count",lambda x:int(x.gt(0).sum())),
                                                         two_plus_pairs=("count",lambda x:int(x.ge(2).sum()))).to_csv(stage/"language_coverage.csv")
            tokens = pd.read_csv(args.exp017_dir/"text/clip/tokens.csv")
            tokens = selected[["annotation_id", "language"]].merge(tokens.drop(columns=["language"]), on="annotation_id", validate="one_to_one")
            if len(tokens) != len(selected):
                raise ValueError("Original tokenizer diagnostics miss training annotation rows")
            tokens.groupby("language").agg(annotation_rows=("tokens","size"),tokens_mean=("tokens","mean"),
                                             tokens_max=("tokens","max"),truncation_rate=("truncated","mean")).to_csv(stage/"tokenizer_languages.csv")
            pairaudit = records.copy()
            pairaudit["emotion_categories"] = (counts>0).sum(axis=1)
            pairaudit["has_two_caption_emotion"] = (counts>=2).any(axis=1)
            pairaudit["mean_raw_polarity_cosine"] = np.nanmean(similarities, axis=1)
            pairaudit.to_csv(stage/"pair_reliability.csv", index=False)
            summary = {"pairs":len(records), "paintings":ctx.train.painting.nunique(), "annotation_rows":len(selected),
                "distinct_emotion_counts":{str(n):int(((counts>0).sum(1)==n).sum()) for n in range(9)},
                "one_category_pairs":int(((counts>0).sum(1)==1).sum()),"two_category_pairs":int(((counts>0).sum(1)==2).sum()),
                "three_plus_category_pairs":int(((counts>0).sum(1)>=3).sum()),
                "duplicate_annotation_rows_retained":int(selected.duplicated(data.KEYS+["emotion","caption_id"]).sum()),
                "positive_caption_rows":int(counts[:,:4].sum()),"negative_caption_rows":int(counts[:,4:].sum()),
                "positive_pair_coverage":int((counts[:,:4].sum(1)>0).sum()),"negative_pair_coverage":int((counts[:,4:].sum(1)>0).sum()),
                "available_channels":int((counts>0).sum()), "missing_channels":int((counts==0).sum()),
                "raw_geometry":rawsummary, "raw_mean_degenerate":int(sum(((counts[:,e]>0)&(np.linalg.norm(means[:,e],axis=1)<=1e-8)).sum() for e in range(8))),
                "tokenizer_source_sha256":file_sha256(args.exp017_dir/"text/clip/tokens.csv")}
            json_output(summary, stage/"summary.json")
            means.flush()
            means._mmap.close()
            del means
            seal(stage, ctx.identity, {"seconds":time.perf_counter()-started,"status":"RAW_AUDITED","locked_captions_retained":0})
        del polarity
        raw = load_audit(args, ctx)
    # Project all original inner/full partitions once; never fit PCA here.
    if vectors is None:
        _, _, audit = old_inputs(old_args(args))
        vectors, _ = load_teacher(args.exp017_dir/"text/clip", ctx.config["teacher"], audit, ctx.train.painting.unique())
        captions = training_captions(args.data_dir, allowed_paintings=ctx.train.painting.unique())
    pr, polarity, _ = load_raw(old_args(args), ctx.parent_identity)
    for fold in range(1, ctx.config["cv_folds"]+1):
        full = ctx.train[ctx.train.fold.ne(fold)]; inner, _ = inner_partition(full, ctx.config, fold)
        for phase, partition in (("inner",inner),("full",full)):
            budget.check(f"Audit fold {fold} {phase}")
            directory = geometry_directory(args, fold, phase)
            if directory.exists():
                read_targets(args, ctx, fold, phase, partition); continue
            parent = args.exp018_dir/"work"/f"fold_{fold}"/"geometry"/phase
            reference = load_geometry(parent, ctx.parent_identity, partition.painting)
            bundle = project_targets(*raw, pr, polarity, partition, parent/"targets.npz", reference)
            with new_output(directory) as stage:
                bundle["records"].to_csv(stage/"pairs.csv", index=False)
                np.savez(stage/"targets.npz", **{k:bundle[k] for k in ("A8","E8","valid","counts","polarity","polarity_valid")})
                summary, similarity = write_geometry_audit(stage, "projected", bundle["records"],bundle["E8"],bundle["valid"],
                                                          bundle["polarity"],bundle["polarity_valid"],ctx.protocol["near_identical_cosine"])
                pd.DataFrame(similarity, columns=EMOTIONS).to_csv(stage/"polarity_similarity.csv", index=False)
                selected = allowed_captions(captions, partition)
                projected_dispersion(selected, vectors, parent/"targets.npz").to_csv(stage/"projected_caption_dispersion.csv", index=False)
                seal(stage, ctx.identity, {"fit_paintings":sorted(partition.painting.unique()),
                    "parent_pca_manifest_sha256":file_sha256(parent/"run.json"), "geometry_summary":summary,
                    "emotion_projected_degenerate":bundle["emotion_projected_degenerate"],"common_mask_removed":bundle["common_mask_removed"]})
            print(f"Audited existing PCA fold {fold} {phase}, {len(partition)} pairs", flush=True)
    if not (args.run_dir/"audit_completion").exists():
        with new_output(args.run_dir/"audit_completion") as stage:
            seal(stage, ctx.identity, {"status":"AUDITED", "seconds":time.perf_counter()-started,
                "target_manifests":{f"{fold}/{phase}":file_sha256(geometry_directory(args,fold,phase)/"run.json")
                    for fold in range(1,ctx.config["cv_folds"]+1) for phase in ("inner","full")}})
    print(f"Training-only emotion audit completed in {time.perf_counter()-started:.2f}s", flush=True)


def stage_budget(args, ctx, phase):
    pilot = read_sealed(args.run_dir/"pilot", ctx.identity)
    seconds = pilot["estimated_cv_seconds"] if phase == "CV" else pilot["estimated_verification_seconds"]
    if phase in ("Confirm","Dev"):
        seconds = max(600., pilot["estimated_cv_seconds"]/3.)
    return ExecutionBudget(args.max_seconds or max(120., 1.5*seconds))
