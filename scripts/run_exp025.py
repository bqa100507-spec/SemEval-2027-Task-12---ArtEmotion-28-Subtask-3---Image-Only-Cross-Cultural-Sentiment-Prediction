"""Evaluate complete cached EXP-025 semantics on all five original P2 folds."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from st3.vlm_semantics import SemanticCache, atomic_json, file_hash, fingerprint
from st3.vlm_probe import evaluate_fold, summarize, contribution_evidence
from st3.caption_data import require_isolation


def load_inputs(inputs, cache_root, config):
    original = json.loads((ROOT / "configs/exp024.json").read_text(encoding="utf-8"))
    for name, expected in original["input_sha256"].items():
        if file_hash(inputs / name) != expected:
            raise ValueError("P2 required input checksum differs: " + name)
    train = pd.read_csv(inputs / "training/pairs.csv", float_precision="round_trip")
    protocol = pd.read_csv(inputs / "training/protocol.csv")
    require_isolation(train, protocol[protocol.role.eq("holdout")])
    expected = protocol[protocol.role.eq("development")][["painting", "image", "image_sha256", "fold"]].sort_values("painting").reset_index(drop=True)
    actual = train[expected.columns].drop_duplicates().sort_values("painting").reset_index(drop=True)
    if not actual.equals(expected) or len(actual) != 1335 or len(train) != 13600 or not train.split.eq("train").all():
        raise ValueError("Original P2 training/fold mapping differs")
    reference = pd.read_csv(inputs / "history/exp022_R0.csv", float_precision="round_trip")
    with np.load(inputs / "visual/features.npz", allow_pickle=False) as saved:
        indices = {name: i for i, name in enumerate(saved["names"].tolist())}
        take = [indices[name] for name in actual.image]
        blocks = [saved[name][take] for name in ("clip", "siglip", "large")]
        if any(not np.allclose(np.linalg.norm(b, axis=1), 1., atol=3e-6) for b in blocks):
            raise ValueError("Invalid historical normalized visual blocks")
        visual = dict(zip(actual.image, np.concatenate(blocks, axis=1) / np.sqrt(3)))
    cache = SemanticCache(cache_root, config)
    semantics, records, missing = {}, {}, []
    for row in actual.itertuples():
        record = cache.read(row.image_sha256)
        if not record or "embedding" not in record:
            missing.append(row.image_sha256)
        else:
            semantics[row.image] = np.array(record["embedding"], dtype=np.float32)
            records[row.image_sha256] = file_hash(cache.directory / (row.image_sha256 + ".json"))
    if missing:
        raise ValueError(f"Semantic cache incomplete: {1335-len(missing)}/1335 images. Run extraction first; heads never invoke the VLM.")
    report = json.loads((cache.directory / "extraction_report.json").read_text(encoding="utf-8"))
    if not report["full_P2_coverage"] or report.get("technical_status") != "PASS":
        raise ValueError("Complete technically valid extraction is required before scientific heads")
    return train, reference, visual, semantics, records


def run(inputs, cache, run_dir, config):
    started = time.perf_counter()
    train, reference, visual, semantics, records = load_inputs(inputs, cache, config)
    identity = {"config": config, "semantic_records_sha256": records,
                "source": {p: file_hash(ROOT / p) for p in ("src/st3/vlm_probe.py", "src/st3/vlm_semantics.py", "scripts/run_exp025.py")}}
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_path = run_dir / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Changed experiment/cache identity: choose a new run directory")
    atomic_json(identity_path, identity)
    all_predictions, fold_results = [], []

    def fold(number):
        path, metadata = run_dir / f"fold_{number}.csv", run_dir / f"fold_{number}.json"
        if metadata.exists():
            stored = json.loads(metadata.read_text(encoding="utf-8"))
            if not path.exists() or stored["identity"] != fingerprint(identity) or stored["sha256"] != file_hash(path):
                raise ValueError("Corrupt/incompatible completed fold")
            pred = pd.read_csv(path, float_precision="round_trip")
        else:
            tick = time.perf_counter()
            pred, details = evaluate_fold(train, visual, semantics, reference, config, number)
            temp = path.with_suffix(".partial");pred.to_csv(temp, index=False, float_format="%.6f", lineterminator="\n");temp.replace(path)
            atomic_json(metadata, {"identity": fingerprint(identity), "sha256": file_hash(path), "seconds": time.perf_counter()-tick, **details})
        if not pred.fold.eq(number).all() or set(pred.model) != {"A","B","C","D","C_shuffle","D_shuffle","E50+L"}:
            raise ValueError("Completed fold labels or comparison coverage differ")
        summary, _ = summarize(train[train.fold.eq(number)], pred)
        fold_results.append(summary.assign(fold=number));all_predictions.append(pred)
        print(f"Fold {number} complete\n" + summary.to_string(index=False), flush=True)

    if config["cv_folds"] != [1, 2, 3, 4, 5]:
        raise ValueError("EXP-025 requires all five original P2 folds")
    for number in config["cv_folds"]:
        fold(number)
    predictions = pd.concat(all_predictions, ignore_index=True)
    gold = train[train.fold.isin(predictions.fold.unique())]
    summary, paintings = summarize(gold, predictions)
    by_fold = pd.concat(fold_results)
    for metric in ("contrast", "macro_log_loss"):
        summary["fold_mean_" + metric] = summary.model.map(by_fold.groupby("model")[metric].mean())
    evidence = contribution_evidence(gold, predictions, config)
    atomic_json(run_dir / "contribution_evidence.json", evidence)
    for filename, frame in (("predictions", predictions), ("summary", summary), ("paintings", paintings), ("folds", pd.concat(fold_results))):
        frame.to_csv(run_dir / (filename + ".csv"), index=False, lineterminator="\n")
    atomic_json(run_dir / "completed.json", {"status": "FIVE_FOLD_COMPLETE", "folds": sorted(predictions.fold.unique().tolist()),
                "seconds": time.perf_counter()-started, "painting_count": gold.painting.nunique(), "GPU_used_for_heads": False,
                "production": "E50+L unchanged", "promotion": False})
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp025.json")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/exp025_vlm_probe"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    with threadpool_limits(limits=4):
        run(args.inputs, args.cache, args.run_dir, config)
