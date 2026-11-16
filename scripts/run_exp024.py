"""Run the small nested P2 retrieval comparison on cached features, entirely on CPU."""

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from st3 import data
from st3.caption_data import inner_partition, require_isolation
from st3.metrics import score, painting_scores
from st3.visual_retrieval import Parameters, RetrievalMemory, visual_features, predict_pairs


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_inputs(inputs, config):
    # Verify only the four needed data files. Never open captions, confirmation
    # annotations, Development labels or the original whole annotation CSV.
    for name, expected in config["input_sha256"].items():
        if digest(inputs / name) != expected:
            raise ValueError("Required input checksum differs: " + name)
    train = pd.read_csv(inputs / "training/pairs.csv", float_precision="round_trip")
    protocol = pd.read_csv(inputs / "training/protocol.csv")
    reference = pd.read_csv(inputs / "history/exp022_R0.csv", float_precision="round_trip")
    expected = protocol[protocol.role.eq("development")]
    if (len(train) != config["training_pairs"] or train.painting.nunique() != config["training_paintings"]
            or set(train.painting) != set(expected.painting) or not train.split.eq("train").all()
            or set(train.fold.unique()) != {1, 2, 3, 4, 5}):
        raise ValueError("P2 training population or fold coverage differs")
    mapping = train[["painting", "image", "image_sha256", "fold"]].drop_duplicates().sort_values("painting").reset_index(drop=True)
    if not mapping.equals(expected[mapping.columns].sort_values("painting").reset_index(drop=True)):
        raise ValueError("Exact original P2 artwork/fold mapping differs")
    require_isolation(train, protocol[protocol.role.eq("holdout")])
    aligned = train[data.KEYS + ["fold"]].merge(reference, on=data.KEYS, validate="one_to_one", suffixes=("_gold", "_reference"))
    if len(aligned) != len(train) or len(reference) != len(train) or not aligned.fold_gold.equals(aligned.fold_reference):
        raise ValueError("E50+L reference keys/folds differ")
    with np.load(inputs / "visual/features.npz", allow_pickle=False) as saved:
        names = saved["names"].tolist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate feature image names")
        positions = {name: i for i, name in enumerate(names)}
        images = sorted(train.image.unique())
        selected = [positions[name] for name in images]
        large, siglip = saved["large"][selected], saved["siglip"][selected]
        for block in (large, siglip):
            if not np.allclose(np.linalg.norm(block, axis=1), 1., atol=1e-5):
                raise ValueError("Historical feature normalization differs")
        vectors = visual_features(large, siglip)
    features = dict(zip(images, vectors))
    return train, features, reference


def select_parameters(fitting, features, config, fold):
    inner, validation = inner_partition(fitting, config, fold)
    require_isolation(inner, validation)
    memory = RetrievalMemory(inner, features, config["language_prior_votes"])
    queries = validation[["painting", "image"]].drop_duplicates().sort_values("painting")
    nearest = memory.neighbors(np.stack([features[name] for name in queries.image]), max(config["neighbors"]))
    grid = [Parameters(k, t, s) for k, t, s in product(config["neighbors"], config["temperatures"], config["shrinkage_votes"])]
    records, selected = [], {}
    for variant in ("uncentered", "centered"):
        candidates = []
        for index, parameters in enumerate(grid):
            predicted = predict_pairs(memory, validation[data.INPUT_COLUMNS], features, variant, parameters, neighbors=nearest)
            metrics = score(validation, predicted)
            contrast, loss = metrics["cultural_contrast"], metrics["log_loss"]
            if contrast is None or loss is None or not np.isfinite([contrast, loss]).all():
                raise ValueError("Inner split cannot support official Contrast and Macro Log Loss")
            records.append({"fold": fold, "variant": variant, **asdict(parameters), "contrast": contrast, "macro_log_loss": loss})
            candidates.append((-contrast, loss, index, parameters))
        selected[variant] = min(candidates, key=lambda value: value[:3])[-1]
    return selected, records, {"inner_paintings": sorted(inner.painting.unique()), "selection_paintings": sorted(validation.painting.unique())}


def run(inputs, run_dir, config):
    tick = time.perf_counter()
    if run_dir.exists():
        raise FileExistsError("Preserve previous results; choose a fresh --run-dir")
    train, features, reference = load_inputs(inputs, config)
    run_dir.mkdir(parents=True)
    sources = ["src/st3/visual_retrieval.py", "scripts/run_exp024.py", "configs/exp024.json", "src/st3/metrics.py", "src/st3/caption_data.py"]
    write_json(run_dir / "run.json", {"config": config, "inputs": str(inputs.resolve()), "source_sha256": {p: digest(ROOT / p) for p in sources},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "CPU_threads": 4,
                "scope": "Original P2 training CV only; no captions/confirmation/Development; no E50 blend"})
    all_predictions, folds, searches, partitions, choices = [], [], [], [], []
    for fold in range(1, 6):
        fold_tick = time.perf_counter()
        fitting, held = train[train.fold.ne(fold)], train[train.fold.eq(fold)]
        require_isolation(fitting, held)
        selected, history, inner = select_parameters(fitting, features, config, fold)
        searches.extend(history)
        partitions.append({"fold": fold, "memory_paintings": sorted(fitting.painting.unique()), "held_paintings": sorted(held.painting.unique()), **inner})
        memory = RetrievalMemory(fitting, features, config["language_prior_votes"])
        for variant in config["variants"]:
            parameters = selected.get(variant, Parameters(16, .1, 20.))
            pred = predict_pairs(memory, held[data.INPUT_COLUMNS], features, variant, parameters)
            pred = pred.assign(model=variant, fold=fold)
            all_predictions.append(pred)
            metrics = score(held, pred[data.KEYS + ["p_positive"]])
            folds.append({"model": variant, "fold": fold, "contrast": metrics["cultural_contrast"], "macro_log_loss": metrics["log_loss"]})
            choices.append({"fold": fold, "variant": variant, **asdict(parameters)})
        pred = held[data.KEYS].merge(reference[data.KEYS + ["p_positive"]], on=data.KEYS, validate="one_to_one")
        metrics = score(held, pred)
        all_predictions.append(pred.assign(model="E50+L", fold=fold))
        folds.append({"model": "E50+L", "fold": fold, "contrast": metrics["cultural_contrast"], "macro_log_loss": metrics["log_loss"]})
        print(f"Fold {fold}/5 complete in {time.perf_counter()-fold_tick:.2f}s; selected {selected}", flush=True)
    predictions, per_fold = pd.concat(all_predictions, ignore_index=True), pd.DataFrame(folds)
    summary, by_painting, by_language = [], [], []
    for model, pred in predictions.groupby("model", sort=False):
        metrics = score(train, pred[data.KEYS + ["p_positive"]])
        rows = per_fold[per_fold.model.eq(model)]
        summary.append({"model": model, "pooled_contrast": metrics["cultural_contrast"], "pooled_macro_log_loss": metrics["log_loss"],
                        "fold_mean_contrast": rows.contrast.mean(), "fold_mean_macro_log_loss": rows.macro_log_loss.mean()})
        joined = train[data.KEYS + ["k", "n"]].merge(pred[data.KEYS + ["p_positive"]], on=data.KEYS, validate="one_to_one")
        by_painting.append(painting_scores(joined.assign(p=joined.k / joined.n)).assign(model=model))
        by_language.extend({"model": model, "language": language, **values} for language, values in metrics["per_language"].items())
    table = pd.DataFrame(summary)
    for name, frame in (("predictions", predictions), ("folds", per_fold), ("summary", table), ("inner_search", pd.DataFrame(searches)),
                        ("selected_parameters", pd.DataFrame(choices)), ("paintings", pd.concat(by_painting)), ("languages", pd.DataFrame(by_language))):
        frame.to_csv(run_dir / (name + ".csv"), index=False, lineterminator="\n")
    write_json(run_dir / "partitions.json", partitions)
    write_json(run_dir / "completed.json", {"status": "COMPLETE", "elapsed_seconds": time.perf_counter() - tick,
                 "training_paintings": train.painting.nunique(), "pairs": len(train), "summary": summary, "GPU_used": False})
    print(table.to_string(index=False), flush=True)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Existing verified input directory with training/, visual/ and history/")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/exp024_visual_retrieval"))
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/exp024.json").read_text(encoding="utf-8"))
    with threadpool_limits(limits=4):
        run(args.inputs, args.run_dir, config)
