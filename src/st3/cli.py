"""Command-line entry points for development audit, baseline, evaluation and ZIPs."""

import argparse
import importlib.metadata
import json
import os
import platform
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from . import data, submission
from .metrics import score

ROOT = Path(__file__).resolve().parents[2]


def json_output(report, path=None):
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    return output


def official_verify(root):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_st3.py"), "_verify", str(root)], check=True)


def run_verifier(root):
    os.chdir(root)
    sys.argv = [data.VERIFIER]
    if os.name != "nt":
        runpy.run_path(data.VERIFIER, run_name="__main__")
        return
    # The upstream verifier compares os.path.relpath output to POSIX CSV paths.
    # Patch this function only in the subprocess running the official script.
    relpath = os.path.relpath

    def posix_relpath(*args, **kwargs):
        return relpath(*args, **kwargs).replace(os.sep, "/")

    with patch.object(os.path, "relpath", posix_relpath):
        runpy.run_path(data.VERIFIER, run_name="__main__")


def git_head(path):
    if not (path / ".git").exists():
        return None
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def run_baseline(args, dev):
    from .baseline import C, MAX_ITER, design_matrix, file_sha256, fit_lr, image_features, language_prior

    if args.run_dir.exists():
        raise ValueError(f"Run directory already exists: {args.run_dir}. Choose a new --run-dir.")
    config = json.loads((ROOT / "configs" / "st3.json").read_text(encoding="utf-8"))
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if config["batch_size"] <= 0:
        raise ValueError("Batch size must be positive")
    annotations = data.read_annotations(args.data_dir)
    data.check_dev_isolation(annotations, dev)
    train_annotations = annotations[annotations["split"] == "train"]
    train = data.aggregate(train_annotations)
    image_of = train_annotations.groupby("painting")["image_name"].first()
    train_images = train["painting"].map(image_of)
    languages = sorted(train["language"].unique())
    print(f"{len(train):,} training pairs, {len(dev):,} development rows", flush=True)
    embeddings, feature_metadata = image_features(args.data_dir, set(train_images) | set(dev["image"]),
                                                  args.cache_dir, config, args.device)
    print("Fitting annotator-weighted logistic regression", flush=True)
    model = fit_lr(design_matrix(train_images, train["language"], embeddings, languages), train, config["seed"])
    q = model.predict_proba(design_matrix(dev["image"], dev["language"], embeddings, languages))[:, 1]
    predictions = dev[data.KEYS].assign(p_positive=q.round(6))
    prior = language_prior(train, dev)
    # Development labels enter only here, after both sets of predictions are fixed.
    gold = data.validation_gold(annotations, dev)
    metrics = {"clip_language_lr": score(gold, predictions), "language_prior": score(gold, prior)}
    metadata = {"phase": "dev", "project_revision": git_head(ROOT), "upstream_revision": git_head(args.data_dir),
                "data_dir": str(args.data_dir), "annotations_sha256": file_sha256(args.data_dir / data.ANNOTATIONS),
                "dev_input_sha256": file_sha256(args.data_dir / data.DEV_INPUT), "config": config,
                "requested_device": args.device, "embedding": feature_metadata, "training_pairs": len(train),
                "training_paintings": train["painting"].nunique(), "languages": languages,
                "lr": {"C": C, "solver": "lbfgs", "max_iter": MAX_ITER, "iterations": int(model.n_iter_[0]),
                       "penalty": "l2", "positive_weight": "k", "negative_weight": "n-k"},
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "numpy", "pandas", "scikit-learn", "scipy", "Pillow")}}
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.run_dir.parent) as scratch:
        stage = Path(scratch) / "run"
        stage.mkdir()
        (stage / "language_prior").mkdir()
        for frame, path in ((predictions, stage), (prior, stage / "language_prior")):
            submission.validate_predictions(frame, dev)
            frame.to_csv(path / submission.FILENAME, index=False, lineterminator="\n")
        json_output(metadata, stage / "run.json")
        json_output(metrics, stage / "metrics.json")
        stage.rename(args.run_dir)
    for name, result in metrics.items():
        print(f"{name}: contrast={result['cultural_contrast']}, log_loss={result['log_loss']}, "
              f"pair_mean_log_loss={result['pair_mean_log_loss']}")
    print(f"Saved run to {args.run_dir}")


def resolved_path(path):
    return Path(path).resolve()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_verify":
        run_verifier(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=resolved_path, default=ROOT / "data" / "upstream")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit and decode the public development data")
    audit.add_argument("--official", action="store_true", help="Also run the organizers' verifier")
    audit.add_argument("--output", type=Path)
    baseline = commands.add_parser("baseline", help="Train, predict and score the reference baseline and prior")
    baseline.add_argument("--run-dir", type=resolved_path, default=ROOT / "runs" / "baseline_dev")
    baseline.add_argument("--cache-dir", type=resolved_path, default=ROOT / "cache")
    baseline.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    baseline.add_argument("--batch-size", type=int)
    evaluate = commands.add_parser("evaluate", help="Score development predictions using validation gold")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    pack = commands.add_parser("submission", help="Validate predictions and generate a ZIP for manual upload")
    pack.add_argument("--predictions", type=Path, default=ROOT / "runs" / "baseline_dev" / submission.FILENAME)
    pack.add_argument("--output", type=Path, default=ROOT / "runs" / "baseline_dev" / "submission.zip")
    validate = commands.add_parser("validate", help="Validate a development ZIP")
    validate.add_argument("archive", type=Path)
    args = parser.parse_args()

    if args.command == "audit":
        if args.official:
            official_verify(args.data_dir)
        print(json_output(data.audit(args.data_dir), args.output), end="")
        return
    dev = data.read_dev_input(args.data_dir)
    if args.command == "baseline":
        run_baseline(args, dev)
    elif args.command == "evaluate":
        annotations = data.read_annotations(args.data_dir)
        data.check_dev_isolation(annotations, dev)
        gold = data.validation_gold(annotations, dev)
        print(json_output(score(gold, submission.read_predictions(args.predictions, dev)), args.output), end="")
    elif args.command == "submission":
        submission.pack(args.predictions, args.output, dev)
        print(f"Validated and wrote {args.output} with {len(dev)} development predictions")
    else:
        submission.validate_archive(args.archive, dev)
        print(f"Validated {args.archive} with {len(dev)} development predictions")
