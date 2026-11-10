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
import time
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
    train = data.training_pairs(annotations)
    train_images = train["image"]
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


def run_ablation(args, dev):
    import pandas as pd

    from .ablation import (VARIANTS, fit_variants, grouped_cv, load_study_embeddings, paired_analysis,
                           paired_bootstrap, predict_lr, shuffle_images)
    from .baseline import C, MAX_ITER

    start = time.perf_counter()
    if args.run_dir.exists():
        raise ValueError(f"Run directory already exists: {args.run_dir}. Choose a new --run-dir.")
    if args.plots:
        from .study_plots import render_study

    config = json.loads((ROOT / "configs" / "exp002.json").read_text(encoding="utf-8"))
    baseline_config = json.loads((ROOT / "configs" / "st3.json").read_text(encoding="utf-8"))
    upstream_revision = git_head(args.data_dir)
    if upstream_revision != baseline_config["upstream_revision"]:
        raise ValueError("The ablation requires the pinned organizer checkout")
    annotations = data.read_annotations(args.data_dir)
    data.check_dev_isolation(annotations, dev)
    train = data.training_pairs(annotations)
    embeddings, provenance = load_study_embeddings(args.data_dir, train, dev, args.cache_dir, args.embedding_run,
                                                   upstream_revision, baseline_config)
    print(f"Fitting A-C using {len(embeddings)} cached embeddings on CPU", flush=True)
    predictions, model_c, languages, iterations = fit_variants(train, dev, embeddings, config["fit_seed"])

    shuffled_predictions = []
    shuffled_assignments = []
    first_seed = config["shuffle_first_seed"]
    for seed in range(first_seed, first_seed + config["shuffle_permutations"]):
        shuffled, assignments = shuffle_images(dev, seed)
        shuffled_predictions.append((seed, predict_lr(model_c, shuffled, embeddings, languages)))
        shuffled_assignments.append(assignments.assign(seed=seed))

    # Gold is aggregated only after A-D predictions have been fixed.
    gold = data.validation_gold(annotations, dev)
    metrics = {name: score(gold, frame) for name, frame in predictions.items()}
    shuffled_metrics = []
    for seed, frame in shuffled_predictions:
        result = score(gold, frame)
        shuffled_metrics.append({"seed": seed, **{key: result[key] for key in
                                ("cultural_contrast", "log_loss", "contrast_paintings", "log_loss_languages")}})
    shuffle_table = pd.DataFrame(shuffled_metrics)
    shuffle_summary = shuffle_table[["cultural_contrast", "log_loss"]].agg(["mean", "std", "min", "max"]).to_dict()
    pairs, paintings, language = paired_analysis(gold, predictions, metrics)
    bootstrap, intervals = paired_bootstrap(paintings, config["bootstrap_replicates"], config["bootstrap_seed"])
    cv_start = time.perf_counter()
    if args.cv:
        folds, oof = grouped_cv(train, embeddings, config)
    cv_seconds = time.perf_counter() - cv_start if args.cv else None

    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.run_dir.parent) as scratch:
        stage = Path(scratch) / "run"
        for name, frame in predictions.items():
            folder = stage / "predictions" / name
            folder.mkdir(parents=True)
            frame.to_csv(folder / submission.FILENAME, index=False, lineterminator="\n")
        diagnostics = stage / "diagnostics"
        diagnostics.mkdir()
        tables = {
            "per_painting.csv": paintings, "per_language.csv": language, "observed_vs_predicted.csv": pairs,
            "bootstrap_distribution.csv": bootstrap, "diagnostics/shuffled_metrics.csv": shuffle_table,
            "diagnostics/shuffled_predictions.csv": pd.concat([frame.assign(seed=seed) for seed, frame in shuffled_predictions]),
            "diagnostics/shuffled_assignments.csv": pd.concat(shuffled_assignments),
        }
        if args.cv:
            tables.update({"train_only_cv_folds.csv": folds, "train_only_cv_predictions.csv": oof})
        for filename, table in tables.items():
            table.to_csv(stage / filename, index=False, lineterminator="\n")
        if args.plots:
            render_study(stage, pairs, paintings, language, intervals)
        json_output({"models": metrics, "inference_shuffle": shuffle_summary, "paired_bootstrap": intervals},
                    stage / "metrics.json")
        packages = ["numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl"]
        if args.plots:
            packages.append("matplotlib")
        metadata = {
            "phase": "dev", "project_revision": git_head(ROOT), "upstream_revision": upstream_revision,
            "data_dir": str(args.data_dir), **provenance, "config": config, "variants": VARIANTS,
            "lr": {"C": C, "solver": "lbfgs", "max_iter": MAX_ITER, "iterations": iterations,
                   "penalty": "l2", "positive_weight": "k", "negative_weight": "n-k"},
            "training_pairs": len(train), "training_paintings": train["painting"].nunique(), "training_languages": languages,
            "dev_pairs": len(dev), "dev_paintings": dev["painting"].nunique(), "dev_languages": dev["language"].nunique(),
            "device": "CPU", "lr_thread_limit": 4, "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "train_only_cv_seconds": cv_seconds, "wall_seconds": time.perf_counter() - start,
        }
        json_output(metadata, stage / "run.json")
        stage.rename(args.run_dir)
    for name, result in metrics.items():
        print(f"{name}: contrast={result['cultural_contrast']:.6f}, log_loss={result['log_loss']:.6f}")
    print(f"Saved ablation and diagnostics to {args.run_dir}")


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
    ablation = commands.add_parser("ablation", help="Run the cached-embedding EXP-002 reliability study")
    ablation.add_argument("--run-dir", type=resolved_path, default=ROOT / "runs" / "exp002_baseline_ablation")
    ablation.add_argument("--cache-dir", type=resolved_path, default=ROOT / "cache")
    ablation.add_argument("--embedding-run", type=resolved_path, default=ROOT / "runs" / "baseline_dev" / "run.json")
    ablation.add_argument("--cv", action="store_true", help="Also run train-only painting-grouped cross-validation")
    ablation.add_argument("--plots", action="store_true", help="Write figures using requirements-analysis.txt")
    rank_cv = commands.add_parser("low-rank-cv", help="Freeze EXP-003 selection using train-only folds")
    rank_dev = commands.add_parser("low-rank-dev", help="Evaluate the frozen EXP-003 candidate and analyze Model C")
    for rank_command in (rank_cv, rank_dev):
        rank_command.add_argument("--run-dir", type=resolved_path, default=ROOT / "runs" / "exp003_low_rank")
        rank_command.add_argument("--cache-dir", type=resolved_path, default=ROOT / "cache")
        rank_command.add_argument("--embedding-run", type=resolved_path, default=ROOT / "runs" / "baseline_dev" / "run.json")
        rank_command.add_argument("--exp002-dir", type=resolved_path, default=ROOT / "runs" / "exp002_baseline_ablation")
    rank_dev.add_argument("--plots", action="store_true", help="Write the Model C singular-value spectrum figure")
    contrast_cv = commands.add_parser("contrast-cv", help="Freeze EXP-004 selection using train-only folds")
    contrast_dev = commands.add_parser("contrast-dev", help="Evaluate the frozen EXP-004 decision")
    for contrast_command in (contrast_cv, contrast_dev):
        contrast_command.add_argument("--run-dir", type=resolved_path, default=ROOT / "runs" / "exp004_contrast_aware")
        contrast_command.add_argument("--cache-dir", type=resolved_path, default=ROOT / "cache")
        contrast_command.add_argument("--embedding-run", type=resolved_path, default=ROOT / "runs" / "baseline_dev" / "run.json")
        contrast_command.add_argument("--exp002-dir", type=resolved_path, default=ROOT / "runs" / "exp002_baseline_ablation")
    visual_embed = commands.add_parser("visual-embed", help="Extract or validate the pinned DINOv2 cache")
    visual_cv = commands.add_parser("visual-cv", help="Freeze the EXP-005 train-only decision")
    visual_dev = commands.add_parser("visual-dev", help="Evaluate and package a qualifying EXP-005 candidate")
    for visual_command in (visual_embed, visual_cv, visual_dev):
        visual_command.add_argument("--cache-dir", type=resolved_path, default=ROOT / "cache")
        visual_command.add_argument("--embedding-run", type=resolved_path, default=ROOT / "runs" / "baseline_dev" / "run.json")
        visual_command.add_argument("--exp002-dir", type=resolved_path, default=ROOT / "runs" / "exp002_baseline_ablation")
        visual_command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    for visual_command in (visual_cv, visual_dev):
        visual_command.add_argument("--run-dir", type=resolved_path, default=ROOT / "runs" / "exp005_visual_representation")
    siglip_embed = commands.add_parser("siglip-embed", help="Extract the pinned SigLIP 2 cache in the dedicated runtime")
    siglip_cv = commands.add_parser("siglip-cv", help="Freeze the EXP-006 train-only decision")
    siglip_dev = commands.add_parser("siglip-dev", help="Evaluate and package the qualifying EXP-006 candidate")
    for command in (siglip_embed, siglip_cv, siglip_dev):
        command.add_argument("--cache-dir", type=resolved_path, default=ROOT / "cache")
        command.add_argument("--embedding-run", type=resolved_path, default=ROOT / "runs" / "baseline_dev" / "run.json")
        command.add_argument("--exp002-dir", type=resolved_path, default=ROOT / "runs" / "exp002_baseline_ablation")
        command.add_argument("--run-dir", type=resolved_path, default=ROOT / "runs" / "exp006_siglip2")
    siglip_embed.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    fusion_cv = commands.add_parser("fusion-cv", help="Compare the fixed EXP-007 fusion models and freeze selection")
    fusion_dev = commands.add_parser("fusion-dev", help="Fit and evaluate the frozen EXP-007 winner")
    for command in (fusion_cv, fusion_dev):
        command.add_argument("--cache-dir", type=resolved_path, default=ROOT / "cache")
        command.add_argument("--embedding-run", type=resolved_path, default=ROOT / "runs" / "baseline_dev" / "run.json")
        command.add_argument("--exp002-dir", type=resolved_path, default=ROOT / "runs" / "exp002_baseline_ablation")
        command.add_argument("--exp006-dir", type=resolved_path, default=ROOT / "runs" / "exp006_siglip2")
        command.add_argument("--run-dir", type=resolved_path, default=ROOT / "runs" / "exp007_fusion")
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
    elif args.command == "ablation":
        run_ablation(args, dev)
    elif args.command == "visual-embed":
        from .dino import run_visual_embed

        run_visual_embed(args, dev)
    elif args.command == "siglip-embed":
        from .siglip2 import run_siglip_embed

        run_siglip_embed(args, dev)
    elif args.command in ("visual-cv", "visual-dev"):
        from .visual_study import run_visual_cv, run_visual_dev

        runner = run_visual_cv if args.command == "visual-cv" else run_visual_dev
        runner(args, dev)
    elif args.command in ("siglip-cv", "siglip-dev"):
        from .visual_study import run_visual_cv, run_visual_dev

        runner = run_visual_cv if args.command == "siglip-cv" else run_visual_dev
        runner(args, dev, "exp006")
    elif args.command in ("fusion-cv", "fusion-dev"):
        from .fusion_study import run_fusion_cv, run_fusion_dev

        runner = run_fusion_cv if args.command == "fusion-cv" else run_fusion_dev
        runner(args, dev)
    elif args.command in ("contrast-cv", "contrast-dev"):
        from .contrast_study import run_contrast_cv, run_contrast_dev

        runner = run_contrast_cv if args.command == "contrast-cv" else run_contrast_dev
        runner(args, dev)
    elif args.command in ("low-rank-cv", "low-rank-dev"):
        from .low_rank_study import run_rank_cv, run_rank_dev

        runner = run_rank_cv if args.command == "low-rank-cv" else run_rank_dev
        runner(args, dev)
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
