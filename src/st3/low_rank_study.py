"""Train-only rank selection and one frozen development evaluation."""

import importlib.metadata
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, xlog1py, xlogy

from . import data, submission
from .ablation import load_study_embeddings, paired_analysis, paired_bootstrap, painting_folds, predict_lr
from .baseline import C, MAX_ITER, design_matrix, file_sha256, fit_lr, language_prior
from .cli import ROOT, git_head, json_output
from .low_rank import binomial_nll, factor_logits, fit_factors, main_logits, spectrum, split_lr, truncate_svd
from .metrics import painting_scores, score

FOLD_FILE = "train_only_cv_predictions.csv"


def image_index(frame, embeddings, languages):
    paths = sorted(frame["image"].unique())
    images = np.stack([embeddings[path] for path in paths]).astype(np.float64)
    image_rows = frame["image"].map({path: idx for idx, path in enumerate(paths)}).to_numpy()
    language_rows = frame["language"].map({lang: idx for idx, lang in enumerate(languages)})
    return images, image_rows, language_rows.fillna(len(languages)).to_numpy(dtype=int)


def training_grid(frame, embeddings, languages):
    images, image_rows, language_rows = image_index(frame, embeddings, languages)
    positives = np.zeros((len(images), len(languages)))
    totals = np.zeros_like(positives)
    if len(set(zip(image_rows, language_rows))) != len(frame):
        raise ValueError("Expected one eligible pair per training image and language")
    positives[image_rows, language_rows] = frame["k"]
    totals[image_rows, language_rows] = frame["n"]
    return images, positives, totals


def grid_predictions(inputs, logits, rows):
    _, image_rows, language_rows = rows
    return inputs[data.KEYS].assign(p_positive=expit(logits[image_rows, language_rows]).round(6))


def dense_logits(images, main, interaction):
    logits = main_logits(images, main)
    if interaction is not None:
        logits[:, :-1] += images @ interaction.T
    return logits


def fit_references(train, inputs, embeddings, languages, grid, seed):
    start = time.perf_counter()
    means = train.groupby("language")["p"].mean()
    q = train["language"].map(means).to_numpy()
    nll = -(xlogy(train["k"], q) + xlog1py(train["n"] - train["k"], -q)).sum() / train["n"].sum()
    predictions = {} if inputs is None else {"A": language_prior(train, inputs)}
    records = {"A": {"train_nll": nll, "parameters": len(languages), "status": "closed form",
                     "converged": True, "seconds": time.perf_counter() - start, "seed": seed}}
    models = {}
    images, k, n = grid
    for name, interactions in (("B", False), ("C", True)):
        start = time.perf_counter()
        matrix = design_matrix(train["image"], train["language"], embeddings, languages, interactions=interactions)
        model = fit_lr(matrix, train, seed)
        del matrix
        if inputs is not None:
            predictions[name] = predict_lr(model, inputs, embeddings, languages, interactions=interactions)
        main, weights = split_lr(model, images.shape[1], len(languages))
        models[name] = (main, weights)
        train_nll = binomial_nll(dense_logits(images, main, weights)[:, :-1], k, n) / n.sum()
        iterations = int(model.n_iter_[0])
        records[name] = {"train_nll": train_nll, "objective": train_nll + np.sum(model.coef_ ** 2) / (2 * C * n.sum()),
                         "parameters": model.coef_.size + 1, "iterations": iterations, "seed": seed,
                         "status": "converged" if iterations < MAX_ITER else "iteration limit",
                         "converged": iterations < MAX_ITER, "seconds": time.perf_counter() - start}
    return predictions, records, models


def verify_folds(train, saved, config):
    data.require_same_pairs(train, saved)
    assignment = saved.groupby("painting")["fold"]
    if not assignment.nunique().eq(1).all() or saved["fold"].isna().any():
        raise ValueError("EXP-002 fold artifact has split painting groups or missing assignments")
    expected = train["painting"].map(assignment.first())
    folds = list(painting_folds(train, config["cv_folds"], config["cv_seed"]))
    for fold, (fit, held) in enumerate(folds, 1):
        if (not expected.iloc[held].eq(fold).all()
                or not set(train.iloc[fit]["painting"]).isdisjoint(train.iloc[held]["painting"])):
            raise ValueError("Regenerated fold disagrees with the exact EXP-002 membership")
    return folds


def select_candidate(summary, config):
    threshold = float(summary.loc["C", "log_loss_mean"] + config["log_loss_guardrail_bits"])
    candidates = summary.loc[[f"LR-{rank}" for rank in config["ranks"]]]
    eligible = candidates[candidates["all_converged"] & candidates["contrast_mean"].notna()
                          & candidates["log_loss_mean"].le(threshold)]
    model = "C"
    if not eligible.empty:
        close = eligible[eligible["contrast_mean"] >= eligible["contrast_mean"].max() - config["contrast_tie_band"]]
        model = min(close.index, key=lambda name: int(name.split("-")[1]))
    return {"model": model, "eligible_candidates": list(eligible.index), "max_log_loss": threshold,
            "tie_band": config["contrast_tie_band"], "final_seed": config["final_seed"] if model != "C" else None}


def study_inputs(args, dev, config_name="exp003"):
    config = json.loads((ROOT / "configs" / f"{config_name}.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "configs" / "st3.json").read_text(encoding="utf-8"))
    upstream = git_head(args.data_dir)
    if upstream != baseline["upstream_revision"]:
        raise ValueError("EXP-003 requires the pinned organizer checkout")
    annotations = data.read_annotations(args.data_dir)
    data.check_dev_isolation(annotations, dev)
    train = data.training_pairs(annotations)
    embeddings, provenance = load_study_embeddings(args.data_dir, train, dev, args.cache_dir, args.embedding_run,
                                                   upstream, baseline)
    previous_run = args.exp002_dir / "run.json"
    previous = json.loads(previous_run.read_text(encoding="utf-8"))
    if any(provenance[key] != previous[key] for key in ("annotations_sha256", "dev_input_sha256")) \
            or provenance["embedding"]["sha256"] != previous["embedding"]["sha256"]:
        raise ValueError("EXP-003 inputs differ from the EXP-002 folds")
    provenance.update({"upstream_revision": upstream, "config": config,
                       "exp002_config": previous["config"], "exp002_run_sha256": file_sha256(previous_run),
                       "exp002_folds_sha256": file_sha256(args.exp002_dir / FOLD_FILE)})
    return config, annotations, train, embeddings, provenance


def run_metadata(provenance, start):
    return {**provenance, "project_revision": git_head(ROOT), "device": "CPU",
            "processor": platform.processor(), "platform": platform.platform(), "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")},
            "wall_seconds": time.perf_counter() - start}


def score_result(gold, predictions, context, records):
    result = score(gold, predictions)
    joined = gold[data.KEYS + ["p"]].merge(predictions, on=data.KEYS, validate="one_to_one")
    by_painting = painting_scores(joined).assign(**context)
    by_language = pd.DataFrame.from_dict(result["per_language"], orient="index").rename_axis("language").reset_index()
    records["paintings"].append(by_painting)
    records["languages"].append(by_language.assign(**context))
    records["predictions"].append(predictions.assign(**context))
    return result["cultural_contrast"], result["log_loss"]


def run_rank_cv(args, dev):
    start = time.perf_counter()
    destination = args.run_dir / "cv"
    if destination.exists():
        raise ValueError(f"CV results already exist at {destination}")
    config, _, train, embeddings, provenance = study_inputs(args, dev)
    saved = pd.read_csv(args.exp002_dir / FOLD_FILE, float_precision="round_trip")
    folds = verify_folds(train, saved, provenance["exp002_config"])
    records = {key: [] for key in ("predictions", "paintings", "languages", "spectrum")}
    rows = []
    fit_seed = provenance["exp002_config"]["fit_seed"]
    for fold, (fit_rows, held_rows) in enumerate(folds, 1):
        print(f"EXP-003 train-only fold {fold}/{len(folds)}", flush=True)
        fit, held = train.iloc[fit_rows], train.iloc[held_rows]
        languages = sorted(fit["language"].unique())
        grid = training_grid(fit, embeddings, languages)
        indices = image_index(held, embeddings, languages)
        predictions, ref, model = fit_references(fit, held, embeddings, languages, grid, fit_seed)
        original = held[data.KEYS].merge(saved.loc[saved["fold"].eq(fold)], on=data.KEYS, validate="one_to_one")
        for name, frame in predictions.items():
            if not np.array_equal(frame["p_positive"].to_numpy(), original[f"p_{name}"].to_numpy()):
                raise ValueError(f"Fold {fold} model {name} failed the EXP-002 prediction regression check")
            context = {"fold": fold, "model": name, "seed": fit_seed}
            contrast, loss = score_result(held, frame, context, records)
            rows.append({**context, **ref[name], "contrast": contrast, "log_loss": loss})

        main, weights = model["C"]
        decomposition = np.linalg.svd(weights, full_matrices=False)
        energy, effective = spectrum(decomposition[1])
        records["spectrum"].append(pd.DataFrame(energy).assign(
            fold=fold, rank=np.arange(1, len(energy["singular_value"]) + 1), effective_rank=effective))
        images, k, n = grid
        for rank in config["ranks"]:
            if rank > min(weights.shape):
                raise ValueError(f"Rank {rank} exceeds the fold interaction matrix dimensions")
            tick = time.perf_counter()
            approximation = truncate_svd(decomposition, rank)
            pred = grid_predictions(held, dense_logits(indices[0], main, approximation), indices)
            nll = binomial_nll(dense_logits(images, main, approximation)[:, :-1], k, n) / n.sum()
            objective = nll + (np.sum(main[1:] ** 2) + np.sum(approximation ** 2)) / (2 * C * n.sum())
            context = {"fold": fold, "model": f"SVD-{rank}", "seed": fit_seed}
            seconds = time.perf_counter() - tick
            contrast, loss = score_result(held, pred, context, records)
            params = len(main) + rank * sum(weights.shape)
            rows.append({**context, "contrast": contrast, "log_loss": loss, "train_nll": nll, "objective": objective,
                         "parameters": params, "status": "post-hoc", "converged": ref["C"]["converged"], "seconds": seconds})
            for seed in config["init_seeds"]:
                tick = time.perf_counter()
                factors, info = fit_factors(*grid, model["B"][0], rank, seed, config)
                info.update({"parameters": params, "seconds": time.perf_counter() - tick})
                context = {"fold": fold, "model": f"LR-{rank}", "seed": seed}
                contrast = loss = None
                if factors is not None:
                    pred = grid_predictions(held, factor_logits(indices[0], *factors), indices)
                    contrast, loss = score_result(held, pred, context, records)
                rows.append({**context, **info, "contrast": contrast, "log_loss": loss})
                print(f"  LR-{rank} seed {seed}: {info['status']}, {info['iterations']} iterations", flush=True)

    table = pd.DataFrame(rows)
    summary = table.groupby("model", sort=False).agg(
        contrast_mean=("contrast", "mean"), contrast_sd=("contrast", "std"),
        log_loss_mean=("log_loss", "mean"), log_loss_sd=("log_loss", "std"), fits=("model", "size"),
        scored=("contrast", "count"), all_converged=("converged", "all"), parameters=("parameters", "max"),
        total_seconds=("seconds", "sum"))
    summary["all_converged"] &= summary["scored"].eq(summary["fits"]) & table.groupby("model")["log_loss"].count().eq(summary["fits"])
    decision = select_candidate(summary, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        stage = Path(scratch) / "output"
        stage.mkdir()
        table.to_csv(stage / "fold_metrics.csv", index=False, lineterminator="\n")
        summary.to_csv(stage / "summary.csv", lineterminator="\n")
        for key, frames in records.items():
            pd.concat(frames, ignore_index=True).to_csv(stage / f"{key}.csv", index=False, lineterminator="\n")
        json_output(run_metadata(provenance, start), stage / "run.json")
        decision["summary_sha256"] = file_sha256(stage / "summary.csv")
        decision["run_sha256"] = file_sha256(stage / "run.json")
        json_output(decision, stage / "selection.json")
        stage.rename(destination)
    print(f"Frozen CV selection: {decision['model']} ({destination / 'selection.json'})")


def write_dev_analysis(stage, gold, predictions, candidate, config):
    metrics = {name: score(gold, frame) for name, frame in predictions.items()}
    references = {name: predictions[name] for name in "ABC"}
    pairs, paintings, languages = paired_analysis(gold, references, {name: metrics[name] for name in references})
    pairs = pairs.merge(predictions[candidate].rename(columns={"p_positive": "p_LR"}), on=data.KEYS, validate="one_to_one")
    contrast = painting_scores(pairs, "p_LR").set_index("painting")["contrast"]
    paintings["contrast_LR"] = paintings["painting"].map(contrast)
    paintings["contrast_LR_minus_C"] = paintings["contrast_LR"] - paintings["contrast_C"]
    paintings["contrast_LR_minus_B"] = paintings["contrast_LR"] - paintings["contrast_B"]
    languages["log_loss_LR"] = languages["language"].map(
        {lang: result["log_loss"] for lang, result in metrics[candidate]["per_language"].items()})
    languages["log_loss_LR_minus_C"] = languages["log_loss_LR"] - languages["log_loss_C"]
    languages["log_loss_LR_minus_B"] = languages["log_loss_LR"] - languages["log_loss_B"]
    draws, intervals = paired_bootstrap(paintings, config["bootstrap_replicates"], config["bootstrap_seed"],
                                        columns=("contrast_LR_minus_C",))
    for filename, frame in (("observed_vs_predicted", pairs), ("per_painting", paintings),
                            ("per_language", languages), ("bootstrap_distribution", draws)):
        frame.to_csv(stage / f"{filename}.csv", index=False, lineterminator="\n")
    for name, frame in predictions.items():
        output = stage / "predictions" / name
        output.mkdir(parents=True)
        frame.to_csv(output / submission.FILENAME, index=False, lineterminator="\n")
    json_output({"models": metrics, "paired_bootstrap": intervals}, stage / "metrics.json")
    return metrics


def run_rank_dev(args, dev):
    start = time.perf_counter()
    destination = args.run_dir / "dev"
    if destination.exists():
        raise ValueError(f"Final results already exist at {destination}")
    if args.plots:
        from .study_plots import render_spectrum
    cv = args.run_dir / "cv"
    decision_path = cv / "selection.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (decision["summary_sha256"] != file_sha256(cv / "summary.csv")
            or decision["run_sha256"] != file_sha256(cv / "run.json")):
        raise ValueError("Frozen CV artifacts have changed")
    config, annotations, train, embeddings, provenance = study_inputs(args, dev)
    previous = json.loads((cv / "run.json").read_text(encoding="utf-8"))
    if any(previous[key] != value for key, value in provenance.items()):
        raise ValueError("Development inputs differ from the frozen CV run")
    summary = pd.read_csv(cv / "summary.csv", index_col="model", float_precision="round_trip")
    if any(decision[key] != value for key, value in select_candidate(summary, config).items()):
        raise ValueError("Saved decision disagrees with the frozen selection rule")
    candidate = decision["model"]
    languages = sorted(train["language"].unique())
    grid = training_grid(train, embeddings, languages)
    seed = provenance["exp002_config"]["fit_seed"]
    predictions, records, models = fit_references(train, dev if candidate != "C" else None, embeddings, languages, grid, seed)
    main, weights = models["C"]
    values = np.linalg.svd(weights, full_matrices=False, compute_uv=False)
    energy, effective = spectrum(values)
    spectral = pd.DataFrame(energy).assign(rank=np.arange(1, len(values) + 1))
    status = "guardrail_or_convergence_rejected_all" if candidate == "C" else "scored"
    if candidate != "C":
        rank = int(candidate.split("-")[1])
        tick = time.perf_counter()
        factors, info = fit_factors(*grid, models["B"][0], rank, decision["final_seed"], config)
        records[candidate] = {**info, "seed": decision["final_seed"], "seconds": time.perf_counter() - tick,
                              "parameters": len(main) + rank * sum(weights.shape)}
        if info["converged"]:
            indices = image_index(dev, embeddings, languages)
            predictions[candidate] = grid_predictions(dev, factor_logits(indices[0], *factors), indices)
        else:
            status = "selected_fit_failed_convergence"

    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        stage = Path(scratch) / "output"
        stage.mkdir()
        np.savez(stage / "full_interaction.npz", weights=weights, languages=np.array(languages))
        spectral.to_csv(stage / "singular_spectrum.csv", index=False, lineterminator="\n")
        if args.plots:
            render_spectrum(stage, spectral)
        if status == "scored":
            # This is the only place the study reads public development labels for scoring.
            gold = data.validation_gold(annotations, dev)
            metrics = write_dev_analysis(stage, gold, predictions, candidate, config)
            for name, result in metrics.items():
                print(f"{name}: contrast={result['cultural_contrast']:.6f}, log_loss={result['log_loss']:.6f}")
        metadata = run_metadata(provenance, start)
        metadata.update({"selection_sha256": file_sha256(decision_path), "selection": decision, "status": status,
                         "fits": records, "interaction_shape": list(weights.shape), "effective_rank": effective})
        if args.plots:
            metadata["packages"]["matplotlib"] = importlib.metadata.version("matplotlib")
        json_output(metadata, stage / "run.json")
        stage.rename(destination)
    print(f"Full training interaction spectrum and final study status ({status}): {destination}")
