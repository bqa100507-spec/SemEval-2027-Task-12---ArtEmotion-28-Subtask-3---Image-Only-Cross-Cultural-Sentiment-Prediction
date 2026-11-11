"""Controlled CLIP/DINO comparison and optional frozen development stage."""

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from . import data, submission
from .ablation import predict_lr
from .baseline import MAX_ITER, design_matrix, file_sha256, fit_lr
from .cli import json_output
from .contrast_study import choose, new_output, summarize, write_development
from .dino import extract_dino, normalize_rows
from .low_rank_study import FOLD_FILE, run_metadata, score_result, study_inputs, verify_folds
from .metrics import score

REFERENCE = "C-CLIP"
DINO = "C-DINO768"
PCA_DINO = "C-DINO512"
SIGLIP = "C-SigLIP2"


def pca_embeddings(embeddings, fit_paths, held_paths, width):
    fit, held = sorted(set(fit_paths)), sorted(set(held_paths))
    if set(fit) & set(held):
        raise ValueError("PCA fit and held-out image sets overlap")
    x = np.stack([embeddings[path] for path in fit]).astype(np.float64)
    y = np.stack([embeddings[path] for path in held]).astype(np.float64)
    with threadpool_limits(limits=4, user_api="blas"):
        projector = PCA(n_components=width, svd_solver="full", whiten=False).fit(x)
        projected_fit = normalize_rows(projector.transform(x))
        projected_held = normalize_rows(projector.transform(y))
    return dict(zip(fit + held, np.vstack((projected_fit, projected_held)))), projector


def fit_head(fit, held, embeddings, seed):
    start = time.perf_counter()
    languages = sorted(fit["language"].unique())
    matrix = design_matrix(fit["image"], fit["language"], embeddings, languages)
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = fit_lr(matrix, fit, seed)
    fit_seconds = time.perf_counter() - fit_start
    del matrix
    convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    predict_start = time.perf_counter()
    predictions = predict_lr(model, held, embeddings, languages)
    iterations = int(model.n_iter_[0])
    converged = iterations < MAX_ITER and not convergence_warnings and np.isfinite(model.coef_).all() \
        and np.isfinite(model.intercept_).all()
    info = {"converged": bool(converged), "warnings": convergence_warnings, "iterations": iterations,
            "parameters": model.coef_.size + model.intercept_.size, "fit_seconds": fit_seconds,
            "predict_seconds": time.perf_counter() - predict_start, "seconds": time.perf_counter() - start}
    return predictions, info


def dino_features(args, dev, train, config):
    return extract_dino(args.data_dir, set(train["image"]) | set(dev["image"]), args.cache_dir, config, args.device)


def candidate_features(args, dev, train, config, experiment):
    if experiment == "exp006":
        from .siglip2 import load_siglip2_features

        return load_siglip2_features(args, dev, train, config)
    return dino_features(args, dev, train, config)


def deltas(rows, index, metric, reference=REFERENCE, label="CLIP"):
    original = rows[rows.model.eq(reference)].set_index(index)[metric]
    reference_column = f"{metric}_{label}"
    merged = rows.join(original.rename(reference_column), on=index, validate="many_to_one")
    return merged.assign(**{f"delta_vs_{label.lower()}": merged[metric] - merged[reference_column]})


def run_visual_cv(args, dev, experiment="exp005"):
    start = time.perf_counter()
    is_siglip = experiment == "exp006"
    candidate = SIGLIP if is_siglip else DINO
    embedding_key = "siglip2_embedding" if is_siglip else "dino_embedding"
    with new_output(args.run_dir / "cv") as stage:
        config, _, train, clip, provenance = study_inputs(args, dev, experiment)
        alternative, feature_meta = candidate_features(args, dev, train, config, experiment)
        saved = pd.read_csv(args.exp002_dir / FOLD_FILE, float_precision="round_trip")
        folds = verify_folds(train, saved, provenance["exp002_config"])
        seed = provenance["exp002_config"]["fit_seed"]
        rows = []
        records = {name: [] for name in ("predictions", "paintings", "languages")}
        for fold, (fit_ids, held_ids) in enumerate(folds, 1):
            print(f"{experiment.upper()} train-only fold {fold}/{len(folds)}", flush=True)
            fit, held = train.iloc[fit_ids], train.iloc[held_ids]
            conditions = [(REFERENCE, clip), (candidate, alternative)]
            if not is_siglip:
                tick = time.perf_counter()
                pca, projector = pca_embeddings(alternative, fit["image"], held["image"], config["pca_width"])
                pca_seconds = time.perf_counter() - tick
                conditions.append((PCA_DINO, pca))
            expected = held[data.KEYS].merge(saved.loc[saved.fold.eq(fold)], on=data.KEYS, validate="one_to_one")["p_C"]
            for name, features in conditions:
                pred, info = fit_head(fit, held, features, seed)
                if name == REFERENCE and (not info["converged"] or not np.array_equal(pred["p_positive"], expected)):
                    raise ValueError(f"Model C-CLIP fold {fold} failed EXP-002 regression")
                context = {"fold": fold, "model": name, "seed": seed}
                contrast, log_loss = score_result(held, pred, context, records)
                info.update({"pca_seconds": pca_seconds, "pca_fit_paintings": int(projector.n_samples_),
                             "pca_explained_variance": float(projector.explained_variance_ratio_.sum())}
                            if name == PCA_DINO else {"pca_seconds": 0.})
                rows.append({**context, **info, "contrast": contrast, "log_loss": log_loss})
                print(f"  {name}: contrast={contrast:.6f}, log_loss={log_loss:.6f}, "
                      f"iterations={info['iterations']}, converged={info['converged']}", flush=True)
        table = pd.DataFrame(rows)
        summary = summarize(table, REFERENCE)
        decision = choose(summary, config, len(folds), REFERENCE)
        if is_siglip:
            predictions = pd.concat(records["predictions"], ignore_index=True)
            pooled = {model: score(train, frame[[*data.KEYS, "p_positive"]])
                      for model, frame in predictions.groupby("model", sort=False)}
            json_output(pooled, stage / "pooled_metrics.json")
        for name, frames in records.items():
            frame = pd.concat(frames, ignore_index=True)
            if name == "paintings":
                frame = deltas(frame, ["fold", "painting"], "contrast")
            elif name == "languages":
                frame = deltas(frame, ["fold", "language"], "log_loss")
            frame.to_csv(stage / f"{name}.csv", index=False, lineterminator="\n")
        table.to_csv(stage / "fold_metrics.csv", index=False, lineterminator="\n")
        summary.to_csv(stage / "summary.csv", lineterminator="\n")
        json_output({**run_metadata(provenance, start), embedding_key: feature_meta}, stage / "run.json")
        decision.update({"summary_sha256": file_sha256(stage / "summary.csv"), "run_sha256": file_sha256(stage / "run.json")})
        json_output(decision, stage / "selection.json")
    print(f"Frozen {experiment.upper()} decision: {decision['model']}", flush=True)


def run_visual_dev(args, dev, experiment="exp005"):
    start = time.perf_counter()
    embedding_key = "siglip2_embedding" if experiment == "exp006" else "dino_embedding"
    with new_output(args.run_dir / "dev") as stage:
        cv = args.run_dir / "cv"
        selection = cv / "selection.json"
        decision = json.loads(selection.read_text(encoding="utf-8"))
        metadata = json.loads((cv / "run.json").read_text(encoding="utf-8"))
        if (decision["summary_sha256"] != file_sha256(cv / "summary.csv")
                or decision["run_sha256"] != file_sha256(cv / "run.json")):
            raise ValueError(f"Frozen {experiment.upper()} CV artifacts have changed")
        summary = pd.read_csv(cv / "summary.csv", index_col="model", float_precision="round_trip")
        expected = choose(summary, metadata["config"], metadata["exp002_config"]["cv_folds"], REFERENCE)
        if any(decision[key] != value for key, value in expected.items()):
            raise ValueError(f"Decision disagrees with the frozen {experiment.upper()} rule")
        candidate = decision["model"]
        result = {"selection": decision, "selection_sha256": file_sha256(selection), "status": "no_go"}
        if candidate != REFERENCE:
            config, annotations, train, _, provenance = study_inputs(args, dev, experiment)
            if any(metadata[key] != value for key, value in provenance.items()):
                raise ValueError(f"Development inputs differ from frozen {experiment.upper()} CV")
            features, feature_meta = candidate_features(args, dev, train, config, experiment)
            if feature_meta["features_sha256"] != metadata[embedding_key]["features_sha256"]:
                raise ValueError("Development features differ from the frozen CV cache")
            if candidate == PCA_DINO:
                tick = time.perf_counter()
                features, projector = pca_embeddings(features, train["image"], dev["image"], config["pca_width"])
                result.update({"pca_seconds": time.perf_counter() - tick, "pca_fit_paintings": int(projector.n_samples_),
                               "pca_explained_variance": float(projector.explained_variance_ratio_.sum())})
            pred, info = fit_head(train, dev, features, provenance["exp002_config"]["fit_seed"])
            result.update({**run_metadata(provenance, start), "fit": info,
                           embedding_key: feature_meta, "status": "scored" if info["converged"] else "fit_failed"})
            if info["converged"]:
                original_path = args.exp002_dir / "predictions" / "C" / submission.FILENAME
                original = submission.read_predictions(original_path, dev)
                result["reference_predictions_sha256"] = file_sha256(original_path)
                gold = data.validation_gold(annotations, dev)
                write_development(stage, gold, {REFERENCE: original, candidate: pred}, candidate, config, REFERENCE)
                submission.pack(stage / "predictions" / candidate / submission.FILENAME, stage / "submission.zip", dev)
                submission.validate_archive(stage / "submission.zip", dev)
        json_output(result, stage / "run.json")
    print(f"{experiment.upper()} development stage: {result['status']}", flush=True)
