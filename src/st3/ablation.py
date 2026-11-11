"""Controlled cached-embedding experiments and painting-level analysis for ST3."""

import json
from pathlib import PureWindowsPath

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .baseline import design_matrix, feature_cache_path, file_sha256, fit_lr, language_prior, load_feature_cache
from .data import ANNOTATIONS, DEV_INPUT, KEYS
from .metrics import painting_scores, score

VARIANTS = {
    "A": "Task 001 language prior",
    "B": "Weighted LR on shared image features and language one-hot",
    "C": "Official weighted LR with image-language interactions",
    "D": "Model C with a dev-painting derangement at inference",
}
CONTRAST_DELTAS = ["contrast_C_minus_A", "contrast_C_minus_B"]


def load_study_embeddings(root, train, dev, cache_dir, source_run, upstream_revision, baseline_config):
    """Resolve the original cache from its provenance without importing Torch."""
    source = json.loads(source_run.read_text(encoding="utf-8"))
    hashes = {"annotations_sha256": file_sha256(root / ANNOTATIONS), "dev_input_sha256": file_sha256(root / DEV_INPUT)}
    if (source["phase"] != "dev" or source["upstream_revision"] != upstream_revision
            or any(source[key] != value for key, value in hashes.items())):
        raise ValueError("Embedding run and current upstream data differ")
    identity = source["embedding"].copy()
    recorded_cache = PureWindowsPath(identity.pop("cache")).name
    if (identity["model"], identity["revision"]) != (baseline_config["clip_model"], baseline_config["clip_revision"]):
        raise ValueError("Embedding run must use the pinned ST3 checkpoint")
    names = sorted(set(train["image"]) | set(dev["image"]))
    cache = feature_cache_path(root, names, cache_dir, identity)
    if cache.name != recorded_cache or not cache.is_file():
        raise ValueError(f"Missing or mismatched Task 001 feature cache: {cache}. Run the baseline first.")
    features = load_feature_cache(cache, names)
    provenance = {**hashes, "embedding": {**identity, "path": str(cache), "sha256": file_sha256(cache), "rows": len(names)},
                  "embedding_run": {"path": str(source_run), "sha256": file_sha256(source_run),
                                    "project_revision": source["project_revision"]}}
    return dict(zip(names, features)), provenance


def predict_lr(model, inputs, embeddings, languages, *, interactions=True):
    matrix = design_matrix(inputs["image"], inputs["language"], embeddings, languages, interactions=interactions)
    return inputs[KEYS].assign(p_positive=model.predict_proba(matrix)[:, 1].round(6))


def fit_variants(train, inputs, embeddings, seed):
    languages = sorted(train["language"].unique())
    predictions = {"A": language_prior(train, inputs)}
    models = {}
    iterations = {}
    for name, interactions in (("B", False), ("C", True)):
        matrix = design_matrix(train["image"], train["language"], embeddings, languages, interactions=interactions)
        models[name] = fit_lr(matrix, train, seed)
        del matrix
        predictions[name] = predict_lr(models[name], inputs, embeddings, languages, interactions=interactions)
        iterations[name] = int(models[name].n_iter_[0])
    return predictions, models["C"], languages, iterations


def shuffle_images(dev, seed):
    """Uniformly draw a derangement of sorted painting IDs and map every row."""
    by_painting = dev[["painting", "image"]].drop_duplicates().set_index("painting")["image"].sort_index()
    if not by_painting.index.is_unique or not by_painting.is_unique or len(by_painting) < 2:
        raise ValueError("Shuffling requires distinct image paths for at least two distinct paintings")
    ids = by_painting.index.to_numpy()
    rng = np.random.default_rng(seed)
    while True:
        sources = rng.permutation(ids)
        if np.all(sources != ids):
            break
    assignments = pd.Series(sources, index=by_painting.index, name="source_painting")
    return dev.assign(image=dev["painting"].map(assignments).map(by_painting)), assignments.reset_index()


def paired_analysis(gold, predictions, metrics):
    pairs = gold[[*KEYS, "k", "n"]].assign(p=gold["k"] / gold["n"])
    for name, frame in predictions.items():
        pairs = pairs.merge(frame.rename(columns={"p_positive": f"p_{name}"}), on=KEYS, validate="one_to_one")

    painting = painting_scores(pairs, "p_A").rename(columns={"contrast": "contrast_A",
                                                              "constant_prediction": "constant_prediction_A"})
    for name in ("B", "C"):
        details = painting_scores(pairs, f"p_{name}").set_index("painting")
        for column in ("contrast", "constant_prediction"):
            painting[f"{column}_{name}"] = painting["painting"].map(details[column])
    painting["contrast_C_minus_A"] = painting["contrast_C"] - painting["contrast_A"]
    painting["contrast_C_minus_B"] = painting["contrast_C"] - painting["contrast_B"]
    painting["contrast_B_minus_A"] = painting["contrast_B"] - painting["contrast_A"]

    language = pairs.groupby("language")["p"].agg(mean_pair_share="mean", median_pair_share="median",
                                                  min_pair_share="min", max_pair_share="max")
    counts = pairs.groupby("language")[["k", "n"]].sum().rename(columns={"k": "positive_votes", "n": "binary_votes"})
    language = language.join(counts)
    language["pooled_positive_share"] = language["positive_votes"] / language["binary_votes"]
    for name, result in metrics.items():
        loss = pd.DataFrame.from_dict(result["per_language"], orient="index")
        if name == "A":
            language = language.join(loss[["pairs", "included"]].rename(columns={"included": "macro_eligible"}))
        language[f"log_loss_{name}"] = loss["log_loss"]
    language["log_loss_C_minus_A"] = language["log_loss_C"] - language["log_loss_A"]
    language["log_loss_C_minus_B"] = language["log_loss_C"] - language["log_loss_B"]
    return pairs, painting, language.reset_index()


def paired_bootstrap(painting, replicates, seed, columns=CONTRAST_DELTAS):
    """Resample each scored painting's vector of paired deltas as one unit."""
    scored = painting.loc[painting["status"] == "scored"].sort_values("painting")
    values = scored[list(columns)].to_numpy()
    if len(values) == 0 or not np.isfinite(values).all() or replicates < 1:
        raise ValueError("Bootstrap requires scored paintings, finite paired deltas, and positive replicates")
    draws = np.random.default_rng(seed).integers(len(values), size=(replicates, len(values)))
    samples = pd.DataFrame(values[draws].mean(axis=1), columns=columns)
    samples.insert(0, "replicate", np.arange(replicates))
    summary = {}
    for index, column in enumerate(columns):
        low, high = samples[column].quantile([.025, .975])
        summary[column] = {"difference": float(values[:, index].mean()), "ci95": [float(low), float(high)]}
    return samples, summary


def painting_folds(train, folds, seed):
    """Yield row indices for disjoint painting groups, with each group held out once."""
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    yield from splitter.split(train, groups=train["painting"])


def grouped_cv(train, embeddings, config):
    """Score the same three variants on train-only held-out painting groups."""
    fold_metrics = []
    oof = []
    folds = config["cv_folds"]
    for fold, (fit_idx, heldout_idx) in enumerate(painting_folds(train, folds, config["cv_seed"]), 1):
        print(f"Fitting train-only grouped CV fold {fold}/{folds}", flush=True)
        fit, heldout = train.iloc[fit_idx], train.iloc[heldout_idx]
        inputs = heldout[[*KEYS, "image"]]
        predictions, _, _, iterations = fit_variants(fit, inputs, embeddings, config["fit_seed"])
        result = {"fold": fold, "fit_paintings": fit["painting"].nunique(),
                  "heldout_paintings": heldout["painting"].nunique(), "heldout_pairs": len(heldout)}
        rows = inputs[KEYS].assign(fold=fold)
        for name, frame in predictions.items():
            metrics = score(heldout, frame)
            for key in ("cultural_contrast", "log_loss", "contrast_paintings", "log_loss_languages"):
                result[f"{key}_{name}"] = metrics[key]
            rows[f"p_{name}"] = frame["p_positive"]
        for name, count in iterations.items():
            result[f"iterations_{name}"] = count
        fold_metrics.append(result)
        oof.append(rows)
    return pd.DataFrame(fold_metrics), pd.concat(oof, ignore_index=True)
