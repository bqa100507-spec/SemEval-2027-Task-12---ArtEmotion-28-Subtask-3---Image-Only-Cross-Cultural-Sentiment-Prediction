"""Matched linear heads and partition-local semantic-shuffle controls, CPU only."""

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

from . import data
from .baseline import design_matrix, fit_lr
from .caption_data import inner_partition, require_isolation
from .exp023_evaluation import paired_interval
from .metrics import score, painting_scores


def unit(values):
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or (norm <= 1e-12).any():
        raise ValueError("Invalid or degenerate projected features")
    return values / norm


def derangement(names, seed):
    names = np.array(sorted(names))
    if len(names) < 2:
        raise ValueError("Shuffle needs at least two paintings in each partition")
    rng = np.random.default_rng(seed)
    while True:
        source = rng.permutation(names)
        if (source != names).all():
            return dict(zip(names, source))


def blend(visual, semantic, weight):
    if not 0 <= weight <= 1 or not visual[data.KEYS].equals(semantic[data.KEYS]):
        raise ValueError("Invalid blend weight or component key order")
    return visual.assign(p_positive=np.round((1 - weight) * visual.p_positive + weight * semantic.p_positive, 6))


def choose_weight(gold, visual, semantic, config):
    records = []
    for weight in config["late_semantic_weights"]:
        metrics = score(gold, blend(visual, semantic, weight))
        if metrics["cultural_contrast"] is None or metrics["log_loss"] is None:
            raise ValueError("Inner partition cannot support the required metrics")
        records.append({"weight": weight, "contrast": metrics["cultural_contrast"], "macro_log_loss": metrics["log_loss"]})
    best = min(records, key=lambda r: (-r["contrast"], r["macro_log_loss"], r["weight"]))
    return best["weight"], records


def head_predictions(fitting, query, visual, semantics, config, fold, phase):
    require_isolation(fitting, query)
    fit_images = fitting[["painting", "image"]].drop_duplicates().sort_values("painting")
    query_images = query[["painting", "image"]].drop_duplicates().sort_values("painting")
    # Separate permutations never transfer held-out semantic vectors into fitting.
    fit_map = derangement(fit_images.painting, config["shuffle_seed"] + fold * 10 + phase)
    query_map = derangement(query_images.painting, config["shuffle_seed"] + fold * 10 + phase + 10000)
    image_of = pd.concat([fit_images, query_images]).set_index("painting").image.to_dict()
    splits = []
    for images, mapping in ((fit_images, fit_map), (query_images, query_map)):
        v = np.stack([visual[name] for name in images.image])
        s = np.stack([semantics[name] for name in images.image])
        shuffled = np.stack([semantics[image_of[mapping[name]]] for name in images.painting])
        splits.append({"A": v, "B": s, "C": np.concatenate([v, s], axis=1) / np.sqrt(2),
                       "B_shuffle": shuffled, "C_shuffle": np.concatenate([v, shuffled], axis=1) / np.sqrt(2)})
    languages = sorted(fitting.language.unique())
    predictions, sizes = {}, {}
    for name in splits[0]:
        if min(splits[0][name].shape) < config["pca_dimensions"]:
            raise ValueError("Insufficient fitting images/dimensions for fixed PCA capacity")
        pca = PCA(n_components=config["pca_dimensions"], svd_solver="randomized", random_state=config["pca_seed"])
        pca.fit(splits[0][name])
        fit_values, query_values = unit(pca.transform(splits[0][name])), unit(pca.transform(splits[1][name]))
        bank = {**dict(zip(fit_images.image, fit_values)), **dict(zip(query_images.image, query_values))}
        matrix = design_matrix(fitting.image, fitting.language, bank, languages)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = fit_lr(matrix, fitting, 2027)
        if model.C != config["lr_C"]:
            raise ValueError("Official LR regularization differs from the frozen comparison")
        values = model.predict_proba(design_matrix(query.image, query.language, bank, languages))[:, 1]
        predictions[name] = query[data.KEYS].assign(p_positive=np.round(values, 6))
        sizes[name] = {"features": config["pca_dimensions"], "parameters": model.coef_.size + model.intercept_.size,
                       "iterations": int(model.n_iter_[0])}
    return predictions, {"heads": sizes, "fitting_paintings": sorted(fitting.painting.unique()),
                         "query_paintings": sorted(query.painting.unique()), "fitting_shuffle": fit_map, "query_shuffle": query_map}


def evaluate_fold(train, visual, semantics, reference, config, fold):
    fitting, held = train[train.fold.ne(fold)], train[train.fold.eq(fold)]
    inner, stopping = inner_partition(fitting, config, fold)
    inner_predictions, inner_details = head_predictions(inner, stopping, visual, semantics, config, fold, 0)
    weights, searches = {}, {}
    for suffix in ("", "_shuffle"):
        weights[suffix], searches[suffix] = choose_weight(stopping, inner_predictions["A"], inner_predictions["B" + suffix], config)
    predictions, details = head_predictions(fitting, held, visual, semantics, config, fold, 1)
    predictions["D"] = blend(predictions["A"], predictions["B"], weights[""])
    predictions["D_shuffle"] = blend(predictions["A"], predictions["B_shuffle"], weights["_shuffle"])
    del predictions["B_shuffle"]
    predictions["E50+L"] = held[data.KEYS].merge(reference[data.KEYS + ["p_positive"]], on=data.KEYS, validate="one_to_one")
    frame = pd.concat([pred.assign(model=name, fold=fold) for name, pred in predictions.items()], ignore_index=True)
    return frame, {"fold": fold, "late_weights": weights, "weight_search": searches, "inner": inner_details, "outer": details}


def summarize(gold, predictions):
    rows, paintings = [], []
    for name, pred in predictions.groupby("model", sort=False):
        metrics = score(gold, pred[data.KEYS + ["p_positive"]])
        rows.append({"model": name, "contrast": metrics["cultural_contrast"], "macro_log_loss": metrics["log_loss"]})
        joined = gold[data.KEYS + ["n", "k"]].merge(pred[data.KEYS + ["p_positive"]], on=data.KEYS, validate="one_to_one")
        paintings.append(painting_scores(joined.assign(p=joined.k / joined.n)).assign(model=name))
    return pd.DataFrame(rows), pd.concat(paintings, ignore_index=True)


def contribution_evidence(gold, predictions, config):
    """Five-fold comparisons and uncertainty are evidence, never run admission."""
    if set(gold.fold.unique()) != {1, 2, 3, 4, 5}:
        raise ValueError("Paired final evidence requires the complete five-fold population")
    summary, paintings = summarize(gold, predictions)
    scores = summary.set_index("model")
    pivot = paintings.pivot(index="painting", columns="model", values="contrast").dropna()
    comparisons = {}
    pairs = [("B", "A"), ("C", "A"), ("D", "A"), ("C", "C_shuffle"), ("D", "D_shuffle"),
             ("C", "E50+L"), ("D", "E50+L")]
    for candidate, control in pairs:
        delta = pivot[candidate] - pivot[control]
        evidence = paired_interval(delta, config["bootstrap_replicates"], config["bootstrap_seed"])
        evidence.update(wins=int((delta > 0).sum()), regressions=int((delta < 0).sum()),
                        macro_log_loss_change=float(scores.loc[candidate, "macro_log_loss"] - scores.loc[control, "macro_log_loss"]))
        fold_gains = {}
        for fold in sorted(gold.fold.unique()):
            held = gold[gold.fold.eq(fold)]
            candidate_pred = predictions[predictions.model.eq(candidate) & predictions.fold.eq(fold)]
            control_pred = predictions[predictions.model.eq(control) & predictions.fold.eq(fold)]
            fold_gains[str(fold)] = score(held, candidate_pred[data.KEYS + ["p_positive"]])["cultural_contrast"] \
                                   - score(held, control_pred[data.KEYS + ["p_positive"]])["cultural_contrast"]
        evidence["fold_contrast_changes"] = fold_gains
        comparisons[candidate + "-" + control] = evidence
    return {"status": "FIVE_FOLD_EVIDENCE", "comparisons": comparisons,
            "interpretation": "Exploratory reused-P2 evaluation, not independent confirmation, an execution gate or production promotion"}
