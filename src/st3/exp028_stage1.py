"""Fixed eight-arm EXP-028 Stage 1 logistic comparison on verified cached features."""
import time
import warnings

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.linalg import eigh
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .baseline import design_matrix
from .caption_data import require_isolation
from .exp028_cache import KEYS, partition
from .fusion import blend
from .metrics import score, painting_scores
from .visual_retrieval import unit_rows
from .vlm_semantics import fingerprint

SEED = 2027
ARMS = ("A_raw", "B_raw", "A256", "B256", "A512", "B512", "C512", "C512_shuffled_Q")
PROCESS = psutil.Process()


def fit_pca(x, weights, width):
    """Fit an unwhitened, painting-weighted covariance basis with fixed signs."""
    x, weights = np.asarray(x, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if not np.isfinite(x).all() or not (weights > 0).all() or len(x) <= width or x.shape[1] < width:
        raise ValueError("Insufficient finite PCA fitting data")
    mean = np.average(x, axis=0, weights=weights)
    centered = (x - mean) * np.sqrt(weights / weights.sum())[:, None]
    covariance = centered.T @ centered
    eigenvalues, basis = eigh(covariance, subset_by_index=[x.shape[1] - width, x.shape[1] - 1])
    if eigenvalues[0] <= eigenvalues[-1] * np.finfo(np.float64).eps * max(x.shape):
        raise ValueError(f"Numerical PCA rank below {width}")
    basis = basis[:, ::-1].copy()
    anchors = np.argmax(np.abs(basis), axis=0)
    basis *= np.sign(basis[anchors, np.arange(width)])
    return mean, basis


def shuffle_indices(rows, seed=28028):
    groups = rows.reset_index(drop=True).groupby("language", sort=True).indices
    assignment = np.arange(len(rows))
    for indices in groups.values():
        ranked = sorted(indices, key=lambda i: fingerprint([seed, rows.painting.iloc[i]]))
        assignment[ranked] = np.roll(ranked, -1)
    return assignment


def features_for_fit(fit, evaluation, visual, qwen, spec):
    require_isolation(fit, *evaluation)
    frames = (fit, *evaluation)
    pairs = [list(frame[KEYS].itertuples(index=False, name=None)) for frame in frames]
    if any(len(set(keys)) != len(keys) for keys in pairs):
        raise ValueError("Duplicate feature query")
    unique = fit.drop_duplicates("painting")
    v = [np.stack([visual[p] for p in frame.painting]) for frame in frames]
    q = [unit_rows(np.stack([qwen[key] for key in keys])) for keys in pairs]
    width = spec["features"]["capacity_control_PCA"]
    small = spec["features"]["per_block_PCA"]
    weight = (1.0 / fit.painting.map(fit.painting.value_counts())).to_numpy()
    mv, pv = fit_pca(np.stack([visual[p] for p in unique.painting]), np.ones(len(unique)), width)
    mq, pq = fit_pca(q[0], weight, width)
    outputs = {}
    for i, frame in enumerate(frames):
        vpc, qpc = (v[i] - mv) @ pv, (q[i] - mq) @ pq
        a, b = unit_rows(vpc[:, :small]), unit_rows(qpc[:, :small])
        values = (v[i], q[i], a, b, unit_rows(vpc), unit_rows(qpc),
                  np.hstack([a, b]) / np.sqrt(2),
                  np.hstack([a, b[shuffle_indices(frame)]]) / np.sqrt(2))
        for arm, value in zip(ARMS, values):
            outputs.setdefault(arm, []).append(value)
    return outputs


def matrix(feature, language, vocabulary):
    # Row IDs are private locators only. They do not enter Model C features.
    ids = list(range(len(feature)))
    return design_matrix(ids, language, dict(zip(ids, feature)), vocabulary)


def weighted_bce(logits, frame):
    k, n = frame.k.to_numpy(), frame.n.to_numpy()
    return float(np.dot(k, np.logaddexp(0, -logits)) / n.sum()
                 + np.dot(n - k, np.logaddexp(0, logits)) / n.sum())


def select_c(candidates):
    return min(candidates, key=lambda row: (row["inner_BCE_nats"], row["C"]))["C"]


def fit_model(x, fit, c, *, memory_limit=6 * 1024**3):
    # Bound the principal CSR arrays and temporary construction copies. Read-out
    # RSS samples describe stage boundaries and are not a continuous peak.
    estimate = 6 * (x.data.nbytes + x.indices.nbytes + x.indptr.nbytes)
    rss_before = PROCESS.memory_info().rss
    if estimate + rss_before > memory_limit or estimate > psutil.virtual_memory().available:
        raise MemoryError("Model C sparse matrices exceed the configured memory budget")
    started = time.perf_counter()
    k, n = fit.k.to_numpy(), fit.n.to_numpy()
    positive, negative = k > 0, n > k
    fit_matrix = sparse.vstack([x[positive], x[negative]], format="csr")
    weights = np.r_[k[positive], (n - k)[negative]]
    labels = np.repeat([1, 0], [positive.sum(), negative.sum()])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=c, max_iter=3000, solver="lbfgs", random_state=SEED)
        model.fit(fit_matrix, labels, sample_weight=weights)
    if model.n_iter_[0] >= 3000 or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError("Invalid or unconverged LR model")
    logits = model.decision_function(x)
    expanded = float(np.average(np.logaddexp(0, (1 - 2 * labels) * np.r_[logits[positive], logits[negative]]),
                                weights=weights))
    compressed = weighted_bce(logits, fit)
    if abs(expanded - compressed) > 1e-12:
        raise ValueError("Binary vote-count objective failed parity")
    return model, {
        "C": c, "iterations": int(model.n_iter_[0]), "converged": True,
        "parameters": int(model.coef_.size + model.intercept_.size),
        "objective_parity_abs": abs(expanded - compressed),
        "fit_BCE_nats": compressed, "seconds": time.perf_counter() - started,
        "rss_before_bytes": rss_before, "rss_after_bytes": PROCESS.memory_info().rss,
        "estimated_sparse_workspace_bytes": estimate,
    }


def predicted(model, x, frame):
    return frame[KEYS].assign(p_positive=model.predict_proba(x)[:, 1].round(6))


def matched_reference(fit, held, visual_blocks):
    languages = sorted(fit.language.unique())
    predictions, records = {}, {}
    for name, vectors in visual_blocks.items():
        x = design_matrix(fit.image, fit.language, vectors, languages)
        model, records[name] = fit_model(x, fit, .3)
        del x
        h = design_matrix(held.image, held.language, vectors, languages)
        predictions[name] = predicted(model, h, held)
    return blend(blend(predictions["CLIP-B"], predictions["SigLIP2-Base"], .5), predictions["CLIP-L"], .5), records


def paired(values, ref, *, draws=10000, seed=28029):
    delta = np.asarray(values) - np.asarray(ref)
    if len(delta) == 0 or not np.isfinite(delta).all():
        raise ValueError("Paired metrics require finite common paintings")
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    return {"delta": float(delta.mean()), "wins": int((delta > 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()), "ties": int((abs(delta) <= 1e-12).sum()),
            "ci95": np.quantile(boot, [.025, .975]).tolist(), "eligible_paintings": len(delta)}


def evaluate(held, predictions, spec):
    metrics, painting_results = {}, {}
    for name, pred in predictions.items():
        metrics[name] = score(held, pred)
        merged = held.merge(pred, on=KEYS, validate="one_to_one")
        scores = painting_scores(merged.assign(p=merged.k / merged.n)).set_index("painting")
        painting_results[name] = scores
        if metrics[name]["cultural_contrast"] is None or metrics[name]["log_loss"] is None:
            raise ValueError("Incomplete official scoring coverage")
    visual = max(("A_raw", "A256", "A512"), key=lambda arm: metrics[arm]["cultural_contrast"])
    qwen = max(("B_raw", "B256", "B512"), key=lambda arm: metrics[arm]["cultural_contrast"])
    comparisons = [
        (arm, "E50+L") for arm in ARMS
    ] + [("C512", arm) for arm in (visual, qwen, "A512", "B512", "C512_shuffled_Q")]
    comparisons = list(dict.fromkeys(comparisons))
    paired_metrics = {}
    for a, b in comparisons:
        left = painting_results[a].contrast.dropna()
        right = painting_results[b].contrast.dropna()
        if not left.index.equals(right.index):
            raise ValueError("Paired scoring populations differ")
        paired_metrics[f"{a} vs {b}"] = paired(left, right)
        paired_metrics[f"{a} vs {b}"]["macro_log_loss_delta_bits"] = metrics[a]["log_loss"] - metrics[b]["log_loss"]
    gate = spec["pilot_signal"]
    c, r, v = (metrics[k] for k in ("C512", "E50+L", visual))
    signal = {
        "against_reference": c["cultural_contrast"] - r["cultural_contrast"] >= gate["contrast_gain_over_matched_E50L"],
        "against_visual_only": c["cultural_contrast"] - v["cultural_contrast"] >= gate["gain_over_strongest_CLIP_only_control"],
        "macro_loss": c["log_loss"] - r["log_loss"] <= gate["macro_loss_guardrail_bits"],
        "capacity_and_shuffle": all(c["cultural_contrast"] > metrics[k]["cultural_contrast"]
                                    for k in (qwen, "B512", "A512", "C512_shuffled_Q")),
    }
    return {"metrics": metrics, "paired": paired_metrics, "strongest_visual": visual, "strongest_qwen": qwen,
            "exploratory_signal": all(signal.values()), "checks": signal}, painting_results


def run_screen(rows, visual_blocks, qwen, historical, spec):
    expected_keys = set(rows[KEYS].itertuples(index=False, name=None))
    if expected_keys != qwen.keys():
        raise ValueError("Qwen coverage must be complete before any real fit")
    inner, stopping, fitting, held = partition(rows, spec)
    if psutil.virtual_memory().available < 2 * 1024**3:
        raise MemoryError("Stage 1 requires at least 2 GiB available RAM before constructing feature matrices")
    records, predictions = {}, {}
    with threadpool_limits(limits=4):
        _, records["R0_inner"] = matched_reference(inner, stopping, visual_blocks)
        predictions["E50+L"], records["R0_refit"] = matched_reference(fitting, held, visual_blocks)
        actual = predictions["E50+L"].set_index(KEYS).p_positive.sort_index()
        frozen = historical[historical.fold.eq(3)].set_index(KEYS).p_positive.sort_index()
        if not actual.index.equals(frozen.index) or not np.isfinite(frozen).all() or not np.array_equal(actual, frozen):
            raise ValueError("Matched E50+L does not reproduce frozen fold-3 reference at six decimals")
        image_rows = rows.drop_duplicates("painting")
        large = unit_rows(np.stack([visual_blocks["CLIP-L"][image] for image in image_rows.image]))
        siglip = unit_rows(np.stack([visual_blocks["SigLIP2-Base"][image] for image in image_rows.image]))
        visual = dict(zip(image_rows.painting, np.hstack([large, siglip]) / np.sqrt(2)))
        selection = features_for_fit(inner, (stopping,), visual, qwen, spec)
        selected = {}
        for arm, (x, z) in selection.items():
            vocab = sorted(inner.language.unique())
            train_x, stop_x = matrix(x, inner.language, vocab), matrix(z, stopping.language, vocab)
            candidates = []
            for c in spec["LR"]["C_grid"]:
                model, receipt = fit_model(train_x, inner, c)
                receipt["inner_BCE_nats"] = weighted_bce(model.decision_function(stop_x), stopping)
                candidates.append(receipt)
            selected[arm] = select_c(candidates)
            records[arm] = {"selection": candidates, "selected_C": selected[arm]}
        del selection, x, z, train_x, stop_x
        for arm, (x, z) in features_for_fit(fitting, (held,), visual, qwen, spec).items():
            vocab = sorted(fitting.language.unique())
            fit_x = matrix(x, fitting.language, vocab)
            model, records[arm]["refit"] = fit_model(fit_x, fitting, selected[arm])
            del fit_x
            expected_parameters = (x.shape[1] + 1) * (len(vocab) + 1)
            if records[arm]["refit"]["parameters"] != expected_parameters:
                raise ValueError("Matched model parameter count differs")
            predictions[arm] = predicted(model, matrix(z, held.language, vocab), held)
    evaluation, paintings = evaluate(held, predictions, spec)
    return predictions, evaluation, paintings, records
