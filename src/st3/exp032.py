"""Painting-grouped binomial heads and fitting-only task graphs for EXP-032."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata


@dataclass(frozen=True)
class Panel:
    paintings: tuple
    languages: tuple
    k: np.ndarray
    n: np.ndarray

    def subset(self, rows):
        indices = np.flatnonzero(rows)
        present = self.n[indices].sum(axis=0) > 0
        return Panel(
            tuple(self.paintings[i] for i in indices),
            tuple(language for language, keep in zip(self.languages, present) if keep),
            self.k[indices][:, present], self.n[indices][:, present],
        )


def make_panel(frame):
    paintings = tuple(sorted(frame["painting"].unique()))
    languages = tuple(sorted(frame["language"].unique()))
    rows = frame["painting"].map({name: i for i, name in enumerate(paintings)}).to_numpy()
    cols = frame["language"].map({name: i for i, name in enumerate(languages)}).to_numpy()
    if frame.duplicated(["painting", "language"]).any():
        raise ValueError("Repeated painting-language key")
    k = np.zeros((len(paintings), len(languages)))
    n = np.zeros_like(k)
    k[rows, cols] = frame["k"]
    n[rows, cols] = frame["n"]
    return Panel(paintings, languages, k, n)


def painting_splits(paintings, seed, groups):
    """Deterministic balanced folds independent of language coverage and votes."""
    order = sorted(range(len(paintings)), key=lambda i: sha256(f"{seed}:{paintings[i]}".encode()).digest())
    assignment = np.empty(len(order), dtype=int)
    assignment[order] = np.arange(len(order)) % groups
    return assignment


@dataclass(frozen=True)
class Head:
    intercept: float
    language_intercepts: np.ndarray
    shared: np.ndarray
    deviations: np.ndarray
    languages: tuple

    def logits(self, x, languages):
        positions = {language: i for i, language in enumerate(self.languages)}
        shared = self.intercept + x @ self.shared
        logits = np.repeat(shared[:, None], len(languages), axis=1)
        known = [(i, positions[language]) for i, language in enumerate(languages) if language in positions]
        if known:
            target, source = np.asarray(known).T
            logits[:, target] += self.language_intercepts[source] + x @ self.deviations[source].T
        return logits


def laplacian(adjacency):
    """Unit mean eigenvalue on the centered supported subspace."""
    degree = adjacency.sum(axis=1)
    supported = degree > 0
    count = int(supported.sum())
    matrix = np.diag(degree) - adjacency
    projection = np.diag(supported.astype(float))
    if count:
        projection -= np.outer(supported, supported) / count
        matrix *= (count - 1) / degree.sum()
    return matrix, projection


def deviation_precision(ridge, fraction, graph_laplacian, projection):
    # A uniform complete graph gives L = P. Isolates keep the identical ridge.
    return ridge * (np.eye(len(projection)) + fraction * (graph_laplacian - projection))


def objective(panel, x, ridge, precision=None):
    """Weighted NLL in nats and analytic gradient in a Helmert basis."""
    width, count = x.shape[1], len(panel.languages)
    basis = helmert(count, full=False).T
    votes = panel.n.sum()

    def unpack(theta):
        intercept = theta[0]
        contrast = theta[1:count]
        shared = theta[count:count + width]
        dev_coordinates = theta[count + width:].reshape(-1, width)
        deviations = basis @ dev_coordinates if precision is not None else np.zeros((count, width))
        return Head(intercept, basis @ contrast, shared, deviations, panel.languages)

    def loss_grad(theta):
        head = unpack(theta)
        z = (head.intercept + x @ head.shared)[:, None] + head.language_intercepts
        if precision is not None:
            z += x @ head.deviations.T
        nll = (panel.n * np.logaddexp(0, z) - panel.k * z).sum() / votes
        value = nll + 0.5 * ridge * (head.shared @ head.shared + head.language_intercepts @ head.language_intercepts)
        residual = (panel.n * expit(z) - panel.k) / votes
        gradient = [
            np.array([residual.sum()]),
            basis.T @ (residual.sum(axis=0) + ridge * head.language_intercepts),
            x.T @ residual.sum(axis=1) + ridge * head.shared,
        ]
        if precision is not None:
            penalty_grad = precision @ head.deviations
            value += 0.5 * np.sum(head.deviations * penalty_grad)
            gradient.append((basis.T @ (residual.T @ x + penalty_grad)).ravel())
        return value, np.concatenate(gradient)

    size = count + width + ((count - 1) * width if precision is not None else 0)
    return loss_grad, unpack, size


def fit_head(panel, x, rule, *, deviation_ratio=None, fraction=0.0, graph=None):
    """Fit one full-rank head. Additive nuisance fitting omits deviations."""
    ridge = 1 / (rule["reference_c"] * panel.n.sum())
    precision = None
    if deviation_ratio is not None:
        count = len(panel.languages)
        l_mat, proj = (graph["laplacian"], graph["projection"]) if graph is not None else (
            np.zeros((count, count)), np.zeros((count, count)),
        )
        precision = deviation_precision(ridge * deviation_ratio, fraction, l_mat, proj)
    loss_grad, unpack, size = objective(panel, x, ridge, precision)
    theta = np.zeros(size)
    theta[0] = logit((panel.k.sum() + 0.5) / (panel.n.sum() + 1))
    started = perf_counter()
    fitted = minimize(
        loss_grad, theta, jac=True, method="L-BFGS-B",
        options={"gtol": rule["gtol"], "ftol": rule["ftol"], "maxiter": rule["maxiter"], "maxls": 40},
    )
    norm = float(np.linalg.norm(fitted.jac, ord=np.inf))
    if not fitted.success or not np.isfinite(fitted.fun) or not np.isfinite(norm) or norm > rule["max_final_gradient"]:
        raise RuntimeError(f"Head did not converge: {fitted.message}, gradient infinity norm {norm:.3g}")
    head = unpack(fitted.x)
    receipt = {
        "parameters": size, "converged": True, "iterations": int(fitted.nit), "evaluations": int(fitted.nfev),
        "reason": str(fitted.message), "seconds": perf_counter() - started, "max_abs_gradient": norm,
        "objective": float(fitted.fun), "shared_l2": float(np.linalg.norm(head.shared)),
        "intercepts_l2": float(np.linalg.norm(head.language_intercepts)),
        "deviations_l2": float(np.linalg.norm(head.deviations)),
        "deviation_l2_by_language": dict(zip(head.languages, np.linalg.norm(head.deviations, axis=1).tolist())),
        "isotropic_deviation_ridge": ridge * deviation_ratio if deviation_ratio is not None else None,
    }
    if precision is not None:
        receipt["deviation_quadratic"] = float(np.sum(head.deviations * (precision @ head.deviations)))
        receipt["minimum_precision_ratio"] = float(np.linalg.eigvalsh(precision).min() / (ridge * deviation_ratio))
        if graph is not None:
            d = head.deviations
            receipt["graph_energy"] = float(np.sum(d * (l_mat @ d)))
            receipt["supported_energy"] = float(np.sum(d * (proj @ d)))
            receipt["gamma"] = float(ridge * deviation_ratio * fraction)
    return head, receipt


def crossfit_residuals(panel, x, spec, optimizer, seed):
    """No residual uses a nuisance model trained on the painting it describes."""
    splits = painting_splits(panel.paintings, seed, spec["nuisance_splits"])
    logits = np.empty(panel.n.shape)
    shared = np.empty(len(panel.paintings))
    receipts = []
    for split in range(spec["nuisance_splits"]):
        fit, predict = splits != split, splits == split
        fitted, receipt = fit_head(panel.subset(fit), x[fit], optimizer)
        logits[predict] = fitted.logits(x[predict], panel.languages)
        shared[predict] = fitted.intercept + x[predict] @ fitted.shared
        receipts.append(receipt | {"fitting_paintings": int(fit.sum()), "predicted_paintings": int(predict.sum())})
    residuals = logit((panel.k + 0.5) / (panel.n + 1)) - logits
    return residuals, shared, receipts


def association(a, b, weight, shared):
    """Excess paired covariance within four quantiles of the shared forecast."""
    groups = np.digitize(shared, np.quantile(shared, [0.25, 0.5, 0.75]))
    numerator, denominator = 0.0, 0.0
    for group in np.unique(groups):
        mask = groups == group
        w = weight[mask]
        da = a[mask] - np.average(a[mask], weights=w)
        db = b[mask] - np.average(b[mask], weights=w)
        numerator += float(2 * w @ (da * db))
        denominator += float(w @ (da**2 + db**2))
    return numerator / denominator if denominator else 0.0


def estimate_graph(panel, residuals, shared, spec):
    """Co-observed residual difference dispersion with bounded vote precision."""
    count = len(panel.languages)
    edges = {}
    for i, j in combinations(range(count), 2):
        both = (panel.n[:, i] > 0) & (panel.n[:, j] > 0)
        overlap = int(both.sum())
        ni, nj = panel.n[both, i], panel.n[both, j]
        weight = np.minimum(2 * ni * nj / (ni + nj), spec["vote_weight_cap"])
        effective = float(weight.sum() ** 2 / (weight @ weight)) if overlap else 0.0
        edge = {"indices": (i, j), "overlap": overlap, "effective_overlap": effective, "supported": False}
        edges[panel.languages[i], panel.languages[j]] = edge
        if effective < spec["minimum_overlap"]:
            continue
        a, b = residuals[both, i], residuals[both, j]
        delta = a - b
        centered = delta - np.average(delta, weights=weight)
        dispersion = float(weight @ centered**2 / (weight.sum() - (weight @ weight) / weight.sum()))
        edge.update(
            supported=True, dispersion=dispersion, paired_association=association(a, b, weight, shared),
            confidence=effective / (effective + spec["overlap_shrinkage"]),
        )
    valid = [e for e in edges.values() if e["supported"]]
    scale = max(float(np.median([e["dispersion"] for e in valid])), 1e-8) if valid else 1.0
    adj = np.zeros((count, count))
    overlap_adj = np.zeros_like(adj)
    for e in valid:
        i, j = e["indices"]
        e["similarity"] = float(np.exp(-e["dispersion"] / scale))
        e["weight"] = e["similarity"] * e["confidence"]
        adj[i, j] = adj[j, i] = e["weight"]
        overlap_adj[i, j] = overlap_adj[j, i] = e["confidence"]
    l_mat, p_mat = laplacian(adj)
    overlap_l, _ = laplacian(overlap_adj)
    units = max(np.linalg.norm(p_mat), 1)
    return {
        "laplacian": l_mat, "projection": p_mat, "edges": edges, "scale": scale, "languages": panel.languages,
        "unsupported": [language for language, degree in zip(panel.languages, adj.sum(axis=1)) if degree == 0],
        "isotropic_distance": float(np.linalg.norm(l_mat - p_mat) / units),
        "overlap_only_distance": float(np.linalg.norm(l_mat - overlap_l) / units),
    }


def rank_correlation(a, b):
    if len(a) < 2 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def graph_stability(full, left, right, spec):
    """Independent complementary fitting halves, with their own nuisance models."""
    shared = sorted(
        key for key in left["edges"].keys() & right["edges"].keys()
        if left["edges"][key]["supported"] and right["edges"][key]["supported"]
    )
    nodes = len({language for pair in shared for language in pair})
    a, b = left["edges"], right["edges"]
    ranks = rank_correlation([a[key]["similarity"] for key in shared], [b[key]["similarity"] for key in shared])
    overlap_ranks = rank_correlation(
        [a[key]["effective_overlap"] for key in shared], [b[key]["effective_overlap"] for key in shared],
    )
    association_ranks = rank_correlation(
        [a[key]["paired_association"] for key in shared], [b[key]["paired_association"] for key in shared],
    )
    best = max(1, len(shared) // 3)
    replicated = []
    if shared:
        for train, check in ((a, b), (b, a)):
            strongest = sorted(shared, key=lambda key: (-train[key]["similarity"], key))[:best]
            replicated.append(float(np.mean([check[key]["paired_association"] for key in strongest])))
    checks = {
        "edges": len(shared) >= spec["minimum_common_edges"],
        "languages": nodes >= spec["minimum_common_languages"],
        "rank_stability": ranks is not None and ranks >= spec["minimum_similarity_rank_correlation"],
        "replicated_association": bool(replicated) and min(replicated) >= spec["minimum_replicated_association"],
        "beyond_overlap": full["overlap_only_distance"] >= spec["minimum_overlap_only_distance"],
        "anisotropy": full["isotropic_distance"] >= spec["minimum_isotropic_distance"],
    }
    return {
        "pass": all(checks.values()), "checks": checks, "common_edges": len(shared), "common_languages": nodes,
        "similarity_rank_correlation": ranks, "overlap_rank_correlation": overlap_ranks,
        "paired_association_rank_correlation": association_ranks,
        "replicated_top_third_association": replicated,
        "disjoint_fitting_halves": True,
    }
