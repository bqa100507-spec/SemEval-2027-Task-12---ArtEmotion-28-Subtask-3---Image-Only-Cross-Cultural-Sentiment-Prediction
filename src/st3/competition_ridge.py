"""EXP-032's no-graph binomial head, recovered from 61ee4be9d629533cc5b03ef6bd8f925ac04dcc1e."""

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import expit, logit


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


def fit_head(panel, x, rule):
    """Fit the archived no-graph arm with its fixed deviation penalty."""
    ridge = 1 / (rule["reference_c"] * panel.n.sum())
    deviation_ratio = rule["deviation_ratio"]
    precision = (ridge * deviation_ratio) * np.eye(len(panel.languages))
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
    receipt["deviation_quadratic"] = float(np.sum(head.deviations * (precision @ head.deviations)))
    return head, receipt
