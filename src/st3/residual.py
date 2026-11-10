"""Small offset-logit corrections on frozen E50+L probabilities."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from .contrast import PairwiseHuber
from .low_rank import binomial_nll

MODELS = ("M0", "M1", "M2", "M3")
COMPONENTS = ("clip_b", "siglip2", "clip_l")


def calibration_inputs(frame, epsilon):
    values = frame[[*COMPONENTS, "q0"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Calibration needs finite probabilities in [0, 1]")
    b, s, large, baseline = logit(np.clip(values, epsilon, 1 - epsilon)).T
    return baseline, large - .5 * (b + s)


def design(languages, disagreement, support):
    matrix = np.zeros((len(disagreement), 2 + 2 * len(support)))
    matrix[:, 0], matrix[:, 1] = 1., disagreement
    indices = pd.Index(support).get_indexer(languages)
    rows = np.flatnonzero(indices >= 0)
    cols = indices[rows]
    matrix[rows, 2 + cols] = 1.
    matrix[rows, 2 + len(support) + cols] = disagreement[rows]
    return matrix


@dataclass
class Residual:
    model: str
    parameters: np.ndarray
    seen: tuple[str, ...]
    support: tuple[str, ...]
    center: float
    scale: float
    epsilon: float

    def predict(self, frame):
        baseline, disagreement = calibration_inputs(frame, self.epsilon)
        d = (disagreement - self.center) / self.scale
        shift = self.parameters[0] + self.parameters[1] * d
        indices = pd.Index(self.support).get_indexer(frame["language"])
        rows = np.flatnonzero(indices >= 0)
        columns = indices[rows]
        shift[rows] += self.parameters[2 + columns] + self.parameters[2 + len(self.support) + columns] * d[rows]
        if self.model != "M1":
            shift[~frame["language"].isin(self.seen).to_numpy()] = 0.
        return np.where(shift == 0., frame["q0"].to_numpy(), expit(baseline + shift).round(6))

    def to_dict(self):
        return {"model": self.model, "parameters": self.parameters.tolist(), "seen": list(self.seen),
                "support": list(self.support), "center": self.center, "scale": self.scale, "epsilon": self.epsilon}

    @classmethod
    def from_dict(cls, saved):
        return cls(**{**saved, "parameters": np.asarray(saved["parameters"], dtype=float),
                      "seen": tuple(saved["seen"]), "support": tuple(saved["support"])})


class PairwiseRows:
    """Apply EXP-004's painting-balanced Huber loss to the original row order."""

    def __init__(self, frame, delta):
        paintings, painting = np.unique(frame["painting"], return_inverse=True)
        languages, language = np.unique(frame["language"], return_inverse=True)
        shape = (len(paintings), len(languages))
        self.rows = painting * shape[1] + language
        self.shape = shape
        positives, totals = np.zeros(shape), np.zeros(shape)
        positives.ravel()[self.rows] = frame["k"]
        totals.ravel()[self.rows] = frame["n"]
        self.huber = PairwiseHuber(positives, totals, delta)

    def __call__(self, probabilities):
        grid = np.zeros(self.shape)
        grid.ravel()[self.rows] = probabilities
        loss, derivative = self.huber(grid)
        return loss, derivative.ravel()[self.rows]


def objective(parameters, baseline, x, k, n, penalty, auxiliary=None, auxiliary_weight=0.):
    logits = baseline + x @ parameters
    q = expit(logits)
    votes = n.sum()
    bce = binomial_nll(logits, k, n) / votes
    derivative = (n * q - k) / votes
    regularization = .5 * np.dot(penalty, parameters ** 2)
    pairwise = 0.
    if auxiliary is not None:
        pairwise, gradient = auxiliary(q)
        derivative += auxiliary_weight * gradient * q * (1 - q)
    return (bce + regularization + auxiliary_weight * pairwise,
            x.T @ derivative + penalty * parameters, bce, regularization, pairwise)


def fit_residual(frame, model, config):
    if (model not in MODELS[1:] or frame.empty or not frame["split"].eq("train").all()
            or frame.duplicated(["painting", "language"]).any()):
        raise ValueError("Fit requires distinct original training pairs and a residual model")
    k, n = frame["k"].to_numpy(dtype=float), frame["n"].to_numpy(dtype=float)
    if not np.isfinite([k, n]).all() or ((k < 0) | (k > n) | (n <= 0)).any():
        raise ValueError("Invalid training vote counts")
    baseline, disagreement = calibration_inputs(frame, config["logit_clip"])
    center, scale = float(disagreement.mean()), float(disagreement.std()) or 1.
    counts = frame.groupby("language")["painting"].nunique()
    support = tuple(counts.index[counts.ge(config["min_language_paintings"])]) if model != "M1" else ()
    x = design(frame["language"], (disagreement - center) / scale, support)
    penalty = np.r_[np.full(2, config["global_l2"]), np.full(2 * len(support), config["language_l2"])]
    auxiliary = PairwiseRows(frame, config["huber_delta"]) if model == "M3" else None
    args = baseline, x, k, n, penalty, auxiliary, config["pairwise_weight"]

    def loss(parameters):
        return objective(parameters, *args)[:2]

    initial = np.zeros(x.shape[1])
    with threadpool_limits(limits=1, user_api="blas"):
        initial_loss = loss(initial)[0]
        result = minimize(loss, initial, jac=True, method="L-BFGS-B", options=config["optimizer"])
        value, gradient, bce, regularization, pairwise = objective(result.x, *args)
    if not (np.isfinite(result.x).all() and np.isfinite(gradient).all()
            and np.isfinite([value, bce, regularization, pairwise]).all()):
        raise ValueError(f"{model} produced nonfinite calibration parameters or objective")
    corrector = Residual(model, result.x, tuple(counts.index), support, center, scale, config["logit_clip"])
    return corrector, {
        "converged": bool(result.success and value <= initial_loss + 1e-14), "status": str(result.message),
        "iterations": int(result.nit), "objective": value, "bce_nats_per_vote": bce, "regularization": regularization,
        "pairwise_huber": pairwise, "gradient_max_abs": float(np.max(np.abs(gradient))),
        "language_paintings": counts.to_dict(),
    }
