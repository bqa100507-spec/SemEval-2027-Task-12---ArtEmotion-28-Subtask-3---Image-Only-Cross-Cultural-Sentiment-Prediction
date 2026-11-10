"""Painting-balanced auxiliary objectives for the full ST3 interaction model."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .baseline import C
from .low_rank import binomial_nll, main_logits


class PairwiseHuber:
    def __init__(self, positives, totals, delta):
        left, right = np.triu_indices(totals.shape[1], k=1)
        rows, pairs = np.nonzero((totals[:, left] > 0) & (totals[:, right] > 0))
        counts = np.bincount(rows, minlength=len(totals))
        self.weights = 1. / (counts[rows] * np.count_nonzero(counts))
        self.left = rows * totals.shape[1] + left[pairs]
        self.right = rows * totals.shape[1] + right[pairs]
        shares = np.divide(positives, totals, out=np.zeros_like(positives, dtype=float), where=totals > 0).ravel()
        self.target = shares[self.left] - shares[self.right]
        self.delta = delta

    def __call__(self, probabilities):
        q = probabilities.ravel()
        error = q[self.left] - q[self.right] - self.target
        bounded = np.clip(error, -self.delta, self.delta)
        losses = .5 * bounded ** 2 + self.delta * (np.abs(error) - np.abs(bounded))
        gradient = self.weights * bounded
        derivative = np.bincount(self.left, gradient, minlength=q.size)
        derivative -= np.bincount(self.right, gradient, minlength=q.size)
        return float(losses @ self.weights), derivative.reshape(probabilities.shape)


def center(values, mask, counts):
    visible = np.where(mask, values, 0.)
    return np.where(mask, visible - (visible.sum(axis=1) / counts)[:, None], 0.)


class PaintingPearson:
    def __init__(self, positives, totals, epsilon):
        mask = totals > 0
        counts = mask.sum(axis=1)
        shares = np.divide(positives, totals, out=np.zeros_like(positives, dtype=float), where=mask)
        centered = center(shares, mask, np.maximum(counts, 1))
        norms = np.linalg.norm(centered, axis=1)
        # Check equality directly, as in the official scorer, before centering.
        low = np.min(np.where(mask, shares, np.inf), axis=1)
        high = np.max(np.where(mask, shares, -np.inf), axis=1)
        self.eligible = (counts >= 3) & (high > low)
        self.mask = mask[self.eligible]
        self.counts = counts[self.eligible]
        self.target = centered[self.eligible] / norms[self.eligible, None]
        self.epsilon = epsilon

    def __call__(self, probabilities):
        derivative = np.zeros_like(probabilities)
        if len(self.counts) == 0:
            return 0., derivative
        q = center(probabilities[self.eligible], self.mask, self.counts)
        denominator = np.sqrt(np.sum(q ** 2, axis=1) + self.epsilon ** 2)
        correlation = (q * self.target).sum(axis=1) / denominator
        gradient = -self.target / denominator[:, None] + q * (correlation / denominator ** 2)[:, None]
        derivative[self.eligible] = center(gradient, self.mask, self.counts) / len(self.counts)
        return float(1. - correlation.mean()), derivative


def unpack_full(parameters, width, languages):
    offset = 1 + width + languages
    return parameters[:offset], parameters[offset:].reshape(languages, width)


def objective(parameters, images, positives, totals, auxiliary, weight):
    main, interaction = unpack_full(parameters, images.shape[1], totals.shape[1])
    logits = main_logits(images, main)[:, :-1] + images @ interaction.T
    q = expit(logits)
    vote_count = totals.sum()
    nll = binomial_nll(logits, positives, totals) / vote_count
    penalty = parameters[1:] @ parameters[1:] / (2 * C * vote_count)
    residual = ((totals - positives) * q - positives * expit(-logits)) / vote_count
    auxiliary_loss = 0.
    if weight:
        auxiliary_loss, aux_gradient = auxiliary(q)
        residual += weight * aux_gradient * q * expit(-logits)
    by_image = residual.sum(axis=1)
    gradient = np.r_[by_image.sum(), images.T @ by_image, residual.sum(axis=0), (residual.T @ images).ravel()]
    gradient[1:] += parameters[1:] / (C * vote_count)
    return nll + penalty + weight * auxiliary_loss, gradient, nll, auxiliary_loss


def fit_contrast(grid, initial, auxiliary, weight, config):
    parameters = np.r_[initial[0], initial[1].ravel()]
    args = (*grid, auxiliary, weight)

    def loss(x):
        return objective(x, *args)[:2]

    options = config["optimizer"] if weight else config["zero_optimizer"]
    with threadpool_limits(limits=config["blas_threads"], user_api="blas"):
        start = objective(parameters, *args)
        result = minimize(loss, parameters, jac=True, method="L-BFGS-B", options=options)
        final = objective(result.x, *args)
    finite = np.isfinite(result.x).all() and np.isfinite(final[1]).all() and np.isfinite([final[0], *final[2:]]).all()
    info = {"converged": bool(finite and result.success and final[0] <= start[0] + 1e-14),
            "status": str(result.message) if finite else "nonfinite",
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "initial_objective": start[0], "initial_auxiliary": start[3]}
    if not finite:
        return None, info
    info.update({"objective": final[0], "train_nll": final[2], "auxiliary": final[3],
                 "gradient_max_abs": float(np.max(np.abs(final[1])))})
    return unpack_full(result.x, grid[0].shape[1], grid[1].shape[1]), info
