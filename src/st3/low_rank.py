"""Binomial low-rank visual interactions and post-hoc SVD diagnostics."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .baseline import C


def split_lr(model, width, languages):
    """Split Model B or C coefficients in the order of the baseline design."""
    weights = model.coef_[0]
    main = np.r_[model.intercept_, weights[:width + languages]]
    interaction = weights[width + languages:]
    return main, interaction.reshape(languages, width) if interaction.size else None


def main_logits(images, main):
    width = images.shape[1]
    shared = main[0] + images @ main[1:1 + width]
    # Reserve the final column for unseen languages with zero language-specific effects.
    return shared[:, None] + np.r_[main[1 + width:], 0.]


def unpack_factors(parameters, width, languages, rank):
    offset = 1 + width + languages
    return (parameters[:offset], parameters[offset:offset + languages * rank].reshape(languages, rank),
            parameters[offset + languages * rank:].reshape(width, rank))


def factor_logits(images, main, u, v):
    logits = main_logits(images, main)
    logits[:, :-1] += (images @ v) @ u.T
    return logits


def binomial_nll(logits, positives, totals):
    """Sum vote-level Bernoulli losses in nats, omitting the binomial constant."""
    return float(np.sum(positives * np.logaddexp(0., -logits) + (totals - positives) * np.logaddexp(0., logits)))


def factor_objective(parameters, images, positives, totals, rank, factor_penalty):
    width = images.shape[1]
    main, u, v = unpack_factors(parameters, width, totals.shape[1], rank)
    projected = images @ v
    logits = main_logits(images, main)[:, :-1] + projected @ u.T
    votes = totals.sum()
    residual = (totals - positives) * expit(logits) - positives * expit(-logits)
    by_image = residual.sum(axis=1)
    base_penalty = 1 / C
    loss = (binomial_nll(logits, positives, totals) + .5 * base_penalty * (main[1:] @ main[1:])
            + .5 * factor_penalty * (np.sum(u ** 2) + np.sum(v ** 2))) / votes
    grad_main = np.r_[by_image.sum(), images.T @ by_image, residual.sum(axis=0)]
    grad_main[1:] += base_penalty * main[1:]
    grad_u = residual.T @ projected + factor_penalty * u
    grad_v = images.T @ (residual @ u) + factor_penalty * v
    return loss, np.r_[grad_main, grad_u.ravel(), grad_v.ravel()] / votes


def fit_factors(images, positives, totals, main, rank, seed, config):
    width, languages = images.shape[1], totals.shape[1]
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, config["init_scale"], (width + languages) * rank)
    initial = np.r_[main, factors]
    factor_penalty = config["factor_penalty_multiplier"] / C
    args = (images, positives, totals, rank, factor_penalty)
    with threadpool_limits(limits=config["factor_threads"], user_api="blas"):
        initial_objective, _ = factor_objective(initial, *args)
        result = minimize(factor_objective, initial, args=args, jac=True, method="L-BFGS-B", options=config["optimizer"])
    finite = np.isfinite(result.x).all() and np.isfinite(result.fun) and np.isfinite(result.jac).all()
    if not finite:
        return None, {"converged": False, "status": "nonfinite", "iterations": int(result.nit),
                      "initial_objective": initial_objective if np.isfinite(initial_objective) else None}
    main, u, v = unpack_factors(result.x, width, languages, rank)
    gradient = float(np.abs(result.jac).max())
    info = {"converged": bool(result.success and result.fun <= initial_objective), "status": str(result.message),
            "iterations": int(result.nit), "function_evaluations": int(result.nfev), "gradient_max_abs": gradient,
            "initial_objective": float(initial_objective), "objective": float(result.fun),
            "train_nll": binomial_nll(factor_logits(images, main, u, v)[:, :-1], positives, totals) / totals.sum()}
    return (main, u, v), info


def truncate_svd(decomposition, rank):
    left, values, right = decomposition
    return (left[:, :rank] * values[:rank]) @ right[:rank]


def spectrum(values):
    energy = values ** 2
    total = energy.sum()
    normalized = energy / total if total else energy
    mass = values[values > 0]
    mass = mass / mass.sum() if len(mass) else mass
    effective = float(np.exp(-np.sum(mass * np.log(mass)))) if len(mass) else 0.
    tail_error = np.sqrt(np.r_[np.cumsum(energy[::-1])[::-1][1:], 0.])
    return {"singular_value": values, "energy": normalized, "cumulative_energy": np.cumsum(normalized),
            "frobenius_error": tail_error, "relative_error": tail_error / np.sqrt(total) if total else tail_error}, effective
