"""H1 Gate 0: do a painting's en/ar/zh cultural deviations inform its deviations in the other languages?

Pure CPU analysis on official Train annotations only (Development unused, nothing external).

Notation: per eligible (painting, language) pair, ``y`` is the empirical logit with a +0.5/+1 continuity
correction and ``v`` its binomial sampling variance ``1/(k+.5) + 1/(n-k+.5)``. Centering for a source t and
target l uses the painting's *other* eligible languages (excluding t and l), with at least three of them.

Two references are computed:

* ``shared`` (pre-registered): t and l are centered on the same mean of the other languages. That shared
  reference carries sampling noise and deviations of the other languages into both variables, so under
  independence the expected correlation is about +1/(m+1), not 0.
* ``split`` (bias check): the other languages are split into two disjoint halves; t is centered on one and l
  on the other, so no noise is shared and the null expectation is 0.

Two sampling variances are computed per pair:

* ``delta`` (pre-registered): ``1/(k+.5) + 1/(n-k+.5)``. At n = 5 votes it overstates the variance of the bounded
  empirical logit by about 2x, so the reliability ``1 - mean variance / observed variance`` comes out negative.
* ``exact`` (sensitivity, labelled as a deviation): the exact variance of the empirical logit under
  Binomial(n, p) with the plug-in ``p = (k+.5)/(n+1)`` from the same counts.
"""

import math

import numpy as np
import pandas as pd

SOURCES = ("English", "Arabic", "Chinese")


def exact_logit_variance(k, n):
    """Exact variance of the corrected empirical logit under Binomial(n, (k+.5)/(n+1)), per (k, n)."""
    k, n = np.asarray(k, dtype=int), np.asarray(n, dtype=int)
    out = np.empty(len(k))
    for kk, nn in set(zip(k.tolist(), n.tolist())):
        grid = np.arange(nn + 1)
        p = (kk + .5) / (nn + 1)
        log_pmf = (np.array([math.lgamma(nn + 1) - math.lgamma(g + 1) - math.lgamma(nn - g + 1) for g in grid])
                   + grid * math.log(p) + (nn - grid) * math.log(1 - p))
        w = np.exp(log_pmf)
        y = np.log((grid + .5) / (nn - grid + .5))
        mean = (w * y).sum()
        out[(k == kk) & (n == nn)] = (w * (y - mean) ** 2).sum()
    return out


def logits(pairs, variance="delta"):
    k, n = pairs.k.to_numpy(dtype=float), pairs.n.to_numpy(dtype=float)
    if variance == "delta":
        v = 1 / (k + .5) + 1 / (n - k + .5)
    elif variance == "exact":
        v = exact_logit_variance(k, n)
    else:
        raise ValueError(variance)
    return pairs.assign(y=np.log((k + .5) / (n - k + .5)), v=v)


def wide(pairs, variance="delta"):
    table = logits(pairs, variance)
    return (table.pivot(index="painting", columns="language", values="y"),
            table.pivot(index="painting", columns="language", values="v"))


def centered_pair(y, v, source, target, minimum_others=3, mode="shared"):
    """Deviations of ``source`` and ``target`` per painting, with their sampling variances."""
    rows = []
    others_all = [c for c in y.columns if c not in (source, target)]
    for painting in y.index:
        ys, yt = y.at[painting, source], y.at[painting, target]
        if np.isnan(ys) or np.isnan(yt):
            continue
        others = [c for c in others_all if not np.isnan(y.at[painting, c])]
        if len(others) < minimum_others:
            continue
        if mode == "shared":
            ref_s = ref_t = others
        elif mode == "split":
            ref_s, ref_t = others[0::2], others[1::2]
        else:
            raise ValueError(mode)
        ms, mt = y.loc[painting, ref_s].mean(), y.loc[painting, ref_t].mean()
        vs = v.at[painting, source] + v.loc[painting, ref_s].sum() / len(ref_s) ** 2
        vt = v.at[painting, target] + v.loc[painting, ref_t].sum() / len(ref_t) ** 2
        rows.append((painting, ys - ms, yt - mt, vs, vt, len(others)))
    return pd.DataFrame(rows, columns=["painting", "d_source", "d_target", "v_source", "v_target", "others"])


def weighted_stats(x, y, vx, vy, weights):
    """Pearson r and per-variable reliabilities for (possibly bootstrap-weighted) rows."""
    w = np.asarray(weights, dtype=float)
    total = w.sum(axis=-1, keepdims=True)
    mx, my = (w * x).sum(-1, keepdims=True) / total, (w * y).sum(-1, keepdims=True) / total
    cx, cy = x - mx, y - my
    sxx, syy, sxy = (w * cx * cx).sum(-1) / total[..., 0], (w * cy * cy).sum(-1) / total[..., 0], \
        (w * cx * cy).sum(-1) / total[..., 0]
    r = sxy / np.sqrt(sxx * syy)
    rel_x = 1 - (w * vx).sum(-1) / total[..., 0] / sxx
    rel_y = 1 - (w * vy).sum(-1) / total[..., 0] / syy
    return r, rel_x, rel_y


def disattenuate(r, rel_x, rel_y):
    r, rel_x, rel_y = (np.asarray(a, dtype=float) for a in (r, rel_x, rel_y))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((rel_x > 0) & (rel_y > 0), r / np.sqrt(rel_x * rel_y), np.nan)


def pool(r, n, rel_x=None, rel_y=None):
    """Inverse-variance pooling: Fisher z for raw r; for disattenuated r, weights (n-3) * rel_x * rel_y."""
    r, n = np.asarray(r, dtype=float), np.asarray(n, dtype=float)
    if rel_x is None:
        z = np.arctanh(np.clip(r, -.999999, .999999))
        return float(np.tanh(np.nansum((n - 3) * z) / np.nansum((n - 3) * ~np.isnan(z))))
    weights = (n - 3) * np.clip(rel_x, 0, None) * np.clip(rel_y, 0, None)
    ok = ~np.isnan(r) & (weights > 0)
    return float((weights[ok] * r[ok]).sum() / weights[ok].sum()) if ok.any() else float("nan")


def analysis_a(y, v, targets, mode, minimum_n=100, draws=2000, seed=20261128):
    """Per (t, l) raw and disattenuated r, pooled per source and overall, with painting-bootstrap CIs."""
    paintings = list(y.index)
    position = {p: i for i, p in enumerate(paintings)}
    cells = []
    for source in SOURCES:
        for target in targets:
            frame = centered_pair(y, v, source, target, mode=mode)
            if len(frame) >= minimum_n:
                cells.append((source, target, frame))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(paintings), np.full(len(paintings), 1 / len(paintings)), size=draws)
    table, boot = [], {"raw": {}, "dis": {}}
    per_cell_boot = []
    for source, target, frame in cells:
        idx = frame.painting.map(position).to_numpy()
        arrays = [frame[c].to_numpy() for c in ("d_source", "d_target", "v_source", "v_target")]
        r, rx, ry = weighted_stats(*arrays, np.ones(len(frame)))
        table.append({"source": source, "target": target, "n": len(frame), "r_raw": float(r),
                      "reliability_source": float(rx), "reliability_target": float(ry),
                      "r_disattenuated": float(disattenuate(r, rx, ry)), "mean_others": float(frame.others.mean())})
        br, brx, bry = weighted_stats(*[a[None, :] for a in arrays], counts[:, idx])
        per_cell_boot.append((source, br, brx, bry, len(frame)))
    cells_table = pd.DataFrame(table)
    result = {"cells": len(cells_table)}
    for label, chosen in [(s, [s]) for s in SOURCES] + [("all", list(SOURCES))]:
        part = cells_table[cells_table.source.isin(chosen)]
        if part.empty:
            result[label] = None
            continue
        raw = pool(part.r_raw, part.n)
        dis = pool(part.r_disattenuated, part.n, part.reliability_source, part.reliability_target)
        draws_raw, draws_dis = [], []
        boots = [b for b in per_cell_boot if b[0] in chosen]
        n = np.array([b[4] for b in boots])
        R = np.stack([b[1] for b in boots], axis=1)
        RX = np.stack([b[2] for b in boots], axis=1)
        RY = np.stack([b[3] for b in boots], axis=1)
        for i in range(draws):
            draws_raw.append(pool(R[i], n))
            draws_dis.append(pool(disattenuate(R[i], RX[i], RY[i]), n, RX[i], RY[i]))
        result[label] = {"pairs": len(part), "paintings_max": int(part.n.max()),
                         "pooled_raw": raw, "pooled_raw_ci95": np.nanquantile(draws_raw, [.025, .975]).tolist(),
                         "pooled_disattenuated": dis,
                         "pooled_disattenuated_ci95": np.nanquantile(draws_dis, [.025, .975]).tolist()}
    return result, cells_table


def shared_reference_null(m):
    """Expected r under independence for the shared reference with m other languages (equal variances)."""
    m = np.asarray(m, dtype=float)
    return 1 / (m + 1)


def source_features(gold, c2, sources=SOURCES):
    """Oracle features per painting: gold source logit minus the painting's mean C2 logit (no other gold)."""
    level = c2.groupby("painting").c2_logit.mean()
    table = pd.DataFrame(index=sorted(set(gold.painting)))
    for source in sources:
        rows = gold[gold.language.eq(source)].set_index("painting")
        d = (rows.y - level.reindex(rows.index)).reindex(table.index)
        table[f"d_{source}"] = d.fillna(0.)
        table[f"has_{source}"] = d.notna().astype(float)
    return table


def fit_oracle(train, features, alphas, inner_folds, seed, use_values=True):
    """Per target language ridge on training paintings only; alpha by inner painting-grouped CV."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    columns = [c for c in features.columns if use_values or c.startswith("has_")]
    models = {}
    for language, part in train.groupby("language"):
        x = features.loc[part.painting, columns].to_numpy()
        target = part.residual.to_numpy()
        groups = part.painting.to_numpy()
        best, best_alpha = math.inf, alphas[-1]
        splits = min(inner_folds, len(set(groups)))
        if splits >= 2:
            for alpha in alphas:
                errors = []
                for fit, held in GroupKFold(n_splits=splits).split(x, target, groups):
                    model = Ridge(alpha=alpha).fit(x[fit], target[fit])
                    errors.append(np.mean((model.predict(x[held]) - target[held]) ** 2))
                if np.mean(errors) < best:
                    best, best_alpha = float(np.mean(errors)), alpha
        models[language] = (Ridge(alpha=best_alpha).fit(x, target), columns, best_alpha,
                            frozenset(part.painting))
    return models


def apply_oracle(models, held, features):
    out = held.copy()
    term = np.zeros(len(held))
    for language, (model, columns, _, fitted) in models.items():
        mask = held.language.eq(language).to_numpy()
        if not mask.any():
            continue
        if fitted & set(held.painting[mask]):
            raise ValueError("A held-out painting was used to fit its oracle map")
        term[mask] = model.predict(features.loc[held.painting[mask], columns].to_numpy())
    out["oracle"] = 1 / (1 + np.exp(-(out.c2_logit + term)))
    return out


def learning_curve(x, frame, folds, fractions, seeds, alphas):
    """CV Pearson r of a ridge predicting a centered deviation from frozen visual features."""
    from sklearn.linear_model import RidgeCV

    rows = []
    for fraction in fractions:
        for seed in seeds:
            predictions = pd.Series(index=frame.painting, dtype=float)
            sizes = []
            for fold in sorted(set(folds[frame.painting])):
                held = frame[folds[frame.painting].to_numpy() == fold]
                train = frame[folds[frame.painting].to_numpy() != fold]
                if set(held.painting) & set(train.painting):
                    raise ValueError("Learning-curve fold overlap")
                rng = np.random.default_rng(seed * 1000 + fold)
                size = max(10, int(round(fraction * len(train))))
                chosen = train.iloc[np.sort(rng.choice(len(train), size=size, replace=False))]
                model = RidgeCV(alphas=list(alphas)).fit(x.loc[chosen.painting].to_numpy(), chosen.d.to_numpy())
                predictions.loc[held.painting] = model.predict(x.loc[held.painting].to_numpy())
                sizes.append(size)
            r = float(np.corrcoef(predictions.loc[frame.painting].to_numpy(), frame.d.to_numpy())[0, 1])
            rows.append({"fraction": fraction, "seed": seed, "mean_train_paintings": float(np.mean(sizes)), "r": r})
    return pd.DataFrame(rows)


def slope(curve):
    means = curve.groupby("mean_train_paintings").r.mean()
    x = np.log(means.index.to_numpy())
    return float(np.polyfit(x, means.to_numpy(), 1)[0]) if len(means) > 1 else float("nan")


def stop_clauses(pooled_dis, pooled_ci, oracle_delta):
    """Which pre-registered STOP clause holds; the A clause is None when disattenuation is undefined."""
    evaluable = not (np.isnan(pooled_dis) or np.isnan(pooled_ci[0]) or np.isnan(pooled_ci[1]))
    a = bool(pooled_ci[0] <= 0 <= pooled_ci[1] and pooled_dis < .10) if evaluable else None
    return {"A_null_correlation": a, "B_oracle_below_0_005": bool(oracle_delta < .005)}


def decide(pooled_dis, pooled_ci, oracle_delta, oracle_ci, positive_slopes):
    if (pooled_ci[0] <= 0 <= pooled_ci[1] and pooled_dis < .10) or oracle_delta < .005:
        return "STOP_H1"
    if pooled_dis >= .15 and pooled_ci[0] > 0 and oracle_delta >= .010 and oracle_ci[0] > 0 and positive_slopes >= 2:
        return "PROMISING"
    return "INCONCLUSIVE"
