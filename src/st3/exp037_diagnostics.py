"""Post-hoc ST3 diagnostics on matched painting-language supports."""

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata

from .data import KEYS
from .metrics import MIN_LOSS_PAIRS, painting_scores


def correlation(a, b):
    if np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
        return 0.
    return float(np.corrcoef(a, b)[0, 1])


def paired_bootstrap(a, b, seed=203038, draws=20000):
    """Resample eligible paintings jointly with replacement in contrast space."""
    diff = np.asarray(a, dtype=float)-np.asarray(b, dtype=float)
    if diff.ndim != 1 or not len(diff) or not np.isfinite(diff).all():
        raise ValueError("Bootstrap expects finite paired painting contributions")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(len(diff), size=(draws, len(diff)))].mean(1)
    eps = 1e-12
    removed = math.ceil(.05*len(diff))
    by_absolute = np.argsort(np.abs(diff), kind="stable")
    by_value = np.sort(diff)
    return {"n":len(diff), "delta":float(diff.mean()), "ci95":np.quantile(means,[.025,.975]).tolist(),
            "positive_fraction":float((means>0).mean()),
            "wins":int((diff>eps).sum()),"ties":int((np.abs(diff)<=eps).sum()),"losses":int((diff<-eps).sum()),
            "trim_absolute_5pct":float(diff[by_absolute[:-removed]].mean()) if removed<len(diff) else None,
            "trim_best_5pct":float(by_value[:-removed].mean()) if removed<len(diff) else None,
            "without_single_best":float(by_value[:-1].mean()) if len(diff)>1 else None,
            "largest_wins_sum":float(by_value[-removed:].sum()),"largest_losses_sum":float(by_value[:removed].sum())}


def centered(frame, values):
    return frame[values] - frame.groupby("painting")[values].transform("mean")


def paired_log_loss(frame, model, baseline, seed=203038, draws=20000):
    """Painting resampling with the original panel's qualifying languages fixed."""
    eligible = frame.language.value_counts()
    eligible = eligible[eligible>=MIN_LOSS_PAIRS].index
    if not len(eligible):
        return None
    predictions = frame[[model,baseline]].clip(.01,.99)
    terms = -np.log2(predictions).mul(frame.p,axis=0)-np.log2(1-predictions).mul(1-frame.p,axis=0)
    names = pd.Index(sorted(frame.painting.unique()))
    panels = (frame.assign(difference=terms[model]-terms[baseline], present=1)
              .pivot(index="painting",columns="language",values=["difference","present"])
              .reindex(names).fillna(0))
    loss, count = [panels[key].reindex(columns=eligible,fill_value=0).to_numpy()
                   for key in ("difference","present")]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(names),np.full(len(names),1./len(names)),size=draws)
    support = weights @ count
    keep = (support>0).all(axis=1)
    if not keep.any():
        return None
    gain = ((weights[keep] @ loss)/support[keep]).mean(axis=1)
    return {"delta":float((loss.sum(0)/count.sum(0)).mean()),
            "ci95":np.quantile(gain,[.025,.975]).tolist(),"kept_draws":int(keep.sum())}


def decompose(frame, model):
    candidate = frame[KEYS+["p"]].assign(p_positive=frame[model])
    official = painting_scores(candidate).set_index("painting")
    names = official.index[official.status.eq("scored")]
    rows = []
    order_accumulator = {lang:[0.,0] for lang in frame.language.unique()}
    for name, painting in frame.groupby("painting"):
        y, q = painting.p.to_numpy(), painting[model].to_numpy()
        yc, qc = y-y.mean(), q-q.mean()
        speech = painting.language.to_numpy()
        a, b = np.triu_indices(len(y), 1)
        truth, pred = np.sign(y[a]-y[b]), np.sign(q[a]-q[b])
        correct = np.where(pred == 0, .5, truth == pred)
        compare = truth!=0
        if name in names:
            for l,r,h in zip(a[compare],b[compare],correct[compare]):
                for pos in (l,r):
                    order_accumulator[speech[pos]][0] += h
                    order_accumulator[speech[pos]][1] += 1
        rows.append({"painting":name,"contrast":official.at[name,"contrast"],
                     "spearman":correlation(rankdata(y),rankdata(q)),
                     "ranking":float(correct[compare].mean()) if compare.any() else np.nan,
                     "global_bias":float(q.mean()-y.mean()),"global_mse":float((q.mean()-y.mean())**2),
                     "deviation_mse":float(np.mean((yc-qc)**2)),
                     "gold_variance":float(np.mean(yc**2)), "prediction_variance":float(np.mean(qc**2)),
                     "covariance":float(np.mean(yc*qc)),
                     "languages":len(painting),"votes":int(painting.n.sum())})
    per_painting = pd.DataFrame(rows).set_index("painting")
    scored = per_painting.loc[names]
    summary = {name: float(scored[name].mean()) for name in ("contrast","spearman","ranking",
        "global_bias","global_mse","deviation_mse","gold_variance","prediction_variance","covariance")}
    summary |= {"n":len(scored), "negative":int((scored.contrast<0).sum()),
                "quantiles":scored.contrast.quantile([0,.1,.25,.5,.75,.9,1]).tolist(),
                "global_rmse":math.sqrt(summary["global_mse"]), "deviation_rmse":math.sqrt(summary["deviation_mse"]),
                "relative_deviation_rms":math.sqrt(summary["prediction_variance"]/summary["gold_variance"]),
                "deviation_calibration_slope":summary["covariance"]/summary["prediction_variance"]
                if summary["prediction_variance"] else None}
    return summary, per_painting, order_accumulator


def pair_diversity(frame, names, eligible):
    local = frame[frame.painting.isin(eligible)].copy()
    target = local.p.to_numpy()
    q = centered(local, names)
    centered_gold = centered(local, "p").to_numpy()
    pairs = []
    for a,b in combinations(names,2):
        xa,xb = local[a].to_numpy(),local[b].to_numpy()
        ca,cb = q[a].to_numpy(),q[b].to_numpy()
        agreements = []
        for _, painting in local.groupby("painting"):
            i,j = np.triu_indices(len(painting), 1)
            u,v = [np.sign(painting[n].to_numpy()[i]-painting[n].to_numpy()[j]) for n in (a,b)]
            compared = np.sign(painting.p.to_numpy()[i]-painting.p.to_numpy()[j]) != 0
            agreements.append(float(np.mean((u==v)[compared])))
        pairs.append({"a":a,"b":b,"prediction_r":correlation(xa,xb),"centered_prediction_r":correlation(ca,cb),
                      "residual_r":correlation(xa-target,xb-target),
                      "centered_residual_r":correlation(ca-centered_gold,cb-centered_gold),
                      "order_agreement":float(np.mean(agreements))})
    return pairs


def logit_ablations(frame, prediction, intercept, seed=203038):
    """Observed-support gauge from rounded scores, with no held-out gold involved."""
    p = frame[prediction].clip(1e-8,1-1e-8)
    lang = frame.language.map(intercept).fillna(float(np.mean(list(intercept.values()))))
    eta = logit(p)
    residual = eta-lang
    shared_proxy = residual.groupby(frame.painting).transform("mean")
    interaction = residual-shared_proxy
    shuffled = interaction.copy()
    rng = np.random.default_rng(seed)
    for _, rows in frame.groupby("language").groups.items():
        shuffled.loc[rows] = rng.permutation(interaction.loc[rows])
    variants = {"shared_proxy":expit(shared_proxy), "intercept_only":expit(lang),
                "no_intercept":expit(residual), "no_observed_interaction":expit(shared_proxy+lang),
                "shuffled_interaction":expit(shared_proxy+lang+shuffled)}
    components = {"shared_proxy_sd":float(shared_proxy.groupby(frame.painting).first().std(ddof=0)),
                  "intercept_sd_over_observed_rows":float(lang.std(ddof=0)),
                  "centered_intercept_sd":float((lang-lang.groupby(frame.painting).transform("mean")).std(ddof=0)),
                  "interaction_sd":float(interaction.std(ddof=0)),
                  "interaction_intercept_centered_r":correlation(interaction,lang-lang.groupby(frame.painting).transform("mean"))}
    return variants, components


def oracle_variants(frame, prediction):
    """Yield label-using diagnostics in probability space, with explicit range handling."""
    y = frame.p
    q = frame[prediction]
    y_mean = y.groupby(frame.painting).transform("mean")
    q_mean = q.groupby(frame.painting).transform("mean")
    y_dev, q_dev = y-y_mean, q-q_mean
    y_sd = np.sqrt(y_dev.pow(2).groupby(frame.painting).transform("mean"))
    q_sd = np.sqrt(q_dev.pow(2).groupby(frame.painting).transform("mean"))
    rescaled = q_dev * (y_sd/q_sd.where(q_sd>1e-12)).fillna(0.)
    reordered = q.copy()
    for _, ix in frame.groupby("painting").groups.items():
        order = np.argsort(y.loc[ix].to_numpy(),kind="stable")
        redistributed = np.sort(q.loc[ix])
        # Average within gold ties to avoid arbitrary ordering within equal observed rates.
        sorted_gold = y.loc[ix].to_numpy()[order]
        redistributed = pd.Series(redistributed).groupby(sorted_gold).transform("mean").to_numpy()
        reordered.loc[ix[order]] = redistributed
    return {"gold_mean":y_mean+q_dev, "gold_deviation_rms":q_mean+rescaled,
            "gold_deviations":q_mean+y_dev, "gold_order":reordered}
