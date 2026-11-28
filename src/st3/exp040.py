"""EXP-040: E3-LoRA with a ridge-initialized interaction head (CPU-testable logic).

The trainable head is exactly the Track 1 E3 component written as one linear layer: for a unit-norm pooled
embedding ``x`` and language ``l`` the logit is ``intercept + language_intercept[l] + x @ (shared + deviation[l])``,
so ``W[l] = shared + deviation[l]`` and ``b[l] = intercept + language_intercept[l]``. A language unseen by the
ridge keeps ``W = shared`` and ``b = intercept`` (as ``competition_ridge.Head.logits``). The head starts at the
frozen ridge solution ``(W0, b0)`` and is anchored to it with ``lam * (||W - W0||^2 + ||b - b0||^2)``.

Candidate mixes, banks, corrections, the P2 report, gate arithmetic and the B1/B2 compositions reuse
``st3.exp038`` unchanged. The only new statistical step is injecting the E3L head's own held-out probabilities
as the E3L arm; inside the bank cross-fit E3L is a Track 1 ridge on the adapted embeddings (like any frozen arm).
"""

import math

import numpy as np
import pandas as pd

from . import exp038 as science
from .competition_ridge import Head
from .data import KEYS
from .metrics import score

ARM = "E3L"
BASE_ARMS = ("E1", "E2", "E3")
CANDIDATES = {"C2": ("E1", "E2", "E3"), "F1": ("E1", "E2", "E3L"), "F2": ("E1", "E2", "E3", "E3L")}
GATED = ("F1", "F2")


# Head.

def head_tensors(head, languages):
    """Per-language weight rows and biases of a Track 1 head, in ``languages`` order (float64)."""
    position = {language: i for i, language in enumerate(head.languages)}
    weight = np.repeat(np.asarray(head.shared, dtype=np.float64)[None, :], len(languages), axis=0)
    bias = np.full(len(languages), float(head.intercept))
    for row, language in enumerate(languages):
        if language in position:
            weight[row] += head.deviations[position[language]]
            bias[row] += head.language_intercepts[position[language]]
    return weight, bias


def head_from_tensors(weight, bias, languages):
    """Inverse view for deployment: an equivalent Track 1 head with zero shared slope and intercept."""
    weight, bias = np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    return Head(0., bias.copy(), np.zeros(weight.shape[1]), weight.copy(), tuple(languages))


def ridge_probabilities(head, x, keys):
    """Frozen Track 1 probabilities for ``keys`` from a painting-indexed feature frame (float64)."""
    paintings, languages = sorted(keys.painting.unique()), sorted(keys.language.unique())
    row = keys.painting.map({p: i for i, p in enumerate(paintings)}).to_numpy()
    col = keys.language.map({lang: i for i, lang in enumerate(languages)}).to_numpy()
    z = head.logits(np.ascontiguousarray(x.loc[paintings].to_numpy(dtype=np.float64)), languages)[row, col]
    from scipy.special import expit

    return expit(z)


def interaction_head(head, languages, lam):
    import torch
    from torch import nn

    class InteractionHead(nn.Module):
        """``logits[i, l] = x[i] @ W[l] + b[l]`` with a fixed L2 anchor to the ridge initialization."""

        def __init__(self):
            super().__init__()
            weight, bias = head_tensors(head, languages)
            self.languages = tuple(languages)
            self.lam = float(lam)
            self.linear = nn.Linear(weight.shape[1], len(languages))
            with torch.no_grad():
                self.linear.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
                self.linear.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
            self.register_buffer("weight0", torch.as_tensor(weight, dtype=torch.float32).clone())
            self.register_buffer("bias0", torch.as_tensor(bias, dtype=torch.float32).clone())

        def forward(self, features):
            with torch.autocast(device_type=features.device.type, enabled=False):
                return self.linear(features.float())

        def anchor(self):
            return ((self.linear.weight - self.weight0) ** 2).sum() + ((self.linear.bias - self.bias0) ** 2).sum()

    return InteractionHead()


def binomial_nll(logits, k, n):
    """Binomial negative log-likelihood on counts, per vote: sum(n * softplus(z) - k * z) / sum(n)."""
    import torch.nn.functional as F

    z = logits.float()
    return (n * F.softplus(z) - k * z).sum() / n.sum()


def objective(logits, k, n, head):
    nll = binomial_nll(logits, k, n)
    return nll + head.lam * head.anchor(), nll


# Inner split, epoch choice, deployment epoch.

def inner_split(paintings, fraction, seed):
    """A painting-grouped inner validation split of the outer training fold (sorted, fixed seed)."""
    names = sorted(set(paintings))
    count = max(1, int(round(fraction * len(names))))
    order = np.random.default_rng(seed).permutation(len(names))
    held = sorted(names[i] for i in order[:count])
    return sorted(set(names) - set(held)), held


def test_like_contrast(gold, probabilities, excluded):
    """Filtered (test-like) Cultural Contrast of one prediction frame on a scoring frame."""
    subset = gold.loc[~gold.language.isin(excluded), KEYS + ["k", "n"]]
    joined = subset[KEYS].merge(probabilities[KEYS + ["p_positive"]], on=KEYS, validate="one_to_one")
    return float(score(subset, joined)["cultural_contrast"])


def choose_epoch(history):
    """Best inner test-like Contrast over epochs 0..E; ties (exact) go to the smaller epoch."""
    if not history or history[0]["epoch"] != 0:
        raise ValueError("Epoch selection needs the frozen initialization (epoch 0) first")
    best = max(row["inner_test_like_contrast"] for row in history)
    return min(row["epoch"] for row in history if row["inner_test_like_contrast"] == best)


def median_epoch(selected):
    """Median of the selected fold epochs; with an even count the lower middle value (smaller epoch)."""
    values = sorted(int(e) for e in selected)
    if not values:
        raise ValueError("No selected epochs")
    return values[(len(values) - 1) // 2]


def schedule(total_steps, warmup_fraction):
    """EXP-036 learning-rate multiplier: linear warmup, then cosine down to a 0.1 floor."""
    warmup = max(1, int(total_steps * warmup_fraction))

    def factor(i):
        if i < warmup:
            return (i + 1) / warmup
        return .1 + .9 * (1 + math.cos(math.pi * (i - warmup) / max(1, total_steps - warmup))) / 2

    return factor


def steps_per_epoch(paintings, batch):
    return math.ceil(paintings / batch)


# Candidates around one fit (P2 fold or deployment).

def with_e3l(features, embeddings):
    """Feature dictionary for one fit: the base features plus the fold's adapted E3L embeddings."""
    out = dict(features)
    out[ARM] = embeddings
    return out


def predict_with_banks(heads, bank_set, visual, keys, features, e3l_probabilities, neighborhood,
                       candidates=None, arms=(*BASE_ARMS, ARM)):
    """Outer predictions, E3L injection, candidate mixes and neighbor corrections from fitted pieces.

    ``e3l_probabilities`` are the E3L head's own probabilities for ``keys`` (None: selected epoch 0, so E3L is
    the frozen E3 column). A Track 1 head named E3L, when present, is reported as the ``E3L_ridge`` diagnostic.
    """
    candidates = dict(candidates or CANDIDATES)
    for system, bank in bank_set.items():
        if set(bank.visual.index) & set(keys.painting):
            raise ValueError(f"A requested painting is in the {system} neighbor bank")
    outer = science.predict_frame(heads, keys, features, None, {})
    outer = outer.rename(columns={ARM: f"{ARM}_ridge"}) if ARM in outer else outer
    if e3l_probabilities is None:  # Selected epoch 0: E3L is the frozen E3 (same head, same features).
        injected = outer[KEYS + ["E3"]].rename(columns={"E3": "p_positive"})
    else:
        injected = keys[KEYS].merge(e3l_probabilities[KEYS + ["p_positive"]], on=KEYS, how="left",
                                    validate="one_to_one")
    if injected.p_positive.isna().any() or len(injected) != len(keys):
        raise ValueError("E3L probabilities are missing for requested pairs")
    outer = outer.merge(injected.rename(columns={"p_positive": ARM}), on=KEYS, validate="one_to_one")
    outer[ARM] = outer[ARM].to_numpy(dtype=np.float64).round(6)
    outer = science.mix_columns(outer.drop(columns="fixed"), candidates)
    corrected, details = science.corrections(bank_set, keys, visual, outer, candidates, neighborhood)
    frame = science.systems_frame(outer, corrected, list(arms), candidates)
    if f"{ARM}_ridge" in outer:
        frame[f"{ARM}_ridge"] = outer[f"{ARM}_ridge"].to_numpy()
    return frame, outer, details


def fit_and_predict(train, keys, features, e3l_probabilities, track1, neighborhood, fold, candidates=None,
                    arms=(*BASE_ARMS, ARM)):
    """One outer fit of the C2 recipe with E3L injected.

    Heads for E1/E2/E3 (and a diagnostic E3L ridge on the adapted embeddings) are the Track 1 fits on
    ``train``; the E3L column on ``keys`` is the LoRA head's own probabilities. The neighbor bank is
    cross-fitted inside ``train`` exactly as C2's, with E3L as a ridge on the adapted embeddings.
    """
    candidates = dict(candidates or CANDIDATES)
    if set(train.painting) & set(keys.painting):
        raise ValueError("Outer-fold painting overlap")
    visual = science.spaces(features, candidates)
    heads, fits = science.fit_all(train, features, None, arms, track1)
    inner, details = science.crossfit(train, features, None, arms, track1, int(fold), candidates)
    bank_set = science.banks(train, inner, visual, candidates, neighborhood["minimum_bank_languages"])
    frame, outer, correction_details = predict_with_banks(heads, bank_set, visual, keys, features, e3l_probabilities,
                                                          neighborhood, candidates, arms)
    receipt = {"outer_fits": fits, "inner": details, "bank_paintings": {k: len(v.visual) for k, v in bank_set.items()}}
    return frame, outer.assign(fold=int(fold)), correction_details["R0"], receipt, (heads, bank_set, visual)


# Gate, selection and diagnostics.

def gates(gold, frame, config, available=True, reason=None):
    """The frozen gate of F1/F2 against the C2 recomputed in this run (or NOT_AVAILABLE with a reason)."""
    systems = ["R0_fixed", "R0_corrected", *BASE_ARMS, ARM, *[f"{c}_mix" for c in CANDIDATES], *CANDIDATES]
    if not available:
        systems = ["R0_fixed", "R0_corrected", *BASE_ARMS, "C2_mix", "C2"]
        report, paintings, _ = science.p2_report(gold, frame, systems, {}, config)
        missing = {name: {"status": "NOT_AVAILABLE", "passed": False, "reason": reason} for name in GATED}
        return report, paintings, missing
    gated = {name: CANDIDATES[name] for name in GATED}
    return science.p2_report(gold, frame, systems, gated, config)


def select(gate_results, tolerance):
    """Highest passing filtered Contrast; within ``tolerance`` F1 (the simpler) is preferred."""
    return science.select({k: v for k, v in gate_results.items() if k in GATED}, list(GATED), tolerance)


def diagnostics(gold, frame, config):
    """E3L vs E3 (paired interval) and centered residual correlations vs E3 and Fixed Ridge, per scope."""
    from .exp037_diagnostics import pair_diversity, paired_bootstrap
    from .metrics import painting_scores

    joined = gold[KEYS + ["fold", "k", "n"]].merge(frame.drop(columns="fold"), on=KEYS, validate="one_to_one")
    joined["p"] = joined.k / joined.n
    out = {}
    for scope, block in science.scopes(joined, config["test_like_excluded"]):
        scored = painting_scores(block, prediction="E3")
        eligible = set(scored.loc[scored.status.eq("scored"), "painting"])
        table = science.painting_contrasts(block, [ARM, "E3"])
        out[scope] = {
            "E3L_minus_E3": paired_bootstrap(table[ARM], table["E3"], seed=config["diagnostic_seed"]),
            "centered_residual_r_vs_E3": pair_diversity(block, ["E3", ARM], eligible)[0]["centered_residual_r"],
            "centered_residual_r_vs_fixed": pair_diversity(block, ["R0_fixed", ARM], eligible)[0][
                "centered_residual_r"]}
    return out


def per_language(gold, frame, config):
    """Per-language deltas (reporting only). Contrast is within-painting, so per language we report the Pearson r
    between painting-centered predictions and painting-centered observed rates, and the log loss in bits."""
    joined = gold[KEYS + ["k", "n"]].merge(frame.drop(columns="fold", errors="ignore"), on=KEYS, validate="one_to_one")
    joined["p"] = joined.k / joined.n
    names = ("E3", ARM, "C2", "F1", "F2")
    centered = joined.copy()
    for column in ("p", *names):
        centered[column] = joined[column] - joined.groupby("painting")[column].transform("mean")
    rows = []
    for language, part in centered.groupby("language"):
        raw = joined.loc[part.index]
        row = {"language": language, "test_like": language not in config["test_like_excluded"], "pairs": len(part)}
        for name in names:
            row[f"{name}_deviation_r"] = float(np.corrcoef(part[name], part.p)[0, 1]) if len(part) > 2 else np.nan
            q = raw[name].clip(.01, .99)
            row[f"{name}_log_loss"] = float((-(raw.p * np.log2(q) + (1 - raw.p) * np.log2(1 - q))).mean())
        for name, reference in ((ARM, "E3"), ("F1", "C2"), ("F2", "C2")):
            row[f"{name}_minus_{reference}_deviation_r"] = row[f"{name}_deviation_r"] - row[f"{reference}_deviation_r"]
            row[f"{name}_minus_{reference}_log_loss"] = row[f"{name}_log_loss"] - row[f"{reference}_log_loss"]
        rows.append(row)
    return pd.DataFrame(rows)


# Learning-curve diagnostic (CPU, frozen C2 recipe, report only).

def learning_curve(gold, features, track1, neighborhood, excluded, fractions, seeds, log=print):
    """Filtered P2 Contrast of the frozen C2 recipe refitted on subsamples of each outer training fold."""
    candidates = {"C2": CANDIDATES["C2"]}
    rows = []
    for fraction in fractions:
        for seed in (seeds if fraction < 1 else seeds[:1]):
            pieces = []
            sizes = []
            for fold, held in gold.groupby("fold"):
                train = gold.loc[gold.fold.ne(fold)]
                names = sorted(train.painting.unique())
                if fraction < 1:
                    count = int(round(fraction * len(names)))
                    keep = set(np.random.default_rng(seed * 1000 + int(fold)).choice(names, size=count, replace=False))
                    train = train.loc[train.painting.isin(keep)]
                sizes.append(train.painting.nunique())
                keys = held[KEYS]
                heads, _ = science.fit_all(train, features, None, BASE_ARMS, track1)
                outer = science.predict_frame(heads, keys, features, None, candidates)
                inner, _ = science.crossfit(train, features, None, BASE_ARMS, track1, int(fold), candidates)
                visual = science.spaces(features, candidates)
                bank_set = science.banks(train, inner, visual, candidates, neighborhood["minimum_bank_languages"])
                corrected, _ = science.corrections(bank_set, keys, visual, outer, candidates, neighborhood)
                pieces.append(corrected[KEYS + ["C2"]])
            frame = pd.concat(pieces, ignore_index=True)
            subset = gold.loc[~gold.language.isin(excluded)]
            joined = subset[KEYS].merge(frame, on=KEYS, validate="one_to_one")
            contrast = score(subset[KEYS + ["k", "n"]], joined.rename(columns={"C2": "p_positive"}))["cultural_contrast"]
            rows.append({"fraction": fraction, "seed": seed, "mean_train_paintings": float(np.mean(sizes)),
                         "filtered_contrast": float(contrast)})
            log(f"Learning curve fraction {fraction} seed {seed}: filtered C2 Contrast {contrast:.6f}")
    return pd.DataFrame(rows)


def curve_slope(curve, draws=2000, seed=0):
    """OLS slope of filtered Contrast against log N with a bootstrap CI over the subsample runs."""
    x, y = np.log(curve.mean_train_paintings.to_numpy()), curve.filtered_contrast.to_numpy()
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(draws):
        idx = rng.integers(0, len(x), len(x))
        if len(set(x[idx])) > 1:
            boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    low, high = np.quantile(boots, [.025, .975]).tolist()
    return {"slope_per_log_n": slope, "bootstrap_95": [float(low), float(high)], "runs": len(x),
            "ci_method": "resampling the subsample runs (fraction x seed) with replacement"}
