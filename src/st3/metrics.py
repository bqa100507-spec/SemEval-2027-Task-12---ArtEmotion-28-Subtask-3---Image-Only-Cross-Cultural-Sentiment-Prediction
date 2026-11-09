"""ST3 development scoring reconstructed from the public rules and starter kit."""

import numpy as np
import pandas as pd

from .data import KEYS
from .submission import validate_predictions

MIN_CONTRAST_LANGUAGES = 3
MIN_LOSS_PAIRS = 20


def painting_scores(frame, prediction="p_positive"):
    """Per-painting Contrast and eligibility from a joined gold/prediction frame."""
    rows = []
    for name, painting in frame.groupby("painting"):
        contrast = None
        status = "scored"
        constant_prediction = False
        if len(painting) < MIN_CONTRAST_LANGUAGES:
            status = "too_few_languages"
        elif painting["p"].nunique() == 1:
            status = "constant_gold"
        elif painting[prediction].nunique() == 1:
            constant_prediction = True
            contrast = 0.0
        else:
            contrast = float(np.corrcoef(painting["p"], painting[prediction])[0, 1])
        rows.append((name, len(painting), status, contrast, constant_prediction))
    return pd.DataFrame(rows, columns=["painting", "languages", "status", "contrast", "constant_prediction"])


def score(gold, predictions):
    q = validate_predictions(predictions, gold)
    predictions = predictions.assign(p_positive=q)
    frame = gold.merge(predictions, on=KEYS, validate="one_to_one")
    frame["p"] = frame["k"] / frame["n"]
    paintings = painting_scores(frame)
    correlations = paintings["contrast"].dropna().to_numpy()

    q = frame["p_positive"].clip(.01, .99)
    frame["loss"] = -(frame["p"] * np.log2(q) + (1 - frame["p"]) * np.log2(1 - q))
    by_language = frame.groupby("language")["loss"].agg(pairs="size", log_loss="mean")
    included = by_language["pairs"] >= MIN_LOSS_PAIRS
    return {
        "cultural_contrast": float(np.mean(correlations)) if len(correlations) else None,
        "log_loss": float(by_language.loc[included, "log_loss"].mean()) if included.any() else None,
        "pair_mean_log_loss": float(frame["loss"].mean()),
        "contrast_paintings": len(correlations),
        "contrast_too_few_languages": int(paintings["status"].eq("too_few_languages").sum()),
        "contrast_constant_gold": int(paintings["status"].eq("constant_gold").sum()),
        "contrast_constant_prediction": int(paintings["constant_prediction"].sum()),
        "log_loss_languages": int(included.sum()),
        "per_language": by_language.assign(included=included).to_dict(orient="index"),
    }
