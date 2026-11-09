"""ST3 development scoring reconstructed from the public rules and starter kit."""

import numpy as np

from .data import KEYS
from .submission import validate_predictions

MIN_CONTRAST_LANGUAGES = 3
MIN_LOSS_PAIRS = 20


def score(gold, predictions):
    q = validate_predictions(predictions, gold)
    predictions = predictions.assign(p_positive=q)
    frame = gold.merge(predictions, on=KEYS, validate="one_to_one")
    frame["p"] = frame["k"] / frame["n"]
    correlations = []
    too_few = constant_gold = constant_prediction = 0
    for _, painting in frame.groupby("painting"):
        if len(painting) < MIN_CONTRAST_LANGUAGES:
            too_few += 1
        elif painting["p"].nunique() == 1:
            constant_gold += 1
        elif painting["p_positive"].nunique() == 1:
            constant_prediction += 1
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(painting["p"], painting["p_positive"])[0, 1]))

    q = frame["p_positive"].clip(.01, .99)
    frame["loss"] = -(frame["p"] * np.log2(q) + (1 - frame["p"]) * np.log2(1 - q))
    by_language = frame.groupby("language")["loss"].agg(pairs="size", log_loss="mean")
    included = by_language["pairs"] >= MIN_LOSS_PAIRS
    return {
        "cultural_contrast": float(np.mean(correlations)) if correlations else None,
        "log_loss": float(by_language.loc[included, "log_loss"].mean()) if included.any() else None,
        "pair_mean_log_loss": float(frame["loss"].mean()),
        "contrast_paintings": len(correlations),
        "contrast_too_few_languages": too_few,
        "contrast_constant_gold": constant_gold,
        "contrast_constant_prediction": constant_prediction,
        "log_loss_languages": int(included.sum()),
        "per_language": by_language.assign(included=included).to_dict(orient="index"),
    }
