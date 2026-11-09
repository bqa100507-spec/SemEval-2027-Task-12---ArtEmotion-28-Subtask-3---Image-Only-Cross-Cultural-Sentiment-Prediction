import math

import pandas as pd
import pytest

from st3.metrics import score


def evaluate(rows):
    frame = pd.DataFrame(rows, columns=["painting", "language", "k", "n", "p_positive"])
    return score(frame[["painting", "language", "k", "n"]], frame[["painting", "language", "p_positive"]])


def test_contrast_per_painting_and_undefined_cases():
    cases = {
        "positive": [(0, 0), (2, .4), (5, 1)],
        "negative": [(0, 1), (2, .6), (5, 0), (5, 0)],
        "constant_prediction": [(0, .5), (2, .5), (5, .5)],
        "constant_gold": [(2, 0), (2, .4), (2, 1)],
        "two_languages": [(0, 0), (5, 1)],
    }
    result = evaluate([(name, str(lang), k, 5, q) for name, pairs in cases.items() for lang, (k, q) in enumerate(pairs)])
    assert result["cultural_contrast"] == pytest.approx(0)
    assert result["contrast_paintings"] == 3
    assert result["contrast_too_few_languages"] == result["contrast_constant_gold"] == result["contrast_constant_prediction"] == 1


def test_log_loss_is_clipped_pair_mean_then_language_macro():
    rows = [(f"a{i}", "A", 3, 5, 0) for i in range(19)]
    rows.append(("a19", "A", 50, 100, .5))
    rows += [(f"b{i}", "B", 8, 10, 1) for i in range(25)]
    rows += [(f"c{i}", "C", 1, 5, .5) for i in range(19)]
    result = evaluate(rows)
    clipped_a = -.6 * math.log2(.01) - .4 * math.log2(.99)
    a = (19 * clipped_a + 1) / 20
    b = -.8 * math.log2(.99) - .2 * math.log2(.01)
    assert result["log_loss"] == pytest.approx((a + b) / 2)
    assert result["pair_mean_log_loss"] == pytest.approx((20 * a + 25 * b + 19) / 64)
    assert result["log_loss_languages"] == 2
    assert result["per_language"]["C"]["included"] is False
    assert result["cultural_contrast"] is None


def test_no_eligible_language_yields_undefined_macro_log_loss():
    result = evaluate([("x", "A", 5, 5, .5)])
    assert result["log_loss"] is None
    assert result["pair_mean_log_loss"] == 1.0
