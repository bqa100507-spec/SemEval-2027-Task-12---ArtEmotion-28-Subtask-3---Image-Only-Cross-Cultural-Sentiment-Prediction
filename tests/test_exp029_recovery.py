import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_exp029 import score_pilot
from st3.metrics import score


def test_recovery_scoring_excludes_baseline_column_from_gold():
    gold = pd.DataFrame({"painting": ["p"] * 3, "language": ["a", "b", "c"],
                         "k": [1, 3, 4], "n": [5] * 3})
    val = gold.assign(p_positive=[.9, .5, .1])
    pred = gold[["painting", "language"]].assign(p_positive=[.1, .5, .9])
    with pytest.raises(KeyError, match="p_positive"):
        score(val, pred)
    result = score_pilot(val, {"candidate": pred})["candidate"]
    assert result == score(gold, pred)
    assert result["cultural_contrast"] > .98
    assert val["p_positive"].tolist() == [.9, .5, .1]
