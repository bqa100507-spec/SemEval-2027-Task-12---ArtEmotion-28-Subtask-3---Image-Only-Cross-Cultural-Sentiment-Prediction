import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from st3 import data
from st3.vote_audit import ordering_audit,reliability_summary
from st3.vote_inputs import COUNT_COLUMNS,EMOTIONS,development_votes,empirical_distributions

CONFIG=json.loads((Path(__file__).resolve().parents[1]/"configs/exp021.json").read_text())


def tiny_votes(tmp_path):
    root=tmp_path/"annotations";root.mkdir()
    protocol=pd.DataFrame({"painting":["train","locked"],"image":["train.jpg","locked.jpg"],
        "role":["development","holdout"],"fold":[1,0],"image_sha256":["a","b"]})
    records=[]
    for language,emotions in (("Thai",["awe","awe","amusement","fear","sadness","other","something else"]),
                               ("Tamil",["anger"]*5),("Korean",["excitement"]*5)):
        for emotion in emotions:records.append(["train",language,"train.jpg","train",emotion,"ignored caption"])
    records+=[["locked","Thai","locked.jpg","train","FORBIDDEN_EMOTION","must not be retained"],
              ["train","Thai","train.jpg","val","FORBIDDEN_EMOTION","validation"]]
    with (root/"artelingo28_train_val.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle);writer.writerow(data.ANNOTATION_COLUMNS+["caption"]);writer.writerows(records)
    return protocol


def test_annotation_only_eight_vote_reconstruction_preserves_repeated_rows(tmp_path):
    protocol=tiny_votes(tmp_path);pairs,votes=development_votes(tmp_path,protocol)
    assert set(EMOTIONS[:4])==data.POSITIVE and set(EMOTIONS[4:])==data.NEGATIVE
    assert pairs.n.tolist()==[5,5,5] and len(votes)==15
    thai=pairs[pairs.language.eq("Thai")].iloc[0]
    assert thai.count_awe==2 and thai.k==3 and thai.p==.6
    assert "caption" not in votes and set(votes.painting)=={"train"}
    d,y=empirical_distributions(pairs)
    np.testing.assert_allclose(d.sum(1),1.)
    np.testing.assert_array_equal(y,pairs.p)
    with pytest.raises(ValueError,match="Only original training"):
        empirical_distributions(pairs.assign(split="val"))
    with pytest.raises(ValueError,match="Invalid"):
        empirical_distributions(pairs.assign(n=6))


def test_beta_audit_reproducible_and_never_changes_empirical_targets(tmp_path):
    pairs,_=development_votes(tmp_path,tiny_votes(tmp_path));before=pairs.copy(deep=True)
    cfg={**CONFIG,"posterior_samples":1024}
    first,paintings=ordering_audit(pairs,cfg)
    second,_=ordering_audit(pairs.sample(frac=1,random_state=21),cfg)
    pd.testing.assert_frame_equal(first,second);pd.testing.assert_frame_equal(pairs,before)
    assert first.confidence.between(.5,1.).all()
    assert first[first.abs_difference.eq(1)].confidence.min()>=.95
    summary=reliability_summary(pairs,first,paintings,cfg)
    assert summary["n5_pairs"]==3 and summary["contrast_eligible_paintings"]==1
    assert "noise ceiling" in summary["assumptions"] and summary["training_policy"].startswith("Empirical")
