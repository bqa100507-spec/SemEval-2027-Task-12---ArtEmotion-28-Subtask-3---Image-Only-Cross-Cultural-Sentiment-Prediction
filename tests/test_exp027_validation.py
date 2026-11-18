import numpy as np
import pandas as pd
import pytest
from st3.exp027_validation import evaluate,publish,verify
from st3.metrics import score


def fixture(count=24):
    gold=pd.DataFrame([dict(painting=f"p{i}",language=l,k=k,n=5) for i in range(count) for l,k in [("A",1),("B",2),("C",4)]])
    base=gold[["painting","language"]].assign(p_positive=np.tile([.8,.5,.2],count))
    good=gold[["painting","language"]].assign(p_positive=np.tile([.2,.4,.8],count))
    return gold,{"BCE":base,"P_low":good,"P_high":good.copy()}


def test_metric_parity_and_paired_bootstrap():
    gold,p=fixture();r=evaluate(gold,p)
    for name,frame in p.items():
        official=score(gold,frame)
        assert r["metrics"][name]["mean_contrast"]==official["cultural_contrast"]
        assert r["metrics"][name]["macro_log_loss_bits"]==official["log_loss"]
    pair=r["paired_comparisons"]["P_low"]
    assert pair["wins"]==24 and pair["losses"]==0 and pair["paired_eligible_paintings"]==24
    assert pair["painting_paired_percentile_95_CI"][0]==pytest.approx(pair["mean_contrast_difference_vs_BCE"])
    assert evaluate(gold,p)==r


def test_small_subset_unavailable_loss_and_contrast_eligibility():
    gold,p=fixture(3)
    gold.loc[gold.painting.eq("p0"),"k"]=2
    keep=~(gold.painting.eq("p1") & gold.language.eq("C"))
    gold=gold[keep];p={k:v[keep].copy() for k,v in p.items()}
    p["P_high"]["p_positive"]=.5
    r=evaluate(gold,p)
    assert r["metrics"]["P_high"]["mean_contrast"]==0
    assert r["metrics"]["BCE"]["eligible_paintings"]==1
    assert r["metrics"]["BCE"]["macro_log_loss_bits"] is None
    assert r["metrics"]["BCE"]["log_loss_status"].startswith("UNAVAILABLE")
    assert np.isfinite(r["metrics"]["BCE"]["descriptive_vote_weighted_BCE_nats_from_rounded_predictions"])
    gold["k"]=2
    r=evaluate(gold,p)
    assert r["paired_comparisons"]["P_low"]["painting_paired_percentile_95_CI"] is None


def test_publish_and_independent_integrity_verification(tmp_path):
    gold,p=fixture(3);(tmp_path/"models").mkdir()
    payload={}
    for name,frame in p.items():
        (tmp_path/"models"/(name+".pt")).write_bytes(b"CPU checkpoint fixture")
        payload[name]={"validation_prediction_six_decimals":frame.p_positive.tolist(),"selected_epoch":1}
    result=publish(tmp_path,gold,payload)
    assert verify(tmp_path/"inner_validation")==result
    target=tmp_path/"inner_validation/P_low.csv";target.write_bytes(target.read_bytes()+b"corrupt")
    with pytest.raises(ValueError,match="integrity"):verify(tmp_path/"inner_validation")
