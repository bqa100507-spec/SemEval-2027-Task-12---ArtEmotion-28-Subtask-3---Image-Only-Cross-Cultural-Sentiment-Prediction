import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location("audit027",Path(__file__).resolve().parents[1]/"scripts/analyze_exp027.py")
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_additive_crossfit_recovers_missing_language_profile_without_gold():
    rows=[]
    for i in range(20):
        for j in range(5):
            if (i+j)%7==0:continue
            rows.append(dict(painting=f"p{i}",language=f"l{j}",p_positive=.2+i*.01+j*.03,k=(i+j)%6,n=5))
    frame=pd.DataFrame(rows);a,summary=audit.crossfit(frame)
    assert summary["held_out_explained_variance"]==pytest.approx(1,abs=1e-12)
    assert np.allclose(a.prediction_centered_SS,a.profile_SS+a.residual_SS+a.profile_residual_cross_term)
    changed=frame.copy();changed["k"]=5-changed.k
    _,other=audit.crossfit(changed)
    assert other["held_out_explained_variance"]==summary["held_out_explained_variance"]
    _,repeat=audit.crossfit(frame);assert repeat==summary


def test_selection_tie_is_earliest_and_wrong_choice_fails():
    p={"curves":[{"epoch":1,"validation_BCE_nats":.6},{"epoch":2,"validation_BCE_nats":.6}],"selected_epoch":1}
    audit.validate_selection(p);p["selected_epoch"]=2
    with pytest.raises(ValueError):audit.validate_selection(p)
    assert audit.corr([1,1,1],[1,2,3]) is None
