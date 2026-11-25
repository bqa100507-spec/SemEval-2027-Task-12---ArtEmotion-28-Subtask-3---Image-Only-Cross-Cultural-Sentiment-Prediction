import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from st3.exp037_closure import digest, prediction_bytes, recipe_prediction, verify_export
from st3.exp037_diagnostics import centered, decompose, logit_ablations, oracle_variants, paired_bootstrap, paired_log_loss
from st3.metrics import painting_scores, score


def test_export_rejects_smuggled_duplicate_and_corrupt_members(tmp_path):
    path = tmp_path/"export.zip"
    manifest = {"identity":"pinned","files":{"one.json":digest(b"one")}}
    with ZipFile(path,"w") as z:
        z.writestr("export_manifest.json",json.dumps(manifest))
        z.writestr("one.json",b"one")
    with ZipFile(path) as z:
        assert verify_export(z,"pinned")==manifest
        with pytest.raises(ValueError,match="identity"):
            verify_export(z,"wrong")
    with ZipFile(path,"a") as z, pytest.warns(UserWarning,match="Duplicate"):
        z.writestr("one.json",b"wrong")
    with ZipFile(path) as z, pytest.raises(ValueError,match="membership"):
        verify_export(z,"pinned")
    with ZipFile(path,"w") as z:
        z.writestr("export_manifest.json",json.dumps(manifest))
        z.writestr("one.json",b"wrong")
    with ZipFile(path) as z, pytest.raises(ValueError,match="digest"):
        verify_export(z,"pinned")


def test_hierarchical_rounding_is_preserved():
    dev = pd.DataFrame({"painting":["p"],"language":["l"]})
    values = dict(zip(("ridge","cs","c"),(.123473,.123479,.123452)))
    frames = {name:dev.assign(p_positive=value) for name,value in values.items()}
    nested = [(0.8,[(.75,"ridge"),(.25,"cs")]),(.2,"c")]
    out = recipe_prediction(dev,frames,nested).p_positive.item()
    expected = round(.8*round(.75*.123473+.25*.123479,6)+.2*.123452,6)
    assert out==expected
    # A distinct fixture demonstrates that removing the inner rounding can change a row.
    frames["ridge"]["p_positive"] = .1
    frames["cs"]["p_positive"] = .100001
    frames["c"]["p_positive"] = .100002
    hierarchy = recipe_prediction(dev,frames,nested)
    flat = recipe_prediction(dev,frames,[(.6,"ridge"),(.2,"cs"),(.2,"c")])
    assert not hierarchy.equals(flat)
    with pytest.raises(ValueError,match="sum to 1"):
        recipe_prediction(dev,frames,[(.2,"ridge")])
    reversed_keys = pd.DataFrame({"painting":["p","r"],"language":["l","l"]})
    bad = b"painting,language,p_positive\nr,l,0.1\np,l,0.2\n"
    with pytest.raises(ValueError,match="order"):
        prediction_bytes(bad,reversed_keys)


def fixture():
    frame = pd.DataFrame([(f"p{i:02d}",l,k,10) for i in range(25) for l,k in zip("abcd",[1,7,4,8])],
                         columns=["painting","language","k","n"])
    return frame.assign(p=frame.k/frame.n, a=.40+.5*(frame.k/10-.5), b=.55-.4*(frame.k/10-.5))


def test_decomposition_invariance_and_painting_paired_loss():
    frame = fixture()
    summary, p, scores = decompose(frame,"a")
    assert summary["contrast"] == pytest.approx(1.)
    assert summary["ranking"] == 1 and summary["relative_deviation_rms"] == pytest.approx(.5)
    assert summary["deviation_calibration_slope"] == pytest.approx(2.)
    assert len(scores) == 4 and (p.languages == 4).all()
    for candidate in oracle_variants(frame,"a").values():
        assert painting_scores(frame.assign(p_positive=candidate)).contrast.mean() == pytest.approx(1.)
    x,y = decompose(frame,"a")[1].contrast,decompose(frame,"b")[1].contrast
    b = paired_bootstrap(x,y,draws=30)
    assert b["ci95"] == pytest.approx([2.,2.]) and b["wins"] == 25
    assert centered(frame,"a").groupby(frame.painting).mean().abs().max()<1e-14
    gold = frame[["painting","language","k","n"]]
    difference = [score(gold,frame[["painting","language"]].assign(p_positive=frame[name]))["log_loss"]
                  for name in ("a","b")]
    loss = paired_log_loss(frame,"a","b",draws=30)
    assert loss["delta"] == pytest.approx(difference[0]-difference[1])


def test_intercept_deletion_is_label_free():
    frame = fixture()
    a,parts = logit_ablations(frame,"a",dict(zip("abcd",[.1,.2,.3,.4])))
    b,_ = logit_ablations(frame.assign(p=0),"a",dict(zip("abcd",[.1,.2,.3,.4])))
    assert all(np.allclose(a[name],b[name]) for name in a)
    assert a["shared_proxy"].groupby(frame.painting).nunique().max() == 1
    assert parts["interaction_sd"] > 0
