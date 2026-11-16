import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from st3 import vlm_semantics
from st3.vlm_semantics import SemanticCache, extract, fingerprint
from st3.vlm_smoke_review import (build_comparison, record_review, require_approval,
                                 SmokeReviewRequired, SystematicHallucination)

ROOT = Path(__file__).resolve().parents[1]


def smoke_fixture(tmp_path, monkeypatch):
    config = json.loads((ROOT/"configs/exp025.json").read_text(encoding="utf-8"))
    config["smoke_images"] = 2
    config["vlm"]["assets_sha256"] = {};config["vlm"]["assets_git_sha1"] = {}
    config["text_encoder"]["assets_sha256"] = {}
    quantized = copy.deepcopy(config);quantized["vlm"]["quantization"] = "nf4"
    images = pd.DataFrame([dict(painting=f"p{i}",image=f"p{i}.jpg",image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),aspect_ratio=1+i)
                           for i in range(2)])
    monkeypatch.setattr(vlm_semantics,"training_images",lambda *args:images)
    import st3.vlm_smoke_review as review
    monkeypatch.setattr(review,"training_images",lambda *args:images)
    from PIL import Image
    folder = tmp_path/"artwork/images";folder.mkdir(parents=True)
    for name in images.image:Image.new("RGB",(4,4),"white").save(folder/name)
    roots = [tmp_path/"nf4",tmp_path/"fp16"]
    caches = [SemanticCache(roots[0],quantized),SemanticCache(roots[1],config)]
    vector = np.ones(768,dtype=np.float32)/np.sqrt(np.float32(768))
    for index,cache in enumerate(caches):
        for row in images.itertuples():
            cache.save(row.image_sha256,{"description":"A Japanese-looking room with uncertain details.","generation_hit_limit":False,
                       "embedding":vector.tolist(),"clip_tokens":8,"tokens_encoded":8,"chunks":1,"GPU":"FAKE_L4",
                       "seconds":2.-index,"generation_run_id":"fixture"})
        (cache.directory/"telemetry_fixture.json").write_text(json.dumps({"peak_allocated":100+index,"peak_reserved":200+index,"generated":2}))
    return config,quantized,images,roots,caches


def test_semantic_warnings_allow_smoke_cache_reuse_without_vlm(tmp_path,monkeypatch):
    config,_,images,roots,caches = smoke_fixture(tmp_path,monkeypatch)
    import torch
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    for name in ("reset_peak_memory_stats","empty_cache"):monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,"mem_get_info",lambda:(1000,2000))
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda:"FAKE_L4")
    for name in ("max_memory_allocated","max_memory_reserved"):monkeypatch.setattr(torch.cuda,name,lambda:0)
    monkeypatch.setattr(vlm_semantics,"load_vlm",lambda *args:pytest.fail("Cached descriptions must not regenerate"))
    monkeypatch.setitem(sys.modules,"transformers",None)
    before={p.name:p.read_bytes() for p in caches[1].directory.glob("*.json") if len(p.stem)==64}
    report=extract(tmp_path,roots[1],config,tmp_path,tmp_path)
    assert report["technical_status"]=="PASS" and report["generated"]==0
    assert len(report["semantic_quality_warnings"])==2
    assert all(w["description"] and w["image"] and w["painting"] for w in report["semantic_quality_warnings"])
    assert before=={p.name:p.read_bytes() for p in caches[1].directory.glob("*.json") if len(p.stem)==64}


def test_single_review_allows_isolated_errors_but_rejection_stops(tmp_path,monkeypatch):
    config,nf4,images,roots,caches=smoke_fixture(tmp_path,monkeypatch)
    report=build_comparison(tmp_path,*roots,nf4,config,tmp_path/"comparison")
    assert report["same_images_and_prompt"] and report["NF4_over_FP16_latency_ratio"]==2.
    assert report["NF4"]["peak_allocated_bytes"]==100
    assert len(report["semantic_quality_warnings"]["FP16"])==2
    comparison=tmp_path/"comparison/smoke_comparison.json"
    approved=tmp_path/"approved.json"
    with pytest.raises(SmokeReviewRequired):require_approval(caches[1],images,None)
    record_review(comparison,"BROADLY_USABLE",approved)
    assert require_approval(caches[1],images,approved)["decision"]=="BROADLY_USABLE"
    rejected=tmp_path/"rejected.json"
    record_review(comparison,"SYSTEMATIC_SCENE_HALLUCINATION",rejected)
    with pytest.raises(SystematicHallucination):require_approval(caches[1],images,rejected)
    record=caches[1].read(images.iloc[0].image_sha256)
    caches[1].save(images.iloc[0].image_sha256,{**record,"description":"Changed scene."})
    with pytest.raises(SmokeReviewRequired):require_approval(caches[1],images,approved)


def test_token_loss_remains_a_hard_failure(tmp_path,monkeypatch):
    _,_,images,_,caches=smoke_fixture(tmp_path,monkeypatch)
    record=caches[1].read(images.iloc[0].image_sha256)
    record["tokens_encoded"]-=1
    with pytest.raises(ValueError,match="token coverage"):caches[1].save(images.iloc[0].image_sha256,record)


def test_all_five_folds_run_even_with_zero_fusion_gain_and_resume(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("exp025_completion_runner",ROOT/"scripts/run_exp025.py")
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    rows=[]
    for i in range(125):
        for language,k in zip(("A","B","C"),(1,3,4)):
            rows.append(dict(painting=f"p{i}",language=language,image=f"p{i}.jpg",n=5,k=k,fold=i%5+1))
    train=pd.DataFrame(rows)
    monkeypatch.setattr(runner,"load_inputs",lambda *args:(train,None,{}, {}, {}))
    called=[]
    def fit(train,*args):
        fold=args[-1];called.append(fold)
        base=train[train.fold.eq(fold)][["painting","language","fold"]].assign(p_positive=.5)
        models=("A","B","C","D","C_shuffle","D_shuffle","E50+L")
        return pd.concat([base.assign(model=name) for name in models],ignore_index=True),{}
    monkeypatch.setattr(runner,"evaluate_fold",fit)
    config=json.loads((ROOT/"configs/exp025.json").read_text(encoding="utf-8"));config["bootstrap_replicates"]=20
    runner.run(tmp_path,tmp_path,tmp_path/"run",config)
    assert called==[1,2,3,4,5]
    done=json.loads((tmp_path/"run/completed.json").read_text())
    assert done["status"]=="FIVE_FOLD_COMPLETE" and done["folds"]==[1,2,3,4,5]
    assert (tmp_path/"run/contribution_evidence.json").is_file()
    called.clear();runner.run(tmp_path,tmp_path,tmp_path/"run",config)
    assert not called
