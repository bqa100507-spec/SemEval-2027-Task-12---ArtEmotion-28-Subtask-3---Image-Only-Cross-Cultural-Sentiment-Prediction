"""Small, offline EL8 tests. No dataset, weights download or training jobs."""

import copy
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from st3 import submission
from st3.baseline import file_sha256
from st3.el8_artifacts import (SEEDS, compare_predictions, export_bundle, import_bundle,
                               safe_relative, seed_mean, strict_payload, verify_archive)
from st3.el8_inference import final_bundle, predict_cached_head, task_head
from st3.el8_refit import require_authorization
from st3.fusion import blend
from st3.visual_groups import fusion
from st3.visual_lora_inputs import config022
from st3.vote_model import VoteStudent


def probabilities(values=(.1234564, .8765434)):
    return pd.DataFrame({'painting': ['a', 'a'], 'language': ['en', 'vi'], 'p_positive': values})


def test_three_seed_probability_mean_then_nested_rounding():
    frames = [probabilities() for _ in SEEDS]
    mean = seed_mean(frames)
    np.testing.assert_array_equal(mean.p_positive, [.123456, .876543])
    out = blend(probabilities((.6, .4)), mean, .5)
    np.testing.assert_array_equal(out.p_positive, np.round([(.6+.123456)/2, (.4+.876543)/2], 6))
    assert compare_predictions(out, out)['status'] == 'PASS'
    altered = out.copy(); altered.loc[0, 'p_positive'] += 1e-12
    assert compare_predictions(out, altered)['status'] == 'FAIL'


@pytest.mark.parametrize('mutation', ['missing_seed', 'order', 'nan', 'duplicate', 'range'])
def test_seed_mean_rejects_invalid_coverage(mutation):
    frames = [probabilities() for _ in SEEDS]
    if mutation == 'missing_seed':frames.pop()
    if mutation == 'order':frames[1] = frames[1].iloc[::-1].reset_index(drop=True)
    if mutation == 'nan':frames[1].loc[0, 'p_positive'] = np.nan
    if mutation == 'range':frames[1].loc[0, 'p_positive'] = 2.
    if mutation == 'duplicate':frames[1] = pd.concat([frames[1], frames[1].iloc[:1]])
    with pytest.raises(ValueError):seed_mean(frames)


def payload():
    cfg = config022(); head = VoteStudent(['en', 'vi'], cfg['training'])
    adapters = {f'model.vision_model.encoder.layers.{i}.self_attn.{q}.{m}':
                torch.zeros((8, 768) if m == 'a' else (768, 8))
                for i in range(12) for q in ('q_proj', 'v_proj') for m in ('a', 'b')}
    return {'languages': head.languages, 'head': head.state_dict(), 'adapters': adapters}


def test_strict_checkpoint_and_rng_preservation(tmp_path):
    p = tmp_path/'student.pt'; value = payload(); torch.save(value, p)
    rng = torch.get_rng_state().clone(); strict_payload(p)
    assert torch.equal(rng, torch.get_rng_state())
    value['adapters'].pop(next(iter(value['adapters']))); torch.save(value, p)
    with pytest.raises(ValueError, match='24 visual'):strict_payload(p)


@pytest.mark.parametrize('part', ['head', 'adapters'])
def test_nonfinite_checkpoint_rejected(tmp_path, part):
    value = payload(); next(iter(value[part].values())).view(-1)[0] = float('nan')
    path = tmp_path/'student.pt'; torch.save(value, path)
    with pytest.raises(ValueError, match='Invalid'):strict_payload(path)


def test_cv_bundle_cannot_be_loaded_as_final(tmp_path):
    (tmp_path/'final_manifest.json').write_text(json.dumps({'kind':'historical_cv','status':'VERIFIED','seeds':list(SEEDS),'condition':'L8'}))
    with pytest.raises(ValueError, match='CV checkpoints'):final_bundle(tmp_path)


def test_cached_inference_uses_image_features_and_shared_language_fallback():
    torch.manual_seed(2027); cfg = config022(); value = payload(); head = task_head(value, cfg, torch.device('cpu'))
    frame = pd.DataFrame({'painting':['a','a','b','b'], 'language':['en','new','en','new'], 'image':['first','first','second','second']})
    rng = np.random.default_rng(1)
    def bank(width):
        x = rng.normal(size=(2,width)).astype(np.float32); x /= np.linalg.norm(x,axis=1)[:,None]
        return dict(zip(['first','second'],x))
    adapted = bank(512); frozen = [bank(768),bank(768)]
    pred = predict_cached_head(frame, head, adapted, frozen, torch.device('cpu'), 4)
    assert list(pred.columns) == submission.COLUMNS and len(pred) == 4
    assert pred.groupby('language').p_positive.std().gt(0).all()
    again = predict_cached_head(frame, head, adapted, frozen, torch.device('cpu'), 4)
    assert compare_predictions(pred,again)['status']=='PASS'
    # Lookup identifiers are not predictive features.
    aliases = frame.assign(painting=frame.painting.map({'a':'x','b':'y'}), image=frame.image.map({'first':'alias1','second':'alias2'}))
    renamed = lambda b:dict(zip(['alias1','alias2'],b.values()))
    other = predict_cached_head(aliases, head, renamed(adapted),[renamed(b) for b in frozen],torch.device('cpu'),4)
    np.testing.assert_array_equal(pred.p_positive,other.p_positive)


def test_fusion_preserves_original_blocks():
    a,b,c = [torch.ones(2,w)/np.sqrt(w) for w in (512,768,768)]
    z = fusion(a,b,c)
    assert z.shape==(2,2048)
    torch.testing.assert_close(z,torch.cat([a,b,c],dim=1)/np.sqrt(3),atol=0,rtol=0)


def test_archive_round_trip_corruption_kind_and_conflicting_restore(tmp_path):
    root = tmp_path/'run';root.mkdir();(root/'weights.pt').write_bytes(b'small fixture')
    path = tmp_path/'all.zip'; receipt=export_bundle(path,{'cv':root},{'cv':'historical_cv'},{'source':'frozen'})
    expected={'cv':'historical_cv'}
    verify_archive(path,receipt['sha256'],expected)
    with pytest.raises(ValueError,match='kind'):verify_archive(path,expected_kinds={'cv':'final_refit'})
    restored=tmp_path/'restored';import_bundle(path,restored,receipt['sha256'],expected)
    import_bundle(path,restored,receipt['sha256'],expected)
    (restored/'cv/weights.pt').write_bytes(b'valid different data')
    with pytest.raises(ValueError,match='overwrite'):import_bundle(path,restored,receipt['sha256'])
    with pytest.raises(ValueError,match='SHA-256'):verify_archive(path,'0'*64)
    with zipfile.ZipFile(path) as z:manifest=json.loads(z.read('artifact_manifest.json'))
    broken=tmp_path/'broken.zip'
    with zipfile.ZipFile(broken,'w') as z:
        z.writestr('artifact_manifest.json',json.dumps(manifest));z.writestr('cv/weights.pt',b'corrupt')
    with pytest.raises(ValueError,match='checksum'):verify_archive(broken)
    with zipfile.ZipFile(broken,'w') as z:z.writestr('artifact_manifest.json',json.dumps(manifest))
    with pytest.raises(ValueError,match='Incomplete'):verify_archive(broken)


@pytest.mark.parametrize('name',['../escape','/abs','x\\y','C:/x','a/../b','a//b'])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError,match='Unsafe'):safe_relative(name)


def test_final_refit_requires_explicit_policy_and_budget():
    from st3.cli import ROOT
    from st3.polarity_workflow import StageBlocked
    args=SimpleNamespace(authorize_final_refit=False,policy_sha256=file_sha256(ROOT/'configs/el8_final_refit.json'),max_gpu_hours=1.)
    with pytest.raises(StageBlocked,match='No final training'):require_authorization(args)
    args.authorize_final_refit=True;args.max_gpu_hours=0
    with pytest.raises(StageBlocked):require_authorization(args)
    args.max_gpu_hours=1.;assert require_authorization(args)[0]['inner_partition_fold_argument']==0


def test_complete_cv_export_includes_all_sixty_models_and_latest_attempt(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('notebook_export',root/'scripts/exp022_notebook.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    run=tmp_path/'run';identity={'science':'original'}
    def seal(path):
        path.mkdir(parents=True);(path/'value').write_bytes(b'fixture')
        helper.write_json(path/'run.json',{'identity':identity,'metadata':{},'files_sha256':{'value':helper.sha256(path/'value')}})
    for stage in ('cv','cv_verification','cv_decision'):seal(run/stage)
    for fold in range(1,6):
        for seed in SEEDS:
            for condition in ('F0','F8','L0','L8'):seal(run/'work'/f'fold_{fold}'/f'{condition}_{seed}')
    attempts=run/'cv_verification_attempts';attempts.mkdir();(attempts/'00000002.json').write_text('{"status":"FAILED"}')
    archive=tmp_path/'all.zip';receipt=helper.export_checkpoints(run,archive,identity,{'runtime':'fixture'})
    with zipfile.ZipFile(archive) as z:
        assert sum(n.endswith('/run.json') and '/work/' in n for n in z.namelist())==60
        assert all('run/'+n+'/run.json' in z.namelist() for n in ('cv','cv_verification','cv_decision'))
        assert 'run/cv_verification_attempts/00000002.json' in z.namelist()
    helper.restore_checkpoints(archive,receipt['sha256'],tmp_path/'restored',identity,{'runtime':'fixture'})


def test_one_run_refit_resumes_completed_models_and_exports_on_failure(tmp_path, monkeypatch):
    """Orchestration integration only: every encoder, fit and GPU call is fake."""
    import st3.el8_refit as workflow
    from st3.cli import json_output
    from st3.el8_artifacts import read_json
    from st3.polarity_workflow import seal
    cfg=config022();train=pd.DataFrame([{'painting':f'p{i:04}', 'language':lang, 'image':f'{i}.jpg'}
        for i in range(1335) for lang in ('English','Vietnamese')])
    run=tmp_path/'output';cv=tmp_path/'cv';cv.mkdir();(cv/'evidence').write_text('CV preserved')
    (cv/'cv').mkdir();(cv/'cv/run.json').write_text('{}')
    args=SimpleNamespace(run_dir=run,cv_run=cv,portable_dir=tmp_path/'portable',model_dir=tmp_path/'e50',
        clip_large_model_dir=tmp_path/'large',data_dir=tmp_path/'data',cache_dir=tmp_path/'cache',device='cuda',max_gpu_hours=1.)
    for p in (args.model_dir,args.clip_large_model_dir):p.mkdir();(p/'heads.npz').write_bytes(b'heads')
    ctx=SimpleNamespace(train=train,config=cfg,identity={'new':'final'},device=torch.device('cpu'))
    evidence={'identity':{'source_sha256':{}},'kind':'historical_cv'}
    monkeypatch.setattr(workflow,'require_authorization',lambda a:({'microbatch':4},'policy'))
    monkeypatch.setattr(workflow,'cv_inventory',lambda p:evidence)
    monkeypatch.setattr(workflow,'scientific_sources_unchanged',lambda x:None)
    monkeypatch.setattr(workflow,'load_bundle',lambda *a:({}, {}, {}))
    monkeypatch.setattr(workflow,'load_clip_large_bundle',lambda *a:None)
    monkeypatch.setattr(workflow,'deterministic_runtime',lambda a:torch.device('cuda'))
    monkeypatch.setattr(workflow,'runtime',lambda a:{'gpu':'synthetic'})
    monkeypatch.setattr(workflow,'forecast',lambda a:{'synthetic':True})
    monkeypatch.setattr(workflow,'replay_cv',lambda *a:json_output({'status':'PASS','runtime':{'gpu':'synthetic'},
        'implementation':workflow.implementation_identity(),'cv_manifest_sha256':file_sha256(cv/'cv/run.json')},a[4]))
    monkeypatch.setattr(workflow,'run_preflight',lambda a:(run/'preflight').mkdir(exist_ok=True))
    monkeypatch.setattr(workflow,'run_pilot',lambda a:None)
    monkeypatch.setattr(workflow,'load_context',lambda a:(ctx,object()))
    original_read=workflow.read_sealed
    monkeypatch.setattr(workflow,'read_sealed',lambda p,i:{'status':'PASS','microbatch':4} if p.name=='preflight' else original_read(p,i))
    class Vision:
        def __init__(self,*a):pass
        def requires_grad_(self,*a):return self
        def eval(self):return self
        def frozen_digest(self):return 'base'
    monkeypatch.setattr(workflow,'VisualLoRA',Vision)
    fits=[];fail={'seed':2028}
    def fit(ctx,vision,train,held,fold,seed,*a):
        fits.append(seed)
        if seed==fail['seed']:raise RuntimeError('synthetic interruption')
        return object(),[],{}, {'held_paintings':[], 'fit_paintings':sorted(train.painting.unique()),
                               'base_sha256':'base','adapter_update_norm':1.}
    monkeypatch.setattr(workflow,'fit_model',fit)
    def predicted(ctx,vision,head,frame,*a):
        result=frame[['painting','language']].copy();result['p_positive']=np.arange(len(result))/max(1,len(result))
        return result,{}
    monkeypatch.setattr(workflow,'predict',predicted)
    monkeypatch.setattr(workflow,'save_checkpoint',lambda p,*a:p.write_bytes(b'synthetic checkpoint'))
    monkeypatch.setattr(workflow,'strict_payload',lambda p:{'adapters':{}})
    monkeypatch.setattr(workflow,'load_adapter',lambda *a:None)
    monkeypatch.setattr(workflow,'task_head',lambda *a:object())
    monkeypatch.setattr(workflow,'final_bundle',lambda *a:{'status':'VERIFIED'})
    with pytest.raises(RuntimeError,match='interruption'):workflow.full_refit(args)
    assert (run/'final/L8_2027/student.pt').exists() and not (run/'final/L8_2028').exists()
    assert Path(read_json(run/'last_export.json')['path']).is_file()
    fail['seed']=None
    status=workflow.full_refit(args)
    assert status['status']=='FINAL_BUNDLE_VERIFIED'
    assert fits==[2027,2028,2028,2029]  # complete seed was reused, not retrained
    assert len(list((run/'final').glob('*/student.pt')))==3
