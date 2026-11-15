"""Cloud packaging and authorization contracts, with all GPU work mocked."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('el8_cloud',ROOT/'scripts/el8_cloud.py')
cloud=importlib.util.module_from_spec(spec);spec.loader.exec_module(cloud)


def test_cloud_extract_checks_hash_before_extracting_and_reuses(tmp_path):
    raw=b'fixture';manifest={'format':'st3-el8-artifacts-v1','metadata':{'checkpoints':60},'kinds':{'historical_cv':'historical_cv'},
        'files_sha256':{'historical_cv/file':hashlib.sha256(raw).hexdigest()}}
    archive=tmp_path/'handoff.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('artifact_manifest.json',json.dumps(manifest));z.writestr('historical_cv/file',raw)
    destination=tmp_path/'local'
    with pytest.raises(ValueError,match='SHA-256'):cloud.extract_handoff(archive,'0'*64,destination)
    assert not destination.exists()
    for _ in range(2):cloud.extract_handoff(archive,cloud.sha256(archive),destination)
    (destination/'historical_cv/file').write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs'):cloud.extract_handoff(archive,cloud.sha256(archive),destination)


def test_notebook_templates_compile_and_default_training_is_off():
    import ast
    source=(ROOT/'scripts/build_el8_notebooks.py').read_text(encoding='utf-8')
    ast.parse(source)
    # Actual generated notebooks are checked when present (clean checkout CI).
    for path in (ROOT/'notebooks').glob('EL8_FinalRefit_*.ipynb'):
        notebook=json.loads(path.read_text(encoding='utf-8'));assert notebook['metadata']['el8']['default_training'] is False
        for c in notebook['cells']:
            if c['cell_type']=='code':
                assert not c['outputs'] and c['execution_count'] is None
                ast.parse(''.join(c['source']))
        config={};exec(''.join(notebook['cells'][1]['source']),config)
        assert config['AUTHORIZE_FINAL_REFIT'] is False and config['MAX_GPU_HOURS']==1.5 and config['MAX_TOTAL_WALL_HOURS']==2.0


def test_cloud_defaults_to_no_training():
    import inspect
    sig=inspect.signature(cloud.launch)
    assert sig.parameters['authorize'].default is False
    assert sig.parameters['max_gpu_hours'].default==0


@pytest.mark.parametrize('platform',['Colab','Kaggle'])
def test_notebook_run_all_imports_paths_and_approval_are_safe(tmp_path,monkeypatch,platform):
    import ast,sys,types
    path=ROOT/'notebooks'/f'EL8_FinalRefit_{platform}.ipynb'
    if not path.exists():pytest.skip('Notebook generated after the pinned implementation commit')
    notebook=json.loads(path.read_text());ns={};exec(''.join(notebook['cells'][1]['source']),ns)
    helper=b'def launch(*args, **kwargs):\n    return {"status":"MOCK_NO_TRAINING","authorized":kwargs["authorize"]}\n'
    local=tmp_path/'local';repo=local/'deployment';(repo/'scripts').mkdir(parents=True)
    (repo/'scripts/el8_cloud.py').write_bytes(helper)
    bundle=b'fake Git bundle; clone is mocked'
    manifest={'metadata':{'deployment_revision':ns['IMPLEMENTATION_REVISION']},
              'files_sha256':{'source/source.bundle':hashlib.sha256(bundle).hexdigest()}}
    archive=tmp_path/'handoff.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('artifact_manifest.json',json.dumps(manifest));z.writestr('source/source.bundle',bundle)
    ns.update(HANDOFF_ZIP=str(archive),HANDOFF_SHA256=cloud.sha256(archive),HELPER_SHA256=hashlib.sha256(helper).hexdigest(),
              LOCAL_ROOT=str(local),PERSISTENT_ROOT=str(tmp_path/'persistent'),AUTHORIZE_FINAL_REFIT=True)
    monkeypatch.setattr(cloud.subprocess,'check_output',lambda *a,**k:ns['IMPLEMENTATION_REVISION']+'\n')
    google=types.ModuleType('google');colab=types.ModuleType('google.colab');mounts=[]
    colab.drive=types.SimpleNamespace(mount=lambda path:mounts.append(path))
    monkeypatch.setitem(sys.modules,'google',google);monkeypatch.setitem(sys.modules,'google.colab',colab)
    exec(''.join(notebook['cells'][2]['source']),ns)
    assert ns['RESULT']=={'status':'MOCK_NO_TRAINING','authorized':True}
    assert mounts==(['/content/drive'] if platform=='Colab' else [])


def test_no_authorization_never_installs_or_trains_and_still_releases(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(cloud,'extract_handoff',lambda *a:pytest.fail('Approval must precede extraction/setup'))
    monkeypatch.setattr(cloud,'persist_export',lambda *a:{'path':'fixture'})
    monkeypatch.setattr(cloud,'release_runtime',lambda *a:calls.append('release'))
    result=cloud.launch(tmp_path/'unused.zip','receipt',tmp_path/'local',tmp_path/'persistent')
    assert result['status']=='BLOCKED_AUTHORIZATION' and calls==['release']


@pytest.mark.parametrize('platform',['Colab','Kaggle'])
def test_notebook_bootstrap_failure_releases_colab_in_finally(tmp_path,monkeypatch,platform):
    import sys,types
    nb=json.loads((ROOT/'notebooks'/f'EL8_FinalRefit_{platform}.ipynb').read_text())
    ns={};exec(''.join(nb['cells'][1]['source']),ns)
    ns.update(AUTHORIZE_FINAL_REFIT=True,HANDOFF_ZIP=str(tmp_path/'missing.zip'),
              PERSISTENT_ROOT=str(tmp_path/'persistent'),LOCAL_ROOT=str(tmp_path/'local'))
    colab=types.ModuleType('google.colab');calls=[]
    colab.drive=types.SimpleNamespace(mount=lambda p:None)
    colab.runtime=types.SimpleNamespace(unassign=lambda:calls.append('release'))
    monkeypatch.setitem(sys.modules,'google',types.ModuleType('google'))
    monkeypatch.setitem(sys.modules,'google.colab',colab)
    exec(''.join(nb['cells'][2]['source']),ns)
    assert ns['RESULT']['status']=='FAILED_SETUP'
    assert calls==(['release'] if platform=='Colab' else [])
