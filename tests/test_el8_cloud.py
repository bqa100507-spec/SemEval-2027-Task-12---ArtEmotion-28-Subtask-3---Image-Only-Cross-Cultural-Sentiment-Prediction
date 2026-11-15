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
        assert config['AUTHORIZE_FINAL_REFIT'] is False and config['MAX_GPU_HOURS']==0


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
              LOCAL_ROOT=str(local),PERSISTENT_ROOT=str(tmp_path/'persistent'))
    monkeypatch.setattr(cloud.subprocess,'check_output',lambda *a,**k:ns['IMPLEMENTATION_REVISION']+'\n')
    google=types.ModuleType('google');colab=types.ModuleType('google.colab');mounts=[]
    colab.drive=types.SimpleNamespace(mount=lambda path:mounts.append(path))
    monkeypatch.setitem(sys.modules,'google',google);monkeypatch.setitem(sys.modules,'google.colab',colab)
    exec(''.join(notebook['cells'][2]['source']),ns)
    assert ns['RESULT']=={'status':'MOCK_NO_TRAINING','authorized':False}
    assert mounts==(['/content/drive'] if platform=='Colab' else [])


@pytest.mark.parametrize('authorize',[False,True])
def test_cloud_stage_invocation_never_runs_cv_and_requires_authorization(tmp_path,monkeypatch,authorize):
    from types import SimpleNamespace
    local=tmp_path/'local';deploy=local/'deployment';deploy.mkdir(parents=True)
    persistent=tmp_path/'persistent';revision='a'*40
    monkeypatch.setattr(cloud,'extract_handoff',lambda *a:{'metadata':{'deployment_revision':revision,'policy_sha256':'policy'}})
    monkeypatch.setattr(cloud.subprocess,'check_output',lambda a,**k:revision if 'rev-parse' in a else '')
    calls=[]
    class Session:
        def __init__(self,platform,paths,**kw):
            self.python=tmp_path/'venv/bin/python';self.portable=local/'portable';self.env={}
            self.run_dir=persistent/'setup/runtime-profile'
        def setup(self):calls.append('setup')
    def logged(command,*a,**kw):
        stage=command[2];calls.append(stage)
        if '--output' in command:
            output=Path(command[command.index('--output')+1]);output.parent.mkdir(parents=True,exist_ok=True);output.write_text('{}')
        if stage=='final-refit':
            assert '--authorize-final-refit' in command and '--max-gpu-hours' in command
            run=Path(command[command.index('--run-dir')+1]);(run/'last_export.json').write_text('{"status":"SYNTHETIC_EXPORT"}')
    helper=SimpleNamespace(NotebookRun=Session,run_logged=logged)
    monkeypatch.setattr(cloud,'load_helper',lambda p:helper)
    result=cloud.launch(tmp_path/'unused.zip','receipt',local,persistent,authorize=authorize,max_gpu_hours=1.5 if authorize else 0.)
    assert calls==['setup','validate','forecast']+(['final-refit'] if authorize else [])
    assert result['status']==('SYNTHETIC_EXPORT' if authorize else 'PREPARED_NOT_AUTHORIZED')
