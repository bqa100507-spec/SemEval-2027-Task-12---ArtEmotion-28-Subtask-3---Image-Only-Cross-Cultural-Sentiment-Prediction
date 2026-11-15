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
