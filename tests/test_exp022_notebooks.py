"""Offline synthetic workflow checks; never launch cloud or ML training."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('exp022_notebook',ROOT/'scripts/exp022_notebook.py')
workflow=importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)


def test_platform_paths_missing_and_ambiguous_inputs(tmp_path):
    inputs=tmp_path/'input';inputs.mkdir()
    with pytest.raises(workflow.WorkflowError,match='Found 0'):workflow.locate_zip(inputs)
    first=inputs/'a';first.mkdir();(first/'exp022_colab_inputs.zip').write_bytes(b'fixture')
    assert workflow.locate_zip(inputs).parent==first
    other=inputs/'b';other.mkdir();(other/'exp022_colab_inputs.zip').write_bytes(b'fixture')
    with pytest.raises(workflow.WorkflowError,match='Found 2'):workflow.locate_zip(inputs)
    for platform in ('Colab','Kaggle'):
        paths=workflow.paths_for(platform,input_path=str(first/'exp022_colab_inputs.zip'),local_root=tmp_path/platform,
            persistent_root=tmp_path/(platform+'-persistent'),input_root=inputs)
        assert paths['local']!=paths['persistent'] and paths['archive'].is_file()


def tree_zip(tmp_path,entries):
    archive=tmp_path/'fixture.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name,value in entries.items():z.writestr(name,value)
    return archive


def test_zip_checksum_paths_and_atomic_extraction(tmp_path):
    content=b'exact';manifest={'files_sha256':{'file.txt':hashlib.sha256(content).hexdigest()}}
    archive=tree_zip(tmp_path,{'file.txt':content,'runtime_bundle.json':json.dumps(manifest)})
    dest=tmp_path/'extracted'
    with pytest.raises(workflow.WorkflowError,match='SHA-256 mismatch'):workflow.extract_checked(archive,dest,'0'*64)
    assert not dest.exists()
    workflow.extract_checked(archive,dest,workflow.sha256(archive));workflow.extract_checked(archive,dest,workflow.sha256(archive))
    assert (dest/'file.txt').read_bytes()==content
    (dest/'extra').write_text('unexpected')
    with pytest.raises(workflow.WorkflowError,match='Missing or extra'):workflow.extract_checked(archive,dest,workflow.sha256(archive))
    unsafe=tree_zip(tmp_path,{'../escape':b'x'})
    with zipfile.ZipFile(unsafe) as z,pytest.raises(workflow.WorkflowError,match='Unsafe'):workflow.safe_entries(z)
    with pytest.warns(UserWarning,match='Duplicate'),zipfile.ZipFile(unsafe,'w') as z:z.writestr('same',b'a');z.writestr('same',b'b')
    with zipfile.ZipFile(unsafe) as z,pytest.raises(workflow.WorkflowError,match='duplicate'):workflow.safe_entries(z)


def test_input_manifest_is_checked_before_any_stage(tmp_path,monkeypatch):
    content=b'bad-input-manifest';manifest={'files_sha256':{'inputs/run.json':hashlib.sha256(content).hexdigest()},'paintings':1335,'locked_images_included':False}
    archive=tree_zip(tmp_path,{'inputs/run.json':content,'bundle.json':json.dumps(manifest)})
    # Synthetic bundle hash accepted solely to reach the independent input pin.
    monkeypatch.setattr(workflow,'BUNDLE_SHA',hashlib.sha256(json.dumps(manifest).encode()).hexdigest())
    with pytest.raises(workflow.WorkflowError,match='Published input manifest SHA-256 mismatch'):
        workflow.extract_checked(archive,tmp_path/'extracted',workflow.sha256(archive),input_bundle=True)


@pytest.mark.parametrize('mutation,message',[
    ({'python':'3.11.9'},'Python'),({'cuda_available':False},'CUDA is unavailable'),
    ({'versions':{'torch':'2.8.0','transformers':'4.49.0'}},'torch='),({'cuda':'12.8'},'CUDA runtime')])
def test_environment_mismatches_stop(mutation,message):
    actual={'python':'3.12.10','versions':{'torch':'2.7.1+cu126','transformers':'4.49.0'},'cuda':'12.6','cuda_available':True}
    actual.update(mutation)
    with pytest.raises(workflow.WorkflowError,match=message):workflow.validate_environment(actual,{'torch':'2.7.1+cu126','transformers':'4.49.0'})


def session(tmp_path,platform='Colab'):
    paths={'local':tmp_path/'local','persistent':tmp_path/'persistent','archive':tmp_path/'inputs.zip'}
    paths['local'].mkdir();paths['persistent'].mkdir()
    result=workflow.NotebookRun(platform,paths);result.run_dir=paths['persistent']/'identity';result.run_dir.mkdir()
    result.identity={'source':'frozen','gpu':'fixture'};result.profile={'driver':'fixture'}
    return result


def sealed(folder,identity,metadata=None):
    folder.mkdir(parents=True,exist_ok=True);(folder/'result.txt').write_text('fixture')
    workflow.write_json(folder/'run.json',{'identity':identity,'metadata':metadata or {},'files_sha256':{'result.txt':workflow.sha256(folder/'result.txt')}})


def test_default_benchmark_calls_only_preflight_and_pilot(tmp_path,monkeypatch):
    run=session(tmp_path);called=[]
    def stage(name,seconds):
        called.append((name,seconds));sealed(run.run_dir/name.lower(),run.identity,{'status':'PASS' if name=='Preflight' else 'PILOT_PASS'})
    monkeypatch.setattr(run,'stage',stage);monkeypatch.setattr(run,'summary',lambda:{'fixture':True})
    run.benchmark();assert called==[('Preflight',600),('Pilot',900)] and run.benchmark_ok
    for name in workflow.LONG_STAGES:
        with pytest.raises(workflow.WorkflowError,match='disabled'):workflow.NotebookRun.stage(run,name,900)
    with pytest.raises(workflow.WorkflowError,match='full CV'):workflow.stage_command('python','repo','inputs','run','CV',900)


def test_failed_preflight_never_runs_pilot(tmp_path,monkeypatch):
    run=session(tmp_path);calls=[]
    def stage(name,seconds):calls.append(name);raise workflow.WorkflowError('parity failed')
    monkeypatch.setattr(run,'stage',stage)
    with pytest.raises(workflow.WorkflowError,match='parity failed'):run.benchmark()
    assert calls==['Preflight'] and not run.benchmark_ok


def test_stage_parameters_are_equivalent_across_platforms(tmp_path,monkeypatch):
    commands=[]
    monkeypatch.setattr(workflow,'run_logged',lambda command,*a,**kw:commands.append(command))
    for platform in ('Colab','Kaggle'):
        directory=tmp_path/platform;directory.mkdir();run=session(directory,platform)
        monkeypatch.setattr(run,'export',lambda:None)
        run.stage('Preflight',600);run.stage('Pilot',900)
    for a,b in zip(commands[:2],commands[2:]):
        assert a[2:4]==b[2:4]
        for flag in ('--input-manifest-sha256','--device','--max-seconds'):
            assert a[a.index(flag)+1]==b[b.index(flag)+1]
        assert '--allow-full-cv' not in a+b


def test_checkpoint_export_restore_preserves_failed_attempts_and_rejects_mixing(tmp_path):
    run=session(tmp_path);sealed(run.run_dir/'preflight',run.identity,{'status':'PASS'})
    sealed(run.run_dir/'work/fold_1/L0_2027',run.identity)
    partial=run.run_dir/'work/fold_1/tmp-unpublished/output';partial.mkdir(parents=True);(partial/'student.pt').write_bytes(b'incomplete')
    attempts=run.run_dir/'mini_verification_attempts';attempts.mkdir();workflow.write_json(attempts/'00000002.json',{'status':'FAILED'})
    archive=tmp_path/'checkpoints.zip';result=workflow.export_checkpoints(run.run_dir,archive,run.identity,run.profile)
    with zipfile.ZipFile(archive) as z:assert not any('tmp-unpublished' in s for s in z.namelist())
    restored=tmp_path/'restored';workflow.restore_checkpoints(archive,result['sha256'],restored,run.identity,run.profile)
    workflow.restore_checkpoints(archive,result['sha256'],restored,run.identity,run.profile)
    assert workflow.read_json(restored/'mini_verification_attempts/00000002.json')['status']=='FAILED'
    assert workflow.verify_sealed(restored/'work/fold_1/L0_2027',run.identity)=={}
    with pytest.raises(workflow.WorkflowError,match='identity differs'):
        workflow.restore_checkpoints(archive,result['sha256'],tmp_path/'other',{'source':'different'},run.profile)
    (restored/'preflight/result.txt').write_text('different valid run must survive')
    with pytest.raises(workflow.WorkflowError,match='overwrite'):
        workflow.restore_checkpoints(archive,result['sha256'],restored,run.identity,run.profile)
    assert (restored/'preflight/result.txt').read_text()=='different valid run must survive'


def test_child_runner_logs_first_actionable_error(tmp_path):
    with pytest.raises(workflow.WorkflowError,match='parity blocked'):
        workflow.run_logged([sys.executable,'-c',"import sys;print('Traceback detail');print('VALIDATION_FAILURE: parity blocked');sys.exit(1)"],tmp_path,'fixture')
    assert 'Traceback detail' in next(tmp_path.glob('*.log')).read_text()
    assert workflow.read_json(next(tmp_path.glob('*.json')))['exit_code']==1


def test_offline_setup_uses_attached_source_python_and_wheels_only(tmp_path,monkeypatch):
    original=subprocess.check_output
    source=tmp_path/'original';source.mkdir()
    subprocess.run(['git','init',str(source)],check=True,capture_output=True)
    (source/'requirements.txt').write_text('transformers==4.49.0\n')
    subprocess.run(['git','-C',str(source),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-m','fixture'],check=True,capture_output=True)
    revision=original(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    bundle=tmp_path/'source.bundle';subprocess.run(['git','-C',str(source),'bundle','create',str(bundle),'HEAD'],check=True,capture_output=True)
    monkeypatch.setattr(workflow,'SOURCE_REV',revision)
    session_root=tmp_path/'session';session_root.mkdir();run=session(session_root,'Kaggle')
    payload={'source.bundle':bundle.read_bytes(),'environment.txt':b'torch==2.7.1+cu126\ntransformers==4.49.0\n',
             'python/bin/python3.12':b'synthetic executable; never run','wheels/fixture.whl':b'synthetic wheel; never installed'}
    payload['runtime_bundle.json']=json.dumps({'files_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in payload.items()},'source_revision':revision,'python':'3.12.10'}).encode()
    kit=tmp_path/'runtime.zip'
    with zipfile.ZipFile(kit,'w') as z:
        for n,v in payload.items():z.writestr(n,v)
    run.offline=kit
    inputs=b'fixture';data_manifest={'files_sha256':{'inputs/run.json':hashlib.sha256(inputs).hexdigest()},'paintings':1335,'locked_images_included':False}
    archive=tmp_path/'data.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('inputs/run.json',inputs);z.writestr('bundle.json',json.dumps(data_manifest))
    monkeypatch.setattr(workflow,'ZIP_SHA',workflow.sha256(archive));monkeypatch.setattr(workflow,'INPUT_SHA',hashlib.sha256(inputs).hexdigest())
    monkeypatch.setattr(workflow,'BUNDLE_SHA',hashlib.sha256(json.dumps(data_manifest).encode()).hexdigest())
    run.paths['archive']=archive;commands=[]
    def command(args,label,**kwargs):
        commands.append([str(v) for v in args])
        if str(args[0])=='git':subprocess.run([str(v) for v in args],check=True,capture_output=True)
    monkeypatch.setattr(run,'command',command)
    def check_output(args,**kwargs):
        if str(args[0])=='git':return original(args,**kwargs)
        if str(args[0])=='nvidia-smi':return 'fixture-driver\n'
        if 'freeze' in args:return 'torch==2.7.1+cu126\ntransformers==4.49.0\n'
        pytest.fail('Unexpected process/network request: '+str(args))
    monkeypatch.setattr(subprocess,'check_output',check_output)
    environment={'python':'3.12.10','versions':{'torch':'2.7.1+cu126','transformers':'4.49.0'},'cuda_available':True,'cuda':'12.6',
                 'gpu':'SYNTHETIC','total_vram_bytes':8*1024**3,'free_vram_bytes':7*1024**3,'capability':[8,0],'cudnn':90000,'visible_devices':1}
    monkeypatch.setattr(run,'python_json',lambda code,label:environment if label=='environment' else {'fixture':'identity'})
    run.setup(offline_sha=workflow.sha256(kit))
    assert any('--no-index' in args and '--find-links' in args for args in commands)
    assert not any('https://' in token for args in commands for token in args)
    assert not any('uv' in str(args) for args in commands)
    assert run.env['CUDA_VISIBLE_DEVICES']=='0' and run.identity=={'fixture':'identity'}
