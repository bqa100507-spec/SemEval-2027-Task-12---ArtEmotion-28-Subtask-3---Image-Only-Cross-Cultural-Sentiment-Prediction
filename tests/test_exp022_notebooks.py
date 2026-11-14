"""Offline synthetic workflow checks; never launch cloud or ML training."""

import copy
import ast
import hashlib
import importlib.util
import json
import io
import subprocess
import sys
import types
import urllib.request
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


def test_bootstrap_hashing_does_not_require_python311_api(tmp_path,monkeypatch):
    value=tmp_path/'artifact';value.write_bytes(b'old notebook host; scientific Python stays 3.12.10')
    expected=hashlib.sha256(value.read_bytes()).hexdigest()
    monkeypatch.delattr(hashlib,'file_digest',raising=False)
    assert workflow.sha256(value)==expected
    ast.parse((ROOT/'scripts/exp022_notebook.py').read_text(encoding='utf-8'),feature_version=(3,10))


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


def test_ram_workflow_checks_preflight_and_pixel_parity_before_pilot(tmp_path,monkeypatch):
    run=session(tmp_path);run.pixel_cache='ram';calls=[]
    def stage(name,seconds):
        calls.append(name)
        if name=='PixelBenchmark':raise workflow.WorkflowError('pixel gradient parity failed')
        sealed(run.run_dir/'preflight',run.identity,{'status':'PASS'})
    monkeypatch.setattr(run,'stage',stage)
    with pytest.raises(workflow.WorkflowError,match='gradient parity'):run.benchmark()
    assert calls==['Preflight','PixelBenchmark'] and not run.benchmark_ok


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


@pytest.mark.parametrize('name',['Colab','Kaggle'])
def test_notebook_structure_pins_and_clean_outputs(name):
    notebook=json.loads((ROOT/'notebooks'/f'EXP022_{name}.ipynb').read_text(encoding='utf-8'))
    assert notebook['nbformat']==4 and notebook['nbformat_minor']==5
    assert len({c['id'] for c in notebook['cells']})==len(notebook['cells'])
    metadata=notebook['metadata']['exp022'];assert metadata['training_revision']==workflow.SOURCE_REV
    helper=subprocess.check_output(['git','show',metadata['helper_revision']+':scripts/exp022_notebook.py'],cwd=ROOT)
    assert hashlib.sha256(helper).hexdigest()==metadata['helper_sha256']
    assert metadata['default_stages']==['Preflight','PixelBenchmark','Pilot']
    for cell in notebook['cells']:
        assert isinstance(cell['source'],list)
        if cell['cell_type']=='code':
            assert cell['outputs']==[] and cell['execution_count'] is None
            ast.parse(''.join(cell['source']))
            ast.parse(''.join(cell['source']),feature_version=(3,10))
    combined='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    if name=='Kaggle':assert 'google.colab' not in combined


@pytest.mark.parametrize('name',['Colab','Kaggle'])
def test_run_all_bootstrap_imports_paths_and_benchmark_are_safe(name,tmp_path,monkeypatch):
    notebook=json.loads((ROOT/'notebooks'/f'EXP022_{name}.ipynb').read_text(encoding='utf-8'))
    cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    namespace={};exec(compile(cells[0],name,'exec'),namespace)
    archive=tmp_path/'exp022_colab_inputs.zip';archive.write_bytes(b'fixture; mocked setup never trains')
    namespace.update(INPUT_ZIP_PATH=str(archive),LOCAL_ROOT=str(tmp_path/'local'),PERSISTENT_ROOT=str(tmp_path/'output'))
    helper=subprocess.check_output(['git','show',notebook['metadata']['exp022']['helper_revision']+':scripts/exp022_notebook.py'],cwd=ROOT)
    urls=[]
    def urlopen(url,**kwargs):urls.append(url);return io.BytesIO(helper)
    monkeypatch.setattr(urllib.request,'urlopen',urlopen)
    displays=types.ModuleType('IPython.display');displays.display=lambda *a:None;displays.Markdown=lambda s:s;displays.FileLink=lambda s:s
    monkeypatch.setitem(sys.modules,'IPython.display',displays)
    google=types.ModuleType('google');colab=types.ModuleType('google.colab');mounts=[]
    colab.drive=types.SimpleNamespace(mount=lambda p:mounts.append(p))
    monkeypatch.setitem(sys.modules,'google',google);monkeypatch.setitem(sys.modules,'google.colab',colab)
    calls=[]
    class FakeRun:
        def __init__(self,platform,paths,**kwargs):calls.append(('platform',platform));self.last_export=None
        def setup(self,**kwargs):calls.append(('setup',kwargs))
        def benchmark(self):calls.append(('benchmark',None))
        def summary(self):return {'platform':name,'output':str(tmp_path),'preflight':'SYNTHETIC','pilot':'SYNTHETIC'}
        def failure(self,error):pytest.fail('Unexpected workflow failure: '+str(error))
        def stage(self,*a,**k):pytest.fail('Run all must never invoke an explicit long stage')
    for index,code in enumerate(cells[1:],1):
        if 'SESSION = None' in code:
            assert namespace['BOOTSTRAP_OK']
            namespace['RUNTIME'].NotebookRun=FakeRun
            namespace['RUNTIME'].show_controls=lambda session,ns:calls.append(('controls',ns['RUN_LONG_TRAINING']))
        exec(compile(code,f'{name}-{index}','exec'),namespace)
    assert namespace['BENCHMARK_OK'] and namespace['RUN_LONG_TRAINING'] is False
    assert [v[0] for v in calls]==['platform','setup','benchmark','controls']
    assert mounts==(['/content/drive'] if name=='Colab' else [])
    assert len(urls)==1 and notebook['metadata']['exp022']['helper_revision'] in urls[0]
    assert '/main/' not in urls[0] and '/exp/022' not in urls[0]


def test_offline_notebook_bootstrap_does_not_request_network(tmp_path,monkeypatch):
    notebook=json.loads((ROOT/'notebooks/EXP022_Kaggle.ipynb').read_text(encoding='utf-8'))
    cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    namespace={};exec(cells[0],namespace)
    helper=subprocess.check_output(['git','show',notebook['metadata']['exp022']['helper_revision']+':scripts/exp022_notebook.py'],cwd=ROOT)
    kit=tmp_path/'exp022_runtime.zip'
    with zipfile.ZipFile(kit,'w') as z:z.writestr('exp022_notebook.py',helper)
    kit.with_suffix('.sha256').write_text(workflow.sha256(kit)+'  '+kit.name)
    namespace.update(NETWORK_MODE='offline',OFFLINE_KIT_PATH=str(kit),LOCAL_ROOT=str(tmp_path/'local'),PERSISTENT_ROOT=str(tmp_path/'persistent'))
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:pytest.fail('Offline bootstrap requested network'))
    monkeypatch.delattr(hashlib,'file_digest',raising=False)
    exec(cells[1],namespace);assert namespace['BOOTSTRAP_OK'] and namespace['OFFLINE_KIT']==kit


def test_real_runner_rejects_wrong_manifest_before_loading_model(tmp_path):
    portable=tmp_path/'portable';(portable/'inputs').mkdir(parents=True)
    (portable/'inputs/run.json').write_text('{}')
    result=subprocess.run(workflow.stage_command(sys.executable,ROOT,portable,tmp_path/'run','Preflight',60),text=True,capture_output=True)
    assert result.returncode==1
    assert 'Portable input manifest hash mismatch' in result.stderr
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('platform_name',[None,'Colab','Kaggle'])
def test_bootstrap_missing_mount_flag_is_actionable_not_nameerror(platform_name,tmp_path,monkeypatch,capsys):
    notebook=json.loads((ROOT/'notebooks/EXP022_Kaggle.ipynb').read_text(encoding='utf-8'))
    cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    namespace={}
    if platform_name:
        exec(cells[0],namespace);namespace.pop('MOUNT_OK')
        namespace.update(PLATFORM=platform_name,LOCAL_ROOT=str(tmp_path/'local'),PERSISTENT_ROOT=str(tmp_path/'output'))
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:(_ for _ in ()).throw(OSError('synthetic network unavailable')))
    if platform_name=='Colab':monkeypatch.setattr(Path,'is_dir',lambda p:False)
    exec(cells[1],namespace)
    output=capsys.readouterr().out
    assert namespace['BOOTSTRAP_OK'] is False and 'NameError' not in output
    if platform_name is None:assert 'Configuration cell first' in output
    elif platform_name=='Colab':assert 'Drive is not mounted' in output
    else:assert 'Internet/helper download unavailable' in output


def test_widget_registration_never_launches_training_and_click_requires_opt_in(tmp_path,monkeypatch):
    objects=[]
    class Widget:
        def __init__(self,*args,**kwargs):
            self.description=kwargs.get('description','');self.value=kwargs.get('value',kwargs.get('options',[''])[0])
            self.disabled=False;self.callbacks=[];objects.append(self)
        def on_click(self,callback):self.callbacks.append(callback)
        def observe(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):return False
    widgets=types.ModuleType('ipywidgets')
    for name in ('Dropdown','BoundedIntText','Checkbox','Button','Output','VBox'):setattr(widgets,name,Widget)
    displays=types.ModuleType('IPython.display');displays.display=lambda *a:None;displays.Markdown=lambda s:s
    monkeypatch.setitem(sys.modules,'ipywidgets',widgets);monkeypatch.setitem(sys.modules,'IPython.display',displays)
    run=session(tmp_path);calls=[]
    monkeypatch.setattr(run,'stage',lambda *a,**k:calls.append((a,k)))
    monkeypatch.setattr(run,'summary',lambda:{'platform':'SYNTHETIC','output':'fixture'})
    namespace={'RUN_LONG_TRAINING':False};workflow.show_controls(run,namespace)
    assert calls==[] and namespace['RUN_LONG_TRAINING'] is False
    selector=next(w for w in objects if w.description=='Stage:');selector.value='MiniScreen'
    button=next(w for w in objects if w.description=='Run selected stage')
    approve=next(w for w in objects if w.description.startswith('I reviewed'))
    button.callbacks[0](button);assert calls==[]
    approve.value=True;button.callbacks[0](button)
    assert calls==[(('MiniScreen',14400),{'allow_long':True})]
    assert approve.value is False and namespace['RUN_LONG_TRAINING'] is False
