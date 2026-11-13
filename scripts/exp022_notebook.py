"""Standard-library orchestration outside the immutable EXP-022 training tree.

This file never changes scientific configuration or relaxes a training seal.
Cloud notebooks load a separately pinned copy; training always uses SOURCE_REV.
"""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import zipfile
from pathlib import Path, PurePosixPath

REPOSITORY = 'https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction.git'
SOURCE_REV = '41ca0d4061f129c1821009e6d00614dede08708f'
INPUT_SHA = 'fa2f7c4528b49925c400216037a64381bc6a0b35a1b3b11458a6bcbfda924337'
ZIP_SHA = 'bbedbe2ad7e426040aebb63366ea5978e03d6483503b8ac9c88af14238a3d156'
BUNDLE_SHA = 'e75a180270518f4a315b88b925e801dcbc61240639e003a54473516a80bed1af'
PYTHON_VERSION = '3.12.10'
TORCH_VERSION = '2.7.1+cu126'
UV_VERSION = '0.9.8'
TORCH_INDEX = 'https://download.pytorch.org/whl/cu126'
LONG_STAGES = ('MiniScreen', 'MiniScreenVerify', 'Screen', 'ScreenVerify')
SEALED_STAGES = ('preflight', 'pilot', 'mini', 'mini_verification', 'mini_decision',
                 'screen', 'screen_verification', 'screen_decision')


class WorkflowError(RuntimeError):
    """A short actionable failure; complete diagnostics stay in persistent logs."""


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require_hash(path, expected, description):
    if not Path(path).is_file():
        raise WorkflowError(f'{description} is missing: {path}. Attach/upload the published artifact and rerun.')
    if not re.fullmatch('[0-9a-f]{64}', expected or '') or sha256(path) != expected:
        raise WorkflowError(f'{description} SHA-256 mismatch: {path}. Replace it with the original published file; do not replace the expected hash.')


def locate_zip(root, explicit='', filename='exp022_colab_inputs.zip'):
    if explicit:
        path = Path(explicit)
        if not path.is_file():raise WorkflowError(f'ZIP not found: {path}. Check the configurable input path.')
        return path
    candidates = sorted(Path(root).rglob(filename))
    if len(candidates) != 1:
        action = 'Add the private Dataset under Notebook Inputs' if not candidates else 'Detach duplicate inputs or set the explicit ZIP path'
        raise WorkflowError(f'Found {len(candidates)} copies of {filename} under {root}. {action}.')
    return candidates[0]


def paths_for(name, *, input_path='', local_root=None, persistent_root=None, input_root=None):
    if name not in ('Colab', 'Kaggle'):raise WorkflowError('Platform must be Colab or Kaggle.')
    local = Path(local_root or ('/content/exp022' if name == 'Colab' else '/kaggle/temp/exp022'))
    persistent = Path(persistent_root or ('/content/drive/MyDrive/exp022_runs' if name == 'Colab' else '/kaggle/working/exp022_runs'))
    archive = (Path(input_path or '/content/drive/MyDrive/exp022_colab_inputs.zip') if name == 'Colab'
               else locate_zip(input_root or '/kaggle/input', input_path))
    if not archive.is_file():raise WorkflowError(f'Input ZIP missing: {archive}. Upload exp022_colab_inputs.zip to MyDrive first.')
    local.mkdir(parents=True, exist_ok=True); persistent.mkdir(parents=True, exist_ok=True)
    if name == 'Colab' and Path('/content/drive') in local.resolve().parents:
        raise WorkflowError('LocalRoot must be local /content storage, outside the Drive mount.')
    return {'local':local, 'persistent':persistent, 'archive':archive}


def safe_entries(archive):
    infos = archive.infolist(); names = [i.filename for i in infos]
    if len(names) != len(set(names)):raise WorkflowError('ZIP contains duplicate entries.')
    for item in infos:
        name = item.filename; parts = PurePosixPath(name).parts
        if not parts or name.startswith('/') or '\\' in name or ':' in name or '..' in parts or stat.S_ISLNK(item.external_attr >> 16):
            raise WorkflowError('Unsafe ZIP entry: ' + name)
    return infos


def extract_checked(archive_path, destination, expected, *, input_bundle=False):
    """Validate before publishing; existing extraction is rechecked, never erased."""
    require_hash(archive_path, expected, 'Input ZIP' if input_bundle else 'Offline runtime ZIP')
    destination = Path(destination)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(dir=destination.parent) as temp:
            infos = safe_entries(archive)
            if input_bundle:
                for name, digest, label in (('bundle.json', BUNDLE_SHA, 'Published bundle manifest'),
                                             ('inputs/run.json', INPUT_SHA, 'Published input manifest')):
                    if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                        raise WorkflowError(label+' SHA-256 mismatch before extraction. Replace the artifact; do not replace the pin.')
            size = sum(i.file_size for i in infos)
            if shutil.disk_usage(destination.parent).free < size + 512*1024**2:
                raise WorkflowError(f'Insufficient local disk: extraction needs {size/1024**3:.2f} GiB plus 0.5 GiB margin.')
            stage = Path(temp)/'ready'; stage.mkdir()
            for item in infos:
                if item.is_dir():continue
                target = stage/item.filename; target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open('wb') as output:shutil.copyfileobj(source, output)
                mode = (item.external_attr >> 16) & 0o777
                if mode:target.chmod(mode)
            validate_tree(stage, 'bundle.json' if input_bundle else 'runtime_bundle.json')
            stage.rename(destination)
    manifest_name = 'bundle.json' if input_bundle else 'runtime_bundle.json'
    manifest = validate_tree(destination, manifest_name)
    if input_bundle:
        require_hash(destination/'bundle.json', BUNDLE_SHA, 'Published bundle manifest')
        require_hash(destination/'inputs/run.json', INPUT_SHA, 'Published input manifest')
        if manifest.get('paintings') != 1335 or manifest.get('locked_images_included') is not False:
            raise WorkflowError('Unexpected input population; only the published training bundle is accepted.')
    return manifest


def validate_tree(root, manifest_name):
    root = Path(root); manifest = read_json(root/manifest_name)
    expected = manifest['files_sha256']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root/manifest_name}
    if actual != set(expected):raise WorkflowError(f'Missing or extra files in {root}. Keep the validated extraction intact; use a new local directory to retry.')
    for name, digest in expected.items():
        if '..' in PurePosixPath(name).parts or Path(name).is_absolute():raise WorkflowError('Unsafe manifest path.')
        require_hash(root/name, digest, name)
    return manifest


def run_logged(command, log_dir, label, *, cwd=None, env=None, timeout=1800, progress=False):
    """Do not stream child tracebacks; retain full output and one actionable error."""
    log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    log = log_dir/f'{label}-{tag}.log'; tick = time.monotonic(); interrupted = False
    with log.open('w', encoding='utf-8') as output:
        output.write(json.dumps([str(s) for s in command])+'\n'); output.flush()
        child = subprocess.Popen([str(s) for s in command], stdout=output, stderr=subprocess.STDOUT,
            cwd=cwd, env=env, start_new_session=os.name != 'nt')
        try:
            child.wait(timeout=timeout)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            interrupted = True
            if os.name == 'nt':child.terminate()
            else:os.killpg(child.pid, signal.SIGINT)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':child.kill()
                else:os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        code = child.returncode
    write_json(log.with_suffix('.json'), {'command':[str(s) for s in command], 'exit_code':code,
        'interrupted':interrupted, 'seconds':time.monotonic()-tick, 'log':str(log)})
    if interrupted:raise WorkflowError(f'{label} interrupted/time-limited. Complete models remain intact; rerun the same stage. Log: {log}')
    if code:
        lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
        first = next((s for s in lines if s.startswith(('VALIDATION_FAILURE:', 'STAGE_BLOCKED:', 'CUDA_OOM:', 'RESUMABLE_BUDGET_STOP:'))), None)
        detail = first or next((s for s in reversed(lines) if s.strip()), f'exit {code}')
        advice = {75:'Repeat this stage to reuse complete models; the interrupted model restarts.',
                  4:'Free GPU memory or choose another runtime; do not change scientific settings.',
                  3:'Complete the required prior verification or use a faster runtime after reviewing its forecast.'}.get(code, 'Inspect the full log; dependency/source/parity errors must be fixed before continuing.')
        raise WorkflowError(f'{label} failed (exit {code}): {detail[:500]} {advice} Log: {log}')
    if progress:print(f'{label}: complete ({time.monotonic()-tick:.1f}s). Log: {log}')
    return log


def package_pins(repo):
    result = {}
    for line in (Path(repo)/'requirements.txt').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):continue
        name, version = line.split('=='); result[name] = version
    return {**result, 'torch':TORCH_VERSION}


def validate_environment(actual, pins):
    if actual['python'] != PYTHON_VERSION:
        raise WorkflowError(f'Python {actual["python"]} is incompatible; require {PYTHON_VERSION}. Rebuild the isolated environment or attach the complete offline kit.')
    for name, expected in pins.items():
        if actual['versions'].get(name) != expected:
            raise WorkflowError(f'{name}={actual["versions"].get(name)} is incompatible; require {expected}. Do not substitute versions; rebuild from the pinned dependencies/offline kit.')
    if not actual.get('cuda_available'):
        raise WorkflowError('CUDA is unavailable in the isolated environment. Enable a GPU accelerator; check the driver and the pinned CUDA Torch installation.')
    if actual.get('cuda') != '12.6':raise WorkflowError(f'Torch CUDA runtime {actual.get("cuda")} differs from pinned 12.6.')


def stage_command(python, repo, portable, run_dir, stage, seconds):
    if stage not in ('Preflight', 'Pilot', *LONG_STAGES):raise WorkflowError('Stage is not available in notebooks; full CV needs separate owner authorization.')
    return [str(python), str(Path(repo)/'scripts/run_exp022.py'), stage, '--portable-dir', str(portable),
        '--input-manifest-sha256', INPUT_SHA, '--run-dir', str(run_dir), '--device', 'cuda',
        '--data-dir', str(Path(portable)/'data'), '--cache-dir', str(Path(portable)/'cache'), '--max-seconds', str(seconds)]


def verify_sealed(directory, identity):
    directory = Path(directory); record = read_json(directory/'run.json')
    if record['identity'] != identity:raise WorkflowError('Source/configuration/software/GPU identity differs: '+str(directory))
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p != directory/'run.json'}
    if actual != set(record['files_sha256']):raise WorkflowError('Incomplete or extra files in sealed checkpoint: '+str(directory))
    for name, digest in record['files_sha256'].items():require_hash(directory/name, digest, 'Sealed checkpoint file')
    return record['metadata']


def export_checkpoints(run_dir, destination, identity, execution_profile):
    """Only published stages/models, attempts and diagnostics; no partial fit dirs."""
    run_dir = Path(run_dir); destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for name in SEALED_STAGES:
        folder = run_dir/name
        if folder.exists():verify_sealed(folder, identity); files.extend(p for p in folder.rglob('*') if p.is_file())
    for model in (run_dir/'work').glob('fold_*/*'):
        if re.fullmatch('fold_[1-5]', model.parent.name) and re.fullmatch('(F0|F8|L0|L8)_202[789]', model.name) and (model/'run.json').exists():
            verify_sealed(model, identity); files.extend(p for p in model.rglob('*') if p.is_file())
    for name in ('logs', 'mini_verification_attempts', 'screen_verification_attempts'):
        if (run_dir/name).exists():files.extend(p for p in (run_dir/name).rglob('*') if p.is_file())
    files.extend(p for p in run_dir.glob('*.json') if p.is_file())
    if (run_dir/'environment-lock.txt').exists():files.append(run_dir/'environment-lock.txt')
    files = sorted(set(files)); hashes = {'run/'+p.relative_to(run_dir).as_posix():sha256(p) for p in files}
    manifest = {'identity':identity, 'execution_profile':execution_profile, 'source_revision':SOURCE_REV, 'files_sha256':hashes,
        'resume_scope':'Completed model boundaries only; verification attempts retained, including failures.'}
    temporary = destination.with_suffix('.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for p in files:archive.write(p, 'run/'+p.relative_to(run_dir).as_posix())
        archive.writestr('checkpoint_bundle.json', json.dumps(manifest, indent=2))
    temporary.replace(destination); digest = sha256(destination)
    destination.with_suffix('.sha256').write_text(digest+'  '+destination.name+'\n', encoding='utf-8')
    return {'path':str(destination), 'sha256':digest, 'bytes':destination.stat().st_size}


def restore_checkpoints(archive_path, expected_sha, run_dir, identity, execution_profile):
    require_hash(archive_path, expected_sha, 'Attached checkpoint bundle')
    run_dir = Path(run_dir); run_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(dir=run_dir.parent) as temp:
        infos = safe_entries(archive); manifest = json.loads(archive.read('checkpoint_bundle.json'))
        if manifest['identity'] != identity or manifest['execution_profile'] != execution_profile or manifest['source_revision'] != SOURCE_REV:
            raise WorkflowError('Checkpoint runtime/source identity differs. Keep this bundle archived; start a separate run on this GPU/software profile. Cross-platform transfer does not authorize mixing fits.')
        if {i.filename for i in infos} != set(manifest['files_sha256'])|{'checkpoint_bundle.json'}:
            raise WorkflowError('Checkpoint bundle entry coverage differs.')
        stage = Path(temp)
        for name, digest in manifest['files_sha256'].items():
            if not name.startswith('run/'):raise WorkflowError('Unexpected checkpoint bundle root.')
            target = stage/name; target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open('wb') as output:shutil.copyfileobj(source, output)
            require_hash(target, digest, 'Restored checkpoint file')
            existing = run_dir/Path(name).relative_to('run')
            if existing.exists() and (not existing.is_file() or sha256(existing) != digest):
                raise WorkflowError('Restore would overwrite different existing artifacts: '+str(existing)+'. Choose a fresh persistent output directory; never discard a valid run.')
        # Validate every published stage/model before writing ANY run files.
        for path in (stage/'run').rglob('run.json'):verify_sealed(path.parent, identity)
        for name in manifest['files_sha256']:
            source = stage/name; target = run_dir/Path(name).relative_to('run')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    return len(manifest['files_sha256'])


class NotebookRun:
    def __init__(self, platform_name, paths, *, offline_kit=None):
        self.platform = platform_name; self.paths = paths; self.offline = Path(offline_kit) if offline_kit else None
        self.local = paths['local']; self.persistent = paths['persistent']; self.portable = self.local/'inputs'
        self.repo = self.local/'source'; self.python = self.local/'venv/bin/python'
        self.logs = self.persistent/'setup_logs'; self.run_dir = None; self.identity = None; self.profile = None
        self.benchmark_ok = False; self.reused_benchmark = False; self.last_export = None
        self.stage_status = {}
        self.env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'0', 'PYTHONUNBUFFERED':'1', 'CUBLAS_WORKSPACE_CONFIG':':4096:8',
            'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'PIP_DISABLE_PIP_VERSION_CHECK':'1',
            'PYTHONDONTWRITEBYTECODE':'1',
            'PIP_CACHE_DIR':str(self.local/'pip-cache')}

    def command(self, command, label, **kwargs):
        return run_logged(command, self.logs, label, env=self.env, **kwargs)

    def python_json(self, code, label):
        path = self.command([self.python, '-c', code], label, cwd=self.repo)
        lines = path.read_text(encoding='utf-8').splitlines()
        return json.loads(next(line[len('EXP022_JSON:'):] for line in reversed(lines) if line.startswith('EXP022_JSON:')))

    def setup(self, *, offline_sha=None, restore_path='', restore_sha=''):
        self.logs.mkdir(parents=True, exist_ok=True)
        host_versions = {}
        for name in ('torch', 'transformers'):
            try:host_versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:host_versions[name] = None
        host = {'platform':self.platform, 'host_python':platform.python_version(), 'host_packages':host_versions,
                'internet':'not needed (attached offline kit)' if self.offline else 'required; clone/pip failures stop setup', 'visible_gpu':'0 only'}
        write_json(self.logs/f'host-{uuid.uuid4().hex[:8]}.json', host)
        try:self.command(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader'], 'gpu')
        except (FileNotFoundError, WorkflowError) as error:
            raise WorkflowError('No accessible NVIDIA GPU. Select GPU in Runtime settings (Colab) or Accelerator settings (Kaggle), then Run all again. '+str(error)) from None
        if self.offline:
            runtime = self.local/'offline'; manifest = extract_checked(self.offline, runtime, offline_sha)
            if manifest['source_revision'] != SOURCE_REV or manifest['python'] != PYTHON_VERSION:
                raise WorkflowError('Offline kit source/Python revision differs from the notebook pins.')
        else:runtime = None
        if not self.repo.exists():
            source = runtime/'source.bundle' if runtime else REPOSITORY
            self.command(['git', 'clone', '--no-checkout', str(source), self.repo], 'clone', timeout=600)
            self.command(['git', '-C', self.repo, 'checkout', '--detach', SOURCE_REV], 'checkout')
        self.command(['git', '-C', self.repo, 'diff', '--exit-code', SOURCE_REV, '--'], 'source-diff')
        commit = subprocess.check_output(['git', '-C', self.repo, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', self.repo, 'status', '--porcelain'], text=True).strip()
        if commit != SOURCE_REV or dirty:raise WorkflowError('Training checkout differs or has extra files. Preserve it and use a fresh LocalRoot; no reset/force operation is performed.')
        extract_checked(self.paths['archive'], self.portable, ZIP_SHA, input_bundle=True)
        if not self.python.exists():
            if runtime:
                base_python = runtime/'python/bin/python3.12'
            else:
                bootstrap = self.local/'bootstrap'
                self.command([sys.executable, '-m', 'pip', 'install', '--target', bootstrap, f'uv=={UV_VERSION}'], 'uv-install', timeout=600)
                uv = bootstrap/'bin/uv'
                uv_env = {**self.env, 'UV_PYTHON_INSTALL_DIR':str(self.local/'managed-python')}
                run_logged([uv, 'python', 'install', PYTHON_VERSION], self.logs, 'python-install', env=uv_env, timeout=600)
                found = subprocess.check_output([str(uv), 'python', 'find', '--managed-python', PYTHON_VERSION], env=uv_env, text=True).strip()
                base_python = Path(found)
            self.command([base_python, '-m', 'venv', self.python.parent.parent], 'venv')
        if runtime:
            self.command([self.python, '-m', 'pip', 'install', '--no-index', '--find-links', runtime/'wheels',
                '--requirement', runtime/'environment.txt'], 'offline-dependencies', timeout=900)
        else:
            self.command([self.python, '-m', 'pip', 'install', f'torch=={TORCH_VERSION}', '--index-url', TORCH_INDEX], 'torch-install', timeout=1200)
            self.command([self.python, '-m', 'pip', 'install', '-r', self.repo/'requirements.txt'], 'dependencies', timeout=900)
        self.command([self.python, '-m', 'pip', 'check'], 'pip-check')
        names = list(package_pins(self.repo))
        code = f'''import importlib.metadata as m,json,platform,torch
value={{'python':platform.python_version(),'versions':{{n:m.version(n) for n in {names!r}}},'cuda_available':torch.cuda.is_available(),'cuda':torch.version.cuda}}
if value['cuda_available']:
 p=torch.cuda.get_device_properties(0)
 value.update(gpu=p.name,total_vram_bytes=p.total_memory,free_vram_bytes=torch.cuda.mem_get_info()[0],capability=list(torch.cuda.get_device_capability(0)),cudnn=torch.backends.cudnn.version(),visible_devices=torch.cuda.device_count())
print('EXP022_JSON:'+json.dumps(value))'''
        self.environment = self.python_json(code, 'environment'); validate_environment(self.environment, package_pins(self.repo))
        if self.environment['visible_devices'] != 1:raise WorkflowError('Expected exactly one visible CUDA GPU; refusing multi-GPU execution.')
        args = {'run_dir':str(self.persistent), 'portable_dir':str(self.portable), 'input_manifest_sha256':INPUT_SHA,
                'device':'cuda', 'data_dir':str(self.portable/'data')}
        code = f'''import sys,json,torch
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'src')
from st3.visual_lora_inputs import inputs
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
a={args!r}
a={{k:Path(v) if k.endswith('_dir') else v for k,v in a.items()}}
print('EXP022_JSON:'+json.dumps(inputs(SimpleNamespace(**a)).identity))'''
        self.identity = self.python_json(code, 'source-input-identity')
        self.profile = {k:self.environment[k] for k in ('python','versions','gpu','total_vram_bytes','capability','cuda','cudnn')}
        lock = subprocess.check_output([str(self.python),'-m','pip','freeze','--all'],env=self.env,text=True)
        lock = '\n'.join(sorted(lock.splitlines()))+'\n'
        self.profile['dependency_lock_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
        driver = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
        self.profile['driver'] = driver
        key = hashlib.sha256(json.dumps([self.identity,self.profile],sort_keys=True).encode()).hexdigest()[:16]
        self.run_dir = self.persistent/key; self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs = self.run_dir/'logs'
        if restore_path:restore_checkpoints(restore_path, restore_sha, self.run_dir, self.identity, self.profile)
        write_json(self.run_dir/'notebook_runtime.json', {'identity':self.identity,'execution_profile':self.profile,'source_revision':SOURCE_REV})
        write_json(self.run_dir/'environment.json', self.environment)
        (self.run_dir/'environment-lock.txt').write_text(lock,encoding='utf-8')
        print(f'Pinned Python/Torch/CUDA compatible. GPU: {self.environment["gpu"]}. Run: {self.run_dir}')

    def stage(self, stage, seconds, *, allow_long=False):
        if self.run_dir is None:raise WorkflowError('Complete setup first.')
        if stage in LONG_STAGES:
            if not allow_long:raise WorkflowError('Long training is disabled. Review the benchmark and explicitly approve the selected stage.')
            if not self.benchmark_ok:raise WorkflowError('A passing Preflight/Pilot is required before selecting later stages.')
            if not 60 <= seconds <= 14400:raise WorkflowError('Choose a time budget between 60 and 14,400 seconds.')
            projection = read_json(self.run_dir/'pilot/runtime_projection.json')
            if stage in ('MiniScreen', 'Screen'):
                if not projection['local_screen_within_gpu_budget']:raise WorkflowError('Screen forecast exceeds four GPU-hours on this runtime. Preserve the benchmark and choose a faster GPU; no scientific settings are changed.')
                estimate = projection['miniscreen_seconds' if stage == 'MiniScreen' else 'screen_seconds']
                print(f'{stage}: forecast {estimate/60:.1f} min; hard runner budget {seconds/60:.1f} min; peak pilot VRAM shown above. Complete models resume; partial current models restart.')
            required = {'MiniScreen':('preflight','pilot'), 'MiniScreenVerify':('mini',),
                        'Screen':('mini','mini_verification','mini_decision'), 'ScreenVerify':('screen',)}[stage]
            for name in required:
                if not (self.run_dir/name/'run.json').exists():raise WorkflowError(f'Missing {name}; select the prerequisite stage first.')
                verify_sealed(self.run_dir/name, self.identity)
        self.stage_status[stage] = 'RUNNING'
        try:
            log = run_logged(stage_command(self.python,self.repo,self.portable,self.run_dir,stage,seconds),self.logs,stage,
                cwd=self.repo,env=self.env,timeout=seconds+180,progress=True)
            self.stage_status[stage] = 'PASS'
            return log
        except BaseException:
            self.stage_status[stage] = 'FAIL'
            raise
        finally:
            write_json(self.run_dir/'notebook_stage_status.json',self.stage_status)
            if self.platform == 'Kaggle':
                active_error = sys.exc_info()[0] is not None
                try:self.export()
                except Exception as error:
                    self.failure(error)
                    if not active_error:raise

    def benchmark(self):
        self.reused_benchmark = (self.run_dir/'pilot/run.json').exists()
        self.stage('Preflight',600); self.stage('Pilot',900)
        pre = verify_sealed(self.run_dir/'preflight',self.identity)
        pilot = verify_sealed(self.run_dir/'pilot',self.identity)
        if pre['status'] != 'PASS' or pilot['status'] != 'PILOT_PASS':raise WorkflowError('Preflight or Pilot did not pass; long-stage controls remain disabled.')
        self.benchmark_ok = True
        write_json(self.run_dir/'benchmark_summary.json',self.summary())
        if self.platform == 'Kaggle':self.export()

    def summary(self):
        if self.run_dir is None:return {'platform':self.platform,'status':'SETUP_NOT_COMPLETE','output':str(self.persistent)}
        pre = read_json(self.run_dir/'preflight/run.json')['metadata'] if (self.run_dir/'preflight/run.json').exists() else {}
        pilot = read_json(self.run_dir/'pilot/run.json')['metadata'] if (self.run_dir/'pilot/run.json').exists() else {}
        parity = read_json(self.run_dir/'preflight/preprocessing.json') if (self.run_dir/'preflight/preprocessing.json').exists() else {}
        if not parity and (self.run_dir/'failed_preprocessing_parity.json').exists():
            parity = read_json(self.run_dir/'failed_preprocessing_parity.json')
        conditions = read_json(self.run_dir/'pilot/conditions.json') if (self.run_dir/'pilot/conditions.json').exists() else {}
        return {'platform':self.platform,'environment':self.environment,
            'preflight':'FAIL' if self.stage_status.get('Preflight')=='FAIL' else pre.get('status','NOT_RUN'),
            'pilot':'FAIL' if self.stage_status.get('Pilot')=='FAIL' else pilot.get('status','NOT_RUN'),'reused_saved_benchmark':self.reused_benchmark,
            'parity':{k:parity[k] for k in ('frozen_vs_cache','zero_lora_vs_frozen') if k in parity},
            'conditions':{k:{'epoch_seconds':v['epoch']['seconds'],'paintings_per_second':v['images_per_second'],
                'allocated_bytes':v['memory']['peak_cuda_allocated_bytes'],'reserved_bytes':v['memory']['peak_cuda_reserved_bytes'],
                'bce_lora_gradient':(v['gradients']['lora'] or {}).get('bce_norm'),
                'ce_lora_gradient':(v['gradients']['lora'] or {}).get('text_norm'),'adapter_update_norm':v['adapter_update_norm']} for k,v in conditions.items()},
            'forecast':pilot.get('runtime_projection',{}),'output':str(self.run_dir),
            'interpretation':'Benchmark only; no held-out quality or scientific screening success is established.'}

    def export(self):
        if self.run_dir is None:return None
        tag = time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:6]
        output = self.persistent.parent/f'exp022-checkpoints-{self.run_dir.name}-{tag}.zip'
        self.last_export = export_checkpoints(self.run_dir,output,self.identity,self.profile)
        print('Checkpoint/log export:',self.last_export['path'],'SHA-256:',self.last_export['sha256'])
        return self.last_export

    def failure(self, error):
        log_dir = self.logs; log_dir.mkdir(parents=True,exist_ok=True)
        tag=uuid.uuid4().hex[:8]; (log_dir/f'notebook-error-{tag}.log').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(log_dir/f'notebook-error-{tag}.json',{'status':'FAIL','first_actionable_error':str(error)})
        print('STOP:',str(error),'Full diagnostic logs:',log_dir)

    def build_offline_kit(self):
        """Explicit online owner action, no training. Produces a separate Linux kit."""
        if not self.benchmark_ok:raise WorkflowError('Build the offline kit only after a successful online benchmark.')
        if self.offline:raise WorkflowError('Use the existing attached offline kit; no network refresh is performed.')
        if sys.platform != 'linux':raise WorkflowError('Build Linux Kaggle dependencies on the validated Linux cloud runtime.')
        output = self.persistent.parent/'exp022_runtime.zip'
        if output.exists():raise WorkflowError('An offline kit already exists. Preserve/version it or choose a new output directory.')
        with tempfile.TemporaryDirectory(dir=self.local) as temp:
            stage = Path(temp); wheels=stage/'wheels'; wheels.mkdir()
            frozen=subprocess.check_output([str(self.python),'-m','pip','freeze','--all'],env=self.env,text=True)
            (stage/'environment.txt').write_text(frozen,encoding='utf-8')
            self.command([self.python,'-m','pip','download','--only-binary=:all:','--extra-index-url',TORCH_INDEX,
                '--dest',wheels,'-r',stage/'environment.txt'],'offline-wheel-download',timeout=3600)
            self.command(['git','-C',self.repo,'bundle','create',stage/'source.bundle','HEAD'],'source-bundle')
            base=subprocess.check_output([str(self.python),'-c','import sys; print(sys.base_prefix)'],text=True).strip()
            shutil.copytree(base,stage/'python',symlinks=False)
            shutil.copyfile(__file__,stage/'exp022_notebook.py')
            files={p.relative_to(stage).as_posix():sha256(p) for p in stage.rglob('*') if p.is_file()}
            write_json(stage/'runtime_bundle.json',{'source_revision':SOURCE_REV,'python':PYTHON_VERSION,'platform':'linux-x86_64',
                'helper_sha256':sha256(__file__),'files_sha256':files,'environment':self.environment})
            temporary=output.with_suffix('.partial')
            with zipfile.ZipFile(temporary,'w',zipfile.ZIP_STORED) as archive:
                for path in sorted(stage.rglob('*')):
                    if path.is_file():archive.write(path,path.relative_to(stage).as_posix())
            temporary.replace(output)
        digest=sha256(output); output.with_suffix('.sha256').write_text(digest+'  '+output.name+'\n',encoding='utf-8')
        print('Offline kit ready:',output,'SHA-256:',digest,'Keep it PRIVATE; attach it together with the original data ZIP on Kaggle.')
        return {'path':str(output),'sha256':digest}


def report_markdown(summary):
    env=summary.get('environment',{}); forecast=summary.get('forecast',{})
    lines=[f"### {summary['platform']} — EXP-022 benchmark",f"GPU: **{env.get('gpu','unavailable')}** · Python {env.get('python','?')} · Torch {env.get('versions',{}).get('torch','?')} · CUDA {env.get('cuda','?')}",
        f"Preflight: **{summary.get('preflight','NOT_COMPLETE')}** · Pilot: **{summary.get('pilot','NOT_COMPLETE')}**",
        f"Persistent output: `{summary['output']}`"]
    lines=[part for value in lines for part in (value,'')]
    if env.get('total_vram_bytes'):
        lines.append(f"VRAM total/free before model load: **{env['total_vram_bytes']/1024**3:.2f}/{env['free_vram_bytes']/1024**3:.2f} GiB**. Pinned dependency validation passed; one GPU is visible.")
    if summary.get('reused_saved_benchmark'):lines.append('**Reused saved benchmark:** throughput below is from the sealed original pilot, not a new timing measurement.')
    for name,value in summary.get('parity',{}).items():lines.append(f"- {name}: max absolute error **{value['max_abs_error']:.3g}**, minimum cosine **{value['cosine_min']:.9f}**")
    lines += ['', '| Condition | Epoch s | Paintings/s | Peak allocated/reserved GiB | BCE / CE LoRA gradient norms | Adapter update |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,value in summary.get('conditions',{}).items():
        grad='frozen' if value['bce_lora_gradient'] is None else f"{value['bce_lora_gradient']:.4g} / {value['ce_lora_gradient']:.4g}"
        lines.append(f"| {name} | {value['epoch_seconds']:.2f} | {value['paintings_per_second']:.2f} | {value['allocated_bytes']/1024**3:.3f} / {value['reserved_bytes']/1024**3:.3f} | {grad} | {value['adapter_update_norm']:.4g} |")
    lora = [summary.get('conditions',{}).get(name,{}) for name in ('L0','L8')]
    gradients_ok=all((v.get('bce_lora_gradient') or 0)>0 and (v.get('ce_lora_gradient') or 0)>0 and v.get('adapter_update_norm',0)>0 for v in lora)
    lines.append('LoRA BCE/CE gradients and actual updates: **'+('PASS' if gradients_ok else 'NOT VERIFIED')+'**.')
    if forecast:
        lines += ['', f"MiniScreen forecast: **{forecast['miniscreen_seconds']/60:.1f} min**. Full limited Screen: **{forecast['screen_gpu_hours']:.2f} GPU-hours**.",
            f"Four-hour resource gate: **{'permits explicit opt-in' if forecast['local_screen_within_gpu_budget'] else 'BLOCKED — choose a faster runtime'}**. Forecasts are not guarantees."]
    lines += ['', 'Run all has not launched screening. Pilot losses/gradients are engineering diagnostics, not held-out model quality.']
    return '\n'.join(lines)


def show_controls(session, namespace=None):
    """Only a fresh button click can enter a long stage; no persisted opt-in."""
    try:
        import ipywidgets as widgets
        from IPython.display import display,Markdown
    except ImportError:
        print('Optional controls unavailable: this host lacks ipywidgets. Benchmark results remain saved. Install ipywidgets in the notebook HOST or use the documented advanced runner.');return
    stage=widgets.Dropdown(options=['Setup / Benchmark',*LONG_STAGES],description='Stage:')
    seconds=widgets.BoundedIntText(value=14400,min=60,max=14400,description='Budget (s):')
    approve=widgets.Checkbox(value=False,description='I reviewed the forecast and authorize this selected stage',indent=False)
    button=widgets.Button(description='Run selected stage',button_style='warning')
    export=widgets.Button(description='Export completed artifacts')
    offline=widgets.Button(description='Build offline Kaggle kit')
    output=widgets.Output()
    def selection_changed(_):
        approve.value=False
        forecast=read_json(session.run_dir/'pilot/runtime_projection.json')
        selected=stage.value
        if selected in ('MiniScreen','Screen'):
            estimate=forecast['miniscreen_seconds' if selected=='MiniScreen' else 'screen_seconds']
            seconds.value=min(14400,max(900,math.ceil(estimate*1.25)))
            message=f'{selected}: forecast {estimate/60:.1f} min. Review the measured peak VRAM above; complete model boundaries resume.'
        else:
            seconds.value=1800 if selected=='ScreenVerify' else 900
            message=f'{selected}: replay timing is not yet measured; explicit budget {seconds.value/60:.0f} min. Verification uses the same encoder/GPU and adds no training fits.' if selected!='Setup / Benchmark' else 'Setup / Benchmark reuses valid sealed stages and does not launch screening.'
        with output:print(message)
    stage.observe(selection_changed,names='value')
    def clicked(_):
        with output:
            if button.disabled:return
            if not approve.value:print('Select the authorization checkbox after reviewing the benchmark.');return
            button.disabled=True
            if namespace is not None:namespace['RUN_LONG_TRAINING']=True
            try:
                if stage.value=='Setup / Benchmark':session.benchmark()
                else:session.stage(stage.value,seconds.value,allow_long=approve.value)
                display(Markdown(report_markdown(session.summary())))
            except Exception as error:session.failure(error)
            finally:
                approve.value=False;button.disabled=False
                if namespace is not None:namespace['RUN_LONG_TRAINING']=False
    def exported(_):
        with output:
            try:session.export()
            except Exception as error:session.failure(error)
    def offline_clicked(_):
        with output:
            if not approve.value:print('Offline kit may download several GB. Review storage/network and check authorization first.');return
            offline.disabled=True
            try:session.build_offline_kit()
            except Exception as error:session.failure(error)
            finally:approve.value=False;offline.disabled=False
    button.on_click(clicked);export.on_click(exported);offline.on_click(offline_clicked)
    display(widgets.VBox([stage,seconds,approve,button,export,offline,output]))
