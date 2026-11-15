"""Standard-library Colab/Kaggle final-refit launcher, training off by default."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):digest.update(chunk)
    return digest.hexdigest()


def extract_handoff(archive, expected_sha, destination):
    if sha256(archive) != expected_sha:
        raise ValueError('Handoff ZIP SHA-256 mismatch. Use the producer receipt; do not replace the expected hash.')
    destination = Path(destination)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):raise ValueError('Duplicate ZIP entries')
        for name in names:
            p = PurePosixPath(name)
            if (name != p.as_posix() or p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name
                    or ((z.getinfo(name).external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError('Unsafe ZIP path: ' + name)
        m = json.loads(z.read('artifact_manifest.json'))
        if (m['format'] != 'st3-el8-artifacts-v1' or m['metadata']['checkpoints'] != 60 or
                m['kinds'].get('historical_cv') != 'historical_cv'):
            raise ValueError('Not a complete EL8 final-refit handoff')
        if set(names) != set(m['files_sha256']) | {'artifact_manifest.json'}:
            raise ValueError('Incomplete handoff ZIP')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            for name, digest in m['files_sha256'].items():
                if not (destination / name).is_file() or sha256(destination / name) != digest:
                    raise ValueError('Existing extraction differs. Preserve it and choose a fresh LOCAL_ROOT.')
            return m
        total = sum(z.getinfo(n).file_size for n in names)
        if shutil.disk_usage(destination.parent).free < total + 2 * 1024**3:
            raise ValueError('Insufficient local disk; need ZIP extraction size plus 2 GiB margin.')
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
            ready = Path(tmp) / 'ready'; ready.mkdir()
            for name, digest in m['files_sha256'].items():
                target = ready / name; target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as f, target.open('wb') as out:shutil.copyfileobj(f, out)
                if sha256(target) != digest:raise ValueError('Handoff file checksum failed: ' + name)
            ready.rename(destination)
    return m


def load_helper(path):
    spec = importlib.util.spec_from_file_location('el8_exp022_environment', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def launch(archive, expected_sha, local_root, persistent_root, *, platform='Colab', authorize=False, max_gpu_hours=0):
    """One action: environment, validation, strict parity, missing fits, replay, export.

    Without approval only source/environment/artifact validation and forecasts
    execute. Never starts CV training, Development scoring or an upload.
    """
    local = Path(local_root); persistent = Path(persistent_root)
    if platform not in ('Colab', 'Kaggle'):raise ValueError('Platform must be Colab or Kaggle')
    if platform == 'Colab' and Path('/content/drive') in local.resolve().parents:
        raise ValueError('LOCAL_ROOT must be outside Drive, under /content')
    local.mkdir(parents=True, exist_ok=True); persistent.mkdir(parents=True, exist_ok=True)
    extracted = local / 'handoff'; manifest = extract_handoff(archive, expected_sha, extracted)
    revision = manifest['metadata']['deployment_revision']; deploy = local / 'deployment'
    if not deploy.exists():
        subprocess.run(['git', 'clone', '--no-checkout', str(extracted / 'source/source.bundle'), str(deploy)], check=True)
        subprocess.run(['git', '-C', str(deploy), 'checkout', '--detach', revision], check=True)
    if (subprocess.check_output(['git', '-C', str(deploy), 'rev-parse', 'HEAD'], text=True).strip() != revision or
            subprocess.check_output(['git', '-C', str(deploy), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('Deployment source differs. Preserve it and choose a fresh LOCAL_ROOT.')
    helper = load_helper(deploy / 'scripts/exp022_notebook.py')
    helper.REPOSITORY = str(extracted / 'source/source.bundle')
    paths = {'local': local / 'environment', 'persistent': persistent / 'setup',
             'archive': extracted / 'resources/exp022_colab_inputs.zip'}
    paths['local'].mkdir(parents=True, exist_ok=True); paths['persistent'].mkdir(parents=True, exist_ok=True)
    session = helper.NotebookRun(platform, paths, pixel_cache='ram'); session.setup()
    # New runtime/source directory; never resume fits into the historical CV root.
    run = persistent / ('final-' + revision[:12] + '-' + session.run_dir.name)
    logs = run / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    cv = extracted / 'historical_cv'
    command = [str(session.python), str(deploy / 'scripts/run_el8.py')]
    shared = ['--cv-run', str(cv), '--portable-dir', str(session.portable),
              '--model-dir', str(extracted / 'production/e50'),
              '--clip-large-model-dir', str(extracted / 'production/clip_large'), '--run-dir', str(run)]
    helper.run_logged(command + ['validate', *shared, '--output', str(run / 'recovered_inventory.json')],
                      logs, 'validate', cwd=deploy, env=session.env, timeout=300, progress=True)
    helper.run_logged(command + ['forecast', *shared, '--output', str(run / 'forecast.json')],
                      logs, 'forecast', cwd=deploy, env=session.env, timeout=300, progress=True)
    print(json.dumps(json.loads((run / 'forecast.json').read_text()), indent=2))
    if not authorize:
        print('PREPARATION COMPLETE. Training is OFF. Review the policy and forecast, set AUTHORIZE_FINAL_REFIT=True '
              'and a positive MAX_GPU_HOURS, then Run all once. Verification and export will run automatically.')
        return {'status': 'PREPARED_NOT_AUTHORIZED', 'run_dir': str(run)}
    if max_gpu_hours <= 0:raise ValueError('Owner authorization also requires a positive MAX_GPU_HOURS.')
    # The child enforces exact CV replay before any refit, including on L4.
    helper.run_logged(command + ['final-refit', *shared, '--device', 'cuda', '--authorize-final-refit',
                      '--policy-sha256', manifest['metadata']['policy_sha256'], '--max-gpu-hours', str(max_gpu_hours)],
                      logs, 'final-refit-verify-export', cwd=deploy, env=session.env,
                      timeout=max_gpu_hours * 3600 + 300, progress=True)
    result = json.loads((run / 'last_export.json').read_text())
    print('Final refit, replay, image-only smoke and complete export finished:', result)
    return result
