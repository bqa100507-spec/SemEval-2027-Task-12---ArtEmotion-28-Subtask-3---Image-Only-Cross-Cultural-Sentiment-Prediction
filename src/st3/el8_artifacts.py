"""Read-only validation and portable packaging of EXP-022 CV/final evidence.

Historical identities are checked against their immutable Git source, never
rewritten to match the machine performing deployment validation.
"""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch

from . import data
from .baseline import file_sha256
from .cli import ROOT
from .fusion import blend
from .polarity_workflow import read_sealed
from .visual_lora_inputs import config022
from .vote_model import VoteStudent

TRAINING_REVISION = 'eb4a7fac4ebe79971305b0b3de2c0e5102ea5f91'
SEEDS = (2027, 2028, 2029)
CONDITIONS = ('F0', 'F8', 'L0', 'L8')
STAGES = ('preflight', 'pixel_benchmark', 'pilot', 'mini', 'mini_verification',
          'mini_decision', 'screen', 'screen_verification', 'screen_decision',
          'cv', 'cv_verification', 'cv_decision')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or name != path.as_posix() or path.is_absolute() or
            any(p in ('..', '.') for p in path.parts) or '\\' in name or ':' in name):
        raise ValueError('Unsafe artifact path: ' + name)
    return path


def source_identity(identity, repo=ROOT):
    for name, expected in identity['source_sha256'].items():
        safe_relative(name)
        raw = subprocess.run(['git', '-C', str(repo), 'show', TRAINING_REVISION + ':' + name],
                             check=True, capture_output=True).stdout
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Historical source differs from training revision: ' + name)
    raw = subprocess.run(['git', '-C', str(repo), 'show', TRAINING_REVISION + ':configs/exp022.json'],
                         check=True, capture_output=True).stdout
    if hashlib.sha256(raw).hexdigest() != identity['config_sha256']:
        raise ValueError('Historical scientific configuration changed')
    return len(identity['source_sha256'])


def sealed(directory, identity):
    record = read_json(directory / 'run.json')
    for name in record['files_sha256']:
        safe_relative(name)
        if (directory / name).is_symlink():
            raise ValueError('Symlink in sealed artifact')
    return read_sealed(directory, identity)


def strict_payload(path, cfg=None):
    cfg = cfg or config022()
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if set(payload) != {'languages', 'head', 'adapters'}:
        raise ValueError('Unexpected checkpoint top-level schema')
    languages = payload['languages']
    if (not isinstance(languages, (list, tuple)) or not languages or
            not all(isinstance(n, str) and n for n in languages) or list(languages) != sorted(set(languages))):
        raise ValueError('Invalid checkpoint language vocabulary')
    # Construction must not disturb a caller's training/dropout RNG.
    with torch.random.fork_rng(devices=[]):
        head = VoteStudent(languages, cfg['training'])
    expected = head.state_dict()
    if set(payload['head']) != set(expected):
        raise ValueError('Unexpected task-head keys')
    for name, tensor in payload['head'].items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError('Invalid task-head tensor: ' + name)
    head.load_state_dict(payload['head'], strict=True)
    keys = {f'model.vision_model.encoder.layers.{i}.self_attn.{q}.{m}'
            for i in range(12) for q in ('q_proj', 'v_proj') for m in ('a', 'b')}
    if set(payload['adapters']) != keys:
        raise ValueError('Adapters must cover exactly the 24 visual q/v modules')
    for name, tensor in payload['adapters'].items():
        shape = (8, 768) if name.endswith('.a') else (768, 8)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError('Invalid LoRA tensor: ' + name)
    return payload


def cv_inventory(root, *, check_source=True, check_tensors=True):
    root = Path(root)
    identity = read_json(root / 'cv/run.json')['identity']
    source_count = source_identity(identity) if check_source else None
    metas = {name: sealed(root / name, identity) for name in STAGES if (root / name).exists()}
    for name in ('preflight', 'pilot', 'mini', 'mini_verification', 'mini_decision',
                 'screen', 'screen_verification', 'screen_decision', 'cv', 'cv_verification', 'cv_decision'):
        if name not in metas:
            raise ValueError('Missing required stage: ' + name)
    study, verification, decision = (metas[n] for n in ('cv', 'cv_verification', 'cv_decision'))
    if study['folds'] != list(range(1, 6)) or study['seeds'] != list(SEEDS):
        raise ValueError('Incomplete CV folds/seeds')
    latest = sorted((root / 'cv_verification_attempts').glob('*.json'))
    if not latest or read_json(latest[-1]).get('status') != 'PASS':
        raise ValueError('Latest CV replay failed or incomplete; successful decision revoked')
    digest = file_sha256(root / 'cv/run.json')
    if (verification['status'] != 'PASS' or verification['checkpoints_replayed'] != 60 or
            verification['study_sha256'] != digest or decision['study_sha256'] != digest or
            read_json(latest[-1]).get('study_sha256') != digest or
            decision['verification_sha256'] != file_sha256(root / 'cv_verification/run.json')):
        raise ValueError('CV verification/decision links changed')
    if decision['status'] != 'CV_GATE_QUALIFIED_NO_DEVELOPMENT_AUTHORIZED':
        raise ValueError('CV did not qualify; do not prepare a final refit')
    for info in (verification, decision):
        if info.get('locked_data_accessed', False) or info.get('locked_confirmation_opened', False) or info.get('development_scored', False):
            raise ValueError('Evaluation restriction violated')
    if decision.get('selected') != 'EL8':
        raise ValueError('Unexpected selected candidate')
    # Verify the chronological stage chain as well as each stage's file seal.
    for phase, parent in (('mini', None), ('screen', 'mini'), ('cv', 'screen')):
        stage, verify, gate = (metas[n] for n in (phase, phase + '_verification', phase + '_decision'))
        sha = file_sha256(root / phase / 'run.json')
        if (verify['study_sha256'] != sha or gate['study_sha256'] != sha or
                gate['verification_sha256'] != file_sha256(root / (phase + '_verification') / 'run.json')):
            raise ValueError('Stage verification chain changed: ' + phase)
        for required in ('preflight', 'pilot'):
            if stage[required + '_sha256'] != file_sha256(root / required / 'run.json'):
                raise ValueError('Stage prerequisite changed: ' + phase)
        if parent and stage['parent_decision_sha256'] != file_sha256(root / (parent + '_decision') / 'run.json'):
            raise ValueError('Parent decision changed: ' + phase)
    found = list((root / 'work').glob('fold_*/*/student.pt'))
    expected = {f'{fold}/{c}_{s}' for fold in range(1, 6) for c in CONDITIONS for s in SEEDS}
    if len(found) != 60 or set(study['seed_manifests']) != expected:
        raise ValueError('Expected exactly 60 complete CV checkpoints')
    count = 0
    fold_populations = {}
    replay_checks = read_json(root / 'cv_verification/replay_checks.json')
    if len(replay_checks) != 60 or {f"{r['fold']}/{r['condition']}_{r['seed']}" for r in replay_checks} != expected:
        raise ValueError('Incomplete original replay record coverage')
    for key in sorted(expected):
        fold, model = key.split('/'); path = root / 'work' / ('fold_' + fold) / model
        info = sealed(path, identity)
        if file_sha256(path / 'run.json') != study['seed_manifests'][key]:
            raise ValueError('CV model manifest changed: ' + key)
        condition, seed = model.split('_')
        if (info['fold'], info['condition'], info['seed']) != (int(fold), condition, int(seed)):
            raise ValueError('Model identity mismatch: ' + key)
        if set(info['fit_paintings']) & set(info['held_paintings']):
            raise ValueError('Painting leakage in checkpoint: ' + key)
        populations = (set(info['fit_paintings']), set(info['held_paintings']))
        if int(fold) in fold_populations and fold_populations[int(fold)] != populations:
            raise ValueError('Different painting populations within a fold')
        fold_populations[int(fold)] = populations
        if (info['preflight_manifest_sha256'] != file_sha256(root / 'preflight/run.json') or
                info['base_sha256'] != metas['preflight']['base_sha256'] or
                (condition.startswith('L') and info['adapter_update_norm'] <= 0) or
                (condition.startswith('F') and info['adapter_update_norm'] != 0)):
            raise ValueError('Checkpoint base/update/preflight evidence differs')
        if check_tensors:
            strict_payload(path / 'student.pt')
        count += 1
    population = set.union(*(v[1] for v in fold_populations.values()))
    if len(population) != 1335 or any(fit != population - held for fit, held in fold_populations.values()):
        raise ValueError('CV population or held-out fold partition differs')
    return {'kind': 'historical_cv', 'checkpoints': count, 'source_files_verified': source_count,
            'source_revision': TRAINING_REVISION, 'identity': identity,
            'decision': decision, 'original_replay': verification}


def seed_mean(frames):
    if len(frames) != 3:
        raise ValueError('EL8 requires all three seeds 2027/2028/2029')
    base = frames[0][data.KEYS].copy()
    for frame in frames:
        data.unique_pairs(frame)
        if not frame[data.KEYS].equals(base):
            raise ValueError('Seed pair coverage/order differs')
        if not np.isfinite(frame.p_positive).all() or not frame.p_positive.between(0, 1).all():
            raise ValueError('Invalid seed probabilities')
    return base.assign(p_positive=np.round(np.mean([f.p_positive.to_numpy() for f in frames], axis=0), 6))


def compare_predictions(actual, expected):
    if not actual[data.KEYS].equals(expected[data.KEYS]):
        raise ValueError('Prediction coverage/order differs')
    delta = np.abs(actual.p_positive.to_numpy() - expected.p_positive.to_numpy())
    bad = delta != 0
    return {'status': 'PASS' if not bad.any() else 'FAIL', 'rows': len(delta),
            'max_abs_error': float(delta.max(initial=0)), 'failing_rows': int(bad.sum()),
            'examples': expected.loc[bad, data.KEYS].head(10).to_dict('records')}


def reconstruct_cv(root):
    """Independent strict aggregation of sealed per-seed predictions, no labels."""
    root = Path(root)
    saved = pd.read_csv(root / 'cv/predictions.csv', float_precision='round_trip')
    results = []
    for fold in range(1, 6):
        subset = saved[saved.fold.eq(fold)]
        base = subset[subset.model.eq('R0')][data.KEYS + ['p_positive']].reset_index(drop=True)
        for condition in CONDITIONS:
            frames = []
            for seed in SEEDS:
                folder = root / 'work' / f'fold_{fold}' / f'{condition}_{seed}'
                info = read_json(folder / 'run.json')['metadata']
                frame = pd.read_csv(folder / 'predictions.csv', float_precision='round_trip')
                if set(frame.painting) != set(info['held_paintings']) or set(frame.painting) & set(info['fit_paintings']):
                    raise ValueError('Invalid held-out prediction identities')
                frames.append(frame.sort_values(data.KEYS).reset_index(drop=True))
            mean = seed_mean(frames)
            for name, frame in ((condition, mean), ('E' + condition, blend(base, mean, .5))):
                expected = subset[subset.model.eq(name)][data.KEYS + ['p_positive']].reset_index(drop=True)
                results.append({'fold': fold, 'model': name, **compare_predictions(frame, expected)})
    return {'level': 'SEALED_PREDICTION_AGGREGATION', 'status': 'PASS' if all(r['status'] == 'PASS' for r in results) else 'FAIL', 'checks': results}


def export_bundle(destination, roots, kinds, metadata):
    """Complete trees, explicit artifact kinds, checksum every file; no overwrite."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Bundle already exists; choose a new export filename')
    files = {}
    for label, root in roots.items():
        safe_relative(label)
        for p in Path(root).rglob('*'):
            if p.is_symlink():
                raise ValueError('Do not export symlinks')
            if p.is_file():
                name = label + '/' + p.relative_to(root).as_posix(); safe_relative(name)
                files[name] = (p, file_sha256(p))
    if not files or set(roots) != set(kinds):
        raise ValueError('Explicit bundle roots/kinds required')
    manifest = {'format': 'st3-el8-artifacts-v1', 'kinds': kinds, 'metadata': metadata,
                'files_sha256': {n: value[1] for n, value in sorted(files.items())}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.partial')
    try:
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for n, (p, _) in sorted(files.items()):
                z.write(p, n)
            z.writestr('artifact_manifest.json', json.dumps(manifest, sort_keys=True, indent=2))
        verify_archive(temp)
        temp.rename(destination)
    finally:
        temp.unlink(missing_ok=True)
    digest = file_sha256(destination)
    destination.with_suffix('.sha256').write_text(digest + '  ' + destination.name + '\n', encoding='utf-8')
    return {'path': str(destination), 'sha256': digest, 'files': len(files), 'kinds': kinds}


def verify_archive(path, expected_sha=None, expected_kinds=None):
    if expected_sha and file_sha256(path) != expected_sha:
        raise ValueError('Archive SHA-256 mismatch')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for n in names:
            safe_relative(n)
            if ((z.getinfo(n).external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Symlink archive member')
        m = json.loads(z.read('artifact_manifest.json'))
        if m['format'] != 'st3-el8-artifacts-v1' or (expected_kinds is not None and m['kinds'] != expected_kinds):
            raise ValueError('Incompatible bundle kind/format')
        if set(names) != set(m['files_sha256']) | {'artifact_manifest.json'}:
            raise ValueError('Incomplete/extra archive files')
        for n, digest in m['files_sha256'].items():
            with z.open(n) as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != digest:
                    raise ValueError('Archive file checksum mismatch: ' + n)
        return m


def import_bundle(archive, destination, expected_sha, expected_kinds=None):
    manifest = verify_archive(archive, expected_sha, expected_kinds)
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    # Verify all existing destinations BEFORE publishing anything.
    for name, digest in manifest['files_sha256'].items():
        target = destination / name
        if target.is_symlink() or not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Restore path escapes destination')
        if target.exists() and (not target.is_file() or file_sha256(target) != digest):
            raise ValueError('Restore would overwrite different artifacts: ' + name)
    with zipfile.ZipFile(archive) as z:
        for name in manifest['files_sha256']:
            target = destination / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as f, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                    shutil.copyfileobj(f, out); temp = Path(out.name)
                temp.rename(target)
    return manifest
