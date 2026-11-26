"""Owner-gated final refit. One invocation trains, replays, smokes and exports.

The newly specified fold=0 inner partition is an approval-requiring deployment
policy; it is not represented as an existing EXP-022 scientific decision.
"""

import copy
import hashlib
import json
import math
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import data
from .baseline import file_sha256
from .cli import ROOT, git_head, json_output
from .contrast_study import new_output
from .e50_heads import ENCODERS, inference_config, load_bundle
from .e50l_heads import load_clip_large_bundle
from .el8_artifacts import (SEEDS, TRAINING_REVISION, compare_predictions, cv_inventory,
                            export_bundle, read_json, sealed, strict_payload)
from .el8_inference import deterministic_runtime, final_bundle, implementation_identity, replay_cv, runtime, task_head
from .polarity_workflow import ExecutionBudget, StageBlocked, read_sealed, seal
from .visual_feasibility import load_context, run_preflight, run_pilot
from .visual_lora import VisualLoRA, load_adapter
from .visual_lora_inputs import config022
from .visual_training import fit_model, predict, save_checkpoint


def require_authorization(args):
    policy_path = ROOT / 'configs/el8_final_refit.json'
    digest = hashlib.sha256(policy_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if (not args.authorize_final_refit or args.policy_sha256 != digest or
            not math.isfinite(args.max_gpu_hours) or args.max_gpu_hours <= 0):
        raise StageBlocked('No final training authorized. Review configs/el8_final_refit.json, then explicitly pass '
                           '--authorize-final-refit --policy-sha256 ' + digest + ' --max-gpu-hours HOURS')
    policy = read_json(policy_path)
    if (policy['seeds'] != list(SEEDS) or policy['condition'] != 'L8' or
            policy['training_revision'] != TRAINING_REVISION or
            policy['inner_partition_fold_argument'] != 0 or policy['microbatch'] != 4):
        raise ValueError('Final-refit policy differs from the reviewed implementation')
    return policy, digest


def forecast(root):
    fits = [read_json(p)['metadata'] for p in Path(root).glob('work/fold_*/L8_*/run.json')]
    if len(fits) != 15:
        raise ValueError('Need all 15 historical L8 fits for a cost forecast')
    costs = np.array([m['fit_seconds'] * 1335 / len(m['fit_paintings']) for m in fits])
    return {'basis': 'Recorded L4 L8 nested selection+refit times scaled to 1335 paintings; final stopping epochs unknown',
            'training_hours_mean': float(3 * costs.mean() / 3600),
            'training_hours_p90_with_25_percent_margin': float(3 * np.quantile(costs, .9) * 1.25 / 3600),
            'includes_replay_setup_export': False, 'actual_final_refit_measured': False}


def scientific_sources_unchanged(identity):
    # A new orchestration identity is intentional. Training mathematics must be
    # byte-identical to the reviewed original source; no historical seal bypass.
    for name, digest in identity['source_sha256'].items():
        if name.startswith('src/st3/') and name != 'src/st3/cli.py':
            raw = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
            import hashlib
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('Frozen scientific source changed: ' + name)


def full_refit(args):
    notify = getattr(args, '_progress', lambda *a, **kw: None)
    policy, policy_digest = require_authorization(args)
    notify(stage='ARTIFACT_VALIDATION')
    if args.run_dir.exists() and (args.run_dir / 'cv').exists():
        raise ValueError('Final RunDir must be separate from historical CV artifacts')
    root = args.cv_run.resolve()
    if args.run_dir.resolve() == root or root in args.run_dir.resolve().parents or args.run_dir.resolve() in root.parents:
        raise ValueError('Final RunDir overlaps CV artifacts')
    for protected in (args.portable_dir, args.model_dir, args.clip_large_model_dir, args.data_dir, args.cache_dir):
        protected = protected.resolve(); dest = args.run_dir.resolve()
        if dest == protected or dest in protected.parents or protected in dest.parents:
            raise ValueError('Final RunDir overlaps protected inputs or models')
    evidence = cv_inventory(root); scientific_sources_unchanged(evidence['identity'])
    _, e50, _ = load_bundle(args.model_dir, inference_config(), ENCODERS)
    load_clip_large_bundle(args.clip_large_model_dir, e50)
    device = deterministic_runtime(args.device)
    if device.type != 'cuda':
        raise StageBlocked('Final training requires an authorized CUDA budget; CPU is supported for inference only')
    budget = getattr(args, '_budget', None) or ExecutionBudget(args.max_gpu_hours * 3600)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    attempt = args.run_dir / ('attempt-' + uuid.uuid4().hex + '.json')
    status = {'status': 'STARTED', 'policy_sha256': policy_digest, 'owner_authorized_gpu_hours': args.max_gpu_hours,
              'source_revision': git_head(ROOT), 'forecast': forecast(root), 'runtime': runtime(device)}
    json_output(status, attempt)
    # A new failed verification revokes any previously published bundle.
    active = args.run_dir / 'bundle/final_manifest.json'
    if active.exists():
        old_manifest = read_json(active)
        if old_manifest['status'] not in ('VERIFIED', 'INVALIDATED_BY_FAILED_REPLAY'):
            raise ValueError('Invalid prior final bundle; inspect preserved failure before retry')
    try:
        notify(stage='PARITY')
        cv_replay = args.run_dir / 'deployment_cv_replay.json'
        if cv_replay.exists():
            previous = read_json(cv_replay)
            if previous.get('status') == 'PASS' and (previous['runtime'] != runtime(device) or
                    previous.get('implementation') != implementation_identity() or
                    previous.get('cv_manifest_sha256') != file_sha256(root / 'cv/run.json')):
                raise StageBlocked('Deployment replay identity changed; preserve this run and choose a fresh RunDir')
        if not cv_replay.exists() or read_json(cv_replay).get('status') != 'PASS':
            replay_cv(root, args.portable_dir / 'inputs', args.portable_dir / 'data',
                      args.portable_dir / 'cache', cv_replay, args.device,
                      min(900, max(1, budget.seconds - (time.perf_counter() - budget.start))))
            if read_json(cv_replay)['status'] != 'PASS':
                raise StageBlocked('Strict image-backed CV parity failed; final training not started')
        budget.check('Final preflight', -1)
        notify(stage='PREFLIGHT')
        stage_args = copy.copy(args)
        stage_args.max_seconds = max(1, budget.seconds - (time.perf_counter() - budget.start))
        run_preflight(stage_args)
        budget.check('Final timing pilot', -1)
        notify(stage='BENCHMARK')
        stage_args.max_seconds = max(1, budget.seconds - (time.perf_counter() - budget.start))
        run_pilot(stage_args)
        # Record the exact installed environment alongside all completed seeds.
        import subprocess, sys
        lock = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True)
        (args.run_dir / 'environment-lock.txt').write_text(lock, encoding='utf-8')
        shutil.copyfile(ROOT / 'configs/el8_final_refit.json', args.run_dir / 'approved_policy.json')
        ctx, model = load_context(args)
        pre = read_sealed(args.run_dir / 'preflight', ctx.identity)
        if pre['status'] != 'PASS' or pre['microbatch'] != policy['microbatch']:
            raise StageBlocked('Final preflight requires original microbatch 4 with safe memory; no silent batch change')
        if ctx.train.painting.nunique() != 1335:
            raise ValueError('Final population must be the permitted 1335 paintings')
        vision = VisualLoRA(model, ctx.config)
        smoke = ctx.train[data.KEYS + ['image']].drop_duplicates()
        chosen = sorted(smoke.painting.unique())[:16]
        smoke = smoke[smoke.painting.isin(chosen)].sort_values(data.KEYS).reset_index(drop=True)
        held = ctx.train.iloc[:0].copy()  # no Development/confirmation access
        completed = []
        for seed in SEEDS:
            path = args.run_dir / 'final' / f'L8_{seed}'
            budget.check('Final L8 seed ' + str(seed), -1)
            if not path.exists():
                notify(stage='TRAINING', seed=seed)
                with new_output(path) as stage:
                    head, curves, gradients, info = fit_model(ctx, vision, ctx.train, held, 0, seed, 'L8', 4, budget)
                    predicted, _ = predict(ctx, vision, head, smoke, 'L8', 4)
                    save_checkpoint(stage / 'student.pt', head, vision)
                    predicted.to_csv(stage / 'smoke_predictions.csv', index=False, lineterminator='\n')
                    smoke.to_csv(stage / 'smoke_inputs.csv', index=False, lineterminator='\n')
                    pd.DataFrame(curves).to_csv(stage / 'curves.csv', index=False, lineterminator='\n')
                    json_output(gradients, stage / 'gradients.json')
                    seal(stage, ctx.identity, {**info, 'kind': 'final_refit', 'condition': 'L8', 'seed': seed,
                                              'policy_sha256': policy_digest, 'microbatch': 4})
            notify(stage='FINAL_VERIFICATION', seed=seed)
            info = sealed(path, ctx.identity)
            if (info['kind'] != 'final_refit' or info['seed'] != seed or info['policy_sha256'] != policy_digest or
                    info['held_paintings'] or info['fit_paintings'] != sorted(ctx.train.painting.unique())):
                raise ValueError('Completed final seed provenance differs')
            payload = strict_payload(path / 'student.pt'); load_adapter(vision, payload['adapters'])
            vision.requires_grad_(False).eval(); head = task_head(payload, ctx.config, ctx.device)
            replay, _ = predict(ctx, vision, head, smoke, 'L8', 4)
            original = pd.read_csv(path / 'smoke_predictions.csv', float_precision='round_trip')
            comparison = compare_predictions(replay, original)
            if comparison['status'] != 'PASS' or vision.frozen_digest() != info['base_sha256']:
                raise ValueError('Final checkpoint replay/base parity failed')
            if info['adapter_update_norm'] <= 0 or not replay.groupby(smoke.language).p_positive.std().fillna(0).gt(0).any():
                raise ValueError('Final model failed LoRA/image sensitivity smoke')
            completed.append({'seed': seed, 'checkpoint_sha256': file_sha256(path / 'student.pt'), **comparison})
            print(f'Final seed {seed}: complete, replay and image-only smoke PASS; {len(completed)}/3; '
                  f'elapsed {(time.perf_counter()-budget.start)/60:.1f} min', flush=True)
        verification = {'status': 'PASS', 'seeds': list(SEEDS), 'checks': completed, 'runtime': runtime(device),
                        'locked_data_accessed': False, 'development_scored': False}
        json_output(verification, args.run_dir / 'final_verification.json')
        bundle = args.run_dir / 'bundle'
        notify(stage='BUNDLE')
        if not bundle.exists():
            with new_output(bundle) as stage:
                for seed in SEEDS:
                    # A killed unpublished fit directory is not a final model.
                    shutil.copytree(args.run_dir/'final'/f'L8_{seed}', stage/'final'/f'L8_{seed}')
                shutil.copyfile(args.run_dir / 'final_verification.json', stage / 'verification.json')
                shutil.copyfile(cv_replay, stage / 'cv_replay.json')
                json_output(evidence, stage / 'cv_evidence.json')
                dependencies = {}
                for label, folder in (('e50', args.model_dir), ('clip_large', args.clip_large_model_dir)):
                    shutil.copytree(folder, stage / 'production' / label)
                    dependencies[label] = {p.name: file_sha256(p) for p in folder.iterdir() if p.is_file()}
                manifest = {'kind': 'final_refit', 'status': 'VERIFIED', 'condition': 'L8', 'seeds': list(SEEDS),
                            'scientific_config': ctx.config, 'policy': policy, 'policy_sha256': policy_digest,
                            'training_revision': TRAINING_REVISION, 'inference_revision': git_head(ROOT),
                            'runtime': runtime(device), 'identity': ctx.identity, 'microbatch': 4,
                            'checkpoints': [f'final/L8_{s}/student.pt' for s in SEEDS],
                            'production_dependencies': dependencies,
                            'files_sha256': {p.relative_to(stage).as_posix(): file_sha256(p) for p in stage.rglob('*') if p.is_file()}}
                json_output(manifest, stage / 'final_manifest.json')
                final_bundle(stage, args)
        else:
            existing = read_json(bundle / 'final_manifest.json')
            existing['status'] = 'VERIFIED'
            json_output(existing, bundle / 'final_manifest.json')
            final_bundle(bundle, args)
        status.update(status='FINAL_BUNDLE_VERIFIED', seconds=time.perf_counter() - budget.start)
    except BaseException as error:
        status.update(status='FAILED', error=str(error), seconds=time.perf_counter() - budget.start)
        if active.exists():
            invalid = read_json(active); invalid['status'] = 'INVALIDATED_BY_FAILED_REPLAY'
            json_output(invalid, active)
        raise
    finally:
        json_output(status, attempt)
        # Includes completed checkpoints, failed attempts and evidence, even on
        # interruption. Export outside the run avoids self-inclusion.
        if not getattr(args, '_managed_export', False):
            roots = {'final_run': args.run_dir, 'historical_cv': root,
                     'e50': args.model_dir, 'clip_large': args.clip_large_model_dir}
            destination = args.run_dir.parent / (args.run_dir.name + '-export-' + uuid.uuid4().hex[:8] + '.zip')
            receipt = export_bundle(destination, roots,
                                    {'final_run': 'final_refit_progress', 'historical_cv': 'historical_cv',
                                     'e50': 'production_e50', 'clip_large': 'production_clip_large'}, status)
            json_output(receipt, args.run_dir / 'last_export.json')
            print('Complete artifact export and checksum:', receipt, flush=True)
    return status
