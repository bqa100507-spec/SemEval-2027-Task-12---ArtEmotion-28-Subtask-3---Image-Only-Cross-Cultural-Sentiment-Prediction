"""Explicit, image-only EL8 prediction. Default E50+L inference is untouched."""

import copy
import hashlib
import importlib.metadata
import json
import platform
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import data, submission
from .baseline import file_sha256
from .cli import ROOT, git_head, json_output
from .contrast_study import new_output
from .e50_heads import ENCODERS, MANIFEST, inference_config, load_bundle, load_legacy
from .e50_inference import read_input, run_feature_worker, run_prediction, safe_image_path
from .e50l_heads import CLIP_LARGE, load_clip_large_bundle, load_clip_large_legacy
from .el8_artifacts import SEEDS, compare_predictions, read_json, safe_relative, sealed, seed_mean, strict_payload
from .fusion import blend
from .incremental import collect_features
from .visual_groups import PixelReader
from .visual_lora import VisualLoRA, load_adapter, load_visual
from .visual_lora_inputs import config022
from .visual_training import predict
from .vote_model import VoteStudent


def runtime(device):
    return {'device': str(device), 'python': platform.python_version(), 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy', 'Pillow')},
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_precision': torch.get_float32_matmul_precision()}


def deterministic_runtime(requested):
    device = torch.device('cuda' if requested == 'auto' and torch.cuda.is_available() else
                          'cpu' if requested == 'auto' else requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('EL8 CUDA requested but unavailable; explicitly choose --device cpu or a CUDA runtime')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = False
    return device


def implementation_identity():
    names = ('el8_inference.py', 'el8_artifacts.py', 'visual_lora.py', 'visual_groups.py',
             'visual_training.py', 'vote_model.py', 'caption_student.py')
    # Git's LF content identifies the Python source on Windows and Linux.
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in names}


def final_bundle(directory, args=None):
    directory = Path(directory); manifest = read_json(directory / 'final_manifest.json')
    if (manifest.get('kind') != 'final_refit' or manifest.get('status') != 'VERIFIED' or
            manifest.get('seeds') != list(SEEDS) or manifest.get('condition') != 'L8'):
        raise ValueError('EL8 requires a VERIFIED three-seed final_refit bundle; CV checkpoints are not a final model')
    if manifest['scientific_config'] != config022():
        raise ValueError('Final EL8 scientific configuration changed')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name != 'final_manifest.json'}
    if actual != set(manifest['files_sha256']):
        raise ValueError('Final bundle file coverage differs')
    for name, digest in manifest['files_sha256'].items():
        safe_relative(name)
        if (directory / name).is_symlink() or file_sha256(directory / name) != digest:
            raise ValueError('Final EL8 checksum mismatch: ' + name)
    expected_models = [f'final/L8_{s}/student.pt' for s in SEEDS]
    if manifest['checkpoints'] != expected_models:
        raise ValueError('Final seed checkpoint inventory differs')
    for name in expected_models:
        strict_payload(directory / name)
        seed = int(Path(name).parent.name.split('_')[1])
        info = sealed((directory / name).parent, manifest['identity'])
        if (info.get('kind') != 'final_refit' or info['seed'] != seed or
                info['condition'] != 'L8' or info['held_paintings'] or len(info['fit_paintings']) != 1335 or
                info['policy_sha256'] != manifest['policy_sha256']):
            raise ValueError('Checkpoint is not a permitted full-population final seed')
    verification = read_json(directory / 'verification.json')
    if verification.get('status') != 'PASS' or verification.get('seeds') != list(SEEDS):
        raise ValueError('Final replay is not complete')
    if (len(verification.get('checks', [])) != 3 or
            {c['seed'] for c in verification['checks']} != set(SEEDS) or
            any(c.get('status') != 'PASS' or c.get('checkpoint_sha256') !=
                file_sha256(directory / f"final/L8_{c['seed']}/student.pt") for c in verification['checks'])):
        raise ValueError('Final verification checkpoint links changed')
    cv_replay = read_json(directory / 'cv_replay.json')
    if cv_replay.get('status') != 'PASS' or cv_replay.get('implementation') != implementation_identity():
        raise ValueError('Current EL8 implementation has no strict CV replay PASS')
    if args:
        for label, root in (('e50', args.model_dir), ('clip_large', args.clip_large_model_dir)):
            for name, digest in manifest['production_dependencies'][label].items():
                if file_sha256(root / name) != digest:
                    raise ValueError('Original E50+L dependency mismatch: ' + label + '/' + name)
    return manifest


def frozen_banks(args, inputs, images, hashes, scratch):
    """Use the existing release caches/workers, including their independent SigLIP environment."""
    cfg = inference_config()
    _, e50, index = load_bundle(args.model_dir, cfg, ENCODERS)
    _, large, large_index = load_clip_large_bundle(args.clip_large_model_dir, e50)
    banks, metadata = [], {}
    for name in ('siglip2', CLIP_LARGE):
        encoder = large['encoder'] if name == CLIP_LARGE else cfg['encoders'][name]
        if name == CLIP_LARGE:
            cached, historical = load_clip_large_legacy(images, large_index, large['legacy'], args.cache_dir, 768)
        else:
            cached, historical = load_legacy(name, images, index, e50['legacy'][name], args.cache_dir, 768)
        missing = {d: p for d, p in images.items() if d not in cached}
        incremental = None
        if missing:
            new, incremental = run_feature_worker(name, missing, args, encoder, scratch)
            cached.update(new)
        banks.append({n: cached[d] for n, d in hashes.items()})
        metadata[name] = {'historical': historical, 'worker': incremental}
    return banks, metadata


def task_head(payload, cfg, device):
    with torch.random.fork_rng(devices=[]):
        head = VoteStudent(payload['languages'], cfg['training'])
    head.load_state_dict(payload['head'], strict=True)
    return head.to(device).eval()


class CachedVision:
    """Adapted vectors are already normalized; preserve original head/group math."""
    def __init__(self, bank):
        self.bank = bank
    def __call__(self, value):
        return value


def predict_cached_head(frame, head, adapted, frozen, device, microbatch):
    # Original predict() performs the image-to-language gather and FP32 sigmoid.
    ctx = SimpleNamespace(device=device, banks=[{}, *frozen],
                          pixels=lambda names, dev: torch.as_tensor(np.stack([adapted[n] for n in names]), device=dev))
    return predict(ctx, CachedVision(adapted), head, frame, 'L8', microbatch)[0]


def adapted_features(vision, processor, images, directory, checkpoint, cfg, device, microbatch):
    identity = {'encoder': 'el8-clip-b', 'width': 512, 'checkpoint_sha256': file_sha256(checkpoint),
                'clip': cfg['clip'], 'processor': processor.to_dict(), 'runtime': runtime(device),
                'batch_size': microbatch, 'implementation': implementation_identity()}
    def loader():
        @torch.inference_mode()
        def encode(batch):
            loaded = []
            try:
                for _, path in batch:
                    with Image.open(path) as image:
                        loaded.append(image.convert('RGB'))
                pixels = processor(images=loaded, return_tensors='pt')['pixel_values'].to(device)
                return vision(pixels).cpu().numpy()
            finally:
                for image in loaded:
                    image.close()
        return encode
    return collect_features(images, directory, identity, microbatch, loader)


def run_el8(args):
    tick = time.perf_counter()
    if args.output_dir.exists():
        raise ValueError('Output exists; choose a fresh directory')
    manifest = final_bundle(args.el8_model_dir, args)
    cfg = manifest['scientific_config']; device = deterministic_runtime(args.device)
    inputs = read_input(args.input, inference_config()['official_languages'])
    input_digest = file_sha256(args.input)
    paths = {n: safe_image_path(args.data_dir, n) for n in inputs.image.unique()}
    hashes = {n: file_sha256(p) for n, p in paths.items()}
    images = {hashes[n]: p for n, p in paths.items()}
    microbatch = manifest['microbatch']
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    cache = args.feature_cache_dir or args.cache_dir / 'el8_incremental'
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix='.el8-') as temporary:
        scratch = Path(temporary)
        base_args = copy.copy(args); base_args.mode = 'e50l'; base_args.output_dir = scratch / 'e50l'
        run_prediction(base_args)
        base = pd.read_csv(base_args.output_dir / submission.FILENAME, float_precision='round_trip')
        frozen, frozen_info = frozen_banks(args, inputs, images, hashes, scratch)
        model, processor, encoder = load_visual(args.cache_dir, cfg, device)
        vision = VisualLoRA(model, cfg); frames = []; feature_info = {}
        for seed, name in zip(SEEDS, manifest['checkpoints']):
            checkpoint = args.el8_model_dir / name
            payload = strict_payload(checkpoint, cfg); load_adapter(vision, payload['adapters'])
            vision.requires_grad_(False).eval()
            bank, info = adapted_features(vision, processor, images, cache, checkpoint, cfg, device, microbatch)
            head = task_head(payload, cfg, device)
            frames.append(predict_cached_head(inputs, head, {n: bank[d] for n, d in hashes.items()}, frozen, device, microbatch))
            feature_info[str(seed)] = info
            del head
        mean = seed_mean(frames); result = blend(base, mean, .5)
        # blend sorts its result; public inference preserves the caller's input order.
        result = inputs[data.KEYS].merge(result, on=data.KEYS, validate='one_to_one', sort=False)
        submission.validate_predictions(result, inputs)
        if file_sha256(args.input) != input_digest or any(file_sha256(paths[n]) != d for n, d in hashes.items()):
            raise ValueError('Inputs changed during EL8 inference')
        with new_output(args.output_dir) as stage:
            result.to_csv(stage / submission.FILENAME, index=False, lineterminator='\n')
            submission.pack(stage / submission.FILENAME, stage / 'submission.zip', inputs)
            submission.validate_archive(stage / 'submission.zip', inputs)
            json_output({'mode': 'el8', 'production_default': 'e50l', 'project_revision': git_head(ROOT),
                         'final_manifest_sha256': file_sha256(args.el8_model_dir / 'final_manifest.json'),
                         'input_sha256': input_digest, 'image_sha256': hashes, 'runtime': runtime(device),
                         'encoder': encoder, 'frozen_features': frozen_info, 'adapted_features': feature_info,
                         'formula': 'round6(0.5*E50L + 0.5*round6(mean(seed_probabilities)))',
                         'wall_seconds': time.perf_counter() - tick,
                         'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                         'output_sha256': {n: file_sha256(stage / n) for n in ('predictions.csv', 'submission.zip')}}, stage / 'run.json')
    print('Validated explicit EL8 predictions:', args.output_dir / 'submission.zip', flush=True)


def replay_cv(root, input_dir, data_root, cache_dir, output, requested='auto', max_seconds=900):
    """Actual image-backed replay, strict raw and six-decimal comparisons.

    This is deliberately separate from inspecting historical PASS manifests.
    No gold labels are read. Only keys, image paths and frozen image features.
    """
    from .el8_artifacts import cv_inventory
    from .polarity_workflow import ExecutionBudget
    evidence = cv_inventory(root); identity = evidence['identity']; cfg = config022()
    if file_sha256(input_dir / 'run.json') != identity['input_manifest_sha256']:
        raise ValueError('Replay input manifest differs')
    sealed = read_json(input_dir / 'run.json')
    if file_sha256(input_dir / 'features.npz') != sealed['files_sha256']['features.npz']:
        raise ValueError('Frozen image features changed')
    # usecols prevents reading any vote/label columns or locked data.
    mapping = pd.read_csv(input_dir / 'targets.csv', usecols=[*data.KEYS, 'image', 'fold', 'image_sha256'])
    if file_sha256(input_dir / 'targets.csv') != cfg['parent_targets_sha256']:
        raise ValueError('Replay training identity differs')
    banks = []
    with np.load(input_dir / 'features.npz', allow_pickle=False) as a:
        for name in ('clip', 'siglip', 'large'):
            banks.append(dict(zip(a['names'].tolist(), a[name].copy())))
    device = deterministic_runtime(requested); budget = ExecutionBudget(max_seconds)
    model, processor, encoder = load_visual(cache_dir, cfg, device); vision = VisualLoRA(model, cfg)
    ctx = SimpleNamespace(device=device, config=cfg, banks=banks, pixels=PixelReader(data_root, processor))
    saved = pd.read_csv(root / 'cv/predictions.csv', float_precision='round_trip')
    checks = []; started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        for fold in range(1, 6):
            held = mapping[mapping.fold.eq(fold)].sort_values(data.KEYS).reset_index(drop=True)
            for image, digest in held[['image', 'image_sha256']].drop_duplicates().itertuples(index=False, name=None):
                if file_sha256(safe_image_path(data_root, image)) != digest:
                    raise ValueError('Replay image changed')
            frame_list = []
            for seed in SEEDS:
                budget.check('Image replay ' + str((fold, seed)), -1)
                folder = root / 'work' / f'fold_{fold}' / f'L8_{seed}'
                info = read_json(folder / 'run.json')['metadata']
                if set(held.painting) != set(info['held_paintings']) or set(held.painting) & set(info['fit_paintings']):
                    raise ValueError('Replay training/held isolation failed')
                payload = strict_payload(folder / 'student.pt', cfg)
                load_adapter(vision, payload['adapters']); vision.requires_grad_(False).eval()
                head = task_head(payload, cfg, device)
                predicted, _ = predict(ctx, vision, head, held, 'L8', info['microbatch'])
                expected = pd.read_csv(folder / 'predictions.csv', float_precision='round_trip').sort_values(data.KEYS).reset_index(drop=True)
                checks.append({'fold': fold, 'seed': seed, 'level': 'raw_checkpoint', **compare_predictions(predicted, expected)})
                frame_list.append(predicted)
            mean = seed_mean(frame_list); subset = saved[saved.fold.eq(fold)]
            base = subset[subset.model.eq('R0')][data.KEYS + ['p_positive']].reset_index(drop=True)
            for name, frame in (('L8', mean), ('EL8', blend(base, mean, .5))):
                expected = subset[subset.model.eq(name)][data.KEYS + ['p_positive']].reset_index(drop=True)
                checks.append({'fold': fold, 'level': name, **compare_predictions(frame, expected)})
            print('Image-backed fold replay complete:', fold, flush=True)
        report = {'status': 'PASS' if all(r['status'] == 'PASS' for r in checks) else 'FAIL', 'checks': checks,
                  'level': 'IMAGE_BACKED_CV_REPLAY', 'runtime': runtime(device), 'original_runtime': identity,
                  'implementation': implementation_identity(), 'cv_manifest_sha256': file_sha256(root / 'cv/run.json'),
                  'seconds': time.perf_counter() - started, 'encoder': encoder,
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None}
    except BaseException as error:
        json_output({'status': 'FAILED', 'error': str(error), 'checks': checks}, output)
        raise
    json_output(report, output)
    return report
