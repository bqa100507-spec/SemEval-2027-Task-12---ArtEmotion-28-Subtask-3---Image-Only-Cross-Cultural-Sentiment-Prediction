"""Input-only submission readiness and exact output validation; no scorer/gold."""

from pathlib import Path

import pandas as pd

from . import data, submission
from .baseline import file_sha256
from .cli import json_output
from .e50_heads import ENCODERS, inference_config, load_bundle, load_legacy
from .e50_inference import read_input, safe_image_path
from .e50l_heads import load_clip_large_bundle, load_clip_large_legacy


def readiness(args):
    if not args.eval_input_csv or not args.eval_data_dir:
        raise ValueError('EVAL_INPUT_CSV and EVAL_DATA_DIR must resolve to the official input-only CSV and images')
    inputs = read_input(args.eval_input_csv, inference_config()['official_languages'])
    if inputs.empty:
        raise ValueError('Competition input CSV is empty')
    paths = {n: safe_image_path(args.eval_data_dir, n) for n in inputs.image.unique()}
    hashes = {n: file_sha256(p) for n, p in paths.items()}
    wanted = set(hashes.values())
    # Certify all frozen components before spending time on final fits. The
    # packaged official Development set is fully covered; no downloads needed.
    _, e50, index = load_bundle(args.model_dir, inference_config(), ENCODERS)
    _, large, large_index = load_clip_large_bundle(args.clip_large_model_dir, e50)
    coverage = {}
    for name in ('clip', 'siglip2', 'clip_large'):
        if name == 'clip_large':
            vectors, detail = load_clip_large_legacy(wanted, large_index, large['legacy'], args.cache_dir, 768)
        else:
            vectors, detail = load_legacy(name, wanted, index, e50['legacy'][name], args.cache_dir,
                                          inference_config()['encoders'][name]['width'])
        if set(vectors) != wanted:
            raise ValueError(f'{name}: frozen features missing for {len(wanted-set(vectors))} requested image contents. '
                             'Prepare verified frozen caches for that official input before final training. '
                             'The v2 handoff already covers its packaged Development input.')
        coverage[name] = {'images': len(vectors), 'provenance': detail}
    return {'status': 'PASS', 'input_sha256': file_sha256(args.eval_input_csv), 'pairs': len(inputs),
            'paintings': inputs.painting.nunique(), 'image_sha256': hashes, 'frozen_coverage': coverage,
            'labels_read': False, 'scoring_performed': False}


def validate_output(directory, input_csv, *, final_manifest=None):
    directory = Path(directory)
    inputs = read_input(input_csv, inference_config()['official_languages'])
    csv = directory / 'predictions.csv'; archive = directory / 'submission.zip'
    predictions = submission.read_predictions(csv, inputs)
    archived = submission.validate_archive(archive, inputs)
    if not predictions[data.KEYS].equals(inputs[data.KEYS]) or not archived.equals(predictions):
        raise ValueError('Submission ordering or archived probabilities differ')
    run = __import__('json').loads((directory / 'run.json').read_text(encoding='utf-8'))
    if run.get('mode') != 'el8' or run.get('input_sha256') != file_sha256(input_csv):
        raise ValueError('Submission prediction manifest has wrong mode or input identity')
    if final_manifest is not None and run.get('final_manifest_sha256') != file_sha256(final_manifest):
        raise ValueError('Submission uses a different final model bundle')
    for name in ('predictions.csv', 'submission.zip'):
        if run['output_sha256'][name] != file_sha256(directory / name):
            raise ValueError('Submission manifest checksum differs: ' + name)
    result = {'status': 'PASS', 'mode': 'el8', 'pairs': len(inputs), 'input_sha256': file_sha256(input_csv),
              'files_sha256': {n: file_sha256(directory / n) for n in ('predictions.csv','submission.zip','run.json')},
              'labels_read': False, 'scoring_performed': False}
    json_output(result, directory / 'validation_report.json')
    for name, digest in result['files_sha256'].items():
        (directory / (name + '.sha256')).write_text(digest + '  ' + name + '\n', encoding='utf-8')
    return result
