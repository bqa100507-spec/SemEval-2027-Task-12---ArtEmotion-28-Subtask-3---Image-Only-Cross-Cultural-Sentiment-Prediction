"""One private ZIP for owner-authorized final-refit preparation (no training)."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from st3.baseline import file_sha256
from st3.el8_artifacts import cv_inventory, export_bundle
from st3.el8_submission import readiness
from st3.e50_inference import safe_image_path


def package(args):
    evidence = cv_inventory(args.cv_run)
    if subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'scripts', 'configs'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit deployment code/configuration before packaging a pinned handoff')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    expected_input = 'bbedbe2ad7e426040aebb63366ea5978e03d6483503b8ac9c88af14238a3d156'
    if file_sha256(args.input_zip) != expected_input:
        raise ValueError('Original portable training ZIP differs')
    submission_input=readiness(args)
    release=json.loads((args.model_dir/'manifest.json').read_text(encoding='utf-8'))
    if submission_input['input_sha256']!=release['dev_input_sha256']:
        raise ValueError('Automatic v2 packaging accepts only the released official Development input')
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        temp = Path(temp); resource = temp / 'resources'; resource.mkdir()
        try:os.link(args.input_zip, resource / 'exp022_colab_inputs.zip')
        except OSError:shutil.copyfile(args.input_zip, resource / 'exp022_colab_inputs.zip')
        source = temp / 'source'; source.mkdir()
        subprocess.run(['git', 'bundle', 'create', str(source / 'source.bundle'), 'HEAD'], cwd=ROOT, check=True)
        evaluation=temp/'evaluation';evaluation.mkdir();(evaluation/'images').mkdir()
        shutil.copyfile(args.eval_input_csv,evaluation/'st3_test.csv')
        for name in submission_input['image_sha256']:
            target=evaluation/'images'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(safe_image_path(args.eval_data_dir,name),target)
        (evaluation/'provenance.json').write_text(json.dumps({**submission_input,'phase':'development',
            'official_input_only':True,'gold_included':False,
            'source':'https://github.com/artemotion-28/artemotion-28/tree/cf7cc444cf56617f3b8417497c6930ad172c5c99',
            'use':'Private competition research; WikiArt artwork rights remain with their holders'},indent=2),encoding='utf-8')
        inference_cache=temp/'inference_cache';inference_cache.mkdir()
        for folder in (args.model_dir,args.clip_large_model_dir):
            for entry in json.loads((folder/'manifest.json').read_text(encoding='utf-8'))['legacy'].values():
                old=args.cache_dir/entry['file']
                if file_sha256(old)!=entry['sha256']:raise ValueError('Historical frozen cache changed')
                target=inference_cache/entry['file'];target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(old,target)
        return export_bundle(args.output, {'historical_cv': args.cv_run, 'resources': resource, 'source': source,
                             'production/e50': args.model_dir, 'production/clip_large': args.clip_large_model_dir,
                             'evaluation':evaluation,'inference_cache':inference_cache},
                             {'historical_cv': 'historical_cv', 'resources': 'permitted_training_inputs',
                              'source': 'immutable_deployment_source', 'production/e50': 'production_e50',
                              'production/clip_large': 'production_clip_large','evaluation':'official_development_input_only',
                              'inference_cache':'verified_frozen_representations'},
                             {'deployment_revision': revision, 'training_revision': evidence['source_revision'],
                              'policy_sha256': hashlib.sha256((ROOT / 'configs/el8_final_refit.json').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                              'input_zip_sha256': expected_input, 'checkpoints': 60,
                              'owner_authorization_included': False,'submission_input':{'phase':'development',
                              'pairs':submission_input['pairs'],'images':len(submission_input['image_sha256']),
                              'input_sha256':submission_input['input_sha256'],'gold_included':False}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag, default in (('cv-run', 'runs/exp022_colab_verified/c30c4ad20092b87c'),
                          ('input-zip', 'runs/exp022_visual_lora/exp022_colab_inputs.zip'),
                          ('model-dir', 'runs/e50_inference_models'), ('clip-large-model-dir', 'runs/e50l_inference_models'),
                          ('eval-input-csv','data/upstream/phase_inputs/dev/st3_test.csv'),('eval-data-dir','data/upstream'),
                          ('cache-dir','cache'),('output', 'runs/task023_delivery/el8_refit_handoff_v2.zip')):
        p.add_argument('--' + flag, type=Path, default=ROOT / default)
    a = p.parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(package(a), indent=2))
