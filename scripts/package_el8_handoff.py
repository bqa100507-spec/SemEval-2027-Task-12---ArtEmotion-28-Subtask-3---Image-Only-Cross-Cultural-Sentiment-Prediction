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


def package(args):
    evidence = cv_inventory(args.cv_run)
    if subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'scripts', 'configs'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit deployment code/configuration before packaging a pinned handoff')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    expected_input = 'bbedbe2ad7e426040aebb63366ea5978e03d6483503b8ac9c88af14238a3d156'
    if file_sha256(args.input_zip) != expected_input:
        raise ValueError('Original portable training ZIP differs')
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        temp = Path(temp); resource = temp / 'resources'; resource.mkdir()
        try:os.link(args.input_zip, resource / 'exp022_colab_inputs.zip')
        except OSError:shutil.copyfile(args.input_zip, resource / 'exp022_colab_inputs.zip')
        source = temp / 'source'; source.mkdir()
        subprocess.run(['git', 'bundle', 'create', str(source / 'source.bundle'), 'HEAD'], cwd=ROOT, check=True)
        return export_bundle(args.output, {'historical_cv': args.cv_run, 'resources': resource, 'source': source,
                             'production/e50': args.model_dir, 'production/clip_large': args.clip_large_model_dir},
                             {'historical_cv': 'historical_cv', 'resources': 'permitted_training_inputs',
                              'source': 'immutable_deployment_source', 'production/e50': 'production_e50',
                              'production/clip_large': 'production_clip_large'},
                             {'deployment_revision': revision, 'training_revision': evidence['source_revision'],
                              'policy_sha256': hashlib.sha256((ROOT / 'configs/el8_final_refit.json').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                              'input_zip_sha256': expected_input, 'checkpoints': 60,
                              'owner_authorization_included': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag, default in (('cv-run', 'runs/exp022_colab_verified/c30c4ad20092b87c'),
                          ('input-zip', 'runs/exp022_visual_lora/exp022_colab_inputs.zip'),
                          ('model-dir', 'runs/e50_inference_models'), ('clip-large-model-dir', 'runs/e50l_inference_models'),
                          ('output', 'runs/task023_delivery/el8_refit_handoff.zip')):
        p.add_argument('--' + flag, type=Path, default=ROOT / default)
    a = p.parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(package(a), indent=2))
