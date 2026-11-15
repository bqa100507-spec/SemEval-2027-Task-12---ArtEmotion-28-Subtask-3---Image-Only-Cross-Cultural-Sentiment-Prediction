"""Generate clean, pinned one-action final-refit notebooks from a handoff receipt."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(revision, digest):
    helper = subprocess.check_output(['git', 'show', revision + ':scripts/el8_cloud.py'], cwd=ROOT)
    helper_sha = hashlib.sha256(helper).hexdigest()
    for platform in ('Colab', 'Kaggle'):
        def cell(kind, source, name):
            base = {'cell_type': kind, 'id': name, 'metadata': {}, 'source': source.splitlines(keepends=True)}
            if kind == 'code':base.update(execution_count=None, outputs=[])
            return base
        config = f'''PLATFORM = {platform!r}
HANDOFF_ZIP = {('/content/drive/MyDrive/el8_refit_handoff.zip' if platform == 'Colab' else '/kaggle/input/your-private-dataset/el8_refit_handoff.zip')!r}
HANDOFF_SHA256 = {digest!r}
IMPLEMENTATION_REVISION = {revision!r}
HELPER_SHA256 = {helper_sha!r}
LOCAL_ROOT = {('/content/el8_final' if platform == 'Colab' else '/kaggle/temp/el8_final')!r}
PERSISTENT_ROOT = {('/content/drive/MyDrive/el8_final_runs' if platform == 'Colab' else '/kaggle/working/el8_final_runs')!r}
# Approval includes configs/el8_final_refit.json: fold=0 hashed 10% stopping split,
# original 100 epoch maximum / patience 10 / fresh full-1335-painting refit.
AUTHORIZE_FINAL_REFIT = False
MAX_GPU_HOURS = 0.0
'''
        body = '''import hashlib, importlib.util, json, subprocess, zipfile
from pathlib import Path

if PLATFORM == 'Colab':
    from google.colab import drive
    drive.mount('/content/drive')

archive = Path(HANDOFF_ZIP)
if not archive.is_file():
    raise FileNotFoundError(f'Upload the private el8_refit_handoff.zip and check HANDOFF_ZIP: {archive}')
digest = hashlib.sha256()
with archive.open('rb') as handle:
    for chunk in iter(lambda:handle.read(1024*1024), b''):digest.update(chunk)
if digest.hexdigest() != HANDOFF_SHA256:
    raise RuntimeError('Handoff ZIP hash mismatch. Use the producer receipt; never replace the expected hash.')
local = Path(LOCAL_ROOT);local.mkdir(parents=True, exist_ok=True)
bundle = local/'bootstrap-source.bundle'
with zipfile.ZipFile(archive) as z:
    manifest = json.loads(z.read('artifact_manifest.json'))
    if manifest['metadata']['deployment_revision'] != IMPLEMENTATION_REVISION:
        raise RuntimeError('Handoff deployment revision differs from the notebook pin.')
    raw = z.read('source/source.bundle')
    if hashlib.sha256(raw).hexdigest() != manifest['files_sha256']['source/source.bundle']:
        raise RuntimeError('Source bundle hash mismatch.')
    if bundle.exists() and bundle.read_bytes() != raw:
        raise RuntimeError('Existing source bundle differs; preserve it and choose a fresh LOCAL_ROOT.')
    bundle.write_bytes(raw)
repo = local/'deployment'
if not repo.exists():
    subprocess.run(['git','clone','--no-checkout',str(bundle),str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach',IMPLEMENTATION_REVISION],check=True)
if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() != IMPLEMENTATION_REVISION:
    raise RuntimeError('Source checkout differs; choose a fresh LOCAL_ROOT.')
helper = repo/'scripts/el8_cloud.py'
if hashlib.sha256(helper.read_bytes()).hexdigest() != HELPER_SHA256:
    raise RuntimeError('Cloud launcher hash mismatch; keep the published pins.')
spec = importlib.util.spec_from_file_location('el8_cloud_launch',helper)
workflow = importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
RESULT = workflow.launch(archive,HANDOFF_SHA256,LOCAL_ROOT,PERSISTENT_ROOT,
                        platform=PLATFORM,authorize=AUTHORIZE_FINAL_REFIT,max_gpu_hours=MAX_GPU_HOURS)
print(json.dumps(RESULT,indent=2))
'''
        intro = '''# EXP-022 EL8 final-model handoff

**E50+L remains production. These notebooks do not score Development, open locked labels or upload to Codabench.**

Upload the private `el8_refit_handoff.zip` once (Drive for Colab; private Dataset for Kaggle). Select a CUDA GPU, preferably the original **L4**, and enable Internet for the pinned Python/dependencies. Set the input path below and Run all. Preparation validates the recovered 60 CV models and prints the cost forecast. Training is **off by default**.

After approving the final-fit policy and GPU budget, set `AUTHORIZE_FINAL_REFIT=True` and a positive `MAX_GPU_HOURS` (for example 1.5 hours), then Run all once. This automatically runs strict CV replay, memory/parity preflight, timing pilot, the three missing final seeds, checkpoint replay, image-only smoke and full export. A parity failure stops before final training. **Cross-runtime parity has failed on the local RTX 4060; L4 replay of this deployment implementation remains unverified.**

The new final policy uses original EXP-022 loss, learning rates, LoRA, seeds and nested stopping/refit; it specifies a newly reviewed fold=0 hash for the full-population inner split. No CV model is treated as a final model.

Logs, completed seeds and export receipts persist under `PERSISTENT_ROOT`. On interruption Run all again with the same paths and runtime; completed seeds are checked and reused, while an interrupted current seed restarts. Colab Drive survives runtime deletion. **Kaggle working files require Save Version/output download before session loss; a completed export alone does not guarantee persistence outside the Kaggle session.** Final export ZIP + `.sha256` supports offline import without the old kernel. Setup may take several minutes; no dependency versions are substituted.
'''
        notebook = {'nbformat':4, 'nbformat_minor':5, 'metadata':{
            'kernelspec':{'name':'python3','display_name':'Python 3'},
            'language_info':{'name':'python'}, 'el8':{'deployment_revision':revision,'handoff_sha256':digest,
                'default_training':False,'platform':platform}},
            'cells':[cell('markdown',intro,'instructions'),cell('code',config,'configuration'),cell('code',body,'prepare-or-authorized-run')]}
        (ROOT/'notebooks'/f'EL8_FinalRefit_{platform}.ipynb').write_bytes((json.dumps(notebook,indent=1)+'\n').encode('utf-8'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--revision',required=True);p.add_argument('--sha256',required=True)
    a=p.parse_args();build(a.revision,a.sha256)
