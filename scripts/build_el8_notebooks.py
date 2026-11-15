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
        config = f"""PLATFORM = {platform!r}
HANDOFF_ZIP = {('/content/drive/MyDrive/el8_refit_handoff_v2.zip' if platform == 'Colab' else '/kaggle/input/your-private-dataset/el8_refit_handoff_v2.zip')!r}
HANDOFF_SHA256 = {digest!r}
IMPLEMENTATION_REVISION = {revision!r}
HELPER_SHA256 = {helper_sha!r}
LOCAL_ROOT = {('/content/el8_final_v2' if platform == 'Colab' else '/kaggle/temp/el8_final_v2')!r}
PERSISTENT_ROOT = {('/content/drive/MyDrive/el8_final_runs_v2' if platform == 'Colab' else '/kaggle/working/el8_final_runs_v2')!r}
# ONE authorization section. Approval covers configs/el8_final_refit.json:
# 1335 paintings; seeds 2027/2028/2029; original LoRA/loss/optimizer/precision;
# fold=0 hashed 10% stopping split; BCE, max100, patience10; fresh full refit.
AUTHORIZE_FINAL_REFIT = False  # Set True after reviewing this policy and caps.
MAX_GPU_HOURS = 1.5           # Observed GPU allocation since Run all, including setup.
MAX_TOTAL_WALL_HOURS = 2.0
AUTO_RELEASE_COLAB = True
REQUIRE_SUBMISSION_ZIP = True
EVAL_INPUT_CSV = ""  # Blank uses the packaged official Development CSV (849 pairs).
EVAL_DATA_DIR = ""   # Blank uses its packaged 85 artwork images. No gold labels.
"""
        body = r"""import hashlib, importlib.util, json, subprocess, time, traceback, uuid, zipfile
from pathlib import Path

NOTEBOOK_STARTED = time.monotonic()
LAUNCH_ENTERED = False
LAUNCH_COMPLETED = False
RESULT = None
bootstrap_root = None
release_platform = globals().get('PLATFORM', 'Colab')
release_enabled = globals().get('AUTO_RELEASE_COLAB', True)
try:
    required = ('PLATFORM','HANDOFF_ZIP','HANDOFF_SHA256','IMPLEMENTATION_REVISION','HELPER_SHA256',
                'LOCAL_ROOT','PERSISTENT_ROOT','AUTHORIZE_FINAL_REFIT','MAX_GPU_HOURS',
                'MAX_TOTAL_WALL_HOURS','AUTO_RELEASE_COLAB','REQUIRE_SUBMISSION_ZIP',
                'EVAL_INPUT_CSV','EVAL_DATA_DIR')
    missing = [n for n in required if n not in globals()]
    if missing:raise RuntimeError('Configuration is missing: '+', '.join(missing)+'. Use Run all.')
    if PLATFORM == 'Colab':
        from google.colab import drive
        drive.mount('/content/drive')
    bootstrap_root = Path(PERSISTENT_ROOT)/'bootstrap'/uuid.uuid4().hex[:12]
    bootstrap_root.mkdir(parents=True,exist_ok=True)
    if not AUTHORIZE_FINAL_REFIT:
        raise RuntimeError('Approval is OFF. Review the policy/budgets in Configuration and set AUTHORIZE_FINAL_REFIT=True before your one final Run all.')
    archive = Path(HANDOFF_ZIP)
    if not archive.is_file():
        raise FileNotFoundError(f'Upload el8_refit_handoff_v2.zip to MyDrive and check HANDOFF_ZIP: {archive}')
    digest = hashlib.sha256()
    with archive.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024), b''):digest.update(chunk)
    if digest.hexdigest() != HANDOFF_SHA256:
        raise RuntimeError('Handoff ZIP hash mismatch. Keep the published checksum and replace the ZIP.')
    local = Path(LOCAL_ROOT)
    if PLATFORM=='Colab' and Path('/content/drive') in local.resolve().parents:
        raise RuntimeError('LOCAL_ROOT must be under /content outside Drive.')
    local.mkdir(parents=True,exist_ok=True)
    bundle = local/'bootstrap-source.bundle'
    with zipfile.ZipFile(archive) as z:
        if z.namelist().count('artifact_manifest.json')!=1 or z.namelist().count('source/source.bundle')!=1:
            raise RuntimeError('Ambiguous/missing handoff source or manifest.')
        manifest = json.loads(z.read('artifact_manifest.json'))
        if manifest['metadata']['deployment_revision'] != IMPLEMENTATION_REVISION:
            raise RuntimeError('Handoff source differs from notebook pin.')
        raw = z.read('source/source.bundle')
        if hashlib.sha256(raw).hexdigest() != manifest['files_sha256']['source/source.bundle']:
            raise RuntimeError('Source bundle hash mismatch.')
        if bundle.exists() and bundle.read_bytes()!=raw:
            raise RuntimeError('Local source bundle differs; preserve it and choose a fresh LOCAL_ROOT.')
        bundle.write_bytes(raw)
    repo = local/'deployment'
    if not repo.exists():
        subprocess.run(['git','clone','--no-checkout',str(bundle),str(repo)],check=True,timeout=300)
        subprocess.run(['git','-C',str(repo),'checkout','--detach',IMPLEMENTATION_REVISION],check=True,timeout=60)
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=IMPLEMENTATION_REVISION:
        raise RuntimeError('Source checkout differs; choose a fresh LOCAL_ROOT.')
    helper = repo/'scripts/el8_cloud.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=HELPER_SHA256:
        raise RuntimeError('Cloud launcher checksum differs; keep the published source pin.')
    spec = importlib.util.spec_from_file_location('el8_cloud_launch',helper)
    workflow = importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
    LAUNCH_ENTERED = True
    RESULT = workflow.launch(archive,HANDOFF_SHA256,LOCAL_ROOT,PERSISTENT_ROOT,
        platform=PLATFORM,authorize=AUTHORIZE_FINAL_REFIT,max_gpu_hours=MAX_GPU_HOURS,
        max_total_wall_hours=MAX_TOTAL_WALL_HOURS,auto_release=AUTO_RELEASE_COLAB,
        require_submission_zip=REQUIRE_SUBMISSION_ZIP,eval_input_csv=EVAL_INPUT_CSV,
        eval_data_dir=EVAL_DATA_DIR,started=NOTEBOOK_STARTED)
    LAUNCH_COMPLETED = True
    print(json.dumps(RESULT,indent=2))
except BaseException as error:
    RESULT = {'status':'FAILED_SETUP','reason':str(error),'failed_stage':'NOTEBOOK_BOOTSTRAP',
              'runtime_release':{'status':'PENDING'}}
    print('STOP:',str(error))
    if bootstrap_root is not None:
        try:
            (bootstrap_root/'bootstrap.log').write_text(traceback.format_exc(),encoding='utf-8')
            (bootstrap_root/'final_report.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
            with zipfile.ZipFile(bootstrap_root/'bootstrap_failure.zip','w') as z:
                for n in ('bootstrap.log','final_report.json'):z.write(bootstrap_root/n,n)
            archive_path = bootstrap_root/'bootstrap_failure.zip'
            (bootstrap_root/'bootstrap_failure.sha256').write_text(hashlib.sha256(archive_path.read_bytes()).hexdigest()+'  bootstrap_failure.zip\n',encoding='utf-8')
            print('Bootstrap diagnostics:',bootstrap_root)
        except BaseException as persist_error:print('Cannot persist bootstrap diagnostics:',str(persist_error))
finally:
    # Covers Drive mount, checksum, source/bootstrap failures before launch().
    # launch() owns export -> verify -> release after it has been entered.
    if not LAUNCH_COMPLETED and release_platform=='Colab' and release_enabled:
        if RESULT is None:RESULT={'status':'FAILED_SETUP'}
        RESULT['runtime_release']={'status':'REQUESTED_NOT_CONFIRMED'}
        def persist_release():
            if bootstrap_root is not None:
                try:(bootstrap_root/'final_report.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
                except BaseException as error:print('Cannot record release state:',str(error))
        persist_release()
        try:
            from google.colab import runtime
            runtime.unassign()
        except BaseException as error:
            RESULT['runtime_release']={'status':'FAILED','reason':str(error)}
            print('Runtime release failed:',str(error));persist_release()
        else:
            RESULT['runtime_release']={'status':'API_RETURNED','termination_independently_confirmed':False}
            persist_release()
"""
        intro = """# EL8: one authorized Colab run to a validated submission

Upload **el8_refit_handoff_v2.zip** to MyDrive, select **NVIDIA L4**, enable Internet, review Configuration and set `AUTHORIZE_FINAL_REFIT=True`. Then choose **Runtime -> Run all ONCE** and grant the Drive mount. No separate preparation, replay, Verify, Export, Submission or Cleanup action is required.

The private ZIP includes the official **Development input only (849 pairs / 85 images)** plus the verified frozen caches. No Development gold is included or read; no scoring or Codabench upload occurs. Blank evaluation paths select this packaged set. This produces a Development submission, not an unreleased Evaluation-phase submission.

The run checks all 60 CV checkpoints and input availability, performs strict image-backed L4 replay, then preflight/benchmark, the missing final seeds 2027/2028/2029, final verification, EL8 inference and exact ZIP validation. **A strict parity failure stops before final training.** Cross-runtime RTX4060 replay failed; no new L4 execution is claimed. EXP-022 is closed; its results and E50+L production remain unchanged.

Approval covers the unchanged deployment policy: 1335 permitted paintings, original LoRA/loss/optimizer/precision, fold=0 hashed 10% stopping split, BCE/max100/patience10, and fresh full-population refit. The measured historical training forecast is 0.51-0.77 GPU-hours excluding setup/replay/export. The GPU cap measures observed allocation since this Run all, including setup; prior idle allocation is unobservable. Training-stage occupancy and total wall time are recorded separately; caps stop safely and retain completed seeds.

Results are under `PERSISTENT_ROOT`: `runs/.../submission/submission.zip`, predictions, verified bundle/checkpoints, logs and manifests; `attempts/.../final_report.json`, `outputs.zip` and checksums. Only **SUCCESS_SUBMISSION_READY** means model, inference, ZIP and persistent read-back all passed. On failure, inspect the exact stage/reason. Completed same-identity seeds resume if the run is repeated; an interrupted current seed restarts. Drive stores each completed checkpoint continuously. Abrupt VM loss cannot execute cleanup; completed Drive files survive even if the last ZIP export did not finish.

With `AUTO_RELEASE_COLAB=True`, export/verification and release run in outer `finally` handlers on success or failure. A failed release is recorded; `API_RETURNED` confirms the API returned, not independently observed physical deallocation. If Drive cannot mount, persistent export is unavailable and the notebook prints the error before release. Kaggle never calls Colab APIs; save a Version/download its outputs before session loss.
"""
        notebook = {'nbformat':4, 'nbformat_minor':5, 'metadata':{
            'kernelspec':{'name':'python3','display_name':'Python 3'},
            'language_info':{'name':'python'}, 'el8':{'deployment_revision':revision,'handoff_sha256':digest,
                'default_training':False,'platform':platform}},
            'cells':[cell('markdown',intro,'instructions'),cell('code',config,'configuration'),cell('code',body,'prepare-or-authorized-run')]}
        (ROOT/'notebooks'/f'EL8_FinalRefit_{platform}.ipynb').write_bytes((json.dumps(notebook,indent=1)+'\n').encode('utf-8'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--revision',required=True);p.add_argument('--sha256',required=True)
    a=p.parse_args();build(a.revision,a.sha256)
