"""Standard-library Colab/Kaggle final-refit launcher, training off by default."""

import hashlib
import csv
import io
import importlib.util
import json
import math
import os
import signal
import shutil
import subprocess
import tempfile
import time
import uuid
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


def write_report(path, record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)
    path.with_suffix(path.suffix+'.sha256').write_text(sha256(path)+'  '+path.name+'\n',encoding='utf-8')


def release_runtime(platform, enabled, report, report_path, release=None):
    """Persist the request before unassign(), which may terminate Python itself."""
    def record():
        try:write_report(report_path,report)
        except BaseException as error:print('Cannot persist runtime release state:',str(error),flush=True)
    if platform!='Colab' or not enabled:
        report['runtime_release']={'status':'NOT_REQUESTED','platform':platform}
        record();return
    report['runtime_release']={'status':'REQUESTED_NOT_CONFIRMED','api':'google.colab.runtime.unassign'}
    record()
    try:
        if release is None:
            from google.colab import runtime
            release=runtime.unassign
        release()
    except BaseException as error:
        report['runtime_release']={'status':'FAILED','reason':str(error)}
        record()
        print('Colab release failed; disconnect/delete the runtime manually:',str(error),flush=True)
    else:
        # An API return is observable; physical deallocation is not.
        report['runtime_release']={'status':'API_RETURNED','termination_independently_confirmed':False}
        record()


def persist_export(attempt, run, report, source=None):
    """Publish only a read-back/checksum-verified archive; exclude hot caches/partial fits."""
    attempt=Path(attempt);files={}
    for label,root in (('run',run),('attempt',attempt)):
        if root is None or not Path(root).exists():continue
        for p in Path(root).rglob('*'):
            rel=p.relative_to(root)
            if (not p.is_file() or p.is_symlink() or any(n.startswith('.') for n in rel.parts)
                    or p.suffix in ('.partial','.sqlite3') or 'inference_cache' in rel.parts
                    or p.name in ('final_report.json','final_report.json.sha256','outputs.zip','outputs.sha256')):continue
            if label=='run' and rel.parts[0]=='final':
                if len(rel.parts)<3 or not (Path(root)/'final'/rel.parts[1]/'run.json').is_file():continue
            files[label+'/'+rel.as_posix()]=p
    if source is not None and Path(source).is_file():files['source/source.bundle']=Path(source)
    hashes={n:sha256(p) for n,p in files.items()}
    payload=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode('utf-8')
    hashes['final_report.json']=hashlib.sha256(payload).hexdigest()
    archive=attempt/'outputs.zip';temporary=attempt/'outputs.zip.partial'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        z.writestr('final_report.json',payload)
        z.writestr('export_manifest.json',json.dumps({'format':'el8-complete-output-v2','files_sha256':hashes},indent=2))
    with zipfile.ZipFile(temporary) as z:
        if set(z.namelist())!=set(hashes)|{'export_manifest.json'}:raise ValueError('Output archive coverage differs')
        for name,digest in hashes.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=digest:raise ValueError('Output archive checksum failed: '+name)
    temporary.replace(archive)
    digest=sha256(archive)
    (attempt/'outputs.sha256').write_text(digest+'  outputs.zip\n',encoding='utf-8')
    if sha256(archive)!=digest or (attempt/'outputs.sha256').read_text().split()[0]!=digest:
        raise ValueError('Persistent output read-back failed')
    return {'path':str(archive),'sha256':digest,'files':len(hashes),'persistence_check':'reopened files and SHA-256 on persistent filesystem'}


def run_child(command, logs, env, cwd, timeout, progress_path):
    """Stream stage/seed/budget updates; terminate the process tree on deadline."""
    logs.mkdir(parents=True,exist_ok=True);log=logs/('complete-'+uuid.uuid4().hex[:8]+'.log')
    start=time.monotonic();last=None
    with log.open('w',encoding='utf-8') as out:
        child=subprocess.Popen([str(s) for s in command],cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT,
                               start_new_session=os.name!='nt')
        try:
            while True:
                remaining=timeout-(time.monotonic()-start)
                if remaining<=0:raise TimeoutError('Overall GPU allocation/wall budget exhausted')
                try:code=child.wait(timeout=min(10,remaining));break
                except subprocess.TimeoutExpired:
                    if progress_path.is_file():
                        try:state=json.loads(progress_path.read_text())
                        except (ValueError,OSError):continue
                        key=(state.get('stage'),state.get('seed'),int((time.monotonic()-start)//60))
                        if key!=last:
                            print(f"Stage={state.get('stage')} seed={state.get('seed')} elapsed={time.monotonic()-start:.0f}s remaining={remaining:.0f}s detail={state.get('detail','')}",flush=True);last=key
        except BaseException:
            if os.name=='nt':child.terminate()
            else:os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name=='nt':child.kill()
                else:os.killpg(child.pid,signal.SIGKILL)
                child.wait()
            raise
    return {'exit_code':code,'log':str(log),'seconds':time.monotonic()-start}


def check_persistent_submission(directory,input_csv):
    """Independent standard-library check after inference, before export/SUCCESS."""
    with Path(input_csv).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['painting','language','image']:raise ValueError('Official input schema differs')
        required=[(r['painting'],r['language']) for r in reader]
    directory=Path(directory);raw=(directory/'predictions.csv').read_bytes()
    with zipfile.ZipFile(directory/'submission.zip') as z:
        if z.namelist()!=['predictions.csv'] or z.read('predictions.csv')!=raw:
            raise ValueError('Submission ZIP structure or CSV bytes differ')
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if reader.fieldnames!=['painting','language','p_positive']:raise ValueError('Prediction columns differ')
    rows=list(reader);keys=[(r['painting'],r['language']) for r in rows]
    if keys!=required or len(set(keys))!=len(keys):raise ValueError('Submission pair coverage/order/uniqueness differs')
    if not rows or any(not math.isfinite(float(r['p_positive'])) or not 0<=float(r['p_positive'])<=1 for r in rows):
        raise ValueError('Invalid submission probabilities')
    return {'status':'PASS','pairs':len(rows)}


def launch(archive, expected_sha, local_root, persistent_root, *, platform='Colab', authorize=False,
           max_gpu_hours=0, max_total_wall_hours=0, auto_release=True, require_submission_zip=True,
           eval_input_csv='',eval_data_dir='',started=None):
    """One authorization/run: prepare -> replay/refit -> real ZIP -> export -> release."""
    started=started if started is not None else time.monotonic()
    local=Path(local_root);persistent=Path(persistent_root)
    attempt=persistent/'attempts'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())/uuid.uuid4().hex[:8]
    run=None;source=None;gpu_confirmed=False;stage='AUTHORIZATION';report={'status':'RUNNING','stage':stage,'attempt_dir':str(attempt),
        'require_submission_zip':require_submission_zip,'production_model':'E50+L',
        'runtime_release':{'status':'PENDING'},'development_scored':False,'locked_data_accessed':False,
        'timing_definition':'GPU allocation observed since notebook execution start; prior idle allocation is unknown. Training wall occupancy is reported separately.'}
    def remaining():
        elapsed=time.monotonic()-started
        return min(max_gpu_hours,max_total_wall_hours)*3600-elapsed
    try:
        if platform not in ('Colab','Kaggle'):raise ValueError('Unknown notebook platform')
        attempt.mkdir(parents=True,exist_ok=True)
        if not authorize:
            report['status']='BLOCKED_AUTHORIZATION'
            raise ValueError('Set AUTHORIZE_FINAL_REFIT=True only after approving the displayed policy and budgets; no training was started')
        if any(not math.isfinite(v) or v<=0 for v in (max_gpu_hours,max_total_wall_hours)):
            raise TimeoutError('Finite positive MAX_GPU_HOURS and MAX_TOTAL_WALL_HOURS are required')
        if platform=='Colab' and Path('/content/drive') in local.resolve().parents:
            raise ValueError('LOCAL_ROOT must be local /content storage outside Drive')
        stage='HANDOFF';report['stage']=stage
        local.mkdir(parents=True,exist_ok=True)
        extracted=local/'handoff';manifest=extract_handoff(archive,expected_sha,extracted)
        report.update(handoff_sha256=expected_sha,source_revision=manifest['metadata']['deployment_revision'])
        source=extracted/'source/source.bundle';deploy=local/'deployment';revision=manifest['metadata']['deployment_revision']
        if not deploy.exists():
            subprocess.run(['git','clone','--no-checkout',str(source),str(deploy)],check=True,timeout=max(1,remaining()-60))
            subprocess.run(['git','-C',str(deploy),'checkout','--detach',revision],check=True,timeout=max(1,remaining()-60))
        if (subprocess.check_output(['git','-C',str(deploy),'rev-parse','HEAD'],text=True).strip()!=revision or
                subprocess.check_output(['git','-C',str(deploy),'status','--porcelain'],text=True).strip()):
            raise ValueError('Deployment checkout changed; preserve it and choose a fresh LOCAL_ROOT')
        # Check the evaluation input exists before even installing the environment.
        csv=Path(eval_input_csv) if eval_input_csv else extracted/'evaluation/st3_test.csv'
        data=Path(eval_data_dir) if eval_data_dir else extracted/'evaluation'
        stage='EVALUATION_INPUT';report['stage']=stage
        if require_submission_zip and (not csv.is_file() or not (data/'images').is_dir()):
            report['status']='BLOCKED_EVALUATION_INPUT';raise ValueError('Official input-only CSV/images unavailable; final training blocked')
        helper=load_helper(deploy/'scripts/exp022_notebook.py');helper.REPOSITORY=str(source)
        original_logged=helper.run_logged
        def timed_logged(command,*a,**kw):
            left=remaining()-60
            if left<=0:raise TimeoutError('Budget exhausted during environment preparation')
            kw['timeout']=min(kw.get('timeout',1800),left)
            try:return original_logged(command,*a,**kw)
            except BaseException:
                if remaining()<=60:raise TimeoutError('Environment preparation reached the run budget') from None
                raise
        helper.run_logged=timed_logged
        paths={'local':local/'environment','persistent':persistent/'setup','archive':extracted/'resources/exp022_colab_inputs.zip'}
        for p in (paths['local'],paths['persistent']):p.mkdir(parents=True,exist_ok=True)
        stage='SETUP';report['stage']=stage;print('Stage=SETUP; pinned Python/dependencies; training remains gated',flush=True)
        session=helper.NotebookRun(platform,paths,pixel_cache='ram');session.setup()
        gpu_confirmed=bool(session.environment.get('gpu'))
        if platform=='Colab' and session.environment.get('gpu')!='NVIDIA L4':
            report['status']='BLOCKED_PARITY'
            raise ValueError('Select NVIDIA L4 for strict original-runtime replay; final training was not started')
        # Same-profile completed boundaries resume, never across source/runtime identities.
        run=persistent/'runs'/('final-'+revision[:12]+'-'+session.run_dir.name);run.mkdir(parents=True,exist_ok=True)
        report.update(run_dir=str(run),submission_zip=str(run/'submission/submission.zip'),
                      predictions_csv=str(run/'submission/predictions.csv'),final_bundle=str(run/'bundle'))
        cache=session.portable/'cache'
        for p in (extracted/'inference_cache').rglob('*'):
            if p.is_file():
                target=cache/p.relative_to(extracted/'inference_cache');target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and sha256(target)!=sha256(p):raise ValueError('Existing frozen cache differs')
                if not target.exists():shutil.copyfile(p,target)
        command=[str(session.python),str(deploy/'scripts/run_el8.py')]
        shared=['--cv-run',str(extracted/'historical_cv'),'--portable-dir',str(session.portable),
                '--cache-dir',str(cache),'--model-dir',str(extracted/'production/e50'),
                '--clip-large-model-dir',str(extracted/'production/clip_large'),'--run-dir',str(run)]
        logs=attempt/'logs';stage='ARTIFACT_VALIDATION';report['stage']=stage
        for name in ('validate','forecast','submission-check'):
            if name=='submission-check' and not require_submission_zip and not csv.exists():continue
            args=command+[name,*shared,'--output',str(run/(name+'.json'))]
            if name=='submission-check':args+=['--eval-input-csv',str(csv),'--eval-data-dir',str(data)]
            timed_logged(args,logs,name,cwd=deploy,env=session.env,timeout=300,progress=True)
        print('Frozen fit forecast:',(run/'forecast.json').read_text(),flush=True)
        provenance=run/'provenance';provenance.mkdir(exist_ok=True)
        for stage_name in ('cv','cv_verification','cv_decision'):
            for p in (extracted/'historical_cv'/stage_name).iterdir():
                if p.is_file():
                    dest=provenance/stage_name/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
        for name in ('notebook_runtime.json','environment-lock.txt'):
            p=session.run_dir/name
            if p.exists():shutil.copyfile(p,run/name)
        if remaining()<=120:raise TimeoutError('Insufficient remaining allocation/wall budget before replay and refit')
        stage='COMPLETE';report['stage']=stage
        args=command+['complete',*shared,'--device','cuda','--authorize-final-refit',
             '--policy-sha256',manifest['metadata']['policy_sha256'],
             '--max-gpu-hours',str((max_gpu_hours*3600-(time.monotonic()-started))/3600),
             '--max-total-wall-hours',str((max_total_wall_hours*3600-(time.monotonic()-started))/3600)]
        if csv.exists():args+=['--eval-input-csv',str(csv),'--eval-data-dir',str(data)]
        if not require_submission_zip:args+=['--no-require-submission-zip']
        invocation=uuid.uuid4().hex;args+=['--invocation-id',invocation]
        child_path=run/'child_report.json'
        if child_path.exists():shutil.copyfile(child_path,attempt/'previous_child_report.json')
        write_report(child_path,{'status':'STARTED','invocation_id':invocation})
        result=run_child(args,logs,session.env,deploy,max(1,remaining()-45),run/'progress.json')
        if child_path.is_file():
            child=json.loads(child_path.read_text())
            if child.get('invocation_id')!=invocation:raise RuntimeError('Child completion belongs to a different invocation')
            if child.get('status') in ('STARTED','RUNNING'):
                report['status']='FAILED_TRAINING';raise RuntimeError('Child exited before a terminal report; inspect '+result['log'])
            report.update(child)
        else:raise RuntimeError('Pinned child exited without a completion report; inspect '+result['log'])
        report['child_process']=result
        if report['status']=='SUBMISSION_VALIDATED':
            if result['exit_code']!=0:
                report['status']='FAILED_SUBMISSION';raise RuntimeError('Child exited unsuccessfully despite a submission status')
            # Independent parent-side ZIP/schema/coverage checks run after
            # child completion, before SUCCESS publication.
            from_csv=run/'submission/predictions.csv';from_zip=run/'submission/submission.zip'
            validation=json.loads((run/'submission/validation_report.json').read_text())
            if (not from_csv.is_file() or not from_zip.is_file() or validation.get('status')!='PASS' or
                    sha256(from_csv)!=validation['files_sha256']['predictions.csv'] or
                    sha256(from_zip)!=validation['files_sha256']['submission.zip']):
                report['status']='FAILED_SUBMISSION';raise ValueError('Persistent submission read-back/receipt mismatch')
            report['persistent_submission_check']=check_persistent_submission(run/'submission',csv)
        elif report['status']=='FINAL_BUNDLE_VERIFIED_NO_SUBMISSION' and not require_submission_zip:pass
        elif result['exit_code']==0:raise ValueError('Child returned an invalid success status')
    except BaseException as error:
        if report['status'] in ('RUNNING','SUBMISSION_VALIDATED'):
            if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)):report['status']='STOPPED_BUDGET'
            elif stage=='EVALUATION_INPUT' or ('submission-check' in str(error)):report['status']='BLOCKED_EVALUATION_INPUT'
            elif stage=='ARTIFACT_VALIDATION':report['status']='FAILED_VERIFICATION'
            else:report['status']='FAILED_SUBMISSION' if stage=='COMPLETE' else 'FAILED_SETUP'
        report.update(failed_stage=stage,reason=str(error),error_type=type(error).__name__)
    finally:
        elapsed=time.monotonic()-started
        report.update(wall_seconds=elapsed,observed_gpu_allocation_seconds=elapsed if gpu_confirmed else None,
                      owner_max_gpu_hours=max_gpu_hours,owner_max_total_wall_hours=max_total_wall_hours)
        path=attempt/'final_report.json'
        try:
            if run is not None and (run/'child_report.json').is_file():
                child_path=run/'child_report.json'
                child_path.with_suffix('.json.sha256').write_text(sha256(child_path)+'  child_report.json\n',encoding='utf-8')
            candidate=dict(report)
            if candidate['status']=='SUBMISSION_VALIDATED':candidate['status']='SUCCESS_SUBMISSION_READY'
            candidate['export']={'path':str(attempt/'outputs.zip'),'verification':'All included files and persistent ZIP read-back must pass before publication'}
            receipt=persist_export(attempt,run,candidate,source)
            report=candidate;report['export']=receipt
            elapsed=time.monotonic()-started
            report.update(wall_seconds=elapsed,observed_gpu_allocation_seconds=elapsed if gpu_confirmed else None)
            write_report(path,report)
            print('Final status:',report['status'],'Persistent report:',path,'Export:',receipt['path'],flush=True)
        except BaseException as error:
            report.update(previous_status=report['status'],status='FAILED_EXPORT',export_error=str(error))
            try:write_report(path,report)
            except BaseException as persist_error:print('Cannot persist final report:',str(persist_error),flush=True)
        finally:
            try:release_runtime(platform,auto_release,report,path)
            except BaseException as error:print('Runtime release/reporting error:',str(error),flush=True)
    return report
