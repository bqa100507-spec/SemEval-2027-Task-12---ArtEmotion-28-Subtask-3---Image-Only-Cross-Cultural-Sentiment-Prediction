"""One-run deployment tests with CPU fixtures and entirely mocked training/GPU."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from st3 import submission
from st3.baseline import file_sha256
from st3.el8_complete import complete
from st3.el8_submission import validate_output
from st3.polarity_workflow import BudgetStop, StageBlocked

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('one_run_cloud',ROOT/'scripts/el8_cloud.py')
cloud=importlib.util.module_from_spec(spec);spec.loader.exec_module(cloud)


def input_fixture(tmp_path):
    frame=pd.DataFrame({'painting':['a','a'],'language':['English','Vietnamese'],'image':['new.png','new.png']})
    path=tmp_path/'official_input.csv';frame.to_csv(path,index=False)
    return path,frame


def prediction_fixture(directory,input_csv,frame,final_manifest):
    directory.mkdir(parents=True)
    pred=frame[['painting','language']].assign(p_positive=[.2,.8])
    pred.to_csv(directory/'predictions.csv',index=False,lineterminator='\n')
    submission.pack(directory/'predictions.csv',directory/'submission.zip',frame)
    (directory/'run.json').write_text(json.dumps({'mode':'el8','input_sha256':file_sha256(input_csv),
        'final_manifest_sha256':file_sha256(final_manifest),
        'output_sha256':{n:file_sha256(directory/n) for n in ('predictions.csv','submission.zip')}}))


@pytest.mark.parametrize('failure,expected',[(None,'SUBMISSION_VALIDATED'),('input','BLOCKED_EVALUATION_INPUT'),
    ('parity','BLOCKED_PARITY'),('training','FAILED_TRAINING'),('verification','FAILED_VERIFICATION'),
    ('submission','FAILED_SUBMISSION'),('invalid_zip','FAILED_SUBMISSION'),('budget','STOPPED_BUDGET')])
def test_child_success_and_all_failure_stages(tmp_path,monkeypatch,failure,expected):
    import st3.el8_complete as module
    import st3.el8_inference as inference
    path,frame=input_fixture(tmp_path);run=tmp_path/'run'
    args=SimpleNamespace(run_dir=run,max_gpu_hours=1.5,max_total_wall_hours=2.,cv_run=tmp_path/'cv',
        require_submission_zip=True,eval_input_csv=path,eval_data_dir=tmp_path,device='cpu')
    monkeypatch.setattr(module,'require_authorization',lambda a:None)
    monkeypatch.setattr(module,'forecast',lambda p:{'training_hours_p90_with_25_percent_margin':.77})
    calls=[]
    def ready(a):
        if failure=='input':raise ValueError('missing required official image')
        return {'status':'PASS','input_sha256':file_sha256(path)}
    monkeypatch.setattr(module,'readiness',ready)
    def fit(a):
        calls.append('refit');a._progress(stage='PARITY')
        if (run/'bundle').exists():return {'status':'FINAL_BUNDLE_VERIFIED'}
        if failure=='parity':raise StageBlocked('strict numerical parity mismatch')
        a._progress(stage='TRAINING',seed=2027)
        calls.append('fit_seed')
        if failure=='training':raise ValueError('synthetic training failure')
        if failure=='budget':raise BudgetStop('synthetic exhausted budget')
        a._progress(stage='FINAL_VERIFICATION')
        if failure=='verification':raise ValueError('checkpoint checksum mismatch')
        (run/'bundle').mkdir();(run/'bundle/final_manifest.json').write_text('{}')
        return {'status':'FINAL_BUNDLE_VERIFIED'}
    monkeypatch.setattr(module,'full_refit',fit)
    monkeypatch.setattr(inference,'final_bundle',lambda *a:None)
    def predict(a,budget):
        calls.append('predict')
        if failure=='submission':raise ValueError('synthetic image-only inference failure')
        prediction_fixture(a.output_dir,path,frame,run/'bundle/final_manifest.json')
        if failure=='invalid_zip':(a.output_dir/'submission.zip').write_bytes(b'invalid zip')
    monkeypatch.setattr(module,'predict_official',predict)
    result=complete(args)
    assert result['status']==expected
    assert json.loads((run/'child_report.json').read_text())['status']==expected
    assert 'predict' not in calls if failure in ('input','parity','training','verification','budget') else True
    if failure=='input':assert not calls
    if failure in ('submission','invalid_zip'):assert (run/'bundle/final_manifest.json').exists()
    if failure is None:
        assert (run/'submission/submission.zip').is_file()
        assert result['submission']['pairs']==2
        assert complete(args)['status']=='SUBMISSION_VALIDATED'
        assert calls.count('predict')==1
        assert calls.count('fit_seed')==1


def test_budget_blocks_before_refit(tmp_path,monkeypatch):
    import st3.el8_complete as module
    path,_=input_fixture(tmp_path)
    args=SimpleNamespace(run_dir=tmp_path/'run',max_gpu_hours=.001,max_total_wall_hours=.001,
        cv_run=tmp_path/'cv',require_submission_zip=True,eval_input_csv=path)
    monkeypatch.setattr(module,'require_authorization',lambda a:None)
    monkeypatch.setattr(module,'readiness',lambda a:{'status':'PASS'})
    monkeypatch.setattr(module,'forecast',lambda p:{'training_hours_p90_with_25_percent_margin':.77})
    monkeypatch.setattr(module,'full_refit',lambda a:pytest.fail('Must not fit with insufficient remaining budget'))
    assert complete(args)['status']=='STOPPED_BUDGET'


def test_persistent_export_and_failed_release(tmp_path):
    run=tmp_path/'run';run.mkdir();(run/'predictions.csv').write_text('fixture')
    attempt=tmp_path/'attempt';attempt.mkdir()
    report={'status':'SUCCESS_SUBMISSION_READY'}
    receipt=cloud.persist_export(attempt,run,report)
    assert receipt['sha256']==cloud.sha256(attempt/'outputs.zip')
    calls=[]
    def broken():calls.append('release');raise RuntimeError('API unavailable')
    cloud.release_runtime('Colab',True,report,attempt/'final_report.json',broken)
    assert calls==['release'] and report['runtime_release']['status']=='FAILED'
    assert json.loads((attempt/'final_report.json').read_text())['runtime_release']['reason']=='API unavailable'
    cloud.release_runtime('Kaggle',True,report,attempt/'final_report.json',lambda:pytest.fail('Kaggle cannot call Colab API'))


def test_release_attempted_when_persistence_is_unavailable(tmp_path,monkeypatch):
    monkeypatch.setattr(cloud,'write_report',lambda *a:(_ for _ in ()).throw(OSError('disk unavailable')))
    calls=[];record={}
    cloud.release_runtime('Colab',True,record,tmp_path/'report.json',lambda:calls.append('called'))
    assert calls==['called'] and record['runtime_release']['status']=='API_RETURNED'


@pytest.mark.parametrize('failure', ['parity','training','submission','timeout','export','invalid_zip','early_exit','stale_report',None])
def test_outer_always_exports_and_releases(tmp_path,monkeypatch,failure):
    """No Python install/encoder/fit. Tests the real parent state machine."""
    local=tmp_path/'local';deploy=local/'deployment';deploy.mkdir(parents=True)
    extracted=local/'handoff';(extracted/'evaluation/images').mkdir(parents=True)
    _,input_rows=input_fixture(tmp_path)
    input_rows.to_csv(extracted/'evaluation/st3_test.csv',index=False)
    for stage in ('cv','cv_verification','cv_decision'):
        folder=extracted/'historical_cv'/stage;folder.mkdir(parents=True);(folder/'run.json').write_text('{}')
    revision='a'*40
    monkeypatch.setattr(cloud,'extract_handoff',lambda *a:{'metadata':{'deployment_revision':revision,'policy_sha256':'policy'}})
    monkeypatch.setattr(cloud.subprocess,'check_output',lambda a,**k:revision if 'rev-parse' in a else '')
    calls=[]
    class Session:
        def __init__(self,*a,**k):
            self.python=tmp_path/'python';self.portable=local/'portable';self.env={};self.environment={'gpu':'NVIDIA L4'}
            self.run_dir=tmp_path/'profile'
        def setup(self):pass
    def logged(command,*a,**kw):
        p=Path(command[command.index('--output')+1]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    monkeypatch.setattr(cloud,'load_helper',lambda p:SimpleNamespace(NotebookRun=Session,run_logged=logged))
    def child(command,*a):
        calls.append('child')
        if failure=='timeout':raise TimeoutError('synthetic timeout')
        if failure in ('early_exit','stale_report'):
            run=Path(command[command.index('--run-dir')+1])
            if failure=='stale_report':(run/'child_report.json').write_text(json.dumps({'status':'SUBMISSION_VALIDATED','invocation_id':'old'}))
            return {'exit_code':1,'log':'fixture'}
        run=Path(command[command.index('--run-dir')+1]);status={None:'SUBMISSION_VALIDATED','parity':'BLOCKED_PARITY',
              'training':'FAILED_TRAINING','submission':'FAILED_SUBMISSION','export':'SUBMISSION_VALIDATED',
              'invalid_zip':'SUBMISSION_VALIDATED'}[failure]
        if status=='SUBMISSION_VALIDATED':
            p=run/'submission';p.mkdir()
            input_rows[['painting','language']].assign(p_positive=[.2,.8]).to_csv(p/'predictions.csv',index=False,lineterminator='\n')
            with cloud.zipfile.ZipFile(p/'submission.zip','w') as z:z.write(p/'predictions.csv','predictions.csv')
            if failure=='invalid_zip':(p/'submission.zip').write_bytes(b'invalid')
            (p/'validation_report.json').write_text(json.dumps({'status':'PASS','files_sha256':
                {n:cloud.sha256(p/n) for n in ('predictions.csv','submission.zip')}}))
        invocation=command[command.index('--invocation-id')+1]
        (run/'child_report.json').write_text(json.dumps({'status':status,'invocation_id':invocation}))
        return {'exit_code':0 if status=='SUBMISSION_VALIDATED' else 1,'log':'fixture'}
    monkeypatch.setattr(cloud,'run_child',child)
    def export(*a):
        calls.append('export')
        if failure=='export':raise OSError('synthetic export failure')
        return {'path':'fixture.zip','sha256':'fixture'}
    monkeypatch.setattr(cloud,'persist_export',export)
    monkeypatch.setattr(cloud,'release_runtime',lambda *a:calls.append('release'))
    report=cloud.launch(tmp_path/'input.zip','hash',local,tmp_path/'persistent',authorize=True,
                        max_gpu_hours=1.5,max_total_wall_hours=2.)
    assert calls==['child','export','release']
    expected={None:'SUCCESS_SUBMISSION_READY','parity':'BLOCKED_PARITY','training':'FAILED_TRAINING',
              'submission':'FAILED_SUBMISSION','timeout':'STOPPED_BUDGET','export':'FAILED_EXPORT',
              'invalid_zip':'FAILED_SUBMISSION','early_exit':'FAILED_TRAINING','stale_report':'FAILED_SUBMISSION'}[failure]
    assert report['status']==expected


def test_required_input_missing_prevents_setup_and_training(tmp_path,monkeypatch):
    local=tmp_path/'local';(local/'deployment').mkdir(parents=True);calls=[];revision='a'*40
    monkeypatch.setattr(cloud,'extract_handoff',lambda *a:{'metadata':{'deployment_revision':revision}})
    monkeypatch.setattr(cloud.subprocess,'check_output',lambda a,**k:revision if 'rev-parse' in a else '')
    monkeypatch.setattr(cloud,'load_helper',lambda p:pytest.fail('Missing input must stop before setup'))
    monkeypatch.setattr(cloud,'persist_export',lambda *a:{'path':'fixture'})
    monkeypatch.setattr(cloud,'release_runtime',lambda *a:calls.append('release'))
    result=cloud.launch(tmp_path/'zip','sha',local,tmp_path/'persistent',authorize=True,max_gpu_hours=1.5,max_total_wall_hours=2.)
    assert result['status']=='BLOCKED_EVALUATION_INPUT' and calls==['release']
