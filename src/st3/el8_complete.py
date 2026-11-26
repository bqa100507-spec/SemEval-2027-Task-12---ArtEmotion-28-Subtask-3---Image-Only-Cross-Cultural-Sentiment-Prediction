"""Single child workflow: input check, strict replay/refit, real inference, report.

This module orchestrates the frozen experiment; it never trains CV models.
The host cloud launcher owns persistent export and runtime release on all exits.
"""

import copy
import math
import subprocess
import sys
import time

from .baseline import file_sha256
from .cli import ROOT, json_output
from .el8_artifacts import read_json
from .el8_refit import forecast, full_refit, require_authorization
from .el8_submission import readiness, validate_output
from .polarity_workflow import BudgetStop, ExecutionBudget


class FinalRunBudget(ExecutionBudget):
    def __init__(self, seconds, progress, reserve=60.):
        super().__init__(seconds); self.progress = progress; self.reserve = reserve

    def check(self, phase='stage', epoch=0):
        remaining = self.seconds - (time.perf_counter() - self.start)
        self.progress(detail=phase, epoch=epoch, remaining_child_seconds=remaining)
        if remaining <= self.reserve:
            raise BudgetStop(f'Remaining run budget {remaining:.1f}s reached the {self.reserve:.0f}s persistence reserve')


def complete(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); stage = 'AUTHORIZATION'; training_seconds = 0.
    report_path = args.run_dir / 'child_report.json'
    progress_path = args.run_dir / 'progress.json'
    state = {'status':'RUNNING', 'stage':stage, 'seed':None}; stage_times={}; stage_tick=started
    def progress(stage=None, seed=None, **extra):
        nonlocal stage_tick
        if stage is not None:
            tick=time.perf_counter()
            stage_times[state['stage']]=stage_times.get(state['stage'],0.)+tick-stage_tick; stage_tick=tick
        if stage is not None:state['stage'] = stage
        if seed is not None:state['seed'] = seed
        state.update(extra, elapsed_child_seconds=time.perf_counter()-started)
        json_output(state, progress_path)
        if stage is not None or seed is not None:
            print(f"EL8 stage={state['stage']} seed={state['seed']} elapsed={state['elapsed_child_seconds']:.1f}s", flush=True)
    report = {'status':'RUNNING','invocation_id':getattr(args,'invocation_id','direct'),
              'development_scored':False,'locked_data_accessed':False}
    json_output(report,report_path)
    try:
        require_authorization(args)
        for value in (args.max_gpu_hours,args.max_total_wall_hours):
            if not math.isfinite(value) or value <= 0:raise BudgetStop('A finite positive GPU and wall budget is required')
        budget = FinalRunBudget(min(args.max_gpu_hours,args.max_total_wall_hours)*3600, progress)
        args._budget = budget; args._progress = progress; args._managed_export = True
        stage = 'EVALUATION_INPUT';progress(stage)
        if args.require_submission_zip or args.eval_input_csv:
            evidence = readiness(args)
            json_output(evidence, args.run_dir / 'evaluation_input.json')
        else:evidence = None
        # Completed final fits are separately seal-checked by full_refit before
        # reuse. A corrupt directory never becomes permission to retrain it.
        missing = sum(not (args.run_dir/'final'/f'L8_{s}').exists() for s in (2027,2028,2029))
        predicted = forecast(args.cv_run)
        needed = predicted['training_hours_p90_with_25_percent_margin']*3600*missing/3 + 360
        if budget.seconds < needed:
            raise BudgetStop(f'Remaining budget {budget.seconds/3600:.2f}h is below the conservative remaining-fit/replay/inference forecast {needed/3600:.2f}h')
        stage='REFIT';progress(stage)
        fitted = full_refit(args)
        training_seconds = fitted.get('training_stage_seconds',0.)
        stage='VERIFICATION';progress(stage)
        from .el8_inference import final_bundle
        bundle = args.run_dir/'bundle';final_bundle(bundle,args)
        if evidence is not None:
            budget.check('Before real competition inference',-1)
            stage='SUBMISSION';progress(stage)
            # Recheck image/cache identity after fitting and before publication.
            if readiness(args) != evidence:raise ValueError('Competition input or frozen cache changed during fitting')
            output=args.run_dir/'submission'
            prediction_args=copy.copy(args)
            prediction_args.el8_model_dir=bundle;prediction_args.input=args.eval_input_csv
            prediction_args.data_dir=args.eval_data_dir;prediction_args.output_dir=output
            prediction_args.feature_cache_dir=args.run_dir/'inference_cache'
            if output.exists():
                validate_output(output,args.eval_input_csv,final_manifest=bundle/'final_manifest.json')
            else:predict_official(prediction_args,budget)
            validation=validate_output(output,args.eval_input_csv,final_manifest=bundle/'final_manifest.json')
            report.update(status='SUBMISSION_VALIDATED',submission=validation)
        else:report.update(status='FINAL_BUNDLE_VERIFIED_NO_SUBMISSION')
    except BaseException as error:
        current=state['stage']
        if isinstance(error,BudgetStop):status='STOPPED_BUDGET'
        elif current=='EVALUATION_INPUT':status='BLOCKED_EVALUATION_INPUT'
        elif current=='PARITY':status='BLOCKED_PARITY'
        elif current in ('VERIFICATION','FINAL_VERIFICATION','ARTIFACT_VALIDATION','PREFLIGHT','BENCHMARK','BUNDLE'):status='FAILED_VERIFICATION'
        elif current=='SUBMISSION':status='FAILED_SUBMISSION'
        else:status='FAILED_TRAINING' if current not in ('AUTHORIZATION',) else 'BLOCKED_AUTHORIZATION'
        report.update(status=status,failed_stage=current,reason=str(error),error_type=type(error).__name__)
    finally:
        progress(stage=report['status'],stop_reason=report.get('reason'))
        report.update(elapsed_child_seconds=time.perf_counter()-started,training_stage_seconds=stage_times.get('TRAINING',0.),
                      stage_wall_seconds=stage_times, timing_definition='Training stage wall occupancy, not CUDA kernel-active time',
                      require_submission_zip=args.require_submission_zip)
        json_output(report,report_path)
    return report


def predict_official(args,budget):
    """Exercise the public prediction entry point in the pinned interpreter."""
    remaining=budget.seconds-(time.perf_counter()-budget.start)-budget.reserve
    if remaining<=0:raise BudgetStop('No inference budget remains before the persistence reserve')
    command=[sys.executable,str(ROOT/'scripts/run_st3.py'),'predict','--mode','el8',
             '--input',str(args.input),'--data-dir',str(args.data_dir),'--output-dir',str(args.output_dir),
             '--el8-model-dir',str(args.el8_model_dir),'--model-dir',str(args.model_dir),
             '--clip-large-model-dir',str(args.clip_large_model_dir),'--cache-dir',str(args.cache_dir),
             '--feature-cache-dir',str(args.feature_cache_dir),'--device',args.device]
    try:result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=remaining)
    except subprocess.TimeoutExpired as error:raise BudgetStop('Inference exceeded the remaining run budget') from error
    print(result.stdout,flush=True)
    if result.returncode:raise ValueError('EL8 prediction failed: '+result.stdout[-4000:])
