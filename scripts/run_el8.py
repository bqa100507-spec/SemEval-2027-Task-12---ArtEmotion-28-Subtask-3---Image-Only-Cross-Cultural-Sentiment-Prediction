"""EL8 artifact validation, strict replay, packaging and explicitly gated refit."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import torch
    torch.set_num_threads(1)
    from st3.cli import json_output
    from st3.el8_artifacts import cv_inventory, export_bundle, import_bundle, reconstruct_cv
    from st3.el8_inference import replay_cv
    from st3.el8_refit import forecast, full_refit
    from st3.polarity_workflow import BudgetStop, StageBlocked
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['validate', 'reconstruct', 'replay', 'export', 'import', 'forecast', 'final-refit','complete','submission-check'])
    for flag, default in (('cv-run', 'runs/exp022_colab_verified/c30c4ad20092b87c'),
                          ('input-dir', 'runs/exp022_visual_lora/inputs'), ('data-dir', 'data/upstream'),
                          ('cache-dir', 'cache'), ('run-dir', 'runs/el8_final'),
                          ('model-dir', 'runs/e50_inference_models'), ('clip-large-model-dir', 'runs/e50l_inference_models'),
                          ('portable-dir', 'runs/exp022_visual_lora/portable_bundle')):
        p.add_argument('--' + flag, type=Path, default=ROOT / default)
    p.add_argument('--output', type=Path); p.add_argument('--archive', type=Path); p.add_argument('--sha256')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--max-seconds', type=float, default=900)
    p.add_argument('--authorize-final-refit', action='store_true')
    p.add_argument('--policy-sha256'); p.add_argument('--max-gpu-hours', type=float, default=0)
    p.add_argument('--max-total-wall-hours',type=float,default=0)
    p.add_argument('--eval-input-csv',type=Path);p.add_argument('--eval-data-dir',type=Path)
    p.add_argument('--require-submission-zip',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--invocation-id',default='direct')
    args = p.parse_args()
    if not math.isfinite(args.max_seconds) or not math.isfinite(args.max_gpu_hours) or args.max_seconds <= 0 or args.max_gpu_hours < 0:
        p.error('Resource budgets must be positive')
    for name, value in vars(args).items():
        if isinstance(value, Path):setattr(args, name, value.resolve())
    args.input_manifest_sha256 = 'fa2f7c4528b49925c400216037a64381bc6a0b35a1b3b11458a6bcbfda924337'
    args.pixel_cache = 'ram'; args.pixel_cache_max_gib = 2.; args.pixel_cache_margin_gib = 1.
    if args.output:
        for protected in (args.cv_run, args.input_dir, args.portable_dir, args.data_dir, args.cache_dir,
                          args.model_dir, args.clip_large_model_dir):
            if args.output.is_relative_to(protected):p.error('--output must be outside protected evidence/inputs/models')
    try:
        if args.stage == 'validate':result = cv_inventory(args.cv_run)
        elif args.stage == 'reconstruct':
            cv_inventory(args.cv_run);result = reconstruct_cv(args.cv_run)
        elif args.stage == 'replay':
            if not args.output:p.error('replay requires --output (outside the historical run)')
            result = replay_cv(args.cv_run, args.input_dir, args.data_dir, args.cache_dir, args.output, args.device, args.max_seconds)
        elif args.stage == 'forecast':
            cv_inventory(args.cv_run);result = forecast(args.cv_run)
        elif args.stage == 'final-refit':result = full_refit(args)
        elif args.stage in ('complete','submission-check'):
            if args.stage=='complete':
                from st3.el8_complete import complete
                result=complete(args)
            else:
                from st3.el8_submission import readiness
                result=readiness(args)
        elif args.stage == 'export':
            if not args.archive:p.error('export requires --archive')
            evidence = cv_inventory(args.cv_run)
            result = export_bundle(args.archive, {'historical_cv': args.cv_run}, {'historical_cv': 'historical_cv'},
                                   {'training_revision': evidence['source_revision'], 'checkpoints': 60})
        else:
            if not args.archive or not args.sha256:p.error('import requires --archive and --sha256')
            result = import_bundle(args.archive, args.run_dir, args.sha256)
        if args.output and args.stage != 'replay':json_output(result, args.output)
        print(json.dumps({k: v for k, v in result.items() if k not in ('identity', 'checks', 'files_sha256')}, indent=2))
        if args.stage=='complete':return 0 if result['status'] in ('SUBMISSION_VALIDATED','FINAL_BUNDLE_VERIFIED_NO_SUBMISSION') else 1
        return 0 if result.get('status') not in ('FAIL', 'FAILED') else 1
    except StageBlocked as error:print('STAGE_BLOCKED: ' + str(error), file=sys.stderr);return 3
    except BudgetStop as error:print('RESUMABLE_BUDGET_STOP: ' + str(error), file=sys.stderr);return 75
    except (ValueError, FileNotFoundError, AssertionError, KeyError) as error:
        print('VALIDATION_FAILURE: ' + str(error), file=sys.stderr);return 1


if __name__ == '__main__':sys.exit(main())
