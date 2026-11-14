"""Controlled visual LoRA stages. No locked-confirmation or Development command."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from st3.polarity_workflow import BudgetStop,StageBlocked
from st3.visual_feasibility import run_preflight,run_pilot
from st3.visual_experiment import run_study
from st3.visual_verification import run_verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['PixelBenchmark','Preflight','Pilot','MiniScreen','MiniScreenVerify','Screen','ScreenVerify','CV','Verify'])
    for flag,default in (('run-dir','runs/exp022_visual_lora'),('exp021-dir','runs/exp021_vote_distribution'),
        ('exp020-dir','runs/exp020_emotion_targets'),('exp019-dir','runs/exp019_gradient_scaling'),
        ('exp018-dir','runs/exp018_polarity_captions/prepared'),('exp017-dir','runs/exp017_text_supervision'),
        ('data-dir','data/upstream'),('cache-dir','cache')):p.add_argument('--'+flag,type=Path,default=ROOT/default)
    p.add_argument('--portable-dir',type=Path);p.add_argument('--input-manifest-sha256')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto');p.add_argument('--max-seconds',type=float,default=0.)
    p.add_argument('--allow-full-cv',action='store_true',help='Explicit owner invocation; verified promising Screen still mandatory')
    p.add_argument('--pixel-cache',choices=['off','ram'],default='off')
    p.add_argument('--pixel-cache-max-gib',type=float,default=2.)
    p.add_argument('--pixel-cache-margin-gib',type=float,default=1.)
    args=p.parse_args()
    if args.max_seconds<0:p.error('MaxSeconds must be nonnegative')
    if args.pixel_cache_max_gib<=0 or args.pixel_cache_margin_gib<0:p.error('Invalid RAM cache budget')
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    for protected in [args.exp021_dir,args.exp020_dir,args.exp019_dir,args.exp018_dir,args.exp017_dir,args.data_dir,args.cache_dir,
                      ROOT/'runs/e50_inference_models',ROOT/'runs/e50l_inference_models',args.portable_dir]:
        if protected and (args.run_dir==protected or protected in args.run_dir.parents or args.run_dir in protected.parents):p.error('RunDir overlaps protected resources')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    # Match baseline.py / PyTorch 2.7.1 defaults used for historical CLIP caches.
    # Disabling cuDNN TF32 changes patch-embedding convolution outputs measurably.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    from st3.pixel_benchmark import run_pixel_benchmark
    stages={'PixelBenchmark':run_pixel_benchmark,'Preflight':run_preflight,'Pilot':run_pilot,'MiniScreen':lambda a:run_study(a,'mini'),
        'Screen':lambda a:run_study(a,'screen'),'CV':lambda a:run_study(a,'cv'),
        'MiniScreenVerify':lambda a:run_verify(a,'mini'),'ScreenVerify':lambda a:run_verify(a,'screen'),'Verify':lambda a:run_verify(a,'cv')}
    try:stages[args.stage](args)
    except BudgetStop as error:print('RESUMABLE_BUDGET_STOP: '+str(error),file=sys.stderr);return 75
    except StageBlocked as error:print('STAGE_BLOCKED: '+str(error),file=sys.stderr);return 3
    except torch.cuda.OutOfMemoryError as error:print('CUDA_OOM: completed models preserved; do not change scientific configuration: '+str(error),file=sys.stderr);return 4
    except (ValueError,FileNotFoundError,KeyError,AssertionError) as error:print('VALIDATION_FAILURE: '+str(error),file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
