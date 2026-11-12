"""Annotation-only EXP-021 stages; no confirmation/Development entry point."""

import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from st3.vote_inputs import run_preflight
from st3.vote_audit import run_audit
from st3.vote_experiment import run_pilot,run_study
from st3.vote_verification import run_verify
from st3.polarity_workflow import BudgetStop,StageBlocked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["Preflight","Audit","Pilot","Screen","ScreenVerify","CV","Verify"])
    for flag,default in (("run-dir","runs/exp021_vote_distribution"),("exp020-dir","runs/exp020_emotion_targets"),
        ("exp019-dir","runs/exp019_gradient_scaling"),("exp018-dir","runs/exp018_polarity_captions/prepared"),
        ("exp017-dir","runs/exp017_text_supervision"),("data-dir","data/upstream"),("cache-dir","cache")):
        parser.add_argument("--"+flag,type=Path,default=ROOT/default)
    parser.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    parser.add_argument("--max-seconds",type=float,default=0.)
    args=parser.parse_args()
    if args.max_seconds<0:parser.error("--max-seconds must be nonnegative")
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    for protected in (args.exp017_dir,args.exp018_dir,args.exp019_dir,args.exp020_dir,args.data_dir,args.cache_dir,
                      ROOT/"runs/e50_inference_models",ROOT/"runs/e50l_inference_models"):
        if args.run_dir==protected or protected in args.run_dir.parents:parser.error("RunDir overlaps historical artifacts or data")
    stages={"Preflight":run_preflight,"Audit":run_audit,"Pilot":run_pilot,
            "Screen":lambda a:run_study(a,screen=True),"CV":lambda a:run_study(a,screen=False),
            "ScreenVerify":lambda a:run_verify(a,screen=True),"Verify":lambda a:run_verify(a,screen=False)}
    try:stages[args.stage](args)
    except BudgetStop as error:print(f"RESUMABLE_BUDGET_STOP: {error}",file=sys.stderr);return 75
    except StageBlocked as error:print(f"STAGE_BLOCKED: {error}",file=sys.stderr);return 3
    except (ValueError,FileNotFoundError,KeyError) as error:print(f"VALIDATION_FAILURE: {error}",file=sys.stderr);return 1
    return 0


if __name__=="__main__":sys.exit(main())
