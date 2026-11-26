"""EXP-019 entry point; do not change historical EXP-018 source fingerprints."""

import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from st3.gradient_confirmation import run_confirm,run_dev
from st3.gradient_experiment import run_cv,run_gradient_pilot,run_verify
from st3.gradient_workflow import run_preflight
from st3.polarity_workflow import BudgetStop,StageBlocked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["Preflight","GradientPilot","CV","Verify","Confirm","Dev"])
    for flag,default in (("run-dir","runs/exp019_gradient_scaling"),("exp018-dir","runs/exp018_polarity_captions/prepared"),
                         ("exp017-dir","runs/exp017_text_supervision"),("data-dir","data/upstream"),("cache-dir","cache"),
                         ("full-text-dir","runs/exp017_text_supervision/full_text")):
        parser.add_argument("--"+flag,type=Path,default=ROOT/default)
    parser.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    parser.add_argument("--max-seconds",type=float,default=0.)
    args=parser.parse_args()
    if args.max_seconds<0:parser.error("--max-seconds must be nonnegative")
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    for protected in (args.exp018_dir,args.exp017_dir,args.data_dir,args.cache_dir,
                      ROOT/"runs/e50_inference_models",ROOT/"runs/e50l_inference_models"):
        if args.run_dir==protected or protected in args.run_dir.parents:
            parser.error("EXP-019 must never write inside historical data, caches or model directories")
    runners={"Preflight":run_preflight,"GradientPilot":run_gradient_pilot,"CV":run_cv,"Verify":run_verify,"Confirm":run_confirm,"Dev":run_dev}
    try:runners[args.stage](args)
    except BudgetStop as error:
        print(f"RESUMABLE_BUDGET_STOP: {error}",file=sys.stderr);return 75
    except StageBlocked as error:
        print(f"STAGE_BLOCKED: {error}",file=sys.stderr);return 3
    except (ValueError,FileNotFoundError,KeyError) as error:
        print(f"VALIDATION_FAILURE: {error}",file=sys.stderr);return 1
    return 0


if __name__=="__main__":sys.exit(main())
