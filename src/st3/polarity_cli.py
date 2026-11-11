"""Stage dispatch and meaningful failure codes for the EXP-018 owner runner."""

import sys

from .polarity_confirmation import run_confirm, run_dev
from .polarity_experiment import run_cv, run_pilot, run_verify
from .polarity_workflow import BudgetStop, StageBlocked, run_audit, run_preflight


def run_polarity(args):
    runners = {"polarity-preflight": run_preflight, "polarity-audit": run_audit, "polarity-pilot": run_pilot,
               "polarity-cv": run_cv, "polarity-verify": run_verify, "polarity-confirm": run_confirm, "polarity-dev": run_dev}
    try:
        runners[args.command](args)
    except BudgetStop as error:
        print(f"RESUMABLE_BUDGET_STOP: {error}", file=sys.stderr, flush=True)
        raise SystemExit(75) from error
    except StageBlocked as error:
        print(f"STAGE_BLOCKED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(3) from error
    except (ValueError, FileNotFoundError) as error:
        print(f"VALIDATION_FAILURE: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
