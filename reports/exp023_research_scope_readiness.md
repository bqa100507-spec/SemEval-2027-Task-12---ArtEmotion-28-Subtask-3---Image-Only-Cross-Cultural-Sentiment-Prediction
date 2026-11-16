# TASK-025B â€” EXP-023 execution-scope readiness

**TASK-025C supersedes the historical budget/forecast instructions below.** See [budget readiness](exp023_budget_readiness.md) for the new immutable delivery, configurable 4/5-hour defaults and progressive per-fit admission. Historical validation and delivery pins below remain unchanged.

This patch separates released-data research execution from competition permission. It does not change EXP-023's architecture, losses, PCA/centering/masks, optimizer, folds, seeds, epochs, rounding, bootstrap, scientific gates or budgets. No scientific/GPU run, sealed confirmation-label access, Development scoring or Codabench submission is performed in this coding task. E50+L remains production; EXP-022/TASK-023 stay closed.

## One configured notebook / Run all

The new default scope is `RESEARCH_ONLY_CV`. Upload the new checksum-pinned `exp023_handoff.zip` to MyDrive and select a CUDA GPU. The single advance owner configuration for a later authorized research run is:

```python
EXECUTION_SCOPE = "RESEARCH_ONLY_CV"
AUTHORIZE_EXP023 = True
AUTHORIZE_GUARDED_CONFIRMATION = False
REQUIRE_SUBMISSION_ZIP = False
CAPTION_ELIGIBILITY_STATUS = "UNRESOLVED"
COMPETITION_ELIGIBILITY_EVIDENCE = ""
MAX_GPU_HOURS = 1.0
MAX_TOTAL_WALL_HOURS = 2.0
AUTO_BROWSER_DOWNLOAD = True
AUTO_RELEASE_COLAB = True
```

Distributed notebook authorization remains **False**. This document is a configuration example, not current GPU authorization. All45 CV fits, verification, metrics and the preregistered scientific decision run automatically after that advance authorization and validation. CV_GO is reported as **CV_GO_PENDING_ELIGIBILITY**, otherwise **CV_NO_GO**. Either outcome saves complete persistent artifacts, exports/read-backs outputs.zip, requests its browser download and releases Colab. No manual Verify/Continue/Export/Cleanup is needed. Browser arrival and physical runtime termination remain unverified from API returns.

Research scope cannot call confirmation, final-model, Development inference or submission methods, even with contradictory flags or a direct accidental method call. Contradictory confirmation/submission requests fail closed. All eligible CV pairs and original scientific gates remain intact. Research CV_GO is not competition qualification or promotion.

## Competition continuation and exact reuse

`COMPETITION_READY` still requires genuine independently fetched organizer evidence via the existing verifier. The frozen unresolved eligibility JSON is **unchanged**. A separate `COMPETITION_ELIGIBILITY_EVIDENCE` JSON may hold a genuine official URL, exact response SHA, exact permission excerpt and the existing three required resource/procedure coverage values. Its checksum/proof is recorded with each execution authorization; no fabricated or supplied-as-verified example is shipped. The initial owner flags must additionally authorize guarded confirmation and request the submission when that full scope is desired.

Use the same new pinned source/handoff, runtime and persistent run directory to continue compatible completed research CV. Scope and permission are authorization metadata; every existing scientific identity field is still checked by exact equality. Model seals are not rewritten. Completed fits are checked and independently replayed, not retrained. A changed source revision/file, config, input or runtime fails explicitly. Older TASK-025 source artifacts cannot be silently migrated into this new source identity; original ZIPs and source commits remain preserved. Source-backed permission resolving later does not itself require changing this scientific source or input bundle.

Only verified CV_GO can proceed to the original final-fit/once-only confirmation gates. Confirmation and final-model identity checks remain unchanged apart from the additional explicit competition-scope guards. The existing R0_1335 baseline and exact confirmed-bundle reuse policy remain mandatory. No automatic promotion/upload occurs.

## Verification and delivery boundary

CPU/mocked tests cover both scope paths, missing/contradictory scope, research eligibility isolation, CV_GO/NO_GO, failed verification, forbidden post-CV calls, sealed-data guard, exact identity resume and mismatch/corruption rejection, export/download/release order and competition flow. A rebuilt private handoff and independently restored source are checked before delivery. Final pins/counts are in `docs/exp023_scope_delivery.json` and the delivery message.

The one-run orchestration is testable without GPU execution; actual L4/Kaggle CUDA throughput and scientific scores remain unmeasured. The unchanged benchmark must demonstrate that the authorized remaining work fits the 1 GPU-hour /2 wall-hour caps; otherwise the run stops, exports and releases. No scientific setting is silently changed to meet the cap.

Final coding validation: **221 passed, 0 failed, 1 skipped** (Windows symlink fixture), both from the implementation worktree and independently restored source. The new ZIP restores/verifies 1,693 input files and 117 exact source files, with 13,600 pairs, 69,719 caption rows and all ten historical B1/B2 reconstruction checks passing. Both notebooks pass schema, AST and shared contract validation. Actual new notebook cells/real ZIP were exercised on CPU with mocked cloud callbacks and a deliberate stop before environment installation: research authorization reaches setup with eligibility unresolved; competition remains blocked. All cases export/read-back and release as appropriate; no actual cloud APIs/GPU/scientific stages run.

The private handoff is `D:\ML\Semeval2027_Task12_Subtask3+2\runs\exp023_research_delivery\exp023_handoff.zip`, SHA-256 `b6e664127fe18d0f115f33d624851aa988a222881ede14c539194cca42a206f4`. Manifest SHA: `fe6757e5e15b1bfd98d7096e90e49833881e7201fbdc58aff473775026834691`. Both notebooks pin source `2f3a313453445bce7bce2cd16e79c2a998e4c9b6`. All498 protected files, owner notebook edits and the original TASK-025 ZIP remain byte-identical. [Machine-readable scope delivery](../docs/exp023_scope_delivery.json).
