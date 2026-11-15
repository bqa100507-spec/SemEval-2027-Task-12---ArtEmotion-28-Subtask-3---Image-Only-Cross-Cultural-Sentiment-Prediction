# Future ST3 experiment workflow contract

**Documentation standard only.** Apply when a new experiment is separately authorized. It does not reopen EXP-022/TASK-023 and does not change their archived notebooks, source, metrics or checkpoints. No future notebook or training implementation is created in TASK-024.

## One notebook, one authorization, one Run all

Before execution freeze the hypothesis, exact conditions/controls, eligible training population, partitions, source/config/assets, seeds, loss/optimizer/precision, pilot scope, decision gates and allowable artifacts. The single Configuration approval covers the declared sequence, including any explicitly authorized, guarded confirmation/final-fit branch. No additional scientific approval, Verify, Export, Submission or Cleanup click occurs midway.

Required sequence:

**Setup → Validate → Benchmark → Experiment → CV → Verify → Decision [guarded confirmation if declared] → Final fit only if GO → Inference → Submission validation → Export → Runtime release.**

- Validate actual official input-only CSV/images and every required asset/cache before spending the fit budget when a submission is required. No synthetic substitute. Keep evaluation-phase resources separate from Development workflows.
- Pilot estimates use representative full work/epochs, not tiny-batch extrapolation. Separate training-stage occupancy, observed total GPU allocation and wall time; do not call wall occupancy CUDA kernel-active time. Include setup, verification, inference/export and pre-run idle-allocation limitations in the budget definition.
- Enforce finite GPU/wall budgets, remaining-work forecasts and a persistence reserve. Parent process deadlines terminate children safely. Failed gates stop automatically with a precise stage/reason; no silent hyperparameter, precision, checkpoint or dependency substitution.
- Every complete seed/stage is sealed with source/runtime/input identity, coverage and checksums before reuse. No unnecessary repeat of valid completed stages; interrupted current fits restart unless an independently tested mid-fit resume exists. Corrupt/incompatible completed fits block instead of being silently retrained. Revoke stale PASS evidence before a new verification attempt.
- Fit transforms and component heads only on applicable training paintings. Never select variants from Development gold. A sealed training-confirmation partition opens only under the predeclared successful gate/authorization. Final inference takes pixels and target language only.
- Create `submission.zip` only after verified models, real official inputs, complete pair coverage, correct order/schema, finite [0,1] probabilities and correct ZIP structure. No automatic Codabench upload. Keep production unchanged until a separately documented promotion decision.
- Progress shows current stage/seed/epoch, elapsed/remaining budget and stop reason. Distinguish operational success, scientific GO, and owner-reported hosted outcomes.

## Durable success and failure outputs

Store complete checkpoints continuously on persistent storage; abrupt VM loss cannot execute Python cleanup. An outer `try/finally` exports every available published model, prediction, metric/decision, log and source/config/runtime manifest on success or controlled failure. Exclude unpublished temporary fits/hot caches from a final bundle. Record failed stage/reason and resume scope explicitly.

Validate exported file coverage, every SHA and persistent read-back before publishing a success marker. Do not report submission readiness if the required ZIP is absent. Preserve the original input/archive receipts and all historical experiments. If persistence is unavailable, report that exact failure and still attempt cleanup; never claim a Drive copy exists without a verified write.

## Automatic browser download, then release

Future Colab notebooks must expose **`AUTO_BROWSER_DOWNLOAD = True`**, as well as **`AUTO_RELEASE_COLAB = True`**, in the same pre-run Configuration section. This adds no extra manual execution stage.

1. Finish and verify the Google Drive export first. Keep those copies regardless of browser behavior.
2. If enabled on Colab and files exist, call the supported `google.colab.files.download` for the verified `submission.zip` and `outputs.zip`.
3. Record each filename/hash, `REQUESTED` / `API_RETURNED` / `FAILED` / `SKIPPED_NO_FILE` / `SKIPPED_EXPORT_UNVERIFIED`, and the exact failure when applicable. An API return means a request was issued; **arrival on the owner's computer is UNVERIFIED** unless a separate real observation establishes it.
4. Persist the attempt record when possible. Browser download failure must not discard Drive artifacts, change a scientific decision or prevent runtime cleanup. Do not add unbounded polling or idle sleeps to wait for downloads.
5. Only after the download attempts, in an enclosing `finally`, call **`google.colab.runtime.unassign()`** when enabled. Persist `REQUESTED_NOT_CONFIRMED` before the call because Python may terminate immediately. Record API-return/failure accurately; do not assert independently confirmed physical deallocation from a returned call.
6. Kaggle never imports or calls Colab APIs. Its output/version/download retention requirements must be explicit; do not promise survival of an unsaved session.

Required control-flow pattern (specification, not an executable new notebook):

```text
try:
    execute the frozen, owner-authorized pipeline
except controlled failure:
    retain completed boundaries and exact stop reason
finally:
    try:
        export and verify persistent artifacts
        if Colab and AUTO_BROWSER_DOWNLOAD:
            attempt each available verified file download independently
            record requests; owner-machine arrival remains unverified
    finally:
        persist final status when possible
        attempt enabled runtime release even if reporting/downloads fail
```

## Mandatory CPU/synthetic tests before expensive authorization

Test happy-path ordering and zero-GPU mocks; missing inputs/assets; failed gates/replay; corrupt/stale/incomplete checkpoint/report rejection; finite budgets/timeouts/OOM; exact pair/rounding/ZIP checks; completed-model reuse; export/read-back corruption; interrupted/failing-stage persistence; original production parity; and no forbidden labels/inference inputs. Verify runtime release on success, gate failure, training/verification/submission failure, budget stop and export/report failure, including bootstrap failures.

Browser-download tests must prove **verified Drive export → download requests → release**, handle each download failing/being blocked, skip absent files/unverified exports, preserve Drive copies and never log owner-machine arrival as confirmed from a mock/API return. No test launches costly training or Codabench submission. Actual cloud execution must be reported separately from local fixtures. Archive the complete verified run and stop after NO_GO; do not automatically start a successor experiment.
