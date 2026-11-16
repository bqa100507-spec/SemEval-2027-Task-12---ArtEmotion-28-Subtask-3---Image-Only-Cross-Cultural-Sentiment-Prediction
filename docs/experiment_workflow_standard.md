# Permanent ST3 experiment workflow contract

**Single authoritative execution specification for every future ST3 experiment.** `AGENTS.md` is the mandatory agent entry point. Scientific protocols remain experiment-specific and must be read alongside this document. TASK-025 implements shared enforcement in `src/st3/experiment_workflow.py`, behavioral compliance tests in `tests/test_experiment_workflow.py`, and an EXP-023 cloud adapter. This does not reopen EXP-022/TASK-023 or modify their archived source, notebooks, metrics or checkpoints.

## One notebook, one authorization, one Run all

Before execution freeze the hypothesis, exact conditions/controls, eligible training population, partitions, source/config/assets, seeds, loss/optimizer/precision, pilot scope, decision gates and allowable artifacts. The single Configuration approval covers the declared sequence, including any explicitly authorized, guarded confirmation/final-fit branch. No additional scientific approval, Verify, Export, Submission or Cleanup click occurs midway.

Required sequence:

**Setup → Validate → Benchmark → Experiment → CV → Verify → Decision → Guarded Confirmation when eligible and authorized → Final Model → Inference → Submission Validation → Export → Browser Download → Runtime Release.**

## Explicit execution scope

Choose one scope in the initial owner Configuration. Both require advance GPU authorization, released/permitted resource provenance, the frozen scientific protocol, compliance tests and explicit owner-authorized budget limits.

| Scope | Eligibility and allowed endpoint |
| --- | --- |
| `RESEARCH_ONLY_CV` | Competition eligibility may remain unresolved. Use only the preregistered released official training population and pinned permitted local artifacts. Complete CV, independent replay, metrics and unchanged scientific gates. Stop with `CV_GO_PENDING_ELIGIBILITY` or `CV_NO_GO`, then export/download outputs and release. No sealed confirmation, final competition fits or Development submission. Confirmation authorization and submission request must both be false. |
| `COMPETITION_READY` | Independently verify genuine source-backed organizer permission before scientific GPU work. Confirmation/final models/submission require their original successful gates and advance authorization. No automatic competition promotion or upload. |

Research permission/provenance is distinct from competition eligibility. An unresolved competition rule must not silently block otherwise authorized released-data research; a research CV_GO must never be relabeled competition eligibility. EXP-023's frozen `configs/exp023_eligibility.json` remains unresolved. Competition scope can consume a separate genuine organizer-evidence JSON (official URL, exact source SHA, permission excerpt and required resource/procedure coverage), checked by the original verifier and retained with its own checksum. Owner status strings alone remain insufficient.

Scientific identity excludes scope/permission **only because they do not change training**: the exact original source revision/files, scientific config, complete input hashes and runtime are still compared without exceptions. Keep one compatible persistent run directory when changing from research CV to competition scope. Recheck seals and independently replay complete CV artifacts; do not retrain them. Different source/input/config/runtime identities fail with an explicit explanation. No semantic-equivalence migration from older source releases is allowed. Per-attempt authorization/eligibility records retain the scope history without changing immutable model seals.

For EXP-023, the final 1,335-painting models are fitted and verified before opening confirmation, then reused without retraining after confirmation GO. A scientific NO-GO cannot publish a promotion-ready submission. Only real inference, complete validation and verified persistent artifacts may produce `SUCCESS_SUBMISSION_READY`.

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

## Enforcement and readiness boundary

The reusable `execute_stages` function checks automatic stage order, independent verification and fail-closed scientific decisions. `Budget`, `publish_model`/`verify_model`, `restore_handoff`, `export_artifacts` and `finalize` enforce resource, provenance, persistence and cleanup contracts. `validate_notebook_contract` checks one owner Configuration section, required settings and one orchestration entry point. Structural validation must be accompanied by the behavioral CPU/mocked tests; it is not a GPU-readiness certificate.

Before spending GPU resources, the scope-appropriate provenance/permission checks, advance authorization, frozen protocol, real input/cache validation and current compliance tests must pass. Source-backed **competition** eligibility is additionally mandatory for competition scope; research-only CV does not assert it. An owner status string is not organizer permission. A new run identity is mandatory for source/runtime/scientific-input changes; historical complete boundaries are never relabeled or silently mixed. Initial platform permissions such as Drive mounting are the only permitted mid-run interaction.

Persistent model seals must be atomically published as each fit finishes, before any shutdown handler. Abrupt platform loss can lose an in-progress fit; it must not lose already published persistent models. Kaggle output survival requires Save Version/output capture; no unsaved-session guarantee is made. The supported cloud environment and dependency lock must be pinned and validated independently of the notebook kernel.

## Progressive operational budgets — TASK-025C

GPU/allocation and wall limits are maximum authorizations, not target durations. They must be finite and configurable for each experiment; there is no universal one-hour or four-hour ceiling. EXP-023 defaults to 4 allocation-hours /5 wall-hours, counted since Run all (not CUDA kernel-active time). Finish, export and release immediately after the authorized sequence ends. Setup/restoration, validation and verification consume this allowance too; pre-Run-all platform occupancy remains unobservable.

Separate owner time limits from scientific identity. Changing only these limits permits exact compatible resume, with the same immutable source/config/inputs/runtime and original checkpoint hashes. Changing source still requires a fresh run identity. Operational defaults are pinned in configs/exp023_runtime.json; the unchanged scientific config's old budget block is archival and no longer read. Do not change a source file merely to raise the notebook's owner-authorized limit.

Record two forecasts: theoretical_worst_case_seconds preserves the max-epoch whole-job bound; measured_expected_remaining_seconds is initially null and then uses observed complete same-condition fit durations, selection/refit epochs and measured representative operations. Unobserved conditions explicitly retain conservative planning proxies. Record min/max/sample counts and the safety factor, without statistical confidence claims. EXP-023 benchmarks inner train, validation, full refit, PCA, checkpoint I/O, reload/prediction, scorer and bootstrap independently; export/read-back duration is measured at finalization and reported separately. Export admission remains a conservative allowance plus cleanup reserve, not an alleged prior measurement.

A whole-job theoretical forecast exceeding a session's budget must not veto safe partial progress. Before each original fit, reserve a conservative max-epoch fit estimate plus initial replay/persistence and cleanup; refuse admission if insufficient. Completed slower fits may raise the allowance. Seal/publish only after exact replay succeeds, on persistent storage. Recompute forecasts after each completed model. Parent deadlines and in-loop checks stop overruns, discard unpublished partial fits and retain all published completions for the next authorized run.

Fewer than all preregistered fits or failed replay cannot yield a scientific GO/NO-GO. STOPPED_BUDGET and INCOMPLETE_CV are operational statuses. Re-run with the same compatible source/handoff/runtime/persistent path to continue; do not repeat completed fits. Export/read-back, browser download attempts and runtime release apply on budget interruption as on success. No polling or waiting to consume unused budget.
