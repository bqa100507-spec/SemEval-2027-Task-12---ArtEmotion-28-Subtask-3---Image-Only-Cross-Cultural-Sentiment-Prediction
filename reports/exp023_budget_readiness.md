# EXP-023 operational budget readiness — TASK-025C

## Previous failure and preserved science

The owner-reported L4 attempt finished setup, compliance, data validation and C0/C1/C2 benchmark, then stopped after 124.529 seconds: remaining3475.5s versus6770s plus180s reserve. No scientific CV fit was started. Inspection confirms two engineering causes: cloud/runner rejected owner budgets above1 GPU-hour /2 wall-hours; forecast admission required the entire max150 selection +150 refit epoch job to fit. That theoretical bound was presented as expected work. The supplied failure is owner evidence; no independent original Colab output archive was available here. The owner's edited notebook and previous deliveries are preserved.

All scientific settings remain unchanged: C0/C1/C2, frozen representations, caption centering, local PCA/common masks, network, batch256, FP32, original optimizer/losses, seeds2027/2028/2029, grouped5folds, max150/patience15, stopping/fresh refit, rounding, bootstrap and gates. Exactly45 independently verified fits are required. Source diffs and exact tiny CPU checkpoint/prediction comparisons validate this boundary. No GPU training, confirmation annotation access, Development scoring or upload occurs in this coding task. E50+L remains production; EXP-022/TASK-023 remain closed.

## Runtime design

New operational defaults in configs/exp023_runtime.json are4 allocation-hours /5 wall-hours, not universal ceilings or target durations. The original scientific JSON, including its now-ignored historical budget block, stays byte-identical. Notebook owner limits remain outside model scientific identity; changing only these limits resumes sealed compatible fits. Exact source/files/config/input/runtime/partition/condition/seed/seal checks remain mandatory.

Representative three-epoch C0/C1/C2 benchmarks separately measure inner training, validation, full refit, local PCA preparation, checkpoint writing, reload/prediction, scorer and original10000-draw bootstrap kernel (neutral differences; no pilot performance decision). The whole-job theoretical_worst_case_seconds and original_max_epoch_formula_seconds remain diagnostic. measured_expected_remaining_seconds is null before any complete fit; thereafter same-condition completed durations and observed stopping/refit epochs update it, with explicitly labeled max-epoch fallback for unobserved conditions. Forecast history records counts, observed ranges,1.25 safety factor, remaining allocation and limitations; these are engineering estimates, not statistical intervals.

Admission checks one complete original fit: max configured epochs at measured speeds (or slower completed-fit rates), validation, replay/persistence allowance and separate180s cleanup reserve. A large whole-job bound no longer blocks affordable partial progress. Initial exact checkpoint replay must pass before atomic model publication on Drive. Epoch checks and the parent hard deadline protect the allowance; interruption loses only the unpublished current fit. Every published model has fit curves, timings, raw predictions, exact held keys, independent replay and checksum seal. Full CV replay is still required before scoring/gates. Incomplete status can never become scientific NO-GO. Export/read-back time is measured at finalization; before that it is explicitly an allowance, not an observed duration.

Research-only CV never requires unresolved competition permission and cannot open confirmation or create submission.zip. Terminal scientific outcomes remain CV_GO_PENDING_ELIGIBILITY /CV_NO_GO after all45 replays. Competition scope retains source-backed eligibility, original guards and no automatic promotion/upload.

## Owner execution and resume

Use the new complete private ZIP and pinned notebook listed in docs/exp023_budget_delivery.json. Upload the ZIP as MyDrive/exp023_handoff.zip (retain prior copies under separate names), open the pinned Colab URL, select a CUDA GPU and set the single configuration:

```python
EXECUTION_SCOPE = "RESEARCH_ONLY_CV"
AUTHORIZE_EXP023 = True
AUTHORIZE_EXPERIMENT = AUTHORIZE_EXP023
AUTHORIZE_GUARDED_CONFIRMATION = False
REQUIRE_SUBMISSION_ZIP = False
CAPTION_ELIGIBILITY_STATUS = "UNRESOLVED"
MAX_GPU_HOURS = 4.0
MAX_TOTAL_WALL_HOURS = 5.0
AUTO_BROWSER_DOWNLOAD = True
AUTO_RELEASE_COLAB = True
```

Distributed authorization remains False. Click Run all once. There is no intermediate Verify/Continue/Export/Cleanup or need to wait out the budget. On STOPPED_BUDGET, keep the new compatible source/handoff/runtime and persistent directory and authorize another Run all; complete sealed models are reused. Abrupt platform loss cannot run cleanup, so only already published models are guaranteed boundaries. API returns do not prove browser arrival or physical GPU deallocation. Kaggle never imports Colab APIs and requires Save Version/output retention.

The old benchmark-only attempt supplies timing context, but no reusable CV model exists according to the owner report. Its old source/input identity cannot be reused as this patch's benchmark/PCA/run identity. Existing official data and frozen caches are repackaged unchanged; use the new automatically derived persistent directory. No cross-source migration is implemented.

## Validation and unresolved limits

Final CPU/mocked tests, independent private restore, notebook checks, current source pins and handoff checksums are recorded in docs/exp023_budget_delivery.json and the delivery reply. Actual updated L4 throughput, stopping distributions,45-fit completion and scientific CV outcome remain unmeasured. Four hours authorizes a maximum and does not guarantee completion. Competition eligibility remains unresolved in the unchanged frozen evidence file; this does not block authorized research-only CV. No additional coding or intermediate manual scientific stage is required for the next authorized research run.
