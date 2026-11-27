# Future ST3 experiment workflow contract

Apply when a new experiment is separately authorized. Historical experiments retain their own protocols and artifacts.

## One notebook, one authorization, one Run all

Before execution record the candidate, eligible training population, source/config/assets, seeds, loss/optimizer/precision, Development selection bounds and artifacts. The single Configuration approval covers the declared sequence. For new competition work, prefer a complete Kaggle notebook with pinned setup, automatic resume and an explicit owner Run All authorization.

Required sequence:

**Setup → Validate → Benchmark → Full training → Development selection → Inference → Submission validation → Export → Runtime release.**

- Fit on the full authorized official training split by default. Use official Development for direct scoring, checkpoint selection and bounded hyperparameter or ensemble selection. Repeated selection can overfit Development. Five-fold and nested CV are opt-in. Scientific gates and ablations do not delay ZIP generation after successful training.
- Evaluate Cultural Contrast first and monitor Macro Log Loss. Generate a valid standalone ZIP for every successfully trained competition candidate, including those below the incumbent. Generate applicable ensemble ZIPs and record the selected components. The owner decides what to upload.
- Test upload-safe Kaggle names at the archive-member level. Keep a checksum-pinned logical path mapping for filenames with spaces, parentheses or other unsafe characters. Restore names in working storage automatically and verify extraction from both an intact ZIP and a recursively expanded dataset.

- Validate actual official input-only CSV/images and every required asset/cache before spending the fit budget when a submission is required. No synthetic substitute. Keep evaluation-phase resources separate from Development workflows.
- Pilot estimates use representative full work/epochs, not tiny-batch extrapolation. Separate training-stage occupancy, observed total GPU allocation and wall time; do not call wall occupancy CUDA kernel-active time. Include setup, verification, inference/export and pre-run idle-allocation limitations in the budget definition.
- Enforce resource deadlines and a persistence reserve. Use remaining-work forecasts to plan continuation across sessions. Failed technical checks stop with a precise stage and reason. Preserve the declared hyperparameters, precision policy, checkpoint identity and dependencies.
- Every complete seed/stage is sealed with source/runtime/input identity, coverage and checksums before reuse. No unnecessary repeat of valid completed stages; interrupted current fits restart unless an independently tested mid-fit resume exists. Corrupt/incompatible completed fits block instead of being silently retrained. Revoke stale PASS evidence before a new verification attempt.
- Fit transforms, auxiliary targets and heads only on the applicable training paintings. Keep Development gold in the scorer for evaluation and selection. Hidden Evaluation annotations stay inaccessible. Final inference takes pixels and target language only.
- Create `submission.zip` only after verified models, real official inputs, complete pair coverage, correct order/schema, finite [0,1] probabilities and correct ZIP structure. No automatic Codabench upload. Keep production unchanged until a separately documented promotion decision.
- Progress shows current stage, seed, epoch, elapsed and remaining time, and stop reason. Distinguish operational completion, measured local results, and owner-reported hosted outcomes.

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

## Pre-upload checklist (mandatory before every Kaggle handoff)

Added after EXP-038, where three Kaggle runs stopped before any GPU work on three different packaging or replay issues ([incident log](../reports/exp038_final.md#technical-incident-log)). Every item must hold for the exact ZIP the owner will upload. Record the evidence in the handoff receipt.

1. **No nested archives.** Kaggle expands archives nested inside an uploaded dataset, recursively. An upload must contain no member with an archive extension (`.zip .tar .gz .tgz .7z .bz2 .xz .whl`) and no member that starts with the ZIP magic `PK\x03\x04`. `.npz` files are ZIP containers too. Store such members as base64 `.b64` text with both the stored and the decoded SHA-256 in the manifest. Decode them into `/kaggle/temp`, verify the decoded hash, and keep the mount read-only. The packager must refuse to build otherwise (EXP-038: `encode_members` / `refuse_archives`), and a test must simulate recursive expansion. Exempt inherited base dataset: `exp037_inputs.zip` (`eaf16b93…`), whose one ZIP-magic member `training/targets.npz` already verified on Kaggle.
2. **Pins are generated, never typed.** Hash pins come from code: from the downloaded files at the pinned revision or from the Hugging Face tree API. The packager checks formats and refuses bad pins: 40 lowercase hex characters for git blobs, 64 for every `*_sha256` (EXP-038: `check_config_pins`).
3. **Cross-platform replay uses measured tolerances.** Bit-exact replay is required only within one platform. Iterative solvers and platform math libraries make Windows and Linux differ, by up to 5e-6 per probability and 1e-7 in pooled metrics in EXP-038. Cross-platform replays use tolerances recorded in the config. Receipts record `exact`, `rows_differing`, `max_abs_difference`, `threshold` and the OS/BLAS/thread platform. Exceeding a tolerance is a hard error. Gates always compare against references recomputed on the same platform in the same run.
4. **Linux preflight from the exact upload ZIP.** Before handing off a notebook, run `cpu-validate` on Linux from the exact upload ZIP, mounted both intact and in a simulated recursive expansion, using the notebook's own locate code. Use `scripts/linux_preflight.sh OVERLAY_ZIP HANDOFF_JSON BASE OUTPUT_DIR`, which runs a pinned `python:3.12.10` Docker image with `requirements-exp038-cpu-preflight.txt` and `OPENBLAS_CORETYPE=Haswell`. Without Docker, run `scripts/linux_preflight.py` in WSL or on a Linux machine, or have the analysis side run it. Attach the `linux_preflight.json` receipt to the handoff.
5. **Fail all integrity checks at once.** Validate collects every independent integrity failure (inputs, restore, pin formats, snapshot hashes, replay) and reports them together before stopping. Only checks that depend on an earlier one may stop early.
6. **Optional pieces never take down the core.** The mandatory set contains only the reference replay and the VLM-free core. Every VLM or new-model arm is optional, with explicit dependency skips, so its failure is `FAILED_OPTIONAL` / `SKIPPED_DEPENDENCY` rather than a crash.
7. **Known cosmetic defect.** In the EXP-036/037/038 launcher pattern every Kaggle log line prints twice (launcher relay plus tee). Fix it in the next launcher template; do not change a runtime that produced recorded results.

Review loop: the analysis side reviews each handoff. Each Kaggle failure log is answered with a numbered patch prompt that carries its own root-cause evidence, as EXP-038 patches 1–4 did.

## Mandatory CPU/synthetic tests before expensive authorization

Test happy-path ordering and zero-GPU mocks; missing inputs/assets; failed gates/replay; corrupt/stale/incomplete checkpoint/report rejection; finite budgets/timeouts/OOM; exact pair/rounding/ZIP checks; completed-model reuse; export/read-back corruption; interrupted/failing-stage persistence; original production parity; and no forbidden labels/inference inputs. Verify runtime release on success, gate failure, training/verification/submission failure, budget stop and export/report failure, including bootstrap failures.

Browser-download tests must prove **verified Drive export → download requests → release**, handle each download failing/being blocked, skip absent files/unverified exports, preserve Drive copies and never log owner-machine arrival as confirmed from a mock/API return. No test launches costly training or Codabench submission. Actual cloud execution must be reported separately from local fixtures. Archive the complete verified run and stop after NO_GO; do not automatically start a successor experiment.
