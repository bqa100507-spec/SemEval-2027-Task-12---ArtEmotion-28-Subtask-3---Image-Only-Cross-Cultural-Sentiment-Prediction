# EXP-027 contrast-aware QLoRA — CPU validation and GPU readiness

**Superseded readiness status:** TASK-030 closes the owner-completed pilot as **GPU_PILOT_COMPLETED / NO_GO_FOR_EXPANSION**. [Verified closure/error analysis](exp027_final_closure.md). All preparation material below is preserved as history, not authorization to rerun or expand. E50+L remains production.

**EXP027 = CPU_VALIDATED / GPU_READY.** This means implementation and CPU/workflow checks passed for a future separately authorized inner-only pilot. **No GPU training, outer-fold evaluation, full CV, Development scoring, confirmation access or submission was executed.** Actual L4 mixed-precision/logit replay, memory, throughput and scientific benefit remain unverified. EXP-026 stays EXPLORATORY_COMPLETED / NOT_PROMOTED; E50+L stays production.

## Delivered design

[Frozen protocol](../docs/exp027_protocol.md), [config](../configs/exp027.json), [CPU parity receipt](../docs/exp027_cpu_parity.json), [training-only scale diagnostics](../docs/exp027_scale_diagnostics.json).

Same pinned Qwen/NF4 decoder Q/V LoRA, frozen vision tower and width64 head as EXP-026. BCE uses the original count weighting and effective-window denominator. Pearson uses complete painting groups, not microbatch fragments; it averages an EXP-004-style predicted-norm-smoothed correlation loss equally over eligible paintings. Scorer eligibility (≥3 languages, nonconstant targets) is distinguished from epsilon0.01 training stabilization. No scorer replacement or other loss/architecture is added.

The native BCE control is unchanged. Treatments use grad-enabled forward/discard, complete-window logit VJP, then RNG-replayed forward/backward per microbatch. One graph at a time, one clip/optimizer step per original four-painting window. Replay must preserve logits exactly and model buffers/parameter versions; partial gradients are discarded on failure. Stochastic equality applies to the same partition/draws, not arbitrary stochastic full-batch masks. Floating-point parameter gradients are checked with tolerances, never described as bitwise exact two-pass gradients.

Grid: **0 /0.000682 /0.00196**. Training-only calibration used64 paintings/609 pairs, excluded outer fold2, examined fold1 and inner validation, and used label-independent synthetic logits. Median/p90 unscaled logit-gradient ratio73.34/254.88 supports substantially smaller weights than EXP-004's0.01/0.05. Low/high weights target roughly5%/15% median perturbations, capped near50% p90. This is engineering-scale justification, not Qwen-optimal-lambda evidence. Future GPU preflight logs actual head/LoRA component-gradient norms without retuning this grid.

Common pilot: outer fold2 excluded, original inner split,192 training/48 validation paintings (1931/504 pairs), seed2027, two epochs, earliest minimum vote-weighted inner-validation binary BCE checkpoint for every variant, no refit. Other P2 folds are historically reused, not independent confirmation. All candidates start from original pretrained weights and newly initialized adapters; EXP-026 fitted adapters cannot leak into a different held-out fold. No outer evaluation endpoint exists.

## Executed CPU checks

`python -m pytest -q` → **334 passed,1 Windows symlink skip**. This includes37 new tests and the applicable repository workflow-compliance checks. Notebook nbformat schema, Python AST and `validate_notebook_contract` PASS. No model weights or dataset download is needed for the tests.

- Native BCE regression: bitwise equality of initialized heads, final head/LoRA parameters, AdamW state, vote-normalized loss and optimizer-step count against EXP-026's original trainer on a tiny CPU fixture.
- Pearson value and logit derivatives match EXP-004's double-precision analytic reference to1e-12. Gradcheck, variable groups, too-small/constant-target groups, near-constant predictions, n<5 eligibility and invalid/nonfinite input handling pass.
- Two-pass versus unsplit full-batch autograd at microbatch sizes1/2/4/8/18: maximum parameter-gradient absolute error **1.1175871e-8** in the deterministic FP32 CPU fixture. Parameter updates, head/LoRA gradients and unchanged pretrained base weights pass. Dropout/Python/NumPy/Torch stochastic replay preserves the same partition's gradients and end RNG. Mutating buffers or changed replay logits fail closed.
- Poisoned outer/locked target fixtures prove labels are not indexed outside permitted painting IDs. Original metadata splits/subsets are reproducible/disjoint. No held-out score is used in calibration/selection.
- Checkpoint source/variant/checksum mismatch rejection, exact interrupted-epoch resume, selected-state reload/validation replay, no default GPU authorization, forbidden outer endpoints, bounded parent timeout and single OOM stop pass.
- Notebook mocks cover successful pilot→verified export→download→release, bad manifest, authorization/scope/confirmation/submission/evaluation/budget violations, mount/input/download/export failures and cleanup. Mock success is not an actual GPU run or browser-arrival/deallocation observation.

Local successful commands (dataset paths refer to existing released training input only):

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/calibrate_exp027.py --inputs runs/exp023_budget_delivery/final/restore_check --output runs/task029/calibration.json
```

## Prepared future pilot

[EXP027_Colab.ipynb](../notebooks/EXP027_Colab.ipynb) has one default-OFF authorization/configuration, pinned source and Python3.12.10/unchanged dependency versions. It checks both the existing ZIP SHA and input-manifest SHA, extracts only two input tables and permitted pilot artworks to local `/content`, runs CPU/workflow checks, then the **pilot only** if separately authorized. Model/processor assets are checked by the original pinned loader. Input paths are configurable; no manual data copying or new input ZIP is needed.

Use a new `MyDrive/exp027/pilot_run_<source>` namespace. Vision features may be reused from compatible `MyDrive/exp026/features`; historical models/results are never reused/relabelled. Source/runtime/config/input/partition/variant checks and selected-checkpoint replay are strict. Completed epoch/current+best state, optimizer/scaler/RNG and counters persist on Drive. Unfinished epochs can be lost on abrupt VM failure. OOM/timeouts stop without changing batch size, precision, rank or settings.

Before any fit, the future L4 run must pass official/cached preprocessing parity and a bounded unsplit/two-pass gradient preflight on one complete3–8-language training painting, with no optimizer update. Unequal same-chunk logits, zero/nonfinite component gradients or failed numerical tolerance block the pilot. All selected checkpoints are independently replayed on inner validation before publication and reuse. No scientific CV/GO decision is produced.

Resource forecast: native two-epoch training≈573s; treatments add≈190s forward work each, suggesting≈1.33x training cost (conservative bound2x before unmeasured state-check/I/O overhead). Three fits, repeated validation/replay, preflight, setup/cache/checkpoints/export project **50–80 minutes**, not a measurement. Historical EXP-026 allocated peak≈4GiB; planning range≈4–6GiB for one graph plus small logit/RNG/buffer state is unverified. The unchanged **maximum2-hour** Run-all allocation includes setup and180s export reserve. Do not silently expand it; stop/resume completed boundaries if needed. Real telemetry must replace estimates before any larger experiment.

Verified persistent ZIP/read-back and manifest precede browser download attempts and outer-finally runtime release on success/failure. API returns never prove arrival on the owner machine or physical deallocation. Research-only resource/license caveats remain; no competition promotion is implied.

## TASK-029B — read-only inner-validation reporting

After all three variants finish unchanged minimum-inner-BCE checkpoint selection, the pilot scores their already saved six-decimal validation predictions. It uses the existing published-specification ST3 scorer (not independently obtained hosted scorer source): at least3 languages/nonconstant gold per painting, constant predictions score0; Macro Log Loss clips to[0.01,0.99], uses base2 and includes languages with≥20 pairs. Missing aggregate metrics remain null with explicit availability status. A separately named descriptive vote-weighted BCE in nats is computed from rounded predictions at1e-7 clipping; it is not the original checkpoint-selection scalar.

P_low/P_high each report paired per-painting Contrast differences against BCE, common eligible counts, wins/losses/ties (1e-12 tie tolerance) and percentile95% intervals from10,000 painting-level paired resamples, reporting seed27029. These are descriptive selected-inner-validation diagnostics, not independent generalization evidence. They never feed selection, lambda tuning, resource gates or scientific settings. No new forward pass, fitting step or outer/Development/confirmation target access is introduced.

`inner_validation/` persists keyed prediction CSVs, permitted validation counts, checkpoint hashes, CSV hashes, selected epochs, a receipt and recomputed metrics. Independent CPU reconstruction (without loading the encoder): `python scripts/verify_exp027_validation.py RUN_DIR/inner_validation`. New source identity requires a fresh run namespace; historical PR28 commits and EXP-026 artifacts remain unchanged. GPU authorization stays OFF.

TASK-029B validation:337 CPU tests passed,1 platform skip; original training/selection AST and scientific config unchanged. Notebook source pin: `4bc703af144e9121e58ed4cb75c94306c318dff1`. No actual pilot or validation performance has been measured in this patch.
