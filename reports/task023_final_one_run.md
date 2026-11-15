# TASK-0023 FINAL: one authorized run through a real submission

EXP-022 remains CLOSED. This change completes deployment orchestration in draft PR #22; it changes no architecture, losses, LoRA, optimizer, precision, folds, seeds or scientific gates. E50+L remains production. The original L4 60-checkpoint Verify PASS and CV Contrast/Log Loss are historical evidence; no new L4 run or local GPU training is performed here. Strict RTX4060-to-L4 replay previously failed, and strict replay remains mandatory before final fits.

## Final owner workflow

Upload the private **el8_refit_handoff_v2.zip** to MyDrive, choose **NVIDIA L4** and enable Internet, approve the displayed policy/budgets in the notebook's single Configuration cell, then **Runtime -> Run all once**. Grant the initial Drive mount. No separate preparation, replay, Verify, Export, Submission or Cleanup action is needed.

Required configuration:

```python
AUTHORIZE_FINAL_REFIT = True
MAX_GPU_HOURS = 1.5
MAX_TOTAL_WALL_HOURS = 2.0
AUTO_RELEASE_COLAB = True
REQUIRE_SUBMISSION_ZIP = True
EVAL_INPUT_CSV = ""
EVAL_DATA_DIR = ""
```

Blank input paths select the packaged official **Development input: 849 pairs, 85 paintings/images**. It produces a Development submission, not an unreleased Evaluation-phase submission. The CSV SHA-256 is `d777a7da97e14f7ed88c33dab20feb60c2f78549374b2b1b8ba91b1b3fc7f7e5`. No Development gold is included, parsed or scored. Original training votes remain limited to the permitted 1,335 paintings in the unchanged training bundle. Evaluation/confirmation labels remain sealed.

The pinned [organizer README](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#getting-the-data) supplies the public phase inputs and mirrored artwork for participation. The v2 copy stays private and is used only for this competition research; WikiArt works retain their holders' rights and are not relicensed. The Codabench page fetch returned unusable competition content during this check, so no new rule interpretation relies on that response. Existing pretrained-resource eligibility caveats are unchanged.

## Automatic sequence and stop behavior

The notebook checks the ZIP/source pins, restores locally, validates all 60 CV checkpoints, checks the real input/images/frozen cache coverage, and verifies the pinned Python environment. It then executes strict image-backed L4 replay, original preflight/memory/pilot, missing final L8 seeds, checkpoint replay, bundle verification and **the public `predict --mode el8` entry point**. Both child and parent independently validate ZIP structure, exact pair ordering/coverage and finite probabilities. Cached frozen CLIP-B/SigLIP2/CLIP-L features cover all 85 packaged images; no SigLIP2/CLIP-L model download or separate overlay is required for that input. CLIP-B's pinned pretrained assets are already in the original training ZIP.

Only `SUCCESS_SUBMISSION_READY` indicates verified models, real inference, valid submission and persistent read-back/export checks. Failures report the stage and exact reason: `BLOCKED_PARITY`, `BLOCKED_EVALUATION_INPUT`, `STOPPED_BUDGET`, `FAILED_TRAINING`, `FAILED_VERIFICATION`, `FAILED_SUBMISSION` or `FAILED_EXPORT`. Missing inputs block before training; a false/missing/stale child completion cannot authorize success. An old child PASS is revoked before every new invocation.

The one authorized run retains the approved fold=0 stopping/refit policy. Completed same-source/runtime/input final seeds are seal-checked and reused; corrupt checkpoints fail closed instead of being retrained. Interrupted current fits restart. CV models are never trained again. Final bundles contain only the three published final seed directories, excluding abandoned temporary fits. A submission failure preserves the verified final bundle. No automatic Codabench upload occurs.

## Budget, persistence and release

`MAX_GPU_HOURS` caps observed GPU allocation from the start of this notebook execution, including environment setup and CPU waiting while a GPU is assigned. Prior idle allocation before Run all is not observable. Overall wall time uses its own cap; training-stage wall occupancy is recorded separately and is not presented as CUDA kernel-active time. The original 0.51–0.77 hour training forecast excludes setup/replay/export. Remaining budgets are checked before fitting, with a conservative remaining-model forecast and persistence reserve. Parent subprocess deadlines also terminate interrupted child process trees. Stage, seed, elapsed time, remaining budget and stop reason are visible and durable.

Completed checkpoints are written directly to Drive at model boundaries. The persistent run contains bundle/checkpoints, real predictions/submission, source/config/runtime provenance, CV replay, validation report and individual SHA receipts. Each attempt publishes `outputs.zip`, `outputs.sha256` and `final_report.json` only after archive hashes and persistent file read-back checks. Hot SQLite inference caches and unpublished temporary fits are excluded from exports. The input ZIP remains the complete 60-model CV archive; output exports include its SHA and CV provenance/replay evidence without redundantly copying all CV weights on every attempt.

Outer `try/finally` handlers cover both notebook bootstrap and the launched workflow: **export -> verify -> `google.colab.runtime.unassign()`** on success, parity failure, training failure, budget stop, submission failure or export failure. Release is attempted even if persistent reporting itself fails. The request is recorded before the API call; `API_RETURNED` is not a claim of independently measured physical deallocation. Release failures are explicitly recorded when possible. A fresh Drive authentication failure cannot produce a Drive export; the notebook prints that blocker and still attempts release. Abrupt VM loss cannot execute Python cleanup, which is why completed checkpoints persist continuously. Kaggle never calls Colab APIs and still needs Save Version/output download before losing its session.

The finalized ZIP retains the original v1 ZIP/receipt as a backup, includes all 60 historical checkpoints, source history, the unchanged training-input ZIP and production heads, four verified frozen feature archives and the official input-only CSV/85 images. Every included file is hashed. Notebook pins and the finalized archive SHA are recorded in the delivery section after packaging. Real L4 parity/training/inference remain owner-run gates; CPU/mocked tests are not GPU performance or model-quality measurements.


## Finalized delivery and measured validation

- New private ZIP: `D:\ML\Semeval2027_Task12_Subtask3+2\runs\task023_delivery\el8_refit_handoff_v2.zip` (612,744,562 bytes).
- Producer receipt: `runs/task023_delivery/el8_refit_handoff_v2.sha256`.
- ZIP SHA-256: `a5ca1f5a47aa21a97bc438f59651acf9aa20c341b017bf2d89c66d7e91165318`.
- Immutable deployment source: `f6be82a60ac0a5e6d247d463b29aa2d07fabbcf9`; notebook delivery commit: `6fc5eb473f44ed08883a364ec0f9ae96d74b3db9`.
- [Updated Colab notebook](https://colab.research.google.com/github/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/6fc5eb473f44ed08883a364ec0f9ae96d74b3db9/notebooks/EL8_FinalRefit_Colab.ipynb). Upload the v2 ZIP to MyDrive, use the settings above, select L4 and Run all once.

Actual clean restore passed **all 497 file hashes**, all **60 checkpoints**, all **68 historical source hashes**, the complete training asset tree and official input/image/cache readiness. `git bundle verify` confirmed complete source history and the declared deployment revision. Restored caches/heads reproduced all **849 E50+L predictions** on CPU with CSV SHA-256 `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3` and a valid ZIP. No SigLIP2/CLIP-L weight or overlay downloads were needed. The v1 ZIP retained its original SHA-256; the owner's saved notebook and protected scientific/production files remained byte-identical.

**53 focused CPU tests passed.** The full clean-source suite, with CUDA explicitly hidden to avoid local GPU execution, passed **278 tests / 3 skipped in 152.12 seconds**. Skips: Windows symlink privileges, private production fixture absent from the isolated checkout, and a CUDA-only tiny FP16 test. Coverage includes success, strict parity block, missing input, invalid ZIP, deadline/budget stop, completed-seed resume, corrupt checkpoint rejection, stale/unfinished child reports, export/release failure, release despite unavailable persistence, notebook bootstrap cleanup and Kaggle API separation. Notebook setup/GPU stages are mocked; no cloud performance or model-quality claim follows from these tests.

GitHub CI passed for the notebook/source commit: [PR check](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/actions/runs/43723796818), [push check](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/actions/runs/43723790475). [Draft PR #22](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/pull/22) remains unmerged and stacked on unmerged PR #21.

No required input resource is missing. A real EL8 submission is **not claimed as generated locally**: strict new L4 replay, authorized final refit and final inference execute in the owner's one run. The notebook produces/validates the real Development ZIP only if those gates pass; otherwise it exports diagnostics and attempts release. E50+L remains production.
