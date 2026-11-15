# TASK-0023: recovered EL8 evidence and deployment readiness

Measured locally on 2026-11-15. Training source `eb4a7fac4ebe79971305b0b3de2c0e5102ea5f91`; recovered run `c30c4ad20092b87c`; implementation branch `codex/023-el8-deployment`, stacked on EXP-022. [Workflow and commands](../docs/el8_deployment.md)

## Recovery and historical results

Authenticated Drive connector access succeeded. All 398 files (166,245,575 bytes) and 60 checkpoints were downloaded automatically to `runs/exp022_colab_verified/c30c4ad20092b87c/`. Source SHA verification passed for 68 files. Every stage/model seal, checkpoint tensor schema and chronological CV verification link passed. No files were unavailable. Original manifests:

- CV: `19cb9639cef5799a3a795e0a5d0f00d014cff9f7568139071692afeb8fdd2188`
- CV Verify: `bdf5f0a74c4a84ad1acd133ab456b29ce0064b1a6a427264595bbb9a0bddbd05`
- Original L4 replay: PASS, 60/60, 181.7063 seconds, locked access false.
- Decision: `CV_GATE_QUALIFIED_NO_DEVELOPMENT_AUTHORIZED`, EL8 selected, E50+L production, confirmation unopened, Development unscored.

These numbers are read from the recovered, checksum-validated artifacts, not new independent test measurements:

| System | Pooled training CV Contrast | Macro Log Loss |
| --- | ---: | ---: |
| R0 | 0.447097707 | 0.710554182 |
| EF0 | 0.448276545 | 0.708833834 |
| EF8 | 0.450555098 | 0.706858907 |
| EL8 | 0.452483161 | 0.705591157 |

## Strict parity

Independent reconstruction from all 60 saved seed predictions passed: all five folds, 40 model/ensemble checks, maximum error **0**. Actual image-backed replay loaded all 15 L8 CV checkpoints and used only the corresponding held-out fold for every painting. Runtime: RTX 4060 Laptop, Python3.12.10, Torch2.7.1+cu126, Transformers4.49.0; original microbatch4 and processing retained. The final pinned-source replay took **127.8271 s** (earlier diagnostic: 115.2870 s), with **407,016,448 bytes** peak allocated VRAM.

| Fold | EL8 failing rows / total | EL8 maximum absolute error | L8 seed-mean maximum error |
| --- | ---: | ---: | ---: |
| 1 | 392 / 2764 | 0.000004 | 0.000007 |
| 2 | 420 / 2724 | 0.000002 | 0.000003 |
| 3 | 419 / 2721 | 0.000003 | 0.000006 |
| 4 | 480 / 2692 | 0.000003 | 0.000006 |
| 5 | 491 / 2699 | 0.000002 | 0.000005 |

All 15 raw seed comparisons failed exact equality; largest raw error **0.000010699033737182617**. No tolerances were changed. The independent local original-vs-deployment path check passed exactly on 2,764 fold-1 pairs; cached/live path parity passed on 43 pairs. These checks support correct native implementation but do not establish exact replay of L4 outputs. The precise cross-runtime discrepancy remains unresolved. New L4 deployment replay is required before final training; the workflow fails closed on this prerequisite.

## Inference and regression checks

The new explicit `predict --mode el8` uses a strictly verified final bundle and retains the original seed-probability mean, six-decimal rounding and E50+L 50/50 blend. It rejects CV models as final models. Existing inference math/files/bundles remain unchanged; the shared CLI only adds explicit EL8 dispatch.

Image-only CV diagnostic (not a final model or competition submission): two unseen synthetic image contents, three paintings including a repeated-content alias, six image-language pairs, all three L8 seeds. Warm/cold predictions matched exactly and warm extraction encoded zero images. Total smoke **59.0779 s**, main-process peak allocated **404,272,640 bytes**. Separate SigLIP2 and CLIP-L worker peaks were **429,472,768** and **1,285,044,736 bytes**; worker wall times were 24.1247 and 21.2027 s including model startup. Adapted extraction per seed: cold 0.5104/0.0610/0.0447 s; warm 0.0131/0.0120/0.0129 s. These tiny-fixture timings are not production throughput estimates. Its diagnostic ZIP passed coverage/structure validation.

All 849 public **input-only** release predictions replayed on CPU, with coverage-validated ZIPs and historical CSV SHA-256 parity:

| Mode | CSV SHA-256 |
| --- | --- |
| Default E50+L | `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3` |
| E50 | `89591c9ccefe47b570933ab2f7765e3927882d0912ec3288914b6c692e50b0d1` |
| CLIP | `e4c0fc06ed640f10b702b4b2a95451b1cad4665c86f2cbf170ef2c7c3cce9e47` |

No Development gold was read or scored. All 34 historical frozen-cache hashes in the earlier local EXP-022 snapshot passed; all prepared head and reference file hashes remained intact. The owner's modified/saved `EXP022_Colab.ipynb` was backed up byte-for-byte and left unstaged and unchanged.

## Final-model status

| Readiness level | Result |
| --- | --- |
| SOURCE IMPLEMENTED | Yes: explicit inference, strict archive/import, owner-gated one-run refit and cloud launchers |
| SYNTHETIC TESTS PASS | New deployment/cloud tests pass; complete clean-source suite recorded at delivery |
| CV CHECKPOINT PARITY PASS | **No**: sealed aggregation PASS; strict new image-backed L4-to-4060 replay FAIL |
| FINAL REFIT COMPLETED | **No**: no owner GPU-budget authorization; three full-training models still required |
| FINAL INFERENCE BUNDLE VERIFIED | **No**: no final checkpoints exist |
| CODABENCH-READY EL8 | **No**: E50+L remains the released fallback/default |

Proposed new policy: fold=0 hashed inner stopping partition, original max100/patience10/BCE, fresh full-1335-painting refit for selected epochs, seeds2027/2028/2029; all scientific hyperparameters unchanged. Requires owner approval. Historical L4 fit-based forecast: **0.5132 GPU-hours** at mean, **0.7703 hours** p90 plus margin, excluding setup/replay/export and unknown final stopping behavior. No final fit, MiniScreen, Screen, CV rerun, locked confirmation access or Codabench upload was launched.

The minimum next action is to approve this policy and budget once, then use the new one-ZIP L4 notebook workflow. It automatically attempts strict parity first, stops if parity fails, and otherwise prepares data, trains only missing final seeds, verifies them and exports complete artifacts plus receipts. New cloud execution remains untested; notebook syntax and mocked orchestration tests do not prove GPU compatibility.

## Private handoff and delivery

The checked one-ZIP handoff is `runs/task023_delivery/el8_refit_handoff.zip`, **595,249,375 bytes**, SHA-256 `7cc9a836e4d5eb3db90252d4c2e3bf9652667d5e46c8c47b65398c09d1f7386b`. Actual extraction and all 406 file checks passed, including all 60 CV weights. `git bundle verify` confirmed complete history and implementation revision `a6fd6559e0b08bde78f12100183dab9ff8ce2fe4`. Both new notebooks pin this implementation and archive hash. Subsequent delivery changes add notebooks, regression tests and explicit CLI error handling; inference/training math is unchanged.

Review PR: [#22](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/pull/22), draft and unmerged, stacked on open/unmerged PR #21. Both Python environments passed `pip check`. The 31 targeted deployment/cloud tests pass, including both notebook Run all paths with mocked GPU setup and no automatic training, strict checkpoint failure, archive corruption, exact seed rounding, failed-replay invalidation, completed-seed resume and automatic failure export.
