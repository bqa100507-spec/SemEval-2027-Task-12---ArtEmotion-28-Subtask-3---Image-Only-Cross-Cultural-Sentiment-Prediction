# EL8 artifact recovery and final-model handoff

**Current owner workflow:** use the finalized v2 ZIP and [one-run notebook/runbook](../reports/task023_final_one_run.md). Configure approval/budgets once before Run all; replay, refit, real Development submission, export and release are automatic. The historical preparation/refit-only procedure below records the first implementation and is superseded for owner execution.

E50+L remains the default competition system. EL8 qualified on **training-only CV**; neither independent Development nor locked confirmation has been authorized. Existing pretrained-resource eligibility caveats remain unresolved. This is deployment engineering, with the original EXP-022 scientific source frozen at `eb4a7fac4ebe79971305b0b3de2c0e5102ea5f91`.

## Recovered evidence

The authenticated Google Drive connector recovered the entire owner-specified run into `runs/exp022_colab_verified/c30c4ad20092b87c/`: 398 files, 166,245,575 bytes, 60 checkpoints (five folds, three seeds, four conditions). All model/stage file seals and 68 Git source SHA-256 values were checked. The private recovery receipt is `runs/exp022_colab_verified/recovery_receipt.json`; temporary download URLs are ignored and must never be published. No manual owner download was needed.

```powershell
$py = '.venv/Scripts/python.exe'
& $py scripts/run_el8.py validate --output runs/task023_delivery/cv_inventory.json
& $py scripts/run_el8.py reconstruct --output runs/task023_delivery/cv_aggregation.json
& $py scripts/run_el8.py replay --device cuda --max-seconds 600 --output runs/task023_delivery/replay.json
```

`validate` checks the immutable training Git objects rather than pretending the new deployment machine has the historical L4 identity. It checks the chronological prerequisite/verification chain, latest replay attempt, all 60 tensor schemas, 24 visual q/v adapter targets, fold populations, model update evidence and file hashes. `reconstruct` independently averages all three saved seed probabilities before six-decimal rounding and the existing 50/50 `fusion.blend`; it reads no gold labels. `replay` uses actual images, pinned CLIP weights and frozen feature caches for the 15 held-out L8 models. Only each painting's held-out fold is used.

Local **sealed prediction reconstruction passed exactly for all five folds and all four conditions/ensembles (40 comparisons)**. Actual L4-to-RTX-4060 image-backed replay **failed** strict equality: EL8 maximum absolute errors by fold were 0.000004, 0.000002, 0.000003, 0.000003, 0.000002. No tolerance was relaxed. On the same local GPU, the new loading/inference path matched the original EXP-022 path exactly for 2,764 fold-1 pairs, and cached/live feature inference matched on 43 pairs. The precise cross-runtime kernel discrepancy remains unisolated. Original sealed L4 verification records PASS for 60/60; that is historical evidence, distinct from a new deployment replay PASS.

## Optional EL8 inference

The command is explicit; missing/invalid EL8 models fail, with no silent substitution:

```powershell
& $py scripts/run_st3.py predict --mode el8 --el8-model-dir runs/el8_final/bundle `
  --input data/eval_st3/st3_test.csv --data-dir data/eval_st3 --output-dir runs/el8_eval
& $py scripts/run_st3.py validate-prediction --input data/eval_st3/st3_test.csv runs/el8_eval/submission.zip
```

This command currently stops because **no final bundle exists**. The 15 CV L8 models are rejected as final models. Keep using `predict --mode e50l` (also the unchanged default) until final training, verification and any separately authorized competition checks are complete.

Inputs are exactly `painting,language,image`, with image paths inside the specified `images/` directory. Output is `painting,language,p_positive`, and `submission.zip` contains only the top-level `predictions.csv`. Official languages absent from the learned vocabulary use the original shared image term; unsupported languages/paths/duplicates fail validation. Captions, votes and labels are never inference inputs.

The model reuses the original rank-8/alpha-16 visual CLIP-B adapters (24 q/v modules; 294,912 parameters), frozen visual projection, unit-normalized 512+768+768 blocks concatenated /sqrt(3), and width-64 language-conditioned head. SigLIP2-Base and CLIP-L remain frozen. The auxiliary head is strictly loaded for checkpoint compatibility and is not executed for ordinary prediction. Each seed produces FP32 sigmoid probabilities, their arithmetic mean is rounded to six decimals, then the existing E50+L probability is blended 50/50 and rounded again. No logit averaging.

The existing release feature workers and model bundles provide E50+L and frozen SigLIP2/CLIP-L inputs. New adapted-feature SQLite caches are isolated by image content, checkpoint SHA, pinned processor/assets, implementation fingerprint, runtime/device and batch size. Each completed batch commits atomically. A retry reuses committed features; a failed run never publishes a submission directory. CPU inference is available, but exact cross-device parity is not certified. Use the pinned L4 environment for further parity investigation. Do not edit reference predictions or historical caches.

## Final refit: owner approval still required

The original EXP-022 source has nested CV stopping/refit but **no final-full-population policy**. Proposed [policy](../configs/el8_final_refit.json): use only the permitted 1,335 paintings, retain seeds 2027/2028/2029 and all original scientific settings, use `inner_partition(..., fold=0)` (hashed 10% stopping groups with seed 17029), select by BCE/max100/patience10, then reset and refit on all 1,335 paintings for the selected epoch count. The fold=0 choice is new and requires approval; it was not selected using held-out scores. Locked paintings and public Development gold stay sealed. Original E50+L heads retain their existing provenance and training population.

One explicit command, **only after approving this policy and GPU budget**, performs strict CV replay, preflight/pilot, missing final seeds, automatic checkpoint replay, image-only smoke and complete export:

```powershell
& $py scripts/run_el8.py final-refit --device cuda --authorize-final-refit `
  --policy-sha256 d835564f2466eae857936cb7226de46a712f5d098c1dfd020aa4656d42b33fdb `
  --max-gpu-hours 1.5 --run-dir runs/el8_final
```

This is **not authorized or executed by this handoff**. On the current RTX 4060 the strict CV parity prerequisite blocks it. Approval does not bypass parity. A fresh pinned L4 run can verify the implementation before training; it never retrains CV models. Completed final seed directories are immutable and reused only with matching new source/runtime/input identity. Interrupted current seeds restart from the beginning. A failed replay invalidates readiness. Export in `finally` preserves complete checkpoints and failure records, and Drive storage keeps completed boundaries even if a hard process kill prevents the last ZIP export.

The 15 recorded L4 L8 fits took 7,390.61 seconds total. Scaling their nested selection/refit times from each 1,068-painting fit population to 1,335 gives **0.5132 hours** for three fits at the historical mean; p90 plus 25% margin gives **0.7703 hours**. This is a forecast, excludes setup/replay/export and is not measured final training. Final stopping epochs may differ; the explicit cap may stop an incomplete seed. Recovered L4 peak allocated memory was approximately 0.55 GB for L8 fits; the new run performs its own memory and timing checks before expensive refitting.

## One-ZIP Colab/Kaggle workflow

Prepare the private, self-contained handoff after committing source:

```powershell
& $py scripts/package_el8_handoff.py
```

`runs/task023_delivery/el8_refit_handoff.zip` includes the complete recovered CV tree, the original checksum-pinned training-input ZIP, unchanged E50/E50+L head bundles and a Git source bundle with immutable training/deployment history. Its adjacent `.sha256` is the producer receipt (delivered SHA `7cc9a836e4d5eb3db90252d4c2e3bf9652667d5e46c8c47b65398c09d1f7386b`; 595,249,375 bytes). It contains private training data and weights: **never commit or publish it**.

Use [Colab](../notebooks/EL8_FinalRefit_Colab.ipynb) or [Kaggle](../notebooks/EL8_FinalRefit_Kaggle.ipynb). Upload the one ZIP to Drive/private Kaggle Dataset, select a GPU, and Run all. Default preparation has training off. Review the forecast/policy and authorize once by setting the approval and positive budget in Configuration. A single Run all then trains/verifies/exports sequentially; no separate Verify or Export clicks. Both notebooks clone the exact Git revision from the checked source bundle and install Python 3.12.10/Torch2.7.1+cu126 with unchanged dependency pins. Internet is needed for this environment setup; the handoff is not a wheel/managed-Python offline kit. Colab extracts under `/content` and persists completed models/logs to Drive. Kaggle uses `/kaggle/temp` and `/kaggle/working`; Save Version/download output is necessary before session loss. A same-profile interrupted run reuses completed final seeds, never the CV fits as final models.

Cloud GPU execution of these new final-refit notebooks has **not** been tested here. Strict L4 replay is pending. Do not infer successful final training from synthetic orchestration tests.

## Complete archive export/import

```powershell
& $py scripts/run_el8.py export --archive runs/task023_delivery/verified_cv.zip
& $py scripts/run_el8.py import --archive runs/task023_delivery/verified_cv.zip `
  --sha256 PRODUCER_RECEIPT_SHA256 --run-dir runs/restored_el8
```

The import verifies ZIP SHA, every required file, exact entry coverage, safe paths and artifact kinds, and refuses differing-file overwrites. Original CV seals remain unchanged. A raw complete-folder ZIP fallback can be extracted safely into a fresh ignored directory and passed with `--cv-run`; do not use an incomplete old notebook export. The reusable EXP-022 exporter now includes `cv`, `cv_verification`, `cv_decision` and CV verification attempts. Old notebook source pins and the owner's saved notebook are preserved.

Final exports distinguish historical CV, final-refit progress, production E50 dependencies, source/config/runtime identity and verification. A successful final bundle contains exactly three full-population L8 checkpoints plus prepared production heads, scientific pins, policy, training identities, CV evidence and replay links. Pretrained encoder assets remain checksum-pinned external resources, not duplicated in Git. No Codabench upload or Development scoring is included.
