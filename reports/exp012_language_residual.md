# EXP-012 Language-aware residual calibration

## Frozen protocol

M0 is the unchanged E50+L reference. M1 fits a global logit offset and slope. M2 adds language-specific offsets and slopes, and M3 adds the EXP-004 painting-balanced pairwise Huber term to M2. The offset starts at the logit of the nested, six-decimal E50+L probability. The slope uses CLIP-L logit minus the mean of CLIP-B and SigLIP 2 logits. Only logit inputs are clipped to [0.0001, 0.9999]. Disagreement is standardized by the inner OOF pairs' unweighted mean and population standard deviation, with scale 1 for zero variance. The saved statistics are reused at inference. Corrected probabilities are rounded to six decimals. A zero correction preserves the original probability exactly, including at 0 and 1.

All fits minimize vote-weighted Bernoulli BCE in natural nats divided by the sum of eligible votes. The penalty is `0.5 × (0.02 × (a² + c²) + 0.20 × Σ_lang(a_lang² + c_lang²))`. It is added after vote normalization. M3 also adds 0.10 times Huber loss with δ=0.20 on within-painting probability differences versus observed differences. Each eligible training painting contributes equally to that loss, and each pair contributes equally within a painting. Deterministic, single-threaded BLAS L-BFGS-B starts at zero. M2 and M3 learn deviations only with at least 50 distinct meta-training paintings per language. Seen languages below the threshold get the global terms, and absent languages keep q0. M1's global formula is language-independent.

The exact EXP-002 five painting-grouped folds are the outer evaluation. Four inner GroupKFold partitions per outer fold use seed 12027 plus the outer fold number. Each original head fits only the inner training paintings. All three correction models share that outer fold's inner OOF cache. Outer held-out paintings and their images are checked against both inner partitions. Targets and normalization use only the corresponding inner OOF rows. The outer probabilities replay the frozen EXP-010 base heads, including intermediate E50 rounding. The official scorer is unchanged.

Commit `147dd05` froze the promotion rule before held-out scoring. Require five complete, converged folds, a strictly positive pooled Contrast gain, and fold-mean Macro Log Loss within M0 +0.005 bits. Promote the eligible candidate with the highest pooled Contrast only if the gain is at least 0.001. Bootstrap and fold consistency are diagnostics. The 10,000 painting-paired draws condition on these predictions and the previously reused outer folds.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py residual-cv
.venv/Scripts/python.exe scripts/run_st3.py residual-dev
```

The CV command saves the inner caches and per-fold partitions under `runs/exp012_language_residual/inner/`. The development command reads the frozen decision before any development scoring or packaging.

## Results

Each score cell shows Cultural Contrast ↑ / Macro Log Loss ↓. Pooled scores cover all 16,019 outer OOF pairs.

| Model | Fold mean | Pooled OOF | Δ pooled Contrast | Fold wins | Painting wins / regressions | Pooled pair-order agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **M0 · E50+L** | **0.445498117** / 0.715718675 | **0.445643116** / 0.714879226 | — | — | — | **0.700935236** |
| M1 · Global | 0.445266390 / 0.715658663 | 0.445410186 / 0.714793373 | −0.000232929 | 1/5 | 729 / 767 | 0.700629999 |
| M2 · Language | 0.445276292 / 0.715645927 | 0.445419877 / 0.714780054 | −0.000223238 | 2/5 | 728 / 768 | 0.700423521 |
| M3 · Language + Huber | 0.445280040 / **0.715645670** | 0.445423629 / **0.714779845** | −0.000219486 | 2/5 | 725 / 771 | 0.700428634 |

| Outer fold | M0 | M1 | M2 | M3 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.462693 / 0.708627 | 0.462678 / 0.707906 | 0.462628 / 0.707877 | 0.462628 / 0.707882 |
| 2 | 0.439750 / 0.716017 | 0.438726 / 0.715850 | 0.438759 / 0.715843 | 0.438770 / 0.715843 |
| 3 | 0.418114 / 0.713914 | 0.417428 / 0.714285 | 0.417491 / 0.714273 | 0.417497 / 0.714270 |
| 4 | 0.477616 / 0.718075 | 0.477614 / 0.718415 | 0.477636 / 0.718404 | 0.477642 / 0.718402 |
| 5 | 0.429317 / 0.721959 | 0.429887 / 0.721838 | 0.429867 / 0.721832 | 0.429864 / 0.721832 |

The paired 95% intervals for pooled Contrast change were M1 [−0.000594, +0.000127], M2 [−0.000600, +0.000158], and M3 [−0.000596, +0.000161]. Pair-order agreement averages equally over Contrast-eligible paintings, excludes tied gold pairs, and awards half credit for a tied prediction. All three corrections lowered it slightly despite improving Macro Log Loss. M3's loss improvement was 0.000099 bits, with no evidence of better within-painting ordering.

**NO-GO.** All candidates cleared the loss ceiling of 0.720718675 and all 15 corrections converged, but none improved pooled Contrast. M3 was the highest-scoring correction and still fell below M0. The development command exited before loading validation gold or publishing any output. E50+L remains the hosted-verified system.

## Painting and language diagnostics

M3 improved 725 of 1,496 scored paintings and regressed on 771. Its largest gain was on Toulouse-Lautrec's *Two Friends* (+0.031310), and its largest regression was on Ruth Vollmer's *Oval Construction* (−0.079403). Pooled loss improved in 8 of 21 eligible languages for M1 and 7 of 21 for M2 and M3. In M3, Thai improved most (−0.003468 bits) and Setswana regressed most (+0.001526 bits).

M2 and M3 learned deviations for 21 languages per fold. Darija (11–17 meta-training paintings), Emakhuwa (3–5), Kinyarwanda (6–8), and Urdu (12–16) received only the global terms. The largest absolute M3 language offset was 0.001132 for Tamil, and the largest disagreement slope deviation was 0.005551 for Yoruba. The global slope ranged from 0.023095 to 0.061103. The mean absolute painting-level probability shift was 0.005361, while the RMS shift after subtracting each painting's mean shift was 0.002366. Together with the ordering result, the small loss gain is consistent with improved probability calibration rather than improved cultural distinction.

## Runtime and verification

The CPU inner stage took 331.34 s wall-clock. Across 20 partitions, the 60 original LR fits converged and took 277.04 s of summed fit time. The caches contained 64,076 cross-fitted pairs, with each outer training pair predicted once within its partition. An independent replay verified all membership and targets, checked painting and image isolation, and recomputed both blend steps from the cached component probabilities. The 16,019 E50+L outer rows matched the original EXP-010 bytes, with subset SHA-256 `ff4d779e8ec56b7b9c6cad0d9846472660675ec3107a1ffd5a22ffe3c74c264f`.

Scoring from the inner caches took 13.51 s. Summed correction fit times over the five folds were 0.321 s for M1, 0.300 s for M2, and 0.516 s for M3. Summed correction prediction times were 0.010, 0.015, and 0.016 s respectively. All 64,076 outer predictions passed probability and exact-coverage checks. Fold metrics, language deltas, parameters, and fit audit records are under `runs/exp012_language_residual/cv/`.

The label-free production command independently replayed the 849-row E50+L development CSV byte for byte, and its original EXP-010 ZIP still validates. All 646 pre-existing file hashes matched, including 33 cache and checkpoint files. The Windows suite passed 114 tests and skipped one symlink fixture. Both Python dependency checks passed.

Use the existing E50+L inference pipeline for evaluation readiness. Rehearse the released input layout and ZIP coverage checks with that fixed system. Further calibration should be assessed on an independent labeled split before changing the competition pipeline.
