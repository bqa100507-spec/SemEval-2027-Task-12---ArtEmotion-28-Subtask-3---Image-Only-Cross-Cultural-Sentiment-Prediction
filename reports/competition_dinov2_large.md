# DINOv2-L as a complementary ST3 encoder

**Decision — do not advance the pooled DINOv2-L candidate to Development.** The frozen Large encoder ran successfully, but its global CLS branch trailed Corrected Ridge in every fold. The fixed blend won one fold and regressed on both pooled metrics. Corrected Ridge remains the incumbent, with owner-reported hosted Contrast 0.4655 on submission #1063064.

## Checkpoint and execution

The public Apache-2.0 `facebook/dinov2-large` revision `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` was accessible without authentication. Both metadata files and the 1,217,522,888-byte safetensors file matched the frozen SHA-256 pins. The two historical DINO caches contain DINOv2-Base, so this run created a separate Large cache. The local Python 3.12 environment loaded the checkpoint with PyTorch `2.7.1+cu126` and Transformers `4.49.0`. [Pinned checkpoint](https://huggingface.co/facebook/dinov2-large/tree/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c) · [Frozen configuration](../configs/competition_dinov2_large.json)

The representation is the official RGB processor's 256-pixel shortest-edge bicubic resize and 224×224 center crop, followed by the FP32, L2-normalized 1,024-dimensional final CLS hidden state. All 1,335 P2 artwork hashes were checked against the organizer-image inventory. The frozen GPU stage took **77.78 s** for batch-eight extraction, including image decoding and preprocessing, or **17.16 artworks/s**. Wall time inside the extractor, including model load and asset verification, was **90.03 s** within the 600-second worker ceiling. Peak allocated/reserved memory was **1.30 / 1.32 GiB**. This is a batched throughput result. For an 85-artwork batch workload, the measured rate implies about five additional seconds of DINO extraction plus model loading, rather than an observed Development runtime. The five CPU fits and evaluation took 52.24 s wall time and about 216 MiB peak RSS.

## Matched P2 comparison

Each outer-fold head uses the existing EXP-032 count-aware interaction ridge objective and the same fixed penalties. Models train on the other four painting-grouped folds. The Corrected Ridge reference was checked against its sealed predictions and the original fold assignments. The complementarity arm is the preregistered probability blend with **80% Corrected Ridge / 20% DINOv2-L**. No held-out scores selected the blend share. The two scopes contain 13,600 full and 11,643 test-like pairs. Test-like excludes English, Arabic and Chinese. [Official ST3 metrics](https://github.com/artemotion-28/artemotion-28#evaluation)

| System | Full Contrast | Full Macro Log Loss | Test-like Contrast | Test-like Macro Log Loss |
| --- | ---: | ---: | ---: | ---: |
| Corrected Ridge | **0.454134** | **0.708500** | **0.463108** | **0.717479** |
| DINOv2-L | 0.409439 | 0.775077 | 0.417196 | 0.787378 |
| Fixed 80/20 blend | 0.452629 | 0.712230 | 0.461493 | 0.721537 |

Each cell below reports Contrast / Macro Log Loss in bits.

| Fold | Full Ridge | Full DINOv2-L | Full blend | Test-like Ridge | Test-like DINOv2-L | Test-like blend |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | .443922 / .664179 | .395598 / .741665 | .442257 / .669448 | .450869 / .677251 | .406836 / .757665 | .449992 / .682693 |
| 2 | .459778 / .724501 | .419487 / .776512 | .457489 / .726345 | .466237 / .733380 | .422514 / .788182 | .463167 / .735595 |
| 3 | .437806 / .710339 | .409016 / .749295 | .439780 / .709161 | .441599 / .721110 | .409506 / .760893 | .443058 / .719798 |
| 4 | .469969 / .735697 | .422615 / .798879 | .467969 / .738691 | .485596 / .742602 | .440675 / .808260 | .484395 / .745747 |
| 5 | .458720 / .710922 | .399743 / .811896 | .455166 / .720655 | .471317 / .716054 | .405904 / .824340 | .466865 / .726832 |

| Comparison with Corrected Ridge | Full Contrast Δ [95% paired CI] | Test-like Contrast Δ [95% paired CI] | Test-like Δ after 5% absolute trim | Test-like Δ after dropping top 1% wins |
| --- | ---: | ---: | ---: | ---: |
| DINOv2-L | −.044695 [−.054184, −.035199] | −.045912 [−.056181, −.035616] | −.034973 | −.052323 |
| Fixed blend | −.001505 [−.003604, +.000538] | −.001615 [−.003835, +.000598] | −.001279 | −.003254 |

The fixed blend's test-like Log Loss increased by **0.004058** bits. The point estimate was negative and the paired interval includes zero and excludes the preregistered +0.002 gain. Omitting large deltas does not uncover a benefit. DINOv2-L and Corrected Ridge's painting-centered errors correlate **0.958** on test-like P2. A larger global DINO descriptor supplied too little complementary signal in this readout, consistent with EXP-005's DINOv2-Base findings on a different split. The advances required a +0.002 test-like gain, positive lower paired bound and trimmed gains, at least four test-like fold wins, nonnegative full-P2 gain and at most +0.005 Log Loss bits. Neither candidate qualified. [EXP-005](exp005_visual_representation.md)

## Verification and next candidate

The sealed OOF reproduced pooled scores exactly, including row-for-row Corrected Ridge parity. The clean regression suite passed 319 tests with three skips. The E50+L production CSV and Corrected Ridge ZIP still match their recorded hashes. No full-training refit, Development scoring or submission was triggered. Artifacts and measured receipts live under `runs/competition_dinov2_large/`. The run uses local implementation commit `b12643180cfc14311453e35b25cab7b61bf8f926`. Reproduce with `python scripts/run_competition_dinov2_large.py fetch`, `extract`, then `cv` in the project environment. Extraction reuses a matching checksum-validated Large cache.

**Next candidate — target-language-conditioned region selection.** Use frozen patch tokens from the already available encoder and compare a small attention readout with one shared regional query against the same readout with target-language queries. This tests whether different regions drive the relative response across languages. EXP-033 pooled a shared visual representation before applying language slopes. Moving the target-language input into the region-selection step changes the mechanism without another backbone swap or an additional probability correction. A single matched painting-grouped pilot against Corrected Ridge can justify further work only if the learned region choice adds complementary signal. The current CLS cache would be retained, and the new candidate would need one bounded extraction of patch tokens.
