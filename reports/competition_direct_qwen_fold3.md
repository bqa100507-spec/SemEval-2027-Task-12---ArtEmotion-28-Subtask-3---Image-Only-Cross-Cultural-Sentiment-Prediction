# Direct culture-conditioned Qwen3-VL · Fold-3 pilot

**NO-GO for five-fold evaluation of this recipe.** Rank-8 QLoRA improved on the matched frozen head in both epochs and both language populations. The preregistered epoch-2 model still trailed Corrected Ridge by **0.104474 Contrast on full fold 3** and **0.096974 on the test-like subset**, with approximately 0.10 bits worse Macro Log Loss. Keep Corrected Ridge as the competition candidate. The prior hosted Development score of 0.4655 on submission #1063064 is the owner-reported reference.

## Evaluation and learning curves

The pinned split uses 1,068 fitting paintings and 10,879 pairs, and 267 held-out paintings and 2,721 pairs. Test-like scoring excludes English, Arabic and Chinese, leaving 2,330 pairs. Full and test-like scoring have 253 and 252 Contrast-eligible paintings and 21 and 18 scored languages. Both Qwen arms used the frozen Stage 1 model, deterministic per-epoch painting order, fixed mean-vote normalization and two epochs. Corrected Ridge used its matching sealed fold-3 predictions. No held-out vote informed fitting, hyperparameters or checkpoint selection. Epoch 2 was fixed as the comparison endpoint. [Stage 1 protocol](competition_direct_qwen_stage1.md) · [Pinned configuration](../configs/competition_direct_qwen.json)

| System | Epoch | Online training BCE (nats) | Held-out BCE (nats) | Full Contrast | Full Macro Log Loss (bits) | Test-like Contrast | Test-like Macro Log Loss (bits) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Corrected Ridge | Fixed reference | — | 0.495257 | 0.437806 | 0.710339 | 0.441599 | 0.721110 |
| Frozen Qwen | 1 | 0.623345 | 0.564497 | 0.266003 | 0.809691 | 0.262779 | 0.826544 |
| Frozen Qwen | 2 | 0.598388 | 0.585274 | 0.254139 | 0.836747 | 0.257870 | 0.844074 |
| QLoRA | 1 | 0.612352 | 0.546689 | 0.312866 | 0.786722 | 0.306464 | 0.800384 |
| QLoRA | 2 | 0.558002 | 0.569654 | 0.333332 | 0.813720 | 0.344625 | 0.818096 |

Frozen Qwen's best observed held-out epoch was **1** for Contrast, BCE and Macro Log Loss. QLoRA's best observed Contrast came at **epoch 2** and its best held-out BCE and Macro Log Loss at **epoch 1**. These are descriptions of the two checkpoints. The reused P2 fold is exploratory evidence.

| Scope | Epoch-2 contrast comparison | Δ Contrast | 95% paired painting interval | Painting wins / losses | Δ Contrast after 5% trim | Δ Macro Log Loss |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Full | Frozen − Ridge | −0.183667 | [−0.216567, −0.150052] | 61 / 192 | −0.156625 | +0.126408 |
| Full | QLoRA − Ridge | −0.104474 | [−0.137940, −0.070660] | 85 / 168 | −0.082226 | +0.103381 |
| Full | QLoRA − Frozen | +0.079192 | [+0.051930, +0.106599] | 166 / 87 | +0.061756 | −0.023027 |
| Test-like | Frozen − Ridge | −0.183729 | [−0.220981, −0.146292] | 71 / 181 | −0.161568 | +0.122964 |
| Test-like | QLoRA − Ridge | −0.096974 | [−0.131542, −0.061860] | 80 / 172 | −0.073493 | +0.096986 |
| Test-like | QLoRA − Frozen | +0.086755 | [+0.054129, +0.119029] | 157 / 95 | +0.067093 | −0.025978 |

Intervals use 10,000 paired painting bootstrap draws under the pinned decision configuration. Trimming removes the 5% largest absolute per-painting Contrast deltas. QLoRA already improved on Frozen at epoch 1 by +0.046863 on full fold 3 and +0.043685 on test-like fold 3. Gains over the frozen control and deficits to the incumbent remained after trimming.

## Optimization and failure diagnosis

| System and epoch | Mean / P95 global norm before clipping | Clipped steps | Mean adapter norm after clipping | Adapter parameter change L2 |
| --- | ---: | ---: | ---: | ---: |
| Frozen 1 | 7.047 / 19.076 | 97.75% | — | — |
| Frozen 2 | 5.655 / 14.741 | 97.38% | — | — |
| QLoRA 1 | 7.849 / 21.525 | 97.28% | 0.0709 | 2.2425 |
| QLoRA 2 | 6.205 / 17.659 | 91.85% | 0.1386 | 1.8046 |

- Frozen Qwen improved its online fitting BCE while held-out BCE, Contrast and Macro Log Loss worsened. That is consistent with overfitting in the frozen readout.
- QLoRA's fitting and held-out BCE diverged in epoch 2, while its Contrast improved. The improved within-painting ordering did not carry over to probability quality. Full-cohort vote-weighted mean QLoRA predictions moved from 0.7103 to 0.5674 against a held-out positive rate of 0.6828. Ten-bin descriptive calibration error rose from 0.0380 to 0.1154, while the share of predictions below 0.10 or above 0.90 fell from 20.18% to 9.30%. The later deterioration includes underprediction and miscalibration, rather than increased extremity. These diagnostics use saved probabilities without fitting a recalibration.
- The low frozen score shows a large gap for this head on this native Qwen representation. The positive QLoRA effect shows that adapting the representation helps this recipe. This comparison does not separately identify the effects of the final-token pooling choice, language prompt or optimization settings.
- Every epoch made 1,068 updates. All 16 LoRA matrices updated, gradients were finite, and frozen parameters received no gradients or updates. QLoRA's measured adapter gradient increased in epoch 2 while clipping fell and Contrast rose. Frequent clipping affected both arms. With no matched clip intervention, the pilot cannot establish its causal contribution.

## Cost, verification and decision

Charged RTX 4060 Laptop GPU job time was **9,890.48 seconds (2 h 44 min 50 s)**, including checkpoint verification, model loading, frozen feature extraction, fitting, and held-out inference. This left 909.52 seconds of the 10,800-second authorization unused. The model ran once in one GPU worker. QLoRA fit epochs took **2,870.98 s** and **2,879.43 s**, about **3.78 training pairs/s**. Their held-out forwards took approximately 645 and 640 seconds. Peak allocated / reserved QLoRA VRAM was **3.76 / 3.95 GiB**. Extrapolating this measured workload with a shared frozen feature cache gives approximately **10.64 GPU-hours** for five folds, or **13.30 hours** with the Stage 1 reserve. That cost does not justify further evaluation given the incumbent gap and paired uncertainty.

All four complete epoch checkpoints, their AdamW and scaler states, intermediate resumable checkpoints and held-out prediction files are sealed locally. The CPU replay verified exact held pair coverage, the painting-grouped order, checkpoint provenance, validation BCE and byte-identical `results.json` (SHA-256 `c2b3f90e6ad6a4468b0418ba619fba86048b1677bd9160cff132d9e489d7a850`). The clean suite passed **314 tests with 3 skips**. The source commit that produced the fit is `2c4632dce687fe012095a71d8f5ce9f54537fecc`. All run artifacts are under `runs/competition_direct_qwen/fold3_pilot/`. E50+L and Corrected Ridge matched their protected artifact hashes. No additional folds, Dev inference, blends or submission preparation ran.
