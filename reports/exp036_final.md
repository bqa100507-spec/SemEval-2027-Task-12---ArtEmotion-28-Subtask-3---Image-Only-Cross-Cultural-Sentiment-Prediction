# EXP-036 final closure

**Completed and verified. The 75% Corrected Ridge / 25% CS-CVA ensemble is the active competition incumbent.** Corrected Ridge remains the primary fallback and E50+L remains the historical released fallback. The owner reported Codabench submission **#1065924**, finished on 24 November 2026 with Development Contrast **0.4703**. This agrees with the rounded local result. Hosted evidence comes from the owner.

## Development replay

All three models cover the same 849 official Development pairs over 85 paintings. The scorer used 82 Contrast-eligible paintings and 17 Macro Log Loss languages. Local rescoring used the pinned official Development votes and the exact saved prediction files.

| Model | Cultural Contrast | Macro Log Loss |
| --- | ---: | ---: |
| Corrected Ridge | 0.465462328 | 0.740305114 |
| CS-CVA standalone, epoch 5 | 0.437632191 | 0.742910807 |
| EXP-036 ensemble | **0.470338787** | **0.729708508** |
| Ensemble minus Corrected Ridge | **+0.004876459** | **−0.010596606** |

All coefficients below passed the original +0.01-bit guardrail relative to Ridge. The audit reconstructed each blend using the saved six-decimal component probabilities and rounded the output to six decimals. The selected blend reproduced every archived ensemble probability exactly.

| CS-CVA weight | Contrast | Macro Log Loss |
| ---: | ---: | ---: |
| 0.00 | 0.465462328 | 0.740305114 |
| 0.10 | 0.468314390 | 0.734965257 |
| **0.25** | **0.470338787** | **0.729708508** |
| 0.50 | 0.467563408 | 0.726835473 |
| 0.75 | 0.456789963 | 0.730948694 |
| 1.00 | 0.437632191 | 0.742910807 |

The standalone trailed Ridge on both metrics. Its contribution as a blend component improved both. Development selected the checkpoint and blend weight, so this is a selected Development result. Hidden Evaluation improvement remains unmeasured. Without a matched sentiment-only LoRA run, the gain cannot be isolated to caption supervision.

## Training and artifacts

Worker `abc23cf9b9a3a138c51e85185b8e2003058436f4`, original handoff commit `dbee2ab3c3c1cbb80835a8b5a3073d8f680456c0`, SigLIP2 revision `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`. The verified export matches the original handoff manifest and the packaged config and dependency locks. The worker used one Tesla T4 with CUDA 12.6, Python 3.12.10, Torch 2.7.1+cu126 and Transformers 4.57.6. The real-image preflight recorded zero initial LoRA feature error, 24 LoRA targets, 3,309,209 trainable parameters, microbatch 4 and 8.96 paintings/s.

| Epoch | Fit BCE | Caption cosine loss | Development Contrast | Development Log Loss | Epoch sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.605829 | 0.995442 | 0.375017 | 0.801927 | 178.96 |
| 2 | 0.511125 | 0.926340 | 0.413265 | 0.755153 | 178.59 |
| 3 | 0.485627 | 0.830768 | 0.426839 | 0.747265 | 179.03 |
| 4 | 0.471734 | 0.773966 | 0.435079 | 0.745524 | 178.89 |
| **5** | 0.464043 | 0.752479 | **0.437632** | 0.742911 | 179.11 |
| 6 | 0.459383 | 0.742981 | 0.437359 | **0.741620** | 178.92 |

All six epochs and 1,182 optimizer steps were verified. Epoch 5 wins the frozen primary Contrast selection and its inference tensors match the exported weights. Logged epochs took 1,073.50 seconds and caption preparation took 300.46 seconds, with 16,001 valid centered targets from 16,019 eligible pairs. The archived worker initialization and completion timestamps span about 24 minutes, consistent with the owner's approximately 25-minute session. Peak training CUDA allocated/reserved memory was 1.14/1.26 GiB. The preflight completed its real-gradient gate. The archive omits numerical gradient and clipping traces, but saved adapter tensors changed between every consecutive epoch. Their combined L2 changes in epochs 2–6 were 2.9142, 1.7097, 1.0105, 0.5280 and 0.2500.

| Artifact | SHA-256 |
| --- | --- |
| `exp036_results.zip` | `9c89585579a2af68e8589027d9f1269c1192bd4bf9373c99984f3e61fe554aa1` |
| `export_manifest.json` | `7b52eaaca43d45f7fde5c1c772e2f35b257dbaf2d4a16623ec448ae1b1b7283a` |
| Standalone submission | `d4feaa9dc09422eb78bf530b3cf9d36a6c76bcc2fc8e5cc50f57be5acdb25df1` |
| Ensemble submission | `bd6752c9a8a60f7c3754053afe11eec0187172838e9c67d43880b9c41fb5b274` |
| Selected CS-CVA weights | `302ae761c3d045fa4094362ce0f9495aed0874b117a1abf1438bdf877058f5d0` |
| Recovered deployment manifest | `cabd6e52c7e614dc403f10da995adff3bebdc9a14251cf1150b58649f7574138` |

The archive passed ZIP integrity and verification of all 39 manifest files. Both standalone and ensemble ZIPs passed exact official key coverage, schema and numeric validation. Both separate downloads matched their embedded ZIPs byte for byte. Original archives and the audit receipt are preserved outside Git under `runs/exp036/closure/`. The deployable bundle is under `runs/exp036/deployment/`. The Evaluation runner reproduced both sets of saved Development probabilities, and fresh-pixel Corrected Ridge inference passed on CPU through all three real frozen encoders. Fresh CS-CVA GPU inference using the exact So400m base remains unmeasured locally. [Evaluation commands](../docs/exp036_evaluation.md)
