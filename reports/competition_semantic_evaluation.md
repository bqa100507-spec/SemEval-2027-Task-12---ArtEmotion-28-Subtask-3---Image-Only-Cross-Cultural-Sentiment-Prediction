# Frozen Qwen3-VL semantic evaluation

**NO-GO. Retain Corrected Ridge.** The combined model improved pooled Contrast and Macro Log Loss on both P2 populations and won every fold on Contrast. Its **+0.002687** test-like Contrast gain fell short of the frozen **+0.005** advancement threshold. The owner-reported hosted Corrected Ridge result remains 0.4655 Contrast on submission #1063064.

## Pooled results

| System | Full P2 Contrast | Full P2 Macro Log Loss | Test-like Contrast | Test-like Macro Log Loss |
| --- | ---: | ---: | ---: | ---: |
| A. Corrected Ridge | 0.454134 | 0.708500 | 0.463108 | 0.717479 |
| B. Semantic-cultural ridge | 0.381494 | 0.786210 | 0.388089 | 0.799487 |
| C. Corrected Ridge with the visual-semantic head | 0.456873 | 0.706068 | 0.465795 | 0.715191 |

Full P2 has 13,600 pairs and 1,269 Contrast-eligible paintings. Test-like P2 excludes English, Arabic and Chinese, leaving 11,643 pairs and 1,252 Contrast-eligible paintings. Macro Log Loss is in bits. The scorer averages within-painting Pearson correlations and applies the documented language eligibility rule to Macro Log Loss. These are matched, repeatedly used P2 folds. The bootstrap quantifies painting-sampling uncertainty within these folds. [Evaluation specification](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#evaluation)

## Fold results

Each cell reports Contrast / Macro Log Loss.

| Fold | Full A | Full B | Full C | Test-like A | Test-like B | Test-like C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.443922 / 0.664179 | 0.372637 / 0.758171 | 0.447116 / 0.662033 | 0.450869 / 0.677251 | 0.382348 / 0.775735 | 0.454522 / 0.675324 |
| 2 | 0.459778 / 0.724501 | 0.406802 / 0.788367 | 0.465890 / 0.720001 | 0.466237 / 0.733380 | 0.412340 / 0.800617 | 0.472600 / 0.728830 |
| 3 | 0.437806 / 0.710339 | 0.350939 / 0.797149 | 0.437933 / 0.710624 | 0.441599 / 0.721110 | 0.354513 / 0.813095 | 0.442101 / 0.721812 |
| 4 | 0.469969 / 0.735697 | 0.387069 / 0.786940 | 0.471209 / 0.730994 | 0.485596 / 0.742602 | 0.395837 / 0.796677 | 0.486316 / 0.737891 |
| 5 | 0.458720 / 0.710922 | 0.389547 / 0.804021 | 0.461732 / 0.710051 | 0.471317 / 0.716054 | 0.395278 / 0.814734 | 0.473463 / 0.715334 |

## Paired uncertainty and complementarity

| Scope | Comparison | Δ Contrast | 95% painting-paired CI | Contrast fold wins | Painting wins / losses | Δ Macro Log Loss | Δ Contrast after 5% trim |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Full | B − A | −0.072640 | [−0.085220, −0.060353] | 0/5 | 474 / 795 | +0.077710 | −0.055299 |
| Full | C − A | +0.002739 | [+0.000659, +0.004834] | 5/5 | 680 / 589 | −0.002432 | +0.001095 |
| Test-like | B − A | −0.075019 | [−0.088996, −0.061323] | 0/5 | 470 / 782 | +0.082009 | −0.054759 |
| Test-like | C − A | +0.002687 | [+0.000380, +0.004994] | 5/5 | 660 / 592 | −0.002288 | +0.001176 |

The C gains survive trimming the largest absolute painting deltas, although they shrink by more than half. Fold 2 supplies the largest gain and fold 3 has the smallest. Semantic-only performs substantially worse, and its painting-centered residual correlations with Corrected Ridge are 0.912 on full P2 and 0.910 on test-like P2. The fixed joint blend shows modest practical complementarity within P2. The design compares the full visual-semantic system to Corrected Ridge and lacks a matched visual-only version of the joint head, so the measured blend gain cannot be isolated to semantic features alone.

C passed the +0.002 full-P2 Contrast threshold, positive test-like paired lower bound, at least four test-like fold wins, positive trimmed gain, and Log Loss guardrail. The test-like effect size gate alone failed. The protocol stops at this decision. Development extraction, final refit, packaging, and submission did not run. Organizer confirmation of generated semantic intermediates remains unresolved. [Organizer FAQ](https://github.com/artemotion-28/artemotion-28#faqs)

## Execution and verification

The original checkpoint, compact prompt, schema, SDPA configuration, greedy decoding, and 72-token limit were frozen. Six verified descriptions were reused and 1,565 artworks were generated once. The run sealed all 1,571 training artworks in 262 batches. Qwen used image pixels. The five-fold predictor fitted the semantic vocabulary, SVD, and heads only on each fold's fitting paintings. The 236 historical confirmation artworks contributed pixels to extraction and their votes stayed outside CV. All ten heads converged.

The cached grounding review found that the principal scene cues were mostly visible. One crowd scene was described as a protest without sufficient visual evidence. The raw text was preserved. Four of 1,571 descriptions failed the uncertainty check, three within P2. Their zero semantic vectors preserved the original fit and scoring population. No claim was filled in and no output hit the token cap. [Pinned recovery recipe](../configs/competition_semantic_recovery.json) · [Evaluation config](../configs/competition_semantic_evaluation.json)

The RTX 4060 Laptop GPU job used **5,276.11 seconds** of charged wall time, including process startup, asset verification and model loading, within the owner-authorized 12,600-second cap. Generation averaged 3.31 seconds per new artwork. The measured batches averaged 3.33 seconds per new artwork end to end. Peak allocated/reserved GPU memory was **3.28 / 3.44 GiB**. CPU CV took **87.80 seconds**. The protected E50+L predictions, Fixed Ridge heads and ZIP, Corrected Ridge bank and ZIP, and original benchmark matched their recorded hashes.

The scientific run used local commit `7bf00b09e6566813cb8af9eed12cb2dadce1723d`. All 308 clean-checkout tests passed with two skips. Three legacy notebook tests failed in the working tree because of existing local notebook edits. Those files were preserved. Scores, pair coverage, fold assignments, and the decision replayed exactly from the sealed OOF against the pinned P2 votes. The OOF and metrics SHA-256 values are `6396df2d37092acc01d93c25db4d067c328e3a9858f1b65e262f6fe6533c0b81` and `f85b830b76a51a967ca1590d0a7f01c28e04e477c3bdf64b3310c137cb3878f3`.

Local artifacts are under `runs/competition_semantic_vlm/evaluation/`, with `extraction.json`, `grounding_review.json`, `gpu_budget.json`, `cv/oof.csv`, `cv/metrics.json`, and `verification.json`. The CPU preparation entry point is `python scripts/run_competition_semantic_evaluation.py --prepare-only`. The authorized extraction and CV used the same runner with `--authorize-3p5h-rtx4060` and the existing pinned recovery runtime.
