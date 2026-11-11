# EXP-002 ST3 baseline ablation and reliability

Run on 2026-11-10 from code revision `823072e35785140d5d0218a4fce6abfd06670f94` and organizer revision `cf7cc444cf56617f3b8417497c6930ad172c5c99`. The study tests whether shared visual features improve positive-share estimates, whether language-conditioned visual weights add within-painting Contrast, and whether that gain depends on matching the correct image to its painting.

## Protocol

| Model | Inputs and fit |
| --- | --- |
| A · Language prior | Task 001's unweighted mean of eligible training-pair positive shares per language, with fallback 0.7. |
| B · Shared visual weights | Weighted LR on the frozen CLIP embedding and the target-language one-hot vector. The same image coefficients apply to every language. |
| C · Language-specific weights | Task 001's weighted LR with the image embedding, language one-hot vector and their outer product. |
| D · Shuffled inference images | C's fitted weights and original development keys. Each painting receives the embedding of a different development painting, shared by all its language rows. |

B and C both use L2 regularization with `C = 0.3` and the same L-BFGS solver and vote-count objective. Each pair contributes positive and negative examples weighted by its observed counts. Training uses 16,019 eligible pairs from 1,571 paintings in the original train split. We verified the Task 001 cache identity and reused all 1,656 frozen ViT-B/32 embeddings. Development has 849 pairs, 85 paintings and 22 languages. Predictions use six-decimal rounding, and validation votes enter after predictions are fixed. Contrast uses the same 82 paintings for all models. Two paintings have fewer than three eligible languages and one has constant observed shares. Macro Log Loss averages 17 languages with at least 20 pairs.

## Development results

| Model | Cultural Contrast ↑ | Macro Log Loss ↓ |
| --- | ---: | ---: |
| A · Language prior | 0.364707 | 0.867784 |
| B · Shared visual weights | 0.370386 | 0.766309 |
| C · Language-specific weights | **0.446564** | **0.751995** |
| D · Inference shuffle mean | 0.285993 | 1.015695 |

A and C reproduced both Task 001 prediction CSVs byte for byte and matched their complete metric records. The project owner reported `0.4466` from Codabench submission `#1044742`, which agrees with C's local Contrast rounded to four decimals. B is close to the organizer's shared-weights reference of 0.370 Contrast and 0.766 Log Loss. [Pinned organizer reference](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/baselines/st3/README.md#results-development-data)

B gained **0.005679 Contrast** over A and reduced Macro Log Loss by **0.101475 bits**. This comparison changes both visual input and the estimator of language effects. B's shared image term shifts every language's logit on a painting by the same amount. The sigmoid can change the spacing between predicted shares and therefore their Pearson correlation, while the ranking of languages follows the fitted language offsets. C can also change that ranking as the image changes. C's gain over B isolates the addition of the interaction block under the fixed training protocol and comes with a further **0.014314-bit** reduction in Macro Log Loss.

## Painting-paired reliability

Each of the 10,000 fixed-seed bootstrap draws resamples 82 scored paintings with replacement and uses the same painting indices for all models. We take the mean of the paired Contrast differences and use the 2.5th and 97.5th percentiles. An eligible painting with constant predictions contributes zero under the shared scorer.

| Comparison | Mean Contrast difference | 95% percentile interval | Improved / regressed paintings |
| --- | ---: | ---: | ---: |
| C − A | +0.081857 | [+0.040855, +0.122778] | 57 / 25 |
| C − B | +0.076178 | [+0.033904, +0.118978] | 56 / 26 |

The median C − B change was **+0.062740**. With the three largest improvements omitted, the remaining mean was **+0.056699**. The intervals quantify variation across resampled paintings conditional on the fixed fits and observed votes. They provide evidence about this development split. Individual paintings still show substantial variation.

## Inference-time image shuffle

Each seed from 12000 through 12019 draws a permutation of sorted development painting IDs and rejects draws with any fixed point. Every seed assigns all 85 embeddings once. The permutation uses painting IDs and a seed, and C keeps its original training fit. All 20 draws preserve the 849 language-pair keys and the same metric denominators.

| Metric | Mean | Sample SD | Min–max |
| --- | ---: | ---: | ---: |
| Cultural Contrast | 0.285993 | 0.031820 | 0.213349–0.343720 |
| Macro Log Loss | 1.015695 | 0.033941 | 0.963633–1.086687 |

Correct image assignment improved C's Contrast by **0.160571** relative to the shuffle mean. Even the best shuffled draw fell below the language prior. This supports dependence on the correct visual input under the tested perturbation. The diagnostic replaces inputs at inference and uses the same fitted model throughout. Mismatched images can push predictions away from useful language priors, so the drop measures sensitivity to that perturbation.

## Language-level findings

C reduced Log Loss against A for **20 of 22** languages, including all **17 of 17** macro-eligible languages. Against B, C improved **15 of 22**, including **12 of 17** macro-eligible languages. The gains span multiple languages, with different magnitudes and regressions.

Among macro-eligible languages, the largest reductions from B to C were Tamil (**−0.060654**, 37 pairs) and Kyrgyz (**−0.057515**, 28 pairs), followed by Swahili (**−0.039164**, 55 pairs) and Yoruba (**−0.037077**, 59 pairs). Hausa (**+0.020896**, 49 pairs) and Chinese (**+0.011352**, 61 pairs) regressed most among macro-eligible languages. Indonesian had the largest overall regression (**+0.081030**, 12 pairs), with support below the macro threshold. These comparisons describe model errors for the observed annotation samples. The CSV records pair counts, pooled positive vote shares, the mean, median and range of pair shares, and all model losses and deltas.

## Painting-level findings

| Selected extreme | Eligible languages | Contrast B | Contrast C | C − B |
| --- | ---: | ---: | ---: | ---: |
| Ad Reinhardt, *Yellow Painting* | 9 | −0.443543 | 0.319094 | +0.762637 |
| Dmitry Levitzky, *Portrait of Countess N. F. Vorontsova* | 14 | −0.097371 | 0.471944 | +0.569315 |
| Pierre-Auguste Renoir, *Reading the Part* | 10 | 0.499278 | 0.110693 | −0.388585 |
| Édouard Manet, *The Head of Christ* | 10 | 0.668651 | 0.289927 | −0.378724 |

On *Yellow Painting*, C raised Swahili from 0.720 to 0.860 and Korean from 0.662 to 0.779, against five positive votes out of five for each. C still predicted 0.763 for Tamil, where the observed share was zero out of five. Better Contrast can coexist with large absolute errors. On *Reading the Part*, C raised Chinese from 0.885 to 0.909 despite an observed share of four of six, while lowering Uzbek from 0.924 to 0.886 against five of five. That reverses the ordering B had for those two languages. Seven of ten languages on that painting had all-positive observed shares, leaving limited variation in the gold vector.

## Train-only painting-grouped cross-validation

Five fixed-seed folds partitioned the 1,571 eligible original training paintings, holding out each painting once. All three variants used identical folds and fixed hyperparameters. Each fold had 295–305 paintings eligible for Contrast and 21 languages for Macro Log Loss.

| Fold | Contrast A | Contrast B | Contrast C | C − B | Macro Log Loss A / B / C |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.342913 | 0.349471 | 0.450409 | +0.100938 | 0.900026 / 0.757240 / 0.739601 |
| 2 | 0.336628 | 0.341731 | 0.434233 | +0.092501 | 0.895124 / 0.744410 / 0.731610 |
| 3 | 0.291380 | 0.298652 | 0.411972 | +0.113320 | 0.870434 / 0.758633 / 0.742971 |
| 4 | 0.362457 | 0.374161 | 0.468554 | +0.094394 | 0.887461 / 0.751015 / 0.733164 |
| 5 | 0.323824 | 0.323696 | 0.413355 | +0.089659 | 0.888209 / 0.760922 / 0.742934 |
| Mean | 0.331440 | 0.337542 | 0.435705 | +0.098162 | 0.888251 / 0.754444 / 0.738056 |

C beat both comparators on both metrics in all five folds. The C − B Contrast difference had sample SD **0.009434** across folds. These fold summaries are descriptive because training sets overlap. They are kept separate from the official development comparison.

## Reproducibility and artifacts

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-analysis.txt
.venv/Scripts/python.exe scripts/run_st3.py ablation --cv --plots
.venv/Scripts/python.exe -m pytest -q
```

The complete run took **40.1 seconds** on a Windows 11 Intel Core i7-13620H, with **28.7 seconds** for grouped CV and the LR thread limit set to four. All Task 002 fitting, inference and analysis ran on CPU, using the cache generated in Task 001. `run.json` stores exact revisions, hashes, settings, seeds, hardware, library versions and timings.

All 31 tests passed. A second run with `--cv` and `HF_HUB_OFFLINE=1` reproduced all 13 CSV and metric JSON artifacts byte for byte. The standalone `baseline` command also reproduced Task 001 predictions and full metrics in a fresh output directory. SHA-256 checks confirmed the original run and embedding cache files were preserved. Artifact checks verified 20 distinct derangements, exact out-of-fold train-only coverage and all 10,000 paired bootstrap draws.

Local results are in `runs/exp002_baseline_ablation/`. `predictions/` holds A–C predictions. `per_language.csv`, `per_painting.csv`, `observed_vs_predicted.csv` and `bootstrap_distribution.csv` hold the analysis tables. `diagnostics/` holds the shuffled assignments, predictions and scores. `train_only_cv_folds.csv` and `train_only_cv_predictions.csv` hold fold summaries and out-of-fold predictions. Figures are `contrast_reliability.png`, `language_log_loss.png` and `painting_examples.png`.

## Conclusion and Task 003

**Yes, within the observed ST3 data.** Language-conditioned visual weights improved average Contrast beyond the language prior and shared visual weights. The paired development intervals favor C, the advantage recurred in all five train-only held-out painting groups, and C's Contrast dropped when the painting-to-image mapping was broken. The size of the gains varies across paintings and languages. Many observed shares come from small vote counts, and these runs use a single frozen representation and one public development split. Performance on evaluation paintings remains an open question.

For Task 003, test **low-rank language-conditioned visual interactions** against C with the same embeddings, grouped folds and fixed regularization protocol. The goal is to retain the Contrast gain while checking whether sharing interaction parameters reduces painting and language regressions. Predefine macro Log Loss as a guardrail. This study varied the interaction block while holding the encoder and objective fixed, so it gives a clearer rationale for studying interaction parameterization first.
