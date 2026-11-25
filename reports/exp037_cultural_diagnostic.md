# EXP-037 Cultural Contrast diagnostic

The original C6 blend improved ordering on the reused Development panel, especially after excluding English, Arabic and Chinese. It still shares most of its centered error with EXP-036. The next useful experiment should target image-dependent changes in language ordering and their stability on sparse panels.

## Selection and uncertainty

The [144-row grid](exp037_assets/checkpoint_alpha_grid.csv) crosses all 18 checkpoints with the original eight weights, with both populations and eligibility counts. Every Stage A checkpoint maximizes full Contrast at alpha zero. B6/0.10 and C6/0.20 win their respective full-scope grids. Thus the expanded checkpoint search found no better checkpoint for the original type of blend. On the Evaluation-like diagnostic alone, maxima were A2/0.10 at 0.449294, B5/0.25 at 0.450991 and C6/0.35 at 0.455358. Those three comparisons reuse Development and do not change the original selection.

The [bounded extra recipes](exp037_assets/additional_ensembles.csv) contain six equal-probability averages, each alone and as a fixed 20% addition to EXP-036, and three specified three-way mixes. The six averages use selected A/B/C, selected B/C, A3/A4, B5/B6, C5/C6 and C6/C7/C8. The C5/C6 blend scored 0.472411 / 0.453560 full / Evaluation-like Contrast. The C6/C7/C8 blend scored 0.471745 / 0.452783. Averaging later C predictions did not repair the C7/C8 decline. The only recipe exceeding the original full score was flat 65% Ridge, 15% CS-CVA and 20% C6. It needs the same three component models as the original C6 ensemble. Probability averages use saved predictions. Parameter averaging was not evaluated.

| Comparison | Scope | Contrast Δ | 95% paired interval | Wins / losses | Remove top 5% absolute / top 5% wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Original C6 blend − EXP-036 | Full | +0.002480 | [−0.003694, +0.008834] | 47 / 35 | +0.001511 / −0.001629 |
| Original C6 blend − EXP-036 | Evaluation-like | +0.005290 | [−0.001389, +0.012068] | 47 / 32 | +0.005422 / +0.001807 |
| Post-hoc 65/15/20 − EXP-036 | Full | +0.002805 | [−0.002783, +0.008354] | 49 / 33 | +0.002363 / −0.000773 |
| Post-hoc 65/15/20 − EXP-036 | Evaluation-like | +0.004916 | [−0.001380, +0.011138] | 47 / 32 | +0.007168 / +0.001888 |
| B6 − A4 | Full | +0.019917 | [−0.000724, +0.040117] | 55 / 27 | +0.020749 / +0.008643 |
| B6 − A4 | Evaluation-like | +0.025151 | [+0.002730, +0.047546] | 51 / 28 | +0.021412 / +0.014188 |
| C6 − B6 | Full | +0.005212 | [−0.010913, +0.021588] | 44 / 38 | +0.006607 / −0.004932 |
| C6 − B6 | Evaluation-like | +0.006582 | [−0.011140, +0.024877] | 42 / 37 | +0.005406 / −0.003780 |

No paired differences tied at 1e−12. The percentile bootstrap resamples matched paintings 20,000 times, seed 203038. Macro Log Loss resamples all available paintings in each scope, keeping that scope’s originally qualifying languages fixed. For the original C6 blend versus EXP-036, loss differences were **−0.001621** bits full (95% interval [−0.005550, +0.002209]) and **−0.002012** bits Evaluation-like ([−0.006520, +0.002435]). Removing only the largest win leaves +0.001526 full and +0.004336 Evaluation-like. The full gain relies partly on the best few paintings, with a broader gain on the filtered panel. All intervals condition on the saved checkpoints, recipes and reused paintings. They do not correct for epoch selection, post-hoc weights or past Development tuning.

## Global and cultural errors

Within each artwork, use unweighted means over the **same observed language support** for votes and predictions. On the Contrast-eligible paintings, give each painting equal weight. Global RMSE compares its predicted and observed means. Centered RMSE compares the deviations, averaging squared errors within each painting first. Within-painting Spearman uses averaged ranks. Pair ordering excludes gold ties and gives predicted ties half credit. Deviation RMS ratio is the square root of the ratio of average within-painting variances. Slope is the pooled within-painting covariance divided by predicted variance under the same painting weights. It measures a descriptive linear magnitude correction.

| Model | Full Pearson | Median [10th, 90th] | Negative / 82 | Spearman | Pair order | Global RMSE | Centered RMSE | RMS ratio / slope |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Corrected Ridge | .465462 | .510 [.083, .762] | 4 | .420510 | 71.72% | .137673 | .180153 | .488 / .991 |
| CS-CVA | .437632 | .466 [.076, .766] | 6 | .393583 | 70.05% | .143969 | .182442 | .437 / 1.060 |
| EXP-036 | .470339 | .509 [.077, .759] | 4 | .415927 | 71.43% | .130891 | .179437 | .464 / 1.057 |
| A4 | .420828 | .455 [.109, .720] | 4 | .378729 | 69.46% | .152037 | .184710 | .413 / 1.069 |
| B6 | .440745 | .468 [.130, .757] | 6 | .383864 | 69.68% | .149695 | .181940 | .473 / .988 |
| C6 | .445957 | .485 [.123, .771] | 6 | .413188 | 71.44% | .145006 | .180275 | .509 / .949 |
| Original C6 blend | .472818 | .514 [.120, .749] | 4 | .428565 | 72.20% | .129896 | .178904 | .467 / 1.060 |
| Post-hoc 65/15/20 | .473143 | .514 [.115, .756] | 4 | .424733 | 71.89% | .130580 | .178932 | .471 / 1.050 |

In the Evaluation-like scope the original blend improves Spearman from **.407141 to .420891** and ordered pairs from **70.98% to 71.88%**, with global RMSE **.143400 to .142811** and centered RMSE **.178929 to .178399**. C6 captures more variation than B6 but its test-like slope is **.892**. C7/C8 expand RMS further and their slopes fall to **.857/.864**, while ordering drops from **71.05%** to **69.64%/69.96%**. These changes show a ranking degradation alongside magnitude misalignment. The observed per-language rates include finite-vote noise. A prediction-to-observed RMS ratio below one does not establish harmful under-dispersion, and scaling C6 up simply to match observed RMS is unsupported.

[Full per-painting contributions](exp037_assets/per_painting_full.csv) · [Evaluation-like per-painting contributions](exp037_assets/per_painting_like.csv) · [Quantiles, eligibility and paired measures](exp037_assets/diagnostic_aggregates.json)

### Representative cases

- *Breast-feeding Mother* (**+0.079725** vs EXP-036) shows a woman nursing an infant in a dark setting. Hausa and Yoruba each had 5/5 positive votes in this Development panel. The blend moved their probabilities from .384/.395 to .419/.410. It reduced Korean from .243 to .242 with 1/5 positive, improving relative ordering and spacing while leaving substantial probability errors.
- *Weiss und Schwarz 4* (**+0.075326**) shows overlapping black and white rectangles. The C6 blend drops Korean from .760 to .728 and Uzbek from .868 to .852, both with 4/5 positive versus 5/5 or 6/6 for the other observed rows. All predictions for Uzbek remain high relative to several unanimous rows. The contrast gain occurs even in abstract imagery.
- *Untitled, 1955* (**−0.073881**) shows calligraphic black marks on a light ground. Igbo is the only non-tied observed rate, at 3/5 versus 1/5 in the other four observed languages. The blend lowers Igbo from .333 to .322 and raises Turkish from .227 to .235. It effectively ties or reverses the small incumbent Igbo–Malay predicted gap.
- *Steppe Day* (**−0.059149** full, **−0.069894** Evaluation-like) depicts an open landscape with a small rider and large sky. The blend raises Kyrgyz from .669 to .699 against 1/5 positive and lowers Tagalog from .829 to .820 against 5/6. The panel exposes a ranking and probability miss on the same image. None of these cases establishes a language-group preference.

## Language effects and complementarity

The Beta(1,1) smoothed pooled **Train-only** language prior obtains **0.364353 / 0.867769** full and **0.328228 / 0.903822** filtered Contrast / loss. For the fitted neural heads, the archived logits recover from six-decimal probabilities. Remove the exact fitted intercepts and take a mean over the observed panel to define a **shared proxy**, then center the remaining residual over that panel. This gauge identifies the separation of relative logits on available rows, but the model's true shared logit is defined over its entire fitted language vocabulary. Exact native feature decompositions would require rerunning all 25 head outputs for each image.

| C6 prediction diagnostic | Full Contrast | Evaluation-like Contrast |
| --- | ---: | ---: |
| Full | .445957 | .437744 |
| Fitted intercept alone | .336330 | .303244 |
| Shared proxy alone | .000000 | .000000 |
| Shared proxy + intercept, omit observed interaction | .335950 | .302900 |
| Remove fitted intercept, retain observed residual | .445910 | .439120 |
| Shuffle interaction across paintings within each language | .263944 | .239880 |

C6's observed-support shared-proxy SD is 1.216 logits across paintings. The centered intercept SD is .035 and centered residual SD .598 across rows. The latter still correlates **.634** with the centered fitted intercept. A projection of the learned interaction onto a broadly common feature direction can encode static language effects. Deleting the explicit intercept or observed-support residual alone cannot estimate the benefit of training a matched model without those terms. The label-free shuffle preserves marginal deviations within languages and breaks their painting alignment, producing a large drop. The earlier matched additive/interaction refits establish a stronger causal comparison for their linear recipe. [Bottleneck diagnosis](st3_bottleneck_diagnosis.md)

[Per-language full metrics](exp037_assets/per_language_full.csv) and [filtered metrics](exp037_assets/per_language_like.csv) report painting counts, votes, mean bias, Log Loss, centered RMSE and pair-order involvement for each observed language and model. Languages with no Development rows are outside this audit. Korean (55 paintings, 275 votes) and Hausa (49, 245) gain **0.011590** and **0.008686** loss bits from adding C6 to EXP-036. Turkish (29, 147) loses **0.004270** bits. Indonesian (12, 62) loses **0.014067** and is excluded from macro scoring. Urdu has one panel of five votes. The tables count a pair once for each language involved and use only non-tied observed comparisons on eligible paintings. They describe involvement in ordering errors, not a language-specific version of official Contrast.

| Model pair on full scope | Global prediction r | Centered prediction r | Global residual r | Centered residual r | Pair order agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge / CS-CVA | .85870 | .86616 | .86546 | .96077 | 83.41% |
| Ridge / C6 | .89524 | .89795 | .89646 | .96598 | 86.78% |
| CS-CVA / C6 | .91054 | .92094 | .90915 | .97369 | 89.86% |
| EXP-036 / C6 | .92589 | .92553 | .92349 | .97518 | 88.89% |

Pearson comparisons pool rows on the common eligible paintings. Residuals share noisy observed outcomes, which can inflate their correlation. The C6 blend gains **+0.006591** on the below-median Ridge-Contrast paintings and **−0.001632** on the rest. The gain across low/high CS-CVA paintings is **+0.000397/+0.004562**. Evaluation-like gains also favor lower-scoring Ridge paintings (**+0.013655/−0.003289**), while lower- and higher-scoring CS-CVA strata are similar (**+0.005838/+0.004728**). These post-hoc strata condition on noisy baseline scores and are exposed to regression to the mean. They suggest prioritizing correct image-specific language ordering on cases the strongest existing component misses. They do not justify a deployment gate fitted on those strata.

## Oracles, analysis only

For the original C6 blend, setting each painting's global mean or cross-language RMS to observed gold leaves unclipped Pearson **0.472818** exactly. A shift or a positive scalar multiplication preserves the centered direction. Clipping maps some rows nonlinearly and changes the resulting Contrast slightly. The two transformations produce **0.476581 / 0.652401** (gold mean) and **0.472317 / 0.785915** (gold RMS) after bounding probabilities, with 59 and 75 pre-bounding out-of-range rows. Gold mean replacement has more headroom for Log Loss than for Contrast, and forcing observed dispersion worsens loss. Keeping predicted probability values and sorting into the gold language order with tied gold levels averaged gives **0.974372** Pearson. Substituting gold deviations gives **1.000000** before clipping and **0.997211** with bounds. This direct label use identifies geometry, not an achievable score. No oracle predictions enter a bundle or submission.
