# EXP-013 Nonlinear image–language head

## Frozen protocol

M0 is the unchanged E50+L reference. M1 averages three independently initialized nonlinear heads in probability space. M2 averages M0 and the unrounded M1 probability at fixed 50/50 weight. M1 and M2 each round their final probabilities to six decimals. An official language absent from the fit uses M1's shared terms, while M2 preserves the original E50+L value.

The input concatenates the existing unit CLIP-B/32, SigLIP 2 Base and CLIP-L/14 embeddings in that order and divides by √3. The 2,048 inputs pass through a 64-wide linear layer, LayerNorm, GELU and dropout at 0.20. A shared logit, language offset and language-specific dot product divided by √64 give the final logit. The only inference inputs for M1 are the frozen features and the target language. With 25 training languages, each head has 132,954 parameters.

The optimization target is weighted binary cross-entropy on k/n with weight n, normalized by votes in each minibatch. Reported epoch losses are vote-normalized over the respective training or internal validation partition. The fixed AdamW settings are learning rate 0.001, weight decay 0.01, batch size 256 and gradient clipping at 1.0. The seeds are 2027, 2028 and 2029. Default Torch linear initialization and zero language offsets and embeddings are reproducible within the pinned runtime and device.

The five outer folds are the original EXP-002 painting-grouped folds. Within each outer training set, GroupShuffleSplit reserves 10% of paintings with seed 13027 plus the outer fold number. The three runs use the same internal split and select epochs independently. Select the strict minimum internal validation BCE, with patience 15 and a cap of 150 epochs. Restart each seed and optimizer before refitting on the entire outer training partition for its selected number of epochs. No held-out painting, image, or vote enters either fit. Feature scaling is fixed per image and uses no split statistics.

Commit `3eb0414` froze promotion before scoring. Require five complete folds, all seeded fits and valid predictions, a pooled Contrast gain of at least 0.001, at least three Contrast fold wins and mean Macro Log Loss within M0 +0.005 bits. Among eligible systems, take the highest pooled Contrast. The paired bootstrap uses 10,000 painting draws. The previously examined folds are not independent confirmation of generalization.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py nonlinear-cv
.venv/Scripts/python.exe scripts/run_st3.py nonlinear-dev
```

CUDA is selected when available. Use `--device cpu` for a CPU run. Outputs are under `runs/exp013_nonlinear_head/`. The development command verifies the frozen gate before loading validation gold or creating a candidate archive.

## Train-only results

Each score cell shows Cultural Contrast ↑ / Macro Log Loss ↓. The pooled scores cover all 16,019 outer pairs.

| Model | Fold mean | Pooled OOF | Contrast fold wins vs M0 | Painting wins / regressions | Pooled pair-order agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 · E50+L | 0.445498117 / 0.715718675 | 0.445643116 / 0.714879226 | — | — | 0.700935236 |
| M1 · Nonlinear | 0.437237540 / 0.720845086 | 0.437375561 / 0.720110553 | 0/5 | 714 / 782 | 0.698586858 |
| **M2 · 50/50 blend** | **0.447584438 / 0.711218677** | **0.447722823 / 0.710437122** | **4/5** | 775 / 721 | **0.703050346** |

| Fold | M0 | M1 | M2 |
| --- | ---: | ---: | ---: |
| 1 | 0.462693 / 0.708627 | 0.450840 / 0.724179 | 0.462170 / 0.707542 |
| 2 | 0.439750 / 0.716017 | 0.438157 / 0.715224 | 0.445939 / 0.709347 |
| 3 | 0.418114 / 0.713914 | 0.411748 / 0.711103 | 0.420490 / 0.706206 |
| 4 | 0.477616 / 0.718075 | 0.466447 / 0.728728 | 0.478481 / 0.715410 |
| 5 | 0.429317 / 0.721959 | 0.418996 / 0.724991 | 0.430842 / 0.717588 |

M1 lost 0.008268 pooled Contrast versus M0, with a paired 95% interval of [−0.014709, −0.001942]. The blend gained 0.002080 pooled Contrast, with a paired interval of [−0.001060, +0.005148]. It reduced pooled Macro Log Loss by 0.004442 bits and improved ordering agreement by 0.212 percentage points. Ordering averages equally over Contrast-eligible paintings, excludes tied gold pairs, and gives half credit for a predicted tie.

M2 alone cleared the frozen rule. The mean loss ceiling was 0.720718675 and the minimum pooled Contrast was 0.446643116. M1 was weaker than E50+L on all five folds. Its probabilities complemented E50+L in the blend, though the paired interval includes zero on the reused folds.

M2 improved pooled loss in 20 of 21 eligible languages. Indonesian improved most (−0.018781 bits), and Yoruba regressed by 0.000742 bits. Its largest Contrast gain was on Fred Sandback's *Untitled* (+0.326154), and its largest drop was on Chagall's *King David* (−0.369007). M1 lowered loss in 3 of 21 languages.

## Training behavior and compute

| Seed | Pooled standalone Contrast | Pooled standalone Macro Log Loss |
| --- | ---: | ---: |
| 2027 | 0.434045 | 0.721656 |
| 2028 | 0.428402 | 0.729230 |
| 2029 | 0.434033 | 0.726298 |

The seed Contrast spread was 0.005643. Averaging the seeds improved on each individual head's pooled Contrast. All 15 fits completed. Epoch selection chose 17–31 epochs, with median 23. All runs stopped after 15 epochs without improvement. From the selected epoch to that stop, online training BCE decreased by 0.01870 nats per vote on average and internal validation BCE increased by 0.01065. The pattern occurred for every seed and fold. The short refits use the validation-selected epochs to limit this overfitting.

On an RTX 4060 Laptop GPU, the CV command took 326.34 s wall-clock. Epoch selection and full refits accounted for 305.42 s, and seeded prediction for 0.393 s across the five held-out folds. Both candidate predictions share those fits. Peak CUDA allocation was 79.23 MiB, with 90 MiB reserved. The highest observed host process working set was 1.67 GiB. There were no backbone passes. Each head has 132,954 parameters, so the three heads add 398,862 parameters to M2.

The split audit replayed all five grouped splits with 126 internal validation paintings per fold and checked isolation by painting and image path. The frozen feature index contains distinct content for all 1,571 training images. The 16,019 E50+L outer rows matched EXP-010 byte for byte. All 48,057 final OOF predictions passed probability and exact-coverage checks. Fold scores, painting and language metrics, all 15 training curves, and per-seed runs are under `runs/exp013_nonlinear_head/cv/`.

## Guarded development result

M2 qualified on CV, so the command selected 25, 17, and 12 epochs for the three seeds using an internal split of the full original training set. It then refit all three on the full training set and scored the fixed blend on the 849 development pairs. The development command took 76.66 s on CUDA. The three fits took 67.58 s in total, and peak allocation was 82.72 MiB.

| Model | Cultural Contrast ↑ | Macro Log Loss ↓ |
| --- | ---: | ---: |
| E50+L | **0.46431737014951985** | 0.7409970449028339 |
| M2 | 0.4575245420226242 | **0.7398458749670416** |

M2's Contrast change was −0.006793, with a 10,000-draw painting-paired 95% interval of [−0.023552, +0.010072]. It improved 34 of 82 scored paintings and regressed on 48. Macro Log Loss fell by 0.001151 bits, with improvements in 10 of 17 eligible languages. Kyrgyz improved most (−0.019918 bits), and Turkish regressed most (+0.015073 bits). The development result did not confirm the CV Contrast gain.

The separate `runs/exp013_nonlinear_head/dev/submission.zip` validates against all 849 pairs and contains the exact M2 CSV. Its SHA-256 is `3ed3218e16e57533c0d1c024c29e8ddf4795b29095370ebd9a8a8bbde21de08a`. No submission was made. The original E50+L production command reproduced its development CSV byte for byte. All 682 historical SHA-256 checks matched, including 33 cache and checkpoint files. The Windows suite passed 122 tests, with one symlink fixture skipped, and both Python dependency checks passed.

Keep E50+L as the competition system. Retain the M2 artifact for analysis, and require independent painting evidence of a primary-metric gain before integrating nonlinear heads into production.
