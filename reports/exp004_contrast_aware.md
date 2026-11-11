# EXP-004 contrast-aware learning for ST3

This study tests whether adding a painting-level term to Model C's weighted binomial objective improves the official within-painting Pearson metric while controlling Macro Log Loss. All variants use the same full interaction architecture, frozen 512-dimensional CLIP features, and exact five painting-grouped folds from EXP-002. EXP-002's ablation provides the evidence for visual interactions beyond language priors. This experiment varies the objective while keeping those interactions. [Organizer task and evaluation rules](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#evaluation) · [EXP-002 report](exp002_ablation.md)

## Frozen protocol

- **Base loss.** Sum the Bernoulli NLL over positive and negative votes, add squared L2 over language offsets, shared visual weights and all interaction weights divided by `2C`, then divide the total by the number of eligible votes. `C = 0.3` and the intercept is unpenalized, matching the original weighted objective and penalty geometry. The loss is computed in nats. Held-out Macro Log Loss follows the official scorer and uses bits.
- **Pairwise term P.** Huber on the difference of predicted and observed language gaps, with `delta = 0.2`. Average over unordered pairs within each painting, then over paintings with at least two eligible languages. Constant targets still supply zero-gap supervision.
- **Pearson term R.** Average `1 − r` over paintings with at least three eligible languages and nonconstant observed shares. Use the observed centered unit vector and a predicted centered norm of `sqrt(sum(q_centered²) + 0.01²)`. The term gives a constant prediction correlation zero with a finite gradient. This smoothing is used for fitting. The official scorer is unchanged.
- **Weights.** Predeclare P at `0.1` and `0.5` and R at `0.01` and `0.05`. Huber's quadratic region and capped slope put it on a smaller scale than the bounded zero-to-two Pearson loss. The larger weights provide a fivefold perturbation of each small setting while retaining the vote-level objective.
- **Optimization.** Start every variant from its fold's exact Model C fit. Use deterministic CPU L-BFGS-B with analytic gradients, two BLAS threads, `gtol=1e-6`, `ftol=1e-11` and a 700-iteration cap. The reference fit seed is 2027. The warm start and optimizer use no random initialization or minibatch sampling, so a second seed would repeat the same computation. A separate lambda-zero minimization uses the original solver tolerances, with a raw probability tolerance of `1e-10`. [L-BFGS-B stopping criteria](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-lbfgsb.html)
- **Selection.** Require all five fits to converge, mean Contrast gain of at least `0.005`, a positive gain on at least four folds, and mean Macro Log Loss within `0.01` bits of C. Choose the highest Contrast among eligible candidates. Within `0.002` of the best eligible mean, prefer the simpler Huber objective and then the smaller weight. The gain, win count and guardrail are engineering thresholds. Freeze the decision and artifact hashes before the separate development stage.

## Grouped CV results

The exact EXP-002 memberships cover 1,571 training paintings and 16,019 eligible pairs. Model C reproduced all of its saved EXP-002 out-of-fold probabilities after six-decimal rounding. Each model was scored on the same 1,496 Contrast-eligible held-out paintings across folds. Table entries report fold means and sample SDs.

| Model | Lambda | Cultural Contrast ↑ | Mean change vs C | Macro Log Loss ↓ | Mean change vs C | Fold wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C · Weighted BCE | 0 | 0.435705 ± 0.024291 | — | 0.738056 ± 0.005381 | — | — |
| P-small | 0.1 | 0.436373 ± 0.024733 | +0.000669 | 0.737902 ± 0.005414 | −0.000154 | 5/5 |
| P-large | 0.5 | 0.437319 ± 0.024805 | +0.001614 | **0.737765 ± 0.005406** | −0.000291 | 5/5 |
| R-small | 0.01 | **0.437539 ± 0.024498** | +0.001834 | 0.737824 ± 0.005357 | −0.000232 | 5/5 |
| R-large | 0.05 | 0.436171 ± 0.021635 | +0.000466 | 0.738630 ± 0.005001 | +0.000573 | 2/5 |

Each fold entry below gives Cultural Contrast / Macro Log Loss.

| Fold | C | P-small | P-large | R-small | R-large |
| ---: | --- | --- | --- | --- | --- |
| 1 | 0.450409 / 0.739601 | 0.450486 / 0.739119 | 0.451324 / 0.738894 | 0.452004 / 0.738962 | 0.451088 / 0.739552 |
| 2 | 0.434233 / 0.731610 | 0.434293 / 0.731690 | 0.435338 / 0.731564 | 0.435301 / 0.731776 | 0.433639 / 0.733246 |
| 3 | 0.411972 / 0.742971 | 0.412510 / 0.742891 | 0.414256 / 0.742743 | 0.414584 / 0.742781 | 0.419243 / 0.743593 |
| 4 | 0.468554 / 0.733164 | 0.470607 / 0.732791 | 0.471719 / 0.732693 | 0.471204 / 0.732684 | 0.464390 / 0.733612 |
| 5 | 0.413355 / 0.742934 | 0.413970 / 0.743018 | 0.413957 / 0.742929 | 0.414602 / 0.742917 | 0.412496 / 0.743144 |

**NO-GO for development promotion.** All variants passed the **0.748056-bit** loss guardrail. Every mean Contrast gain fell below **+0.005**, and R-large also missed the fold-win criterion. The frozen decision retained C. The development command recorded the fallback without scoring any new predictions, so the development bootstrap is inapplicable. C's earlier development scores remain **0.446564 Contrast** and **0.751995 Macro Log Loss**. [Original development results](baseline_dev.md)

## Ordering, language and painting effects

- **Relative ordering.** As a descriptive check, average pair-order agreement over the same scored paintings, dropping observed ties and giving predicted ties half credit. Agreement was 69.602% for C, 69.648% for P-small, 69.727% for P-large, 69.710% for R-small and 69.625% for R-large. P-large had the strongest change on this rank-based diagnostic, while R-small led on official Pearson Contrast. Both slightly lowered Macro Log Loss. R-large traded a small mean Contrast gain for worse probability loss and won on only two folds. Macro Log Loss measures overall probabilistic accuracy, including factors beyond calibration.
- **Languages.** Fold-averaged Log Loss improved in 17 of 21 macro-eligible languages for P-large, 15 for R-small and seven for R-large. For R-small, Thai (−0.000932) and IsiZulu (−0.000694) improved most, and Indonesian (+0.000947) regressed most. For R-large, the largest regressions were Arabic (+0.003958) and Uzbek (+0.002332). These are comparisons of prediction errors against the observed samples.
- **Paintings.** R-small improved on 825 and regressed on 671 scored paintings. Its mean painting-level gain remained +0.001398 after removing its ten largest gains. P-large improved 817 paintings and retained +0.001200 after the same check. R-large fell to −0.001484 under that check. Its largest loss was on Ralph Hotere's *Avignon Painting*, where Contrast moved from 0.916172 to −0.286981.
- **Vote support.** Nearly all scored paintings (1,494 of 1,496) had at least half of their eligible language pairs at exactly five votes. That split provides too little balance for a painting-level support comparison. Descriptive all-pair loss changes for R-large were +0.000794 bits on the 14,494 five-vote pairs and −0.000151 on the 1,525 pairs with more votes. For R-small they were −0.000239 and −0.000386, respectively. Vote counts and group composition vary together, and these data alone cannot establish noise amplification.

## Optimization and verification

All 20 experimental fits met the gradient tolerance within the 700-iteration budget. Iteration ranges were 77–92 for P-small, 78–111 for P-large, 78–98 for R-small and 185–366 for R-large. At initialization, weighted auxiliary contributions averaged 0.00237, 0.01184, 0.00470 and 0.02352 nats per vote-equivalent in that order, alongside a BCE NLL of about 0.481 nats per vote. All five lambda-zero minimizations stopped at the original C weights. Maximum raw-probability discrepancy was **2.50e-15**, below the **1e-10** tolerance.

The main CV run took **152.1 seconds** on an Intel Core i7-13620H using the cached embeddings and CPU only. It used code revision `61edc10bbce64dbd443d31bc34dd652eaf0143cc`. The scientific protocol is in `configs/exp004.json`. All 47 tests passed. A replay of R-large on fold 1 reproduced all 3,212 predictions and its optimizer diagnostics exactly. Artifact checks validated all 80,095 CV probabilities and preserved the hashes of 35 earlier run files and both embedding caches.

A separate train-only diagnostic after the selection freeze initialized from the same fold C weights and ran `fit_contrast` with `weight=0` and `zero_optimizer=optimizer`. This controls for taking more optimization steps at the tighter experimental tolerances. The resulting BCE-only fit scored **0.436082 Contrast** and **0.737946 Macro Log Loss**, changes of **+0.000377** and **−0.000110** versus official C. R-small's Contrast advantage over this control was **+0.001457**. The diagnostic kept the frozen promotion decision fixed and saved fold metrics to `runs/exp004_checks/bce_polish.csv`.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py contrast-cv
.venv/Scripts/python.exe scripts/run_st3.py contrast-dev
.venv/Scripts/python.exe -m pytest -q
```

Local artifacts under `runs/exp004_contrast_aware/cv/` include per-fold metrics and deltas, per-language Log Loss, per-painting Contrast and support, all predictions, zero-weight controls, provenance and the frozen decision. `dev/run.json` records the verified fallback.

## Interpretation and Task 005

The smaller auxiliary perturbations produced consistent but sub-threshold improvements under these fixed folds and weights. Stronger Pearson weighting increased optimizer cost and concentrated more of its gains in a few paintings. The overlapping fold training sets and small vote counts limit what these comparisons establish about generalization. EXP-002's controlled image and interaction ablations remain the evidence for visual conditioning. EXP-004 tested the loss within that architecture.

For Task 005, keep C's vote-level objective and use the same grouped protocol to test a single stronger frozen visual representation against CLIP ViT-B/32, subject to the competition's external-resource rules. The earlier ablation showed a material contribution from correctly matched images, while changing interaction rank and adding the tested auxiliary losses fell short of the promotion criteria.
