# EXP-003 low-rank language-conditioned visual interactions

The study asks whether a factorized image–language interaction can retain Cultural Contrast and control Macro Log Loss with fewer parameters than the full Model C interaction. It uses the original Task 001 cache and the exact five EXP-002 train-only painting folds. Models A, B and C are the unchanged language prior, shared visual weights and full visual interaction baselines.

## Frozen protocol

- **Direct factorization.** For image embedding `x` and language `l`, the logit is `b0 + b[l] + w @ x + U[l] @ (V.T @ x)`. Ranks 2, 4, 8 and 16 each use initialization seeds 2027 and 4027 in every fold. Initialize the main effects with the same fold's fitted B and both factors independently from a zero-mean normal distribution with standard deviation 0.05. The full-training seed is fixed at 2027.
- **Objective.** Minimize `(sum vote NLL + (||w||² + ||b||²)/(2C) + (||U||²_F + ||V||²_F)/(2C)) / sum votes` with `C = 0.3`. The vote term is the aggregated binomial NLL in nats, omitting the data-only combinatorial constant. The intercept is unpenalized. Missing painting–language pairs have zero vote weight. Use the same main-effect penalty scale as the reference and one fixed factor-penalty multiplier of 1 to avoid adding a tuning axis. Factor-norm decay induces different shrinkage from C's squared Frobenius penalty on the full interaction coefficients. [Reference logistic objective](https://scikit-learn.org/1.6/modules/linear_model.html#logistic-regression)
- **Optimizer.** SciPy L-BFGS-B with an analytic gradient, 900 iterations maximum, `ftol=1e-10` and `gtol=2e-7`. The study requires a successful stop, a finite result and an objective no greater than its initialization. Iteration-cap results are retained as diagnostics. [L-BFGS-B termination rules](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-lbfgsb.html)
- **Post-hoc compression.** Extract the 25 × 512 block of C's interaction coefficients. Truncate the SVD at the same four ranks and preserve C's intercept, language offsets and shared visual coefficients. Each fold's diagnostic starts from the C trained on that fold's fit paintings.
- **Selection.** Average held-out metrics equally over folds and seeds. A direct rank needs all 10 fits converged and mean Macro Log Loss within 0.01 bits of C. Select the smallest eligible rank within 0.002 Contrast of the best eligible mean. The Log Loss margin and tie band are predefined engineering choices. The CV command persists the decision before the separate development command reads development votes.

## Exact-fold CV results

All A/B/C out-of-fold predictions matched EXP-002 exactly before scoring the new variants. The 1,571 training paintings and 16,019 eligible pairs were assigned to the same held-out folds. The table shows mean ± sample SD across five folds for references and SVD diagnostics, or across the ten fold–seed combinations for direct fits. The SDs describe dispersion across these fits. SVD parameter counts assume factor storage. The diagnostic evaluates the reconstructed weights.

| Model | Parameters | Cultural Contrast ↑ | Macro Log Loss ↓ | Converged fits |
| --- | ---: | ---: | ---: | ---: |
| A · Language prior | 25 | 0.331440 ± 0.026379 | 0.888251 ± 0.011225 | Closed form |
| B · Shared weights | 538 | 0.337542 ± 0.028306 | 0.754444 ± 0.006704 | 5/5 |
| C · Full interaction | 13,338 | **0.435705 ± 0.024291** | **0.738056 ± 0.005381** | 5/5 |
| LR-2 | 1,612 | 0.393666 ± 0.023700 | 0.764343 ± 0.008879 | 10/10 |
| LR-4 | 2,686 | 0.406309 ± 0.017991 | 0.762764 ± 0.007233 | 10/10 |
| LR-8 | 4,834 | 0.386806 ± 0.019628 | 0.771617 ± 0.006846 | 1/10 |
| LR-16 | 9,130 | 0.367014 ± 0.025618 | 0.781709 ± 0.006332 | 0/10 |
| SVD-2 | 1,612 | 0.381658 ± 0.031591 | 0.748903 ± 0.006019 | Post-hoc |
| SVD-4 | 2,686 | 0.414793 ± 0.029521 | 0.742594 ± 0.005768 | Post-hoc |
| SVD-8 | 4,834 | 0.426493 ± 0.024971 | 0.740782 ± 0.005755 | Post-hoc |
| SVD-16 | 9,130 | 0.434446 ± 0.023300 | 0.738612 ± 0.005515 | Post-hoc |

**Decision: retain Model C.** The guardrail was **0.748056 bits**, and all four direct ranks exceeded it. LR-4 was the strongest direct model on both means. It used 79.9% fewer parameters than C, with **−0.029396 Contrast** and **+0.024708 bits** relative to C. Across individual folds, LR-4 lost **0.015939–0.041917 Contrast** and added **0.021736–0.029448 bits**. Its largest seed difference within a fold was **0.000045 Contrast** and **0.000021 bits**. LR-2 also regressed on both metrics in every fold. C's original development results remain **0.446564 Contrast** and **0.751995 Macro Log Loss**. The frozen fallback produced only training-spectrum artifacts, so a low-rank development score and its paired bootstrap interval are inapplicable. [Task 002 development results](exp002_ablation.md)

At rank 2, direct fitting gained 0.012008 Contrast over truncation and added 0.015440 bits. At rank 4, direct fitting had lower Contrast and higher loss than truncation. Direct ranks 8 and 16 also trailed their SVD counterparts under the tested cap. **SVD-16 preserved the most performance**, finishing **0.001259 Contrast** below C and **0.000556 bits** above it. Storing SVD-16 in factor form would use **31.5% fewer parameters**. SVD-4 and SVD-8 also fell within the numerical loss margin as compression diagnostics. Truncation requires fitting full C first.

## Model C matrix structure

The full original-training interaction matrix was **25 × 512**, with singular values from **7.265108** to **0.177798**. The first component held 16.1% of squared matrix energy. Effective rank was **20.6235**, computed as the exponential Shannon entropy of singular values normalized by their sum. Effective rank ranged from 20.631 to 20.771 in the five fold fits. Squared singular-value energy and the singular-value entropy use different normalizations.

| Rank | Cumulative squared energy | Frobenius error | Relative Frobenius error |
| ---: | ---: | ---: | ---: |
| 2 | 28.20% | 15.3616 | 84.73% |
| 4 | 47.70% | 13.1108 | 72.32% |
| 8 | 69.23% | 10.0571 | 55.47% |
| 16 | 93.01% | 4.7947 | 26.45% |

The spectrum has a compressible tail and substantial energy beyond the first few components. The modest CV drop for SVD-16 shows that coefficient reconstruction error and predictive loss need separate measurements. The spectrum describes the learned coefficient matrix. Cultural interpretation of its axes would require separate evidence.

## Language and painting regressions in train-only CV

LR-4's fold- and seed-averaged Log Loss was higher than C's for all **21 macro-eligible CV languages**. The largest increases were Kyrgyz (**+0.048640**), Malay (**+0.045412**), Tamil (**+0.043723**) and Thai (**+0.043692**). The smallest was Setswana (**+0.005135**). LR-2 also regressed for all 21. The largest language differences describe the observed annotation samples and model fits.

Among 1,496 Contrast-eligible held-out training paintings, LR-4 improved over C on **658** and regressed on **838**, using the mean of the two seeded scores. It improved on 220 of the 473 paintings where C scored below B. LR-4 fell below B on 588 paintings overall. It lessened the worst C-versus-B drop, on Paula Modersohn-Becker's *Portrait of a Woman with Pitcher*, from **−0.971284** to **−0.207392**. It worsened the next two largest drops, from **−0.664295** to **−0.727930** and **−0.633471** to **−0.820094**. The three severe public-development examples from EXP-002 remain untested by the factorized model under the frozen protocol.

## Optimizer and reproducibility

All 40 direct fits returned finite outputs and reduced their initial objectives. LR-2 stopped successfully in 442–591 iterations, and LR-4 in 605–858. Nine LR-8 and all ten LR-16 fits reached the cap. A separate optimizer-only audit repeated fold 1, rank 16, seed 2027 with a cap of 1,800. It stopped successfully at 1,370 iterations and reduced the training objective from 0.466515649 at the cap to 0.466512388, with maximum absolute gradient decreasing from `1.49e-5` to `1.12e-6`. This isolates an iteration-budget issue for that fit. Held-out performance after longer optimization remains unknown.

CV took **340.1 seconds** on an Intel Core i7-13620H under Windows 11, and full-training spectrum generation took **12.6 seconds**. The mean direct optimizer time per seed and fold was 4.62, 6.20, 8.42 and 9.64 seconds at ranks 2, 4, 8 and 16, plus the shared B fit used for initialization. All compute reused the frozen image cache on CPU. Fold 1's LR-4 seed-2027 replay reproduced all 3,212 predictions and recorded optimizer diagnostics exactly. Hash checks preserved 23 Task 001/002 run files and both embedding caches. Independent checks validated all 240,285 CV predictions and recomputed the singular values and reconstruction errors. All 39 tests passed, including a synthetic test of the candidate-eligible development output path.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py low-rank-cv
.venv/Scripts/python.exe scripts/run_st3.py low-rank-dev --plots
.venv/Scripts/python.exe -m pytest -q
```

The run at `runs/exp003_low_rank/` uses code revision `83e77820196e97b0f5ba37486432e8f309815e9c`. `cv/selection.json` binds the decision to the CV summary and provenance hashes. `cv/` contains fold objectives, convergence, timing, predictions, language scores, painting scores and fold spectra. `dev/` contains the full-training interaction matrix, all 25 singular values and their normalized energy, the spectrum figure and the verified fallback record. The separate optimizer audit is in `runs/exp003_checks/optimizer_audit.json`.

## Conclusion and Task 004

Keep full C for the current system. The converged factorized fits provided meaningful parameter savings but underperformed C across the same held-out painting groups. The high-rank direct results are provisional because most fits reached the iteration cap. The comparison also changes regularization geometry, so it cannot isolate the effect of rank alone. The study used one frozen encoder, one factor penalty, two initializations and the original training corpus. Small vote counts limit the precision of individual painting–language shares, and Macro Log Loss summarizes probabilistic accuracy rather than calibration alone.

For Task 004, predeclare **SVD-16 as a compression candidate** against full C and test it on held-out painting groups. Track the saved parameter count, latency, Contrast and Macro Log Loss. This follows the smallest observed predictive trade-off in EXP-003 while keeping the established model as the reference.
