# EXP-027 closure and CPU failure analysis — TASK-030

**EXP027 = GPU_PILOT_COMPLETED / NO_GO_FOR_EXPANSION. Production = E50+L.** Neither Pearson treatment improves selected inner-validation Contrast against matched BCE. P_high's−0.000188-bit Log Loss change does not offset its negative primary-metric result. Preserve source, configs, notebooks, predictions, checkpoints and logs. No additional GPU run, retraining, lambda/epoch/rank search, outer scoring, full CV, Development or confirmation access is authorized. This is not a rejection of all Pearson objectives or QLoRA.

## Evidence and independent verification

Original Drive run: `MyDrive/exp027/pilot_run_4bc703af144e/`; source `4bc703af144e9121e58ed4cb75c94306c318dff1`. TASK-030 downloaded the saved report/identity, predictions/targets/receipts, three selected CPU-loadable checkpoints and their SHA receipts, plus the persistent export ZIP. No encoder weights or new inference/training were run.

**VERIFIED OBSERVATION:** source-file hashes match pinned Git contents; scientific config/run identity matches all three checkpoints. All504 validation keys/values align, probabilities are finite/in range, targets match the original checksum-pinned released input after filtering to these validation painting IDs only. Three model SHA receipts and published prediction hashes pass; checkpoint prediction vectors match CSVs exactly. All models select epoch1, the earliest minimum of logged inner-validation BCE. Metrics and10,000 paired painting bootstrap replicates at seed27029 reproduce the stored metrics exactly. Fifteen languages meet the≥20-pair Macro Log Loss cutoff. Of48 paintings,47 meet Contrast eligibility; the excluded painting is retained in diagnostic CSV with eligibility false.

The110,576,148-byte export ZIP matches SHA `e2e478802d24b1a5969b0c9fc8e7515b59daf35eec40b1c8193bf88639c28d63`; its35-file manifest, every archived file hash and corresponding standalone run files independently pass. Raw downloads remain outside Git under `runs/task030/`. [Verified receipt and analysis](../docs/exp027_closure_evidence.json).

**Reported engineering, not independently replayed CUDA:** NVIDIA L4; gradient preflight PASS with maximum absolute parameter-gradient discrepancy0.00062704 under the combined AMP tolerance; exact same-chunk two-pass replay recorded for every treatment epoch. Worker elapsed2774.68s; allocated peak4,304,538,624 bytes. Archived checkpoint curves corroborate the successful run, but gradient arrays/encoder inference cannot be reconstructed from saved predictions alone. Browser arrival and physical runtime deallocation remain unverified; file readback integrity is independently verified.

| Selected model | Contrast | Macro Log Loss, bits | Epoch | Delta vs BCE | Wins / losses | Paired percentile95% CI |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| BCE |0.195952|0.944456|1|—|—|—|
| P_low |0.184009|0.945056|1|−0.011943|15 /32|[−0.020683,−0.002435]|
| P_high |0.188607|0.944269|1|−0.007346|16 /31|[−0.017516,+0.003806]|

These are descriptive intervals on a single, selected inner-validation population, not independent outer-test confidence statements. Population/seed is fixed:192 training paintings/1931 pairs,48 validation paintings/504 pairs, two epochs. Only selected epoch1 validation predictions were saved; **epoch2 validation Contrast is NOT TESTED**. No new fold or epoch was evaluated here.

## Painting and language errors

The [painting CSV](exp027_diagnostics/painting_diagnostics.csv) contains each model's correlation, delta, probability/target variance (population definition), absolute/relative probability change, rank agreement, vote support and language count, including excluded paintings. [Language CSV](exp027_diagnostics/language_diagnostics.csv) gives support, observed/predicted shares, offsets and Log Loss with explicit eligibility. No per-pair votes or raw predictions are committed.

**VERIFIED OBSERVATION:** losses are widespread:68.1% /66.0% of eligible paintings deteriorate under P_low/P_high. Median deltas are−0.015577 /−0.012254; trimming two observations from each tail still gives−0.013111 /−0.009086. Thus a few extreme paintings do not explain the negative mean.

- Largest gains include `john-constable_selfportrait` (+0.120685 /+0.162620) and `pietro-perugino_birth-of-the-virgin-1473` (+0.032386 /+0.050097). The selfportrait remains a high-target-variance, negative-correlation case even after improvement. Complete best/worst and high-variance/poor-correlation lists are in the receipt.
- P_high's worst cases include `nikolai-ge_death-of-virginia-study` (−0.090819), `camille-corot_l-arbre-tordu-les-chenes-du-mont-usey-fontainebleau` (−0.076424), and low-variance `fyodor-solntsev_st-george` (−0.060158).
- Fixed descriptive bottom-quartile target-variance groups have mean absolute changes0.02033 /0.02595, versus0.02883 /0.02751 elsewhere. Rank correlations between target variance and treatment delta are−0.099 /−0.019. Low-variance instability exists in examples but is not supported as the dominant aggregate explanation. Quartiles/zero correlation are descriptive definitions, not optimized cutoffs.
- The largest ordering changes are modest: P_high's lowest rank agreement versus BCE is0.9286 on Ivan Bilibin's Rosehip illustration (delta−0.04025); the Perugino example changes rank agreement to0.9333 while improving. Median agreements are0.9890 /0.9860. Direction changes are not uniformly useful.

## Language-offset cross-fit

Use all48 paintings' prediction-only rows, one fixed shuffle (seed30027), disjoint24/24 halves, then reverse. Fit a two-way additive model `prediction = training-painting intercept + language offset` by least squares; training intercepts account for unequal language availability. Center fitted language coefficients to fix their gauge. For each test painting, recenter its available-language profile and compare with its centered predictions. One language row is unseen in its fitting half; all variants use the identical mean-offset-zero fallback. No gold is used to fit the profile; label correlations are subsequent diagnostics.

| Model | OOS explained centered variance | Median profile-rank agreement | Mean residual/gold correlation |
| --- | ---: | ---: | ---: |
| BCE |0.979239|0.983516|0.029116|
| P_low |0.982807|0.983333|0.015260|
| P_high |0.982100|0.983333|0.011154|

[Cross-fit per-painting output](exp027_diagnostics/language_offset_crossfit.csv) records test-half assignments; [fitted language profiles](exp027_diagnostics/language_profiles.csv) record both directions. The estimator/population differ from EXP-026's owner-reported profile diagnostic, so those percentages must not be compared directly.

**VERIFIED OBSERVATION:** treatments do not reduce dependence on shared language offsets. Centered prediction SS rises1.928→2.206/2.324, mostly through the fitted shared profile (SS1.898→2.178/2.294). Residual SS changes0.04002→0.03793/0.04159. Mean absolute probability changes versus BCE are0.00538/0.00677. Most language ordering remains stable, with small but real changes that overall align worse with votes.

**SUPPORTED HYPOTHESIS:** these mild objective perturbations mainly strengthen/rearrange a shared language-response structure rather than produce substantially more useful painting-specific interactions. Residual/gold correlations stay near zero and decline, consistent with weak useful interaction signal.

**UNRESOLVED:** cross-fit residual includes interaction, finite-sample profile error, noise and misspecification; it is not a purified causal interaction component. Out-of-sample profile and residual are not orthogonal. Their SS decomposition explicitly retains `2*dot(profile,residual)` (negative cross terms); it is incorrect to interpret the two SS values as additive variance percentages. More residual interaction would not imply better Contrast. Learned offsets describe model/task behavior, not inherent language-community traits.

## Training behavior

| Model | Train BCE epoch1→2 | Train auxiliary epoch1→2 | Validation BCE epoch1→2 | Pairs/s epoch1→2 | Selected-adapter update norm |
| --- | --- | --- | --- | --- | ---: |
| BCE |0.655019→0.603501|Not supervised|0.642757→0.648103|6.758→6.709|1.02041|
| P_low |0.654771→0.604403|0.795150→0.701174|0.643333→0.648216|4.987→4.975|1.01732|
| P_high |0.654458→0.604782|0.793644→0.704971|0.642574→0.645157|4.989→4.973|1.00857|

[Complete logged curves](exp027_diagnostics/training_curves.csv) include **minimum per-window**, not mean/global, head and LoRA gradient norms. All are positive. Clipping frequency is0.50 at epoch1 and0.541667 cumulative at epoch2; AMP scale remains1024. Treatment training costs≈1.35x control, consistent with the extra forward pass. No OOM/nonfinite failure is recorded.

**VERIFIED OBSERVATION:** auxiliary loss decreases, all models' train BCE decreases, and all validation BCEs worsen at epoch2. This is consistent with shared overfitting, not a Pearson-only optimization failure. Training Pearson reduction does not establish temporal validation-Contrast improvement: epoch2 Contrast is unavailable. All saved selected checkpoints are epoch1.

Treatment selected states differ from BCE: head L2 distances0.3863/0.5749 and adapter distances0.05464/0.15295. These establish a real trajectory change, not causal attribution to individual parameters. Preflight on **one six-language painting** reports head auxiliary/BCE norm1.834% and LoRA2.886% for the high treatment. Weak auxiliary influence is plausible, but full-training component-gradient ratios/cosines were not logged and are **NOT TESTED**. State divergence is not a substitute for those measurements, and it does not justify a lambda search.

Selected update norms and per-epoch gradient diagnostics are logged values, not independently recomputed gradients from encoder inference. Distances between saved selected states are independently recomputed on CPU. Prediction-only profile fits use all48 paintings; undefined label correlations are omitted, and paired Contrast uses the same47 eligible paintings for all models.

## Explicit interpretation

1. **Q1 — VERIFIED / limited:** saved artifact/selection/metric integrity passes; previous CPU autograd tests and recorded CUDA replay/preflight support a functioning implementation that did not improve this pilot. No implementation defect was found. CUDA derivatives were not independently rerun, so arbitrary latent bugs are not disproven.
2. **Q2 — VERIFIED:** nonzero parameter/probability/ordering changes occur, but common offset dominance remains≈98%; evidence for a useful new interaction mechanism is weak.
3. **Q3 — VERIFIED:** selected-checkpoint alignment worsens overall (31–32 losses/47, negative mean/median/trimmed deltas), despite isolated gains. P_high's interval crosses zero; neither establishes improvement.
4. **Q4 — SUPPORTED HYPOTHESIS:** common epoch2 overfitting is supported by train/validation BCE divergence. Inadequate cultural supervision or noisy votes is plausible but **UNRESOLVED** without reliability/power controls; no causal claim follows from48 validation paintings.
5. **Q5 — UNRESOLVED:** loss geometry, small training population, learned language conditioning and checkpoint criterion may jointly limit learning. A selected-validation endpoint cannot identify the main cause. BCE-based checkpointing may differ from a Contrast-based endpoint, but changing it post hoc is not justified.
6. **Q6 — evidence needed:** controlled image/language interventions can test whether joint conditioning is nearly additive; matched changes in conditioning with objective fixed could isolate representational restrictions. Distinguishing data scale/noise would require separately preregistered matched populations/seeds/reliability analysis, not extrapolation from this pilot. Independent generalization requires future authorized data/protocol; reused P2 is not untouched confirmation.

[EXP-004](exp004_contrast_aware.md) already found small, non-promotable Pearson/pairwise improvements on a different frozen-CLIP linear architecture. [EXP-026](exp026_exploratory.md) found QLoRA better than frozen VLM yet below a larger-data refitted E50+L reference, with rigidity and epoch2 overfitting. Neither establishes that loss changes alone solve language conditioning; neither supports declaring QLoRA universally ineffective.

## One preferred next direction — proposal only

**Factorial image–language conditioning probe before any more adaptation.** Hypothesis: the existing decoder/head mostly expresses shared language effects; the loss cannot create useful image-specific differences under the current conditioning path. Keep saved BCE/Pearson models and their original source fixed. On a predeclared small set of original **inner-training** images and target-language pairs, evaluate complete2×2 image/language swaps and logit mixed differences `z(i,l)-z(j,l)-z(i,m)+z(j,m)`. Compare frozen versus adapted states and BCE versus Pearson; use a neutral-language-input control while preserving the head's true target-language input. No generated captions or new checkpoint. This probes the forward mechanism rather than retuning a predictor.

Expected signal: weak mixed differences despite substantial main effects would support additive conditioning rigidity; strong mixed differences with poor vote-gap alignment would instead implicate supervision/alignment. **Failure criterion for the rigidity hypothesis:** clear, reproducible mixed effects already present with the correct input image but no better alignment; then do not prescribe a conditioning redesign as the cause. Small/no effects are diagnostic evidence, not proof that a replacement will improve competition scores. Logit-scale analysis separates sigmoid scale effects from interaction changes.

Compute proposal: no training, a few hundred bounded forward passes after one model load; roughly≤5–10 minutes GPU inference based on prior≈20 pairs/s, plus cache/load overhead, **not benchmarked or authorized**. CPU analysis uses caches where compatible; actual adapted counterfactuals need inference, not merely re-scoring current predictions. Use training-only images and no examined validation targets to select interventions or thresholds. Reuse of existing checkpoints is restricted to their training population, never as initialization for a newly held-out fold. Preregister identities/contrasts before inference; this is not an independent accuracy test.

**Recommendation:** QLoRA merits this bounded mechanism investigation, but no further GPU *training*, larger lambda/rank/model, extra epochs, full CV or competition expansion is currently justified. Do not implement EXP-028 or launch even the inference probe in TASK-030. A separate owner task/authorization is required.

## Reproduction and preservation

```powershell
.venv/Scripts/python.exe scripts/verify_exp027_validation.py runs/task030/evidence/inner_validation
.venv/Scripts/python.exe scripts/analyze_exp027.py --run-dir runs/task030/evidence --output runs/task030/new_analysis --inputs runs/exp023_budget_delivery/final/restore_check --export-zip runs/task030/export/outputs.zip
```

Use a fresh output path; audits refuse overwriting. Analysis reads only permitted validation vote fields, saved logs/weights and prediction-derived profile fits. It never invokes an encoder, optimizer or training runner. Full raw evidence/download receipts/ZIP remain ignored. Aggregate diagnostics and receipt are versioned for review. Original source/config/notebooks and all experiment artifacts are retained; existing notebook execution outputs, if locally edited by the owner, are not committed or discarded.

Validation:339 tests passed,1 Windows symlink skip on a clean Git-source tree with this analysis patch. The owner-executed notebook was preserved rather than resetting its authorization/outputs to satisfy the default-OFF test. All499 protected files,70 local semantic-cache files, original training/config source and E50+L prediction SHA remain unchanged. No GPU or student training was run in TASK-030.
