# EXP-018 Polarity-aware CLIP caption supervision

## Frozen scope and protocol

This follow-up starts from EXP-017 head `5203258aa45c8c9dd3c695c35d9001bb5b8bafda`. It reuses the exact 1,335 development-training paintings, five folds, internal stopping splits and 236-painting locked confirmation set. These folds have already been examined in EXP-017 and earlier studies. The protocol supplies experiment-level control, with confirmation still requiring a successful predeclared gate.

Only the existing pinned CLIP-L text cache supplies semantic supervision. The teacher is `openai/clip-vit-large-patch14` at `32bd64288804d66eefd0ccbe215aa642df71cc41`. No text or visual encoder extraction is needed for audit, pilot or CV. Training-caption annotation rows retain their original vote association, including repeated rows. Ambiguous emotions are excluded. The caption reader filters allowed painting IDs before retaining caption strings. All current-stage polarity statistics exclude locked paintings.

For each eligible pair, the audit constructs mean all-caption, positive-caption and negative-caption embeddings. One unwhitened 64-dimensional PCA fits only the applicable partition's all-caption means. Its same mean and components transform all three target kinds. Valid projections are normalized to unit length; absent or degenerate projections are masked. Each inner and full refit projector is shared exactly across M2 and M3.

The sentiment branch is the EXP-017 width-64 LayerNorm/GELU/dropout architecture over the three frozen visual embeddings divided by sqrt(3). The shared semantic module concatenates the visual hidden state, learned language embedding and a learned polarity embedding. The chosen polarity embedding width is 64, initialized with standard deviation 0.02. Its shared Linear(192,64), GELU and Linear(64,64) module produces two normalized queries. Every condition allocates this complete architecture. The sentiment branch receives no ground-truth polarity and inference ignores the auxiliary module.

M1 uses vote-weighted BCE only. M2 supervises both queries with the pooled target and averages their cosine losses. M3 supervises available positive and negative queries separately and averages the available channels within each pair. Pair weights are inverse valid-pair counts within each painting. The fixed auxiliary coefficient is 0.1. For unit 64-dimensional vectors, cosine loss is 32 times coordinate-mean MSE; this is a deliberate scaling change from EXP-017. Optimizer defaults, three seeds, stopping BCE, 150-epoch maximum and patience 15 remain fixed. [Frozen config](../configs/exp018.json)

M0 reuses only the verified EXP-017 fold-specific E50+L OOF predictions. B1, B2 and B3 preserve the fixed rounded 50/50 blend with the respective new student. The only promotion candidates are B2 and B3. Each requires at least +0.003 pooled Contrast over M0, +0.002 over B1, three fold wins, mean Macro Log Loss within M0 +0.005 bits, all seeds finite and completed, and successful isolation and checkpoint replay. The highest qualifying pooled Contrast wins. Source hashes and artifact identities bind all stages.

The existing resource-eligibility assessment remains **PROVISIONAL_PENDING_ORGANIZER_CLARIFICATION**. This task introduces no text teacher or external annotation source. Future official submission requires resource permission to be confirmed. [Pinned organizer FAQ](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#faqs)

## Execution budget

Before launching this study, EXP-017's measured 987.31 s CV and 526.36 s independent verification implied at least 26.73 minutes including 90 s of checks and loading. The owner budget therefore calls for a measured pilot and runnable handoff. Full EXP-018 CV is not launched to discover its runtime.

The pilot measures one warmup and three timed training epochs per condition on the first fold's real inner training partition, internal BCE evaluation, two shared PCA fits and data/cache loading. The planning estimate uses EXP-017's 90th-percentile selected epoch count, adds the same patience and a 25% overhead allowance, and accounts separately for scoring and verification. The configured maximum also allows all 150 selection epochs and 150 refit epochs. Stronger cosine supervision can change stopping duration, so the estimate remains conditional on the historical stopping assumption.

## Owner runner and safeguards

Run from the project root with the prepared `.venv`. The active prepared run is `runs/exp018_polarity_captions/prepared/`, which is the default for both the PowerShell runner and Python commands. All new outputs live under `runs/exp018_polarity_captions/`. Each successful stage is sealed with configuration, source, runtime and input identities plus file checksums. Existing successful outputs are validated and reused. Incompatible or corrupt outputs stop execution.

```powershell
.\scripts\run_exp018.ps1 -Stage Preflight
.\scripts\run_exp018.ps1 -Stage Audit
.\scripts\run_exp018.ps1 -Stage Pilot
.\scripts\run_exp018.ps1 -Stage CV
.\scripts\run_exp018.ps1 -Stage Verify -DeepVerify
.\scripts\run_exp018.ps1 -Stage Confirm
.\scripts\run_exp018.ps1 -Stage Dev
```

CV, replay verification and confirmation are separate invocations. CV results await replay before any promotion classification. Verify replays each seed checkpoint, checks PCA boundaries and means, recomputes metrics and constructs the frozen decision. `-DeepVerify` additionally re-fits each PCA. Its completed replay folds are reusable. Every replay attempt records its state. A later interrupted or failed replay blocks confirmation despite any earlier successful verification, until a new replay completes successfully.

Completed seeds persist individually. A time-budget stop discards only the unpublished in-progress seed; the next invocation restarts that seed and reuses completed ones. Each epoch checks the time budget. The default heavy-stage budget is 150% of its pilot estimate, with an allowance for deep PCA replay. Override it explicitly with `-MaxSeconds 1800` if needed. Logs retain unique timestamps and exact exit codes. Exit 0 means stage success or a completed scientific guard, 1 means validation/execution failure, 2 means invalid invocation or missing environment, 3 means a missing/unpublished prerequisite, and 75 means a resumable budget stop.

Confirm opens locked labels only after a verified CV candidate passes. It refits M0 and the matched BCE/text students on the 1,335 paintings and requires at least +0.001 Contrast over M0, positive gain over B1 and loss within M0 +0.005 bits. Dev requires successful confirmation and then uses the full original training split. Any new official-training CLIP captions are encoded into an isolated EXP-018 cache; the original EXP-017 cache and outputs remain intact. Public development is scored once and a separate ZIP is validated for manual upload.

Use a fresh `-RunDir` when changing code, scientific configuration, runtime versions or training device. Passing paths and budget overrides as runner parameters preserves the recorded scientific identity. Full CV and heavy replay remain owner commands under the execution-budget policy.

## Measured preparation results

**READY_FOR_LOCAL_RUN.** Preflight, Audit and Pilot completed successfully for frozen source commit `0c71eed94c0778adecec22f01bbbe0bc926d1f5e`. Full CV, heavy replay, locked confirmation and public-development evaluation have not run. There is no EXP-018 model promotion decision or candidate ZIP. The earlier preparation under the run root is preserved; the active `prepared/` run was created after strengthening the latest-verification guard. The gradient diagnostics reproduced exactly between the two preparations.

### Polarity audit

All 13,600 eligible development-training pairs reused the original 69,719 valid binary-caption annotation rows. Of these pairs, 7,797 (57.33%) have both polarities, 4,795 (35.26%) are positive-only and 1,008 (7.41%) are negative-only. Positive means are available for 12,592 pairs and negative means for 8,805. No raw mean is degenerate.

The within-pair positive/negative cosine has minimum 0.185392, 10th percentile 0.854430, median 0.939862, 90th percentile 0.972593 and maximum 0.996998. The high median leaves substantial common content, while the nonzero variation supplies a testable distinction. It establishes neither a Contrast gain nor successful polarity supervision.

| Mean kind | Component variance | Between paintings within language, group-mean cosine | Between languages within painting, group-mean cosine |
| --- | ---: | ---: | ---: |
| All binary captions | 0.000256352 | 0.952830 | 0.770659 |
| Positive captions | 0.000281399 | 0.929655 | 0.753110 |
| Negative captions | 0.000317351 | 0.907385 | 0.728870 |

Each variation entry is an unweighted mean over the corresponding groups, excluding groups with fewer than two nonzero means. Full per-pair counts, raw means, availability masks and group variation are retained in `audit/pairs.csv`, `audit/raw_means.npz` and `audit/variation.csv`. These diagnostics exclude the 236 locked paintings.

The representative first outer-training partition contains 1,068 paintings; its inner training subset contains 961 paintings and 9,759 pairs. The inner pooled-target PCA retains 89.2759% of variance and its full outer-training counterpart retains 89.2368%. Neither has a degenerate available projected target. Positive-only and negative-only channels remain explicitly masked.

### Gradient diagnostics

Both conditions used the same 256-pair batch and exactly matched initialization and dropout state for diagnostics. Norms below refer to the first shared visual Linear layer, including its weight and bias. Both learned polarity-query embedding rows received finite nonzero gradients. The coefficient remains 0.1.

| Quantity | M2 pooled target | M3 polarity targets |
| --- | ---: | ---: |
| BCE, nats | 0.720694005 | 0.720694005 |
| Cosine auxiliary | 1.000896454 | 1.000708818 |
| Scaled auxiliary | 0.100089647 | 0.100070886 |
| BCE gradient norm | 4.836877346 | 4.836877346 |
| Unscaled auxiliary gradient norm | 0.433001429 | 0.430891931 |
| Scaled auxiliary gradient norm | 0.043300144 | 0.043089192 |
| Unscaled auxiliary/BCE gradient ratio | 0.089520864 | 0.089084737 |
| Scaled auxiliary/BCE gradient ratio | 0.008952087 | 0.008908473 |
| Gradient-vector cosine | -0.074254557 | -0.055602506 |
| Equivalent coordinate-mean MSE | 0.031278014 | 0.031272151 |
| Cosine/MSE ratio | 32.000000000 | 32.000000000 |

M2 query-gradient norms were 0.003257956 and 0.003256953. M3 norms were 0.004370557 and 0.004463574. M3 had 235 positive and 162 negative targets in the batch; M2 supervised both 256-row channels against the pooled target. The scaled shared auxiliary gradients were about 0.9% of the BCE gradient norm at this initialization. Their mildly negative alignment is a diagnostic of this batch, with no coefficient adjustment.

### Runtime estimate and handoff

| Measured or estimated item | Time |
| --- | ---: |
| Actual Preflight, Audit and Pilot including process startup | 38.91 s |
| Python pilot including loading, PCA, gradients and 12 training epochs | 9.56 s |
| Two measured shared PCA fits | 0.69 s |
| Estimated five-fold CV including scoring and overhead | 19.72 min |
| Estimated replay verification | 60 s |
| Estimated preparation + CV + verification | 21.03 min |
| Configured 150-selection/150-refit-epoch CV upper estimate | 53.86 min |

Timed training epochs averaged 0.163 s for M1, 0.257 s for M2 and 0.278 s for M3. The estimate assumes 35 selected refit epochs and 50 stopping epochs, calibrated from EXP-017 and increased by the fixed overhead margin. Actual stopping under the stronger cosine objective is unknown. Deep PCA replay adds a measured-PCA allowance to its separate budget. These are runtime estimates, with no completed CV performance claim.

All three conditions allocate 149,594 parameters. Peak pilot CUDA allocation was 83.44 MiB and reservation was 90 MiB on the RTX 4060 Laptop GPU. No checkpoint was downloaded or text/image embedding re-extracted. The measured 21.03-minute forecast exceeds the 15-minute policy, so Codex stopped after preparation and handed off full execution.

### Previous EXP-017 results for context

These are historical measurements on the reused folds. They are not an independent baseline or new EXP-018 scores, and the new two-query cosine objective differs from the previous single-query MSE objective.

| Historical EXP-017 system | Pooled Contrast | Pooled Macro Log Loss |
| --- | ---: | ---: |
| E50+L M0 | 0.447097707 | 0.710554182 |
| BCE-only M1 | 0.437792268 | 0.716045717 |
| CLIP-text M2 | 0.439016172 | 0.716105044 |
| Matched BCE ensemble B1 | 0.448585747 | 0.706474066 |
| CLIP ensemble B2 | 0.449753673 | 0.706239161 |

The next owner command is the CV stage shown above. Verify must finish before Confirm can open the locked set. Confirm and Dev are guarded separately. Successful per-seed outputs survive a budget stop, and an unfinished seed restarts. On a resumed CV, the reported stage duration covers that invocation; cumulative fitting time is the sum of saved seed fit times, and the timestamped runner logs retain overall process time.

## Verification and remaining execution

The Windows suite passed 144 tests and skipped the existing symlink fixture. Small offline integration fixtures exercised the complete pilot, two-fold CV, shared PCA, seed resume, checkpoint replay, six-decimal ensembles and NO-GO guards. A regression test also confirms that a failed later replay blocks an earlier success until Verify completes again. These fixture scores are test outputs; the real EXP-018 CV remains unexecuted.

The actual PowerShell Preflight, Audit and Pilot commands completed with exit 0, and their current source/configuration signatures revalidated. A premature Confirm invocation returned the expected prerequisite exit 3 without opening the locked set. The production regression reproduced all 849 E50+L, E50 and CLIP development rows byte for byte and validated their archives. E50+L CSV SHA-256 remains `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`. All 291 saved historical artifact hashes matched, including EXP-017 outputs, its original CLIP progressive cache, prepared production bundles and the original E50+L predictions/archive. The pinned Python environment passed its dependency check.

The active pilot metadata is at `runs/exp018_polarity_captions/prepared/pilot/run.json`, with gradient and timing details alongside it. `preparation_verification.json` records the final READY_FOR_LOCAL_RUN checks. Full CV, heavy replay, promotion selection, locked confirmation and public-development scoring are pending owner execution. Resource eligibility remains provisional. No candidate archive has been generated.

The next command from the project root is:

```powershell
.\scripts\run_exp018.ps1 -Stage CV
```

Then run `Verify -DeepVerify`, `Confirm` and `Dev` as separate invocations shown above. The latter two stages enforce the frozen guards. Production E50+L remains the release system while this experiment awaits evaluation.
