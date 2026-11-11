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

Run from the project root with the prepared `.venv`. All new outputs live under `runs/exp018_polarity_captions/`. Each successful stage is sealed with configuration, source, runtime and input identities plus file checksums. Existing successful outputs are validated and reused. Incompatible or corrupt outputs stop execution.

```powershell
.\scripts\run_exp018.ps1 -Stage Preflight
.\scripts\run_exp018.ps1 -Stage Audit
.\scripts\run_exp018.ps1 -Stage Pilot
.\scripts\run_exp018.ps1 -Stage CV
.\scripts\run_exp018.ps1 -Stage Verify -DeepVerify
.\scripts\run_exp018.ps1 -Stage Confirm
.\scripts\run_exp018.ps1 -Stage Dev
```

CV, replay verification and confirmation are separate invocations. CV results await replay before any promotion classification. Verify replays each seed checkpoint, checks PCA boundaries and means, recomputes metrics and constructs the frozen decision. `-DeepVerify` additionally re-fits each PCA. Its completed replay folds are reusable.

Completed seeds persist individually. A time-budget stop discards only the unpublished in-progress seed; the next invocation restarts that seed and reuses completed ones. Each epoch checks the time budget. The default heavy-stage budget is 150% of its pilot estimate, with an allowance for deep PCA replay. Override it explicitly with `-MaxSeconds 1800` if needed. Logs retain unique timestamps and exact exit codes. Exit 0 means stage success or a completed scientific guard, 1 means validation/execution failure, 2 means invalid invocation or missing environment, 3 means a missing/unpublished prerequisite, and 75 means a resumable budget stop.

Confirm opens locked labels only after a verified CV candidate passes. It refits M0 and the matched BCE/text students on the 1,335 paintings and requires at least +0.001 Contrast over M0, positive gain over B1 and loss within M0 +0.005 bits. Dev requires successful confirmation and then uses the full original training split. Any new official-training CLIP captions are encoded into an isolated EXP-018 cache; the original EXP-017 cache and outputs remain intact. Public development is scored once and a separate ZIP is validated for manual upload.

Use a fresh `-RunDir` when changing code, scientific configuration, runtime versions or training device. Passing paths and budget overrides as runner parameters preserves the recorded scientific identity. Full CV and heavy replay remain owner commands under the execution-budget policy.
