# EXP-021 — Vote distribution learning and cultural reliability

## Frozen scope and provenance

Base: EXP-020 `b7f274c8f1a5caea5e7cadda46910588c37f7dcb`. Before implementation, actual local EXP-020 CV and successful independent thirty-checkpoint/ten-target replay were validated. Its decision was COMPETITION_NO_GO. Pooled Contrast: A8 0.440223388; E8 0.439654679; EA8 0.449334031; EE8 0.449189301. These are historical results on repeatedly examined folds, not independent test evidence.

No EXP-017–020 source, manifest, checkpoint or result is changed. R0 reuses exact historical E50+L OOF probabilities. Production E50+L is unchanged. This experiment introduces no new pretrained resource: only official development-training emotion votes, target language and the existing three frozen image feature banks are used. No caption/text embedding, generated data or external emotion annotation enters the new pipeline. Original resource permissions remain unresolved; this task adds no external resource.

Every implemented stage excludes locked emotions before reading/retaining the emotion field. Protocol identity and artifact checks are read-only. No Confirm or Dev command exists for EXP-021; public Development gold and the 236 locked annotation targets are inaccessible from its stage dispatch. Posterior ordering statistics never enter training features, targets, weighting, filtering or selection.

## Phase A — audit completed before new neural implementation

All 13,600 eligible pairs / 1,335 paintings reconstruct exactly: eight integer counts sum to n; the four positive counts sum to k; empirical y equals original k/n. Binary votes: 69,719, including 47,501 positive votes. Ambiguous labels are excluded from both denominators. Every repeated categorical annotation row remains a separate vote; no deduplication occurs. No caption field is retained.

12,279 pairs (90.29%) have n=5. Counts range from 5 to 11, with median 5. Positive-rate median is 0.8; 1,008 pairs are all-negative and 4,795 all-positive (42.67% extreme rates combined). Eight-emotion entropy median is 1.370951 bits. Paintings have median ten eligible languages, range one to nineteen. Normal Contrast eligibility leaves 1,269 paintings.

There are 68,496 within-painting unordered language comparisons, including 23,655 observed ties (34.53%). Median absolute positive-rate difference is 0.2. Under independent Beta(0.5,0.5) posteriors:

| Ordering confidence | Fraction |
| --- | ---: |
| >=0.95 | 13.05% |
| >=0.80 | 48.48% |
| <0.60 | 36.18% |
| <0.80, defined as uncertain | 51.52% |

70.14% of comparisons differ by at most 0.2. For 572 paintings, at least half the comparisons are uncertain. Painting-level near-tie/uncertainty fractions and vote-count/difference bins are recorded, allowing sensitivity to coarse vote resolution to be inspected without discarding or reweighting paintings.

The audit samples 4,096 independent posterior draws per eligible language pair using NumPy seed 212027, with paintings/languages sorted before sampling. Continuous-rate ordering confidence is `max(P(a>b),1-P(a>b))`, with no claim that an observed tie has a correct direction. Sampling is deterministic, not exact integration; maximum Monte Carlo standard error is about 0.0078 per probability. Assumptions are exchangeable independent binary votes and independent language-specific priors/posteriors. Worker dependence, non-binomial variation and uncertainty in these assumptions are not modeled. These estimates are descriptive uncertainty, not proven truth or an established evaluation noise ceiling.

An initial audit was preserved under `initial_audit_v1/` after detecting floating-point boundary differences such as 0.8-0.6. The published current audit computes differences with exact integer cross-products before division. Its posterior draws are unchanged. No original annotation or target was modified.

Machine-readable outputs: `audit/targets.csv`, `language_comparisons.csv`, `painting_reliability.csv`, `ordering_bins.csv`, `vote_count_distribution.csv`, `summary.json`. Audit/data identity is intentionally separate from later neural-source identity: adding training modules does not invalidate the already frozen annotation-only audit.

## Matched model and objective

Both conditions allocate the original width-64 language-conditioned binary architecture and one compact shared `Linear(128,8)` auxiliary classifier on visual hidden state plus learned language representation. No query embeddings or text targets exist. D0 and D8 have identical initialization, binary logits, parameter count, shuffle generator, AdamW defaults, dropout and BCE stopping logic. D0's classifier is allocated but receives no auxiliary loss; its diagnostic predictions are untrained outputs.

For c, n=sum(c), d=c/n and y=sum(c[:4])/n:

`BCE = sum(n * BCEWithLogits(binary,y)) / sum(n)`

`CE8 = -sum(n * sum(d * logsoftmax(emotion_logits))) / sum(n)`

`D8 = BCE + 0.1 * CE8 / ln(8)`; D0 uses BCE alone.

These are natural-log training losses with empirical counts only. No smoothing, balancing, posterior adjustment, consistency loss, Pearson objective or adaptive coefficient is added. The mathematical loss accepts tensors independently of the feature extractor and can later backpropagate through a trainable encoder without redesign.

Inference calls only the binary branch and returns sigmoid(logit), including the shared-head behavior for unseen languages. No auxiliary head, labels or vote counts are required. Auxiliary probability and binary-vs-positive-auxiliary consistency passes are diagnostics only. Training-language priors are evaluated after fitting and are never inference inputs or target replacements. Languages with fewer than two unique images are excluded from prior-collapse diagnostics.

## Screening, optional CV and verification

Screen first: folds 1 and 3, seeds 2027 and 2028, D0/D8: eight complete nested-stopping/refit models. Screen checkpoints live in the same immutable fold/condition/seed directories later used by full CV. ScreenVerify replays their initial/final gradients, state identities, predictions, rounding, seed averages, ensembles and metrics before publishing its decision.

Frozen Screen criteria: PROMISING needs pooled D8-D0 Contrast >=0.001, at least one fold win, pooled Macro Log Loss within D0+0.005 bits, and finite/nondegenerate gradients. UNPROMISING means Contrast <=-0.005, loss worsening >0.01 bits, or a reproducible major treatment failure. Otherwise INCONCLUSIVE. Operational major prediction collapse is defined before scoring: at least 95% constant eligible-painting predictions for D8 in every fold/seed unit, while D0 is at most 5%. A merely prior-like auxiliary distribution is diagnostic and does not automatically trigger this failure rule.

Screen is an exploratory resource-allocation gate, not statistical proof or competition promotion. INCONCLUSIVE/UNPROMISING blocks full CV. A verified PROMISING permits a separate manual CV invocation; no stage automatically launches it. Full CV adds only the missing 22 models, reuses the eight verified screen seeds, and recomputes all averages using three seeds rather than reusing two-seed Screen averages.

Full competition promotion requires ED8 pooled gain >=0.003 over R0 and >=0.002 over ED0, at least three fold wins over R0, fold-mean Macro Log Loss within R0+0.005 bits, complete isolated seeds and successful independent replay. Failure gives COMPETITION_NO_GO. Even a PASS does not replace production or open locked evaluation; those workflows are not implemented in EXP-021.

Outputs include fold/pooled Contrast and Macro Log Loss, training/internal-validation auxiliary CE, binary/auxiliary consistency, language prior diagnostics, per-seed variance, painting/language changes, pair-order agreement and 10,000 paired painting bootstrap draws. Macro Log Loss is the existing base-2 language aggregation; the local scorer is reconstructed from published rules, with hosted scorer parity unverified. Repeatedly examined CV groups are not independent test data.

## Runtime policy and owner commands

Runtime estimates use actual per-seed EXP-018, EXP-019 and EXP-020 records (summing selection/refit work, not a last resumed invocation). The frozen conservative estimate uses the 90th percentile historical fit duration or measured epoch-cost forecast, whichever is larger, with a 1.5 margin. Reliability audit, target construction/pilot, eight-fit Screen, optional full CV and verification are recorded separately. EXP-019's stronger-loss runs contribute a conservative stress reference, not a lambda candidate.

```powershell
.\scripts\run_exp021.ps1 -Stage Preflight
.\scripts\run_exp021.ps1 -Stage Audit
.\scripts\run_exp021.ps1 -Stage Pilot
.\scripts\run_exp021.ps1 -Stage Screen
.\scripts\run_exp021.ps1 -Stage ScreenVerify
```

Only after verified PILOT_PROMISING, and as separate explicit commands:

```powershell
.\scripts\run_exp021.ps1 -Stage CV
.\scripts\run_exp021.ps1 -Stage Verify
```

Default run directory is `runs/exp021_vote_distribution/`. Paths to data/cache and each previous experiment are explicit options; `-Device auto/cuda/cpu` and `-MaxSeconds` are supported. Stage logs are unique. Completed valid seeds are never overwritten or refitted. Interrupted unpublished seeds restart; published seeds persist. Source/config/runtime changes require a fresh run directory. Latest interrupted/failed replay blocks any previous success until replay succeeds again. Exit 0: success; 1: validation failure; 2: invocation/environment; 3: missing prerequisite/blocked gate; 75: safe resumable budget stop.

## Current execution and conclusions

The exact annotation audit is complete. Neural implementation, lightweight tests and measured pilot are being finalized. No real screening decision, full-CV result or competition claim currently exists. The final runtime decision and next owner command will be added after the representative pilot.

Competition improvement, frozen-head learning and possible encoder fine-tuning value remain separate conclusions. A later LoRA comparison must first verify real encoder-parameter gradients. If Screen supports distribution learning, compare BCE-only versus BCE+normalized empirical eight-way CE at the frozen coefficient. If evidence remains insufficient, begin with BCE-only rather than assuming either auxiliary objective is useful. No LoRA training is launched here.
