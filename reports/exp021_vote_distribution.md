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

70.14% of comparisons differ by at most 0.2. For 572 paintings, at least half the comparisons are uncertain; for 1,072, at least half are near ties. Painting-level near-tie/uncertainty fractions and vote-count/difference bins are recorded, allowing sensitivity to coarse vote resolution to be inspected without discarding or reweighting paintings.

| Vote-count stratum | Comparisons | Mean absolute rate difference | Uncertain fraction | Confidence >=0.95 |
| --- | ---: | ---: | ---: | ---: |
| Both n=5 | 55,875 | 0.213834 | 51.97% | 12.45% |
| One n=5, one larger | 11,983 | 0.201698 | 49.59% | 15.69% |
| Both n>5 | 638 | 0.189145 | 48.43% | 15.36% |

These strata contain different painting/language populations. The descriptive associations do not isolate a causal effect of more annotators; joint vote/difference bins are supplied for closer inspection.

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

**READY_FOR_LOCAL_RUN.** Final-source Pilot completed with exit 0. Required eight-fit Screen and ScreenVerify have not run; there is no PILOT_PROMISING/INCONCLUSIVE/UNPROMISING classification yet. Full CV and competition promotion remain unassessed. No locked confirmation or public Development was accessed, and no candidate archive exists.

### Actual matched pilot

One short D0 and one short D8 sequence used fold-1 inner training, seed 2027: four epochs each (one warm-up, three timed), not eight complete screening fits. Both allocate 133,986 parameters and have identical initial state, shuffle generator and binary logits. All eight logits receive finite nonzero diagnostic CE gradients. Training rows/images/languages are retained. Rare languages seen only in the stopping split remain in validation with shared-head/zero-language-modulation behavior, without expanding the fitting vocabulary. The prototype pilot before this validation safeguard remains intact under `initial_pilot_v1/`.

| Condition / checkpoint | BCE nats | CE8 nats | CE8/ln(8) | Shared BCE norm | Raw CE gradient norm | Scaled normalized CE norm | Scaled ratio | Gradient cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 / initialization | 0.700084 | 2.021031 | 0.971911 | 0.013156 | 0.015609 | 0.000000 | 0.000000 | 0.048497 |
| D0 / short_pilot_final | 0.463771 | 2.022475 | 0.972605 | 0.009030 | 0.015553 | 0.000000 | 0.000000 | 0.012866 |
| D8 / initialization | 0.700084 | 2.021031 | 0.971911 | 0.013156 | 0.015609 | 0.000751 | 0.057056 | 0.048497 |
| D8 / short_pilot_final | 0.464008 | 1.555424 | 0.748001 | 0.008983 | 0.014858 | 0.000715 | 0.079542 | 0.435096 |

Generic decomposition fields named `text_norm` in JSON refer here to raw CE8 gradients, not text supervision. The scaled norm is `0.1/ln(8)` times that raw gradient. D0 has zero applied auxiliary contribution; its CE/logit gradients are hypothetical diagnostics of the allocated, untrained classifier. All reported gradients concern the head or differentiable feature-input leaf, never pretrained encoder parameters.

At the first visual Linear layer, D8 scaled auxiliary/BCE ratios were 0.045525 initially and 0.140272 after four epochs; cosines were 0.239391 and 0.439750. At the shared representation, orthogonal gradient fractions were 0.998823 and 0.900384. The auxiliary signal is neither identically zero nor wholly redundant with BCE at these checkpoints. This establishes a functioning pathway, not successful backbone adaptation.

Both conditions clipped 2/39 minibatches (5.13%) in the warm-up epoch and none in the next three epochs. Diagnostic global clipping factors at initialization were 0.186101 (D0) and 0.184002 (D8), versus 1.0 after the short pilot. Components are measured before clipping; parameter diagnostics also report the single combined-gradient clipping factor.

| End-of-pilot inner-training evaluation | D0 | D8 |
| --- | ---: | ---: |
| Binary BCE nats | 0.476713 | 0.476416 |
| Eight-class CE nats | 2.047463 | 1.576234 |
| Binary vs auxiliary-positive MAE | 0.245653 | 0.034771 |

D8 internal stopping CE fell from 1.731535 to 1.613801 nats across the four pilot epochs. Its final internal stopping BCE was 0.508809 versus D0 0.508729. That tiny difference is not a screening result, and no outer-fold Contrast was computed. The full-fitting workflow records end-epoch training CE and internal validation CE separately; D0 applied training CE is zero because the auxiliary loss is disabled, not because its untrained classifier has perfect predictions.

All 25 training languages had enough images for the fixed prior-like diagnostic, and none met it. Median within-language class-probability standard deviation was 0.020992 for D0 and 0.074978 for D8. Mean training CE across languages was 2.057230 for the untrained D0 classifier and 1.592429 for D8, against training-only language-prior CE 1.736397. These are in-sample diagnostics, not evidence of held-out cultural prediction or language-generalization gains. Full per-language values are in `pilot/prior_diagnostics.csv`.

### Measured budget and execution boundary

| Component | Actual or conservative estimate |
| --- | ---: |
| Actual preflight Python stage | 6.13 s |
| Actual audit Python stage | 7.61 s |
| Actual final Pilot Python stage | 7.25 s |
| Actual final Pilot PowerShell process | 11.90 s |
| Input loading/identity checks within Pilot | 2.096 s |
| Inner/stopping target tensor construction | 0.185 s |
| Timed D0 / D8 epoch | 0.288 / 0.280 s |
| Conservative eight-fit Screen | 23.30 min |
| ScreenVerify allowance | 1.50 min |
| Preparation + Screen + verification | 25.15 min |
| Optional full thirty-model CV, before reuse | 87.36 min |
| Additional 22 fits after verified Screen | 64.06 min |
| Optional full Verify allowance | 3.00 min |

All 75 selected historical fit records are retained in `pilot/historical_fit_times.csv`. Their 90th-percentile fit duration is 116.48 s and total selection/refit epoch count is 104.2. The historical duration, including the slower EXP-019 stress condition, dominates the current faster short-epoch forecast; the fixed 1.5 margin is applied before the eight-fit estimate. Thus 25.15 minutes is a conservative planning estimate, not a measured screen duration. Actual runtime may be substantially shorter. The predeclared estimate exceeds 15 minutes, so Codex hands off Screen. Full CV is never automatic, regardless of budget.

Peak pilot process RSS: 1421.61 MiB; CUDA allocation/reservation: 109.62/114.00 MiB. Environment is Python 3.12.10 / Torch 2.7.1+cu126 / RTX 4060 Laptop, with exact versions and source/config/device hashes in manifests. No new dependency or checkpoint download was needed.

### Verification and separate conclusions

Windows complete suite: **178 passed, 1 existing symlink skip**. Eleven new tests cover exact counts/ambiguous-label exclusion and repeated votes, deterministic uncertainty with no target mutation, rational near-tie boundaries, numerical BCE/soft-CE/ln(8), matched initialization/optimizer RNG, finite shared/logit gradients, inference without counts/captions/auxiliary forward, rare validation languages, frozen screening gates, actual tiny screening/replay, full-CV reuse of existing seeds, budget interruption and corrupt checkpoints. Synthetic fixture decisions used to exercise the optional CV path are never reported as real research results. No historical model was refitted.

The current Preflight protects all 1,066 files under EXP-017–020. Current dataset/protocol and original source hashes are verified, and complete preservation is checked at delivery. Production source/configuration and frozen prediction references remain unchanged. No screening, full CV or independent full replay was executed on real data in EXP-021.

Competition improvement: **not assessed**. Frozen-head learning: finite short-run auxiliary fitting and nonconstant in-sample predictions are observed, but held-out screening benefit is unknown. LoRA usefulness: **unproven**; feature-input gradients and in-sample CE do not establish encoder-level adaptation. Current recommendation is **BCE-only first** as the eventual LoRA reference. If verified Screen becomes PROMISING, the next separately authorized comparison should be BCE-only versus BCE + normalized eight-way vote CE, keeping this coefficient and the existing backbone fixed. A negative/inconclusive frozen-head screen would not prove the objective cannot help LoRA.

Next owner command:

```powershell
.\scripts\run_exp021.ps1 -Stage Screen
```

Then run `-Stage ScreenVerify`. Stop after INCONCLUSIVE or UNPROMISING. Only a verified PROMISING permits an explicit later `-Stage CV`, followed by `-Stage Verify`. No LoRA, model upload, merge, confirmation or Development evaluation is performed here.

Competition improvement, frozen-head learning and possible encoder fine-tuning value remain separate conclusions. A later LoRA comparison must first verify real encoder-parameter gradients. If Screen supports distribution learning, compare BCE-only versus BCE+normalized empirical eight-way CE at the frozen coefficient. If evidence remains insufficient, begin with BCE-only rather than assuming either auxiliary objective is useful. No LoRA training is launched here.
