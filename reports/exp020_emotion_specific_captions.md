# EXP-020 — Fine-grained emotion caption supervision

## Frozen design and provenance

Base: EXP-019 `c63bc02075a5d03b9ca53903c9f118a185f8d4f6`. Production E50+L is unchanged. Task 020 verified actual local EXP-019 completion before implementation: fifteen checkpoints passed independent replay, and its frozen competition decision was COMPETITION_NO_GO. Lambda=1 student pooled Contrast was 0.417569215 versus 0.440121979 for EXP-018 lambda=0.1; the lambda=1 ensemble scored 0.446633867 versus 0.449566497 for the prior polarity ensemble. Those are historical, repeatedly examined CV measurements, not new EXP-020 results.

Only A8 and E8 are trained: two conditions, five original painting folds, three original seeds, totaling thirty new models and sixty selection/refit stages. R0, R1, RB1 and RB3 reuse exact EXP-018 M0, M3, B1 and B3 predictions. Old model heads, visual/text caches, folds and PCA are never overwritten or refitted during CV. `configs/exp020.json` pins every completed EXP-019 stage/seed manifest and its configuration; the existing EXP-019 pins protect EXP-018. Original source files remain unchanged; additive EXP-020 modules have independent fingerprints.

Both new conditions allocate one shared eight-query network with learned 64-dimensional emotion embeddings, preserving the original sentiment branch. Emotion order is amusement, awe, contentment, excitement, anger, disgust, fear, sadness. Positive/negative associations follow official binary mapping; ambiguous labels are excluded. A8 uses each emotion's corresponding pooled polarity target, E8 its own emotion-caption mean. Lambda stays 0.1. All other optimizer, clipping, dropout, seeds, group splits, stopping BCE and epoch rules remain EXP-018's.

A8/E8 have identical emotion-level availability, weights, initialization and parameter count. A query is absent when its own emotion is absent, even if another emotion of that polarity occurs. Degenerate emotion/polarity projections remove that channel from both conditions. Each pair averages its valid channel losses before the original inverse-painting-pair weighting. Missing targets are zero and excluded. No category label enters the sentiment branch or inference; auxiliary outputs are not computed at inference.

## Audit and geometry definitions

Only the 1,335 development-training paintings contribute captions, emotion counts and audit statistics. Annotation multiplicities are retained, including repeated identical captions; associations stay attached to their original emotion. Raw means are stored in a disk-backed NPY array to avoid duplicating the full eight-channel tensor in RAM. The original pinned CLIP-L vectors and tokenizer diagnostics are reused without extraction or checkpoint downloads.

Raw and all ten original inner/full training-partition projections report 8x8 co-observed cosine matrices, comparison counts, within/across-polarity means, between-language and between-painting variation, multi-caption dispersion, and emotion/polarity similarity. PCA components/means come directly from the original pooled-binary-caption geometry. There is no emotion-specific PCA fitting. Normalization and the 1e-8 degeneracy rule follow EXP-018. Existing polarity projections are checked against saved normalized targets before constructing A8.

Fixed descriptive thresholds, chosen before the audit: near-indistinguishable emotion/polarity cosine >=0.99; high subgroup polarity similarity >=0.95; high language truncation >=0.5; near-collapsed predictions have mean off-diagonal query cosine >=0.99. Raw/normalized matrices exclude missing channels; diagonals and undefined comparisons remain explicit. Within/across-polarity aggregation weights each available within-pair channel comparison equally. Variation summaries retain per-group counts. Single-caption targets are counted separately and have no invented multi-caption dispersion.

## Diagnostics, scoring and decisions

The representative pilot uses fold 1, seed 2027, three deterministic inner-training batches and four short epochs per condition (one warm-up, three timed). It checks matched initial sentiment outputs/masks/weights/architecture, finite BCE/cosine losses, available-query gradients and exactly zero missing-channel target gradients. Diagnostics record raw/scaled shared, language and feature-input gradients, per-emotion shared-gradient components/diversity, query-embedding gradients, predicted/target cosine matrices and collapse. Evaluation mode preserves comparison RNG. These feature-input gradients do not pass through pretrained encoders and do not prove successful encoder adaptation. Final checkpoint diagnostics use folds 1/3/5 and all seeds; true zero gradients at convergence are recorded explicitly.

CV uses seed averaging and the authoritative six-decimal nested blend implementation. Outputs include pooled and fold-mean Contrast/Macro Log Loss, fold/language/painting changes, seed variance, pair-order agreement, and 10,000 paired painting draws. Primary research comparison is E8-A8; competition comparison is EE8-R0. Other comparisons: A8-R1, E8-R1, EE8-EA8, EE8-RB3, EE8-RB1, EA8-RB1. Comparisons with historical two-query R1 also change architecture and channel averaging; E8-A8 is the matched target-information comparison. Scoring uses the existing published-specification reconstruction; exact hosted scorer parity is unverified (`docs/competition.md`). Pretrained-resource eligibility remains provisional.

Post-CV subgroups are descriptive only: one/multiple emotions, at least one two-caption emotion versus all singleton targets, raw high/low polarity similarity, and lower/higher language tokenizer truncation. Pair subsets are aggregated at painting level using normal Contrast eligibility (at least three languages, nonconstant gold); undefined correlations are not manufactured. These subgroups do not select models. CV paintings have been repeatedly examined, so bootstrap intervals are not independent final-test evidence.

EA8/EE8 qualify only with >=0.003 pooled gain over R0, >=0.002 over RB1, >=3/5 fold wins over R0, fold-mean loss within R0+0.005 bits, and complete finite isolated/replayed seeds. Highest pooled Contrast wins among qualified candidates. No qualifying candidate means COMPETITION_NO_GO; keep confirmation sealed. No classification is made before completed independent Verify.

Research interpretation is independent: a matched E8-A8 gain >=0.001, at least three fold wins and a positive paired CI supports RESEARCH_PROMISING; a loss <=-0.001 with at most two wins and a negative CI supports RESEARCH_UNPROMISING; otherwise RESEARCH_INCONCLUSIVE. Inspect sparsity, target distinctiveness after PCA, prediction collapse, gradient diversity and training/stopping curves alongside that label. Small gains require caution. Negative results do not establish that emotion supervision is inherently useless.

## Runner, recovery and conditional scoring

```powershell
.\scripts\run_exp020.ps1 -Stage Preflight
.\scripts\run_exp020.ps1 -Stage Audit
.\scripts\run_exp020.ps1 -Stage Pilot
.\scripts\run_exp020.ps1 -Stage CV
.\scripts\run_exp020.ps1 -Stage Verify
.\scripts\run_exp020.ps1 -Stage Confirm
.\scripts\run_exp020.ps1 -Stage Dev
```

Default output: `runs/exp020_emotion_targets/`. Override explicit paths with `-RunDir`, `-Exp019Dir`, `-Exp018Dir`, `-Exp017Dir`, `-DataDir`, `-CacheDir`. Runtime/device/library/source signatures must match. Changing code/scientific settings needs a fresh run directory. Stages do not chain long training. `-MaxSeconds` overrides budgets. Interrupted audit resumes published target partitions; interrupted CV retains complete sealed seeds, restarting only an unpublished current seed. Corrupt or incompatible published outputs are rejected. A failed/interrupted later Verify blocks an earlier successful decision until replay succeeds again. Logs record wall time, exit code and RAM before/after; pilot/CV/replay record peak process RSS and CUDA allocation/reservation.

Exit 0 is successful stage completion; 1 validation failure; 2 invalid invocation/environment; 3 missing prerequisite or blocked gate; 75 resumable budget stop. Missing completed parent evidence gives an explicit original-branch recovery command. No force push, deletion of old artifacts, production modification or Codabench upload occurs.

Only a verified PASS opens confirmation. New confirmation heads use the original 1,335 training paintings and compare with refitted R0/matched BCE control. The previously unused confirmation-training partition uses the original pooled-caption PCA procedure if its transform is absent; no PCA is fitted on emotion-specific means or locked paintings. The existing +0.001 Contrast / positive incremental gain / +0.005-bit confirmation gates apply. Optional Dev reuses that confirmed student, trained on 1,335 paintings, and frozen release R0 probabilities. It never fits on the 236 locked captions or emotion labels, nor expands the caption cache. Public Development is scored once and a separate validated ZIP is produced only after both gates pass.

## Execution record

**READY_FOR_LOCAL_RUN.** Preflight, the complete training-only audit and the matched pilot passed. Full A8/E8 CV and heavy Verify have not run. Competition selection, locked confirmation and candidate public Development are pending; no new candidate ZIP exists. Research is **RESEARCH_INCONCLUSIVE**. The measured estimate exceeds 15 minutes, so the remaining experiment is handed off.

### Target coverage and sparsity

The audit covers 1,335 paintings, 13,600 eligible pairs and 69,719 binary-caption annotation rows. There are 38,787 available and 70,013 missing emotion channels. 784 pairs have one category, 4,101 have two and 8,715 have at least three. Maximum observed categories: seven; none has all eight. All 60 repeated pair/emotion/text rows were retained.

| Emotion | Annotation rows | Pairs | Pair coverage | Single-caption targets | Single-caption % | Two-plus targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| amusement | 6,888 | 4,459 | 32.79% | 2,812 | 63.06% | 1,647 |
| awe | 7,992 | 5,363 | 39.43% | 3,470 | 64.70% | 1,893 |
| contentment | 23,875 | 10,247 | 75.35% | 3,351 | 32.70% | 6,896 |
| excitement | 8,746 | 5,501 | 40.45% | 3,230 | 58.72% | 2,271 |
| anger | 1,467 | 1,211 | 8.90% | 1,005 | 82.99% | 206 |
| disgust | 4,656 | 2,863 | 21.05% | 1,819 | 63.53% | 1,044 |
| fear | 6,615 | 4,004 | 29.44% | 2,399 | 59.92% | 1,605 |
| sadness | 9,480 | 5,139 | 37.79% | 2,754 | 53.59% | 2,385 |

Overall 20,840/38,787 emotion targets (53.73%) are single-caption. Positive targets cover 12,592 pairs and negative targets 8,805; their caption-row counts are 47,501 and 22,218. Complete counts/masks and per-language coverage are in `audit/counts.npy` and `audit/language_coverage.csv`.

### Matched-population geometry

The matrices below use the predetermined fold-1 full-training population: 1,068 paintings / 10,836 pairs, before and after its exact saved PCA. The standalone read-only `scripts/audit_exp020_geometry.py` generated the matching raw population without changing any target or training input. The global raw audit additionally covers all 1,335 paintings. Abbreviations follow the fixed emotion order: Am amusement, Aw awe, Ct contentment, Ex excitement, An anger, Di disgust, Fe fear, Sa sadness.

Raw emotion-mean cosine

| | Am | Aw | Ct | Ex | An | Di | Fe | Sa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Am | 1.0000 | 0.8691 | 0.8956 | 0.8800 | 0.8657 | 0.8888 | 0.8807 | 0.8905 |
| Aw | 0.8691 | 1.0000 | 0.8821 | 0.8726 | 0.8622 | 0.8818 | 0.8837 | 0.8737 |
| Ct | 0.8956 | 0.8821 | 1.0000 | 0.8911 | 0.8818 | 0.8914 | 0.8938 | 0.8968 |
| Ex | 0.8800 | 0.8726 | 0.8911 | 1.0000 | 0.8621 | 0.8748 | 0.8740 | 0.8743 |
| An | 0.8657 | 0.8622 | 0.8818 | 0.8621 | 1.0000 | 0.8854 | 0.8906 | 0.8850 |
| Di | 0.8888 | 0.8818 | 0.8914 | 0.8748 | 0.8854 | 1.0000 | 0.9047 | 0.8981 |
| Fe | 0.8807 | 0.8837 | 0.8938 | 0.8740 | 0.8906 | 0.9047 | 1.0000 | 0.8992 |
| Sa | 0.8905 | 0.8737 | 0.8968 | 0.8743 | 0.8850 | 0.8981 | 0.8992 | 1.0000 |

Projected and normalized emotion-mean cosine

| | Am | Aw | Ct | Ex | An | Di | Fe | Sa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Am | 1.0000 | 0.7608 | 0.8063 | 0.7700 | 0.7248 | 0.7868 | 0.7809 | 0.7936 |
| Aw | 0.7608 | 1.0000 | 0.7734 | 0.7434 | 0.7071 | 0.7635 | 0.7867 | 0.7630 |
| Ct | 0.8063 | 0.7734 | 1.0000 | 0.7778 | 0.7539 | 0.7869 | 0.7986 | 0.7994 |
| Ex | 0.7700 | 0.7434 | 0.7778 | 1.0000 | 0.6968 | 0.7368 | 0.7515 | 0.7256 |
| An | 0.7248 | 0.7071 | 0.7539 | 0.6968 | 1.0000 | 0.7351 | 0.7929 | 0.7571 |
| Di | 0.7868 | 0.7635 | 0.7869 | 0.7368 | 0.7351 | 1.0000 | 0.8080 | 0.7965 |
| Fe | 0.7809 | 0.7867 | 0.7986 | 0.7515 | 0.7929 | 0.8080 | 1.0000 | 0.8092 |
| Sa | 0.7936 | 0.7630 | 0.7994 | 0.7256 | 0.7571 | 0.7965 | 0.8092 | 1.0000 |

| Matched statistic | Raw | PCA |
| --- | ---: | ---: |
| Within polarity | 0.887412 | 0.780076 |
| Across polarity | 0.884610 | 0.773461 |
| Emotion/polarity median cosine | 0.983047 | 0.970813 |
| Fraction at cosine >=0.99 | 0.387781 | 0.322876 |

Global raw near-polarity fraction is 0.387217. Across all ten projected partitions it ranges from 0.319782 to 0.323906. There are zero degenerate raw means, zero degenerate projected emotion means and zero common-mask removals in every partition. This establishes distinguishable geometry, not useful held-out learning.

Mean pairwise cosine across groups (each group weighted equally):

| Emotion | Raw between languages | PCA between languages | Raw between paintings | PCA between paintings |
| --- | ---: | ---: | ---: | ---: |
| amusement | 0.701479 | -0.033246 | 0.872857 | 0.765861 |
| awe | 0.700424 | -0.030097 | 0.870654 | 0.756682 |
| contentment | 0.726197 | -0.045245 | 0.897581 | 0.800809 |
| excitement | 0.717483 | -0.010812 | 0.871186 | 0.768613 |
| anger | 0.710580 | -0.023153 | 0.861622 | 0.746679 |
| disgust | 0.716209 | -0.046997 | 0.868119 | 0.761827 |
| fear | 0.713582 | -0.026373 | 0.880158 | 0.775280 |
| sadness | 0.714905 | -0.040598 | 0.892053 | 0.778847 |

Within-emotion multi-caption dispersion uses 14,290 matched groups. Mean caption-pair cosine is 0.867210 raw and 0.755892 projected. No dispersion is invented for singleton groups. Co-observation counts and group-level variation/dispersion CSVs accompany all ten PCA audits.

Saved CLIP token diagnostics cover 25 languages. High-truncation languages (fixed threshold >=0.5): Burmese 98.20%, Kyrgyz 73.58%, Tamil 87.83%. These rates were reused; no tokenizer/model was downloaded.

### Pilot gradients and query collapse

Each entry below is a median over three deterministic 256-pair inner-training batches. Both models allocate 149,978 parameters. Initial sentiment outputs, optimizer settings, masks and weights matched exactly. All available query embeddings received nonzero gradients, all loss/gradient checks were finite, and the maximum missing-target gradient was exactly zero.

| Condition / checkpoint | Aux cosine loss | Shared raw text norm | Shared scaled text norm | Shared text/BCE ratio | Shared gradient cosine | Near-collapse fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A8 / initialization | 1.002334 | 0.007520 | 0.000752 | 0.057964 | 0.017922 | 1.000000 |
| A8 / short_pilot_final | 0.133520 | 0.002712 | 0.000271 | 0.031192 | 0.000657 | 1.000000 |
| E8 / initialization | 1.003195 | 0.007218 | 0.000722 | 0.055636 | 0.015045 | 1.000000 |
| E8 / short_pilot_final | 0.171867 | 0.002628 | 0.000263 | 0.029998 | -0.000134 | 1.000000 |

Both models still meet the >=0.99 near-collapse criterion on all inspected eligible pairs after four epochs. E8 target query cosine is much lower than its predicted cosine. This is a research concern requiring final-checkpoint inspection, not a numerical error or a basis for an early competition NO-GO. The fixed architecture/loss were not changed.

| Condition / checkpoint | Mean predicted query cosine | Mean target query cosine | Mean emotion-gradient cosine |
| --- | ---: | ---: | ---: |
| A8 / initialization | 0.999707 | 0.907626 | 0.191733 |
| A8 / short_pilot_final | 0.997879 | 0.907626 | 0.153270 |
| E8 / initialization | 0.999707 | 0.775162 | 0.166688 |
| E8 / short_pilot_final | 0.997456 | 0.775162 | 0.096254 |

Emotion-gradient vectors have different masked row support; their cosine combines semantic direction with overlap in observed emotion channels. Matching masks control this between A8/E8, but a low cross-emotion cosine alone does not demonstrate useful semantic diversity. First-visual initial scaled ratios were 0.009352 (A8) and 0.009444 (E8), with cosines -0.191396 and -0.245513. Shared-representation and feature-input gradients were mostly orthogonal to BCE. These remain head/feature-input diagnostics, not encoder-parameter gradients.

| Emotion | Covered pairs in pilot batches | Initial A8 component norm | Initial E8 component norm |
| --- | ---: | ---: | ---: |
| amusement | 238 | 0.00124210 | 0.00124273 |
| awe | 287 | 0.00152322 | 0.00151175 |
| contentment | 574 | 0.00256123 | 0.00255275 |
| excitement | 309 | 0.00179892 | 0.00179702 |
| anger | 56 | 0.00056038 | 0.00054704 |
| disgust | 164 | 0.00139248 | 0.00139562 |
| fear | 221 | 0.00171439 | 0.00172027 |
| sadness | 311 | 0.00165936 | 0.00165478 |

Component norms above are unscaled; multiply by 0.1 for scaled contributions. Full gradients, masks, predicted/target matrices and both collapse definitions are preserved in `pilot/gradients.json`. The short E8 stopping BCE was 0.509282 versus 0.509367 for A8; this tiny four-epoch difference is not a CV result.

### Runtime, verification and handoff

| Item | Actual or estimated time |
| --- | ---: |
| Actual Preflight process | 31.63 s |
| Actual Audit process | 260.53 s |
| Actual Pilot process | 54.04 s |
| Successful three-stage total | 346.21 s |
| Timed A8 / E8 epoch | 1.168 / 1.478 s |
| CV estimate, including historical margin | 104.47 min |
| Verification allowance within total | 5.00 min |
| Total preparation + CV + allowance | 114.61 min |
| Optional confirmation, separate | 20.89 min |

Historical EXP-018 M3 used 601 selection +376 refit epochs across fifteen fits, taking 703.17 s. EXP-019 C2 used 799+574 epochs and 1483.02 s. The estimate takes the maximum of twice each historical fitting total and the two-condition measured epoch forecast (4178.74 s), applies the fixed 1.5 margin, then adds actual preparation and the 300-second allowance. It is conservative and stopping duration remains unknown. It does not reuse the last resumed invocation's duration as total fitting time.

Peak pilot process RSS was 1565.16 MiB; CUDA allocation/reservation peaked at 114.23/136.00 MiB. Audit logs record before/after free RAM, not peak audit RSS. Environment: pinned Python 3.12.10, Torch 2.7.1+cu126, RTX 4060 Laptop; exact versions/source hashes are in stage manifests.

Windows tests: **167 passed, 1 existing symlink skip**. Ten new tests cover mappings, repeated means, PCA reuse, common masks, normalization/weighting, sparse gradients, inference without captions, exact rounding, actual tiny A8/E8 fits/replay/resume and blocked confirmation. Frozen synthetic references are explicitly test fixtures, never research scores. The first real Preflight exposed a production inference-manifest schema distinction (`identity` is absent there); validation was corrected and a regression test added before successful preparation. No successful outputs were overwritten.

Preflight preserved 445 EXP-018/019 files and checked the older 291 historical hashes. No old source, model, checkpoint, PCA, cache or prediction was changed. Cached production regression re-generated all 849 E50+L, E50 and CLIP predictions byte for byte and validated their ZIPs plus the original release archive. E50+L SHA-256 remains `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`. This is release parity checking, not new candidate Development scoring.

Full CV, independent thirty-checkpoint/ten-target replay, promotion, confirmation and candidate Development remain owner stages. Competition outcome is pending. Research remains INCONCLUSIVE: targets retain distinct information, but are sparse and the short pilot shows near-collapse. Inspect full E8-A8 effects and final diagnostics before authorizing a LoRA experiment. If collapse persists without reliable matched gains, prefer the existing two-query polarity baseline for a later small, fixed CLIP-L LoRA versus BCE-only comparison. No LoRA is launched here.

Next owner command:

```powershell
.\scripts\run_exp020.ps1 -Stage CV
```

Then run `-Stage Verify` separately. Confirm and Dev enforce the frozen gates. No automatic upload or merge.
