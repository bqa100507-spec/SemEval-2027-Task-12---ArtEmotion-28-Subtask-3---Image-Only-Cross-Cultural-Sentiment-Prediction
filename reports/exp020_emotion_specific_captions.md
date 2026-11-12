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

CV uses seed averaging and the authoritative six-decimal nested blend implementation. Outputs include pooled and fold-mean Contrast/Macro Log Loss, fold/language/painting changes, seed variance, pair-order agreement, and 10,000 paired painting draws. Primary research comparison is E8-A8; competition comparison is EE8-R0. Other comparisons: A8-R1, E8-R1, EE8-EA8, EE8-RB3, EE8-RB1, EA8-RB1. Scoring uses the existing published-specification reconstruction; exact hosted scorer parity is unverified (`docs/competition.md`). Pretrained-resource eligibility remains provisional.

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

Implementation and lightweight validation are in progress. Actual audit/pilot measurements, execution decision and remaining owner command will be recorded after preparation. No new CV score, promotion decision or confirmed candidate currently exists.

The next research step, only if matched CV and final diagnostics support it, is a separately authorized small LoRA pilot on the existing CLIP-L: compare BCE-only against the selected lambda=0.1 auxiliary condition using identical grouped splits and a fixed small trainable attention subset. First verify actual encoder-parameter gradients and memory cost. If E8 does not beat A8 reliably, prefer the cheaper two-query polarity target for that pilot. Do not begin LoRA or raise lambda in EXP-020.
