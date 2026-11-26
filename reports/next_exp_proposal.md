# Proposed EXP-023: painting-centered caption supervision

**PROPOSAL ONLY — NOT IMPLEMENTED, NOT RUN, NOT AUTHORIZED.** EXP numbering is separate from TASK-023 deployment and TASK-024 closure. Existing configs/reports/remote refs contain EXP-001–022; EXP-023 is the next available research identifier. Exactly one primary experiment is recommended: **painting-centered CLIP-caption targets for a lightweight head over frozen visual features**. Do not reopen EL8, train LoRA, search teachers or run this proposal under the closure task.

## Evidence review before choosing a variant

The project review covers EXP-001–022, including the archived EXP-012–015 reports and actual later manifests where preparation reports are stale. [Research index](../docs/research_index.md) separates the protocols and task-only infrastructure numbers.

EXP-017 is directly relevant to the current CV population, rather than merely numerically similar. Read-only comparison found **all 13,600 R0/M0 painting-language rows, five fold assignments and probabilities exactly equal between EXP-017 and EXP-022**, covering the same 1,335 paintings. EXP-017's recorded independent verification is PASS for 45 student checkpoints, 20 partition-local PCA transformations and 15 LR baselines. Its CV selection/source/config/summary hashes and existing CLIP-caption feature checksum were checked; no target was regenerated or model replayed for this closure.

| EXP-017 system | Meaning | Pooled Contrast | Macro Log Loss |
| --- | --- | ---: | ---: |
| M0 | E50+L refit in the current folds | 0.447097707 | 0.710554182 |
| M1 | Frozen-feature BCE student | 0.437792268 | 0.716045717 |
| M2 | Same student + pooled CLIP-caption MSE | 0.439016172 | 0.716105044 |
| B1 | Fixed 50/50 M0 + M1 | 0.448585747 | 0.706474066 |
| B2 | Fixed 50/50 M0 + M2 | 0.449753673 | 0.706239161 |

B2−B1 is **+0.001167926**, with stored paired 95% interval **[+0.000269630, +0.002043707]**. B2−M0 is **+0.002655966**, interval **[−0.000759359, +0.006141706]**. B1−M0's interval also spans zero. B2 wins four folds against M0, but misses EXP-017's preregistered +0.003 baseline and +0.002 incremental gates. M2−M1 standalone uncertainty spans zero. Thus the matched text signal warrants a new falsifiable target hypothesis, **not promotion or a replay of old B2 as a new method**. These intervals are conditional on repeatedly examined paintings, not fresh confirmation.

The E5 ensemble's increment over B1 was smaller and uncertain (+0.000562889, interval crossing zero), with unresolved checkpoint eligibility. EXP-018 pooled/polarity cosine supervision, EXP-019 coefficient escalation, EXP-020 emotion-specific queries and EXP-021 annotation-only CE8 all ended NO-GO. EXP-022's stronger CV gain reversed on hosted Development. A new teacher search, bigger auxiliary coefficient or more LoRA would disregard this evidence.

## Primary hypothesis and distinction

**Falsifiable hypothesis:** removing the painting-common direction from training-only CLIP caption targets yields a better image-language sentiment head than both an otherwise identical uncentered-caption control and a BCE-only control, and the fixed ensemble survives a single guarded confirmation comparison.

Reasoning is a hypothesis, not a causal finding: pooled caption vectors may emphasize painting content shared across languages, while Cultural Contrast concerns within-painting language differences. Centering changes the supervised target direction to align with that question. It does not change cached visual embeddings or adapt an encoder; only the learned head representation can improve. It may instead amplify noise or remove useful information.

This differs from EXP-012's post-hoc logit correction, EXP-018's two polarity queries/cosine scaling, EXP-019's larger coefficient, EXP-020's eight emotion queries, EXP-021's vote-distribution targets and EXP-022's visual adaptation. The sole new scientific treatment is **within-painting centering of caption targets**. No simultaneous language-shrinkage change, additional branch or backbone is allowed.

## Data, resources and eligibility

Use only the original 1,335 CV paintings' permitted training votes/captions, original identifiers/folds and the existing CLIP-B/SigLIP2-Base/CLIP-L frozen image banks. Reuse the checksum-verified EXP-017 CLIP-L text cache (69,719 eligible caption rows / 69,153 exact texts; `features.npz` SHA `93fc6ef0fcc492dd9fff7e59d719841dfbc0e4fda381ebe46dac073d0149139d`). The text checkpoint stays `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41`. No new checkpoint, LLM, VLM, generated caption, translation or external annotation is proposed. Teacher extraction is not needed for this cache-covered experiment.

The [pinned organizer ST3 description](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#st3-image-only-cross-cultural-sentiment-prediction) requires image and target-language inference. Its [external-resource FAQ](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#faqs) leaves guidance pending. Historical competition captures permitted training captions/pretrained models, but the current Codabench fetch returned unrelated content. Therefore retain the existing eligibility caveat: **resolve organizer permission for training-only caption distillation before competition execution/promotion; checkpoint license alone is not permission**. E5 is excluded. No organizer contact is sent in this task.

No captions, votes, language-response distributions, artist/style metadata or training image IDs are inference features. Identifiers only locate images, cached features and groups. Caption-derived targets are built solely for the current fitting paintings; no stopping, outer-held, confirmation or public Development captions enter a learned transform.

## Frozen architecture and controlled conditions

The proposed preregistration uses EXP-017's width-64 sentiment architecture and allocated auxiliary MLP `128→64→64`, with the original normalized 64-dimensional output. Input Z is the three normalized visual blocks concatenated /sqrt(3). Fixed original settings: FP32 head learning, dropout0.2, AdamW LR0.001/decay0.01, clipping1, batch256 pairs, seeds2027/2028/2029, max150/patience15, minimum inner binary BCE, fresh reset/refit. Do not inherit EXP-022's image-grouped/LoRA batching for this comparison.

Three matched conditions, five folds and three seeds give **45 lightweight fits**:

| Condition | Sentiment loss | Auxiliary target |
| --- | --- | --- |
| C0 | Original vote-normalized BCE | Allocated, unsupervised |
| C1 | BCE + 0.1 coordinate-mean MSE | Original uncentered caption target on the shared availability mask |
| C2 — sole treatment | Same BCE + 0.1 MSE | Painting-centered caption target on the same mask |

For a fitting partition, average the original unit CLIP caption embeddings per eligible painting/language, preserving repeated annotation votes exactly as EXP-017. Fit one unwhitened 64-PC transform to those **fitting** pair means using randomized PCA seed2027; C1/C2 share it. Normalize projected pair targets as before, giving t[p,l]. Define `d[p,l] = t[p,l] − mean_language(t[p,*])` over that training painting's eligible languages, with equal language weighting; normalize d when its norm exceeds 1e-8. No cross-painting or held-out response supplies that center.

Use one **common auxiliary mask** for C1/C2: at least three eligible languages, valid original target and nondegenerate centered vector. Both use the original inverse-valid-pair-count painting weights on that common mask; BCE and OOF scoring retain **every eligible pair**. The existing OOF identities show 32 paintings with fewer than three languages, so this mask is a real coverage change. Historical unmasked B2 remains a provenance reference, not an exact matched control for C2. The fresh common-mask C1 is necessary to isolate centering from availability changes. Freeze these rules before computing new targets; do not inspect scores to choose masking thresholds.

Construct separate inner-fit and full-outer-fit transforms/centers. Inner stopping uses BCE only; outer-held predictions use pixels/language only and never need caption targets. Matched initialization, dropout RNG, pair ordering, vocabulary fallback and loss normalization must be identical. No lambda, dimension, temperature, blend or optimizer search. Existing EXP-017 C0/B1 and unmasked M2/B2 should first be independently reconstructed from stored predictions on the exact common folds; a mismatch blocks execution rather than relaxing replay.

## Evaluation, gates and independent confirmation

Reuse the exact 1,335-painting protocol, five 267-painting outer folds and original inner hashing seed17029+fold. Report pooled and fold-mean Contrast/Macro Log Loss, per-seed results, every fold, painting wins/regressions, language loss, target-mask coverage and fixed language-coverage/aspect-ratio subgroups. Diagnostic subgroup findings cannot choose a variant or change a gate. Use 10,000 paired painting bootstrap draws, seed232027, and report leave-one-fold-out **aggregation diagnostics**, not newly trained alternate folds. All learned transforms remain partition-local.

R0 is fit only on each corresponding training fold. Form EC0/EC1/EC2 with a fixed 50/50 probability blend: mean three raw seed probabilities → round6 → blend with rounded R0 → round6. Historical EXP-010/013/014 CV uses 1,571 paintings and different folds; never substitute those scores. Prefer the cheaper released E50+L for ties or incomplete evidence.

Proposed gates are frozen in this document **before execution**, but are resource/decision heuristics, not guarantees or significance tests:

1. **CV_GO** requires all 45 fits/replays, isolated groups, finite/noncollapsed predictions; EC2−R0 pooled Contrast ≥+0.005; EC2−EC1 and EC2−EC0 each ≥+0.002; ≥4/5 folds improve over R0 and ≥3/5 over EC1; no R0 fold regression worse than −0.010; fold-mean Macro Log Loss ≤R0+0.003 bits. Paired 95% lower bounds must exceed zero for EC2−R0 and EC2−EC1. Show individual-seed directions and do not drop a seed. Otherwise **NO_GO**, export and stop; no tuning on these folds.
2. **Guarded confirmation** is permitted only after CV_GO+Verify and only when the single future owner authorization explicitly includes this conditional stage. The 236 original training paintings remain sealed until then; they are not accessed in this closure. Fit R0 and all three heads on the same 1,335 paintings, freeze checkpoints and hashes, then evaluate once on the 236. Require EC2−R0≥+0.003, positive incremental gain ≥+0.001 over each matched ensemble, Macro Log Loss≤R0+0.003 bits and a positive paired lower95 bound versus R0. Failure ends the experiment without a Development candidate. No hyperparameter adjustment follows confirmation feedback.
3. **Final artifact identity:** the exact 1,335-trained models evaluated at confirmation become the final inference bundle after GO; no recipe/population switch afterward. In particular, the released 1,571-trained E50+L heads have already seen those 236 paintings and **must not be used as the confirmation baseline**. Use the separately named R0_1335 component for both confirmation and the candidate's later input-only inference. Production E50+L_1571 stays untouched. If a new recipe is desired, it requires a separately authorized study.
4. **Hosted promotion** is not automatic. Only after both gates may one fixed Development input-only archive be produced for **one manual** Codabench comparison. The owner-reported baseline is 0.4643; predeclare a material hosted gain of at least +0.001 (≥0.4653 at reported precision). Assess a same-hosted-scorer loss guardrail if that baseline/candidate loss evidence is available; do not impute hosted loss from earlier local scoring. No further uploads/tuning to chase this feedback. A failure leaves E50+L production.

This makes confirmation part of the future one-authorization scope, not an extra mid-run prompt. If that scope or an eligibility prerequisite is absent, fail before training. Minimum Codabench uploads: **zero on NO_GO, one manually approved fixed candidate if all gates pass**. No Development gold is read for selection or verification.

## Cost, risks and reproducibility

Measured historical anchors: EXP-017's 45-fit CV command took 987.31 s and independent replay 526.36 s; caption extraction was an additional cost then, but the required CLIP cache now exists. These are prior RTX4060 timings, not a benchmark of this proposal on L4/CPU. Plan **roughly 0.5–1.0 GPU allocation-hours**, plus several CPU minutes for PCA/aggregation, including verification and conditional final/confirmation work. Pure CPU execution is plausible for small heads but has no measured runtime forecast here. Preregister caps of **1.0 GPU-hour / 2.0 wall-hours**; a measured pilot must demonstrate the remaining workflow fits the cap before full fits. Stop rather than silently changing scientific settings. No extraction/backbone training is budgeted.

Main risks: centering amplifies noisy five-vote targets; multilingual CLIP truncation/semantic limitations; masked low-coverage paintings; weak auxiliary gradients; adaptive reuse of CV paintings; confirmation power at 236 paintings; and final-model transfer differences. The target-centering explanation is unproven. A positive within-CV interval is insufficient by itself, as EXP-022 demonstrates. If C2 is indistinguishable from C1 or fails confirmation, record NO_GO and close rather than adding losses/learning rates.

Pin source/config/data/folds/teacher and image cache SHAs, exact package/runtime profiles, all masks/PCA memberships, initialization/RNG traces, optimizer steps, selection/refit epochs, per-seed predictions and nested rounding. Reuse only completed boundaries with matching identities. No cross-runtime fit mixing; failed replay revokes a prior successful decision. Keep code, weights, targets and reports in separately versioned run directories and archive failures. No model or notebook implementation is created by this proposal.

## Ranked choices

Ranks are qualitative judgments from the available evidence, not predicted numerical score gains. All options preserve image-only inference; none is authorized now.

| Rank | Direction | Novelty / evidence | Expected Contrast effect | Rank-reversal risk | Compute / complexity | Paper value |
| --- | --- | --- | --- | --- | --- | --- |
| **1 — primary** | Painting-centered caption targets + matched mask, with prospective confirmation | Modest new target hypothesis; supported only indirectly by B2−B1 | Uncertain; no promised gain | High until confirmation; explicitly guarded | Frozen caches; small heads; 0.5–1h planning cap; moderate implementation | Clear controlled text-information ablation and potential negative result |
| 2 — validation alternative | Generalization audit / prospective blind confirmation of fixed cached systems | Little model novelty; strongest lesson from EXP-013/022 | No direct score improvement promised | Lowers selection risk, cannot eliminate it | Mostly CPU, low/moderate complexity | High methodological value; no new model claim |
| 3 — head alternative | Hierarchical language-interaction shrinkage inside a frozen-feature head | Must differ from EXP-012 logit corrections and EXP-021 reliability reweighting; little direct evidence | Unknown | High on the reused folds | CPU/small GPU; moderate complexity | Useful only with a genuinely matched regularization ablation |
| 4 — defer | Further visual fine-tuning | EXP-014/022 do not support another large run absent a new hypothesis | Unknown; may regress | High | Highest cost/complexity | Low incremental value without new independent evidence |

The future implementation must follow [the one-click workflow contract](../docs/experiment_workflow_standard.md), including optional browser download attempts after verified Drive export and before runtime release. This task stops at this proposal and closure.
