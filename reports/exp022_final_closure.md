# EXP-022 / TASK-023 final closure

**EXP022 = CLOSED · TASK023 = CLOSED · EL8 = NO_GO_FOR_PRODUCTION · PRODUCTION = E50+L.**

EXP-022 completed a controlled research experiment. TASK-023 completed deployment engineering. The resulting EL8 candidate regressed in the owner-reported independent Development comparison. Its train-only CV improvement did not translate into a measured Development gain. No further EL8 training, retuning, promotion or submission is an open action item. This closure performs no model execution, training, Development scoring or new submission.

## Evidence and provenance

The final Colab output was recovered automatically through the authenticated Drive connector from `MyDrive/el8_final_runs_v2/attempts/20261115T145828Z/a4dc9cf4/outputs.zip`. Both the archive and latest standalone `final_report.json` matched their supplied SHA-256 receipts. The archive contains **104 checksummed files**; every entry was verified before extraction into a new ignored directory. Historical files were not changed.

| Evidence | Immutable identifier / location |
| --- | --- |
| EXP-022 training source | `eb4a7fac4ebe79971305b0b3de2c0e5102ea5f91` |
| Historical verified CV run | `runs/exp022_colab_verified/c30c4ad20092b87c/` |
| Deployment implementation | `f6be82a60ac0a5e6d247d463b29aa2d07fabbcf9` |
| Original final delivery | `c3337fa8ae8010d76fa598536f5ff3f506ec6215` |
| Recovered Colab archive | `runs/task024_closure/colab_20261115_a4dc9cf4/outputs.zip` |
| Colab archive SHA-256 | `3a7af734bf3d399723e6b0798630341fdc8aef78b60e83270fd5a919def1e78a` |
| Archive size | 18,822,461 bytes |
| Read-only verification receipt | `runs/task024_closure/colab_verification.json` |
| Committed aggregate evidence | [exp022_closure_evidence.json](../docs/exp022_closure_evidence.json) |
| Permanent research archive | [archive/exp022-task023-no-go](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/tree/archive/exp022-task023-no-go) |

All three final checkpoint state dictionaries passed strict CPU loading, finite-value/shape checks, original model seals, source verification against all **73** recorded deployment files and final-bundle validation. The bundle and standalone copies of each checkpoint match. These are read-only integrity checks, not a new inference replay. The archive's actual L4 replay evidence contains 25 checks: 15 held-out seed predictions plus five L8 means and five EL8 blends. All pass with maximum absolute error **0**. Original EXP-022 CV Verify records **60/60** PASS. Earlier strict RTX4060 replay failures remain preserved as cross-runtime evidence; no tolerance was relaxed to obtain the later L4 PASS.

## Research question and frozen protocol

The factorial study asked whether visual CLIP-B LoRA helps beyond an otherwise matched frozen head, and whether empirical eight-emotion supervision adds value beyond BCE. Hypotheses concern held-out cross-cultural sentiment prediction, not merely lower training loss or nonzero gradients.

Only original official training paintings, eligible binary-mappable votes and identifiers were used. The original 236-painting confirmation partition stayed sealed. CV used **1,335 paintings / 13,600 pairs / 69,719 votes**, five fixed painting-grouped folds, three seeds (2027/2028/2029), and nested BCE stopping followed by a fresh full-outer-training refit. At least five valid annotators per pair were required; ambiguous labels were excluded. Painting/image identifiers locate resources and groups; they are not predictive features. No caption supervision was used in EXP-022.

| Frozen choice | Value |
| --- | --- |
| Visual inputs | CLIP-B 512 + SigLIP2-Base 768 + CLIP-L 768; each block unit L2; concatenate / sqrt(3) |
| Adapted checkpoint | `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| LoRA | Visual q/v in all 12 blocks, 24 modules; r=8, alpha=16, dropout=0; 294,912 parameters |
| Frozen weights | CLIP-B base/projection, SigLIP2-Base and CLIP-L; no text tower |
| Sentiment head | Original width 64, dropout 0.2, shared/image-language terms; compact eight-way auxiliary head allocated in every condition |
| Objective | Vote-normalized BCE; treatment adds 0.1 × CE8 / ln(8) from empirical votes |
| Optimizer | AdamW; head LR 0.001; LoRA LR 0.0001; decay 0.01; global clipping 1.0 |
| Sampling | Four distinct paintings/effective step; shared encoding for their language rows; same order/initialization across conditions |
| Stopping | Inner 10% grouped split; minimum binary BCE; max100 epochs, patience10; seeded reset/refit |
| Precision/preprocessing | Original deterministic 224px CLIP crop; FP32 vision/adapters; FP16 head-training autocast, FP32 losses/prediction; historical TF32 flags preserved |
| Inference | Artwork pixels + target language only; no votes, captions or other-language responses |

Exact configuration, asset pins, gradients, pixel-cache parity and sampling semantics remain in the [archived original report](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/c3337fa8ae8010d76fa598536f5ff3f506ec6215/reports/exp022_visual_lora_vote_distribution.md) and `configs/exp022.json` at the recorded training source. Its early resource-handoff status is historical; the completed CV/Colab manifests and this closure supersede that execution status.

## Completed train-only CV

These figures are read from the verified stored results; closure did not rescore any gold. F0/F8 are matched frozen controls; L0/L8 add visual LoRA. Prefix E means a fixed 50/50 blend with fold-isolated R0. Average the three raw seed probabilities, round to six decimals, then apply the existing rounded probability blend. This is not logit fusion.

| System | Pooled Contrast ↑ | Pooled Macro Log Loss, bits ↓ |
| --- | ---: | ---: |
| R0 — E50+L | 0.447097707 | 0.710554182 |
| F0 — frozen BCE | 0.435973427 | 0.721847137 |
| F8 — frozen BCE + CE8 | 0.442186747 | 0.717401409 |
| L0 — LoRA BCE | 0.439294186 | 0.714772802 |
| L8 — LoRA BCE + CE8 | 0.445211139 | 0.712555031 |
| EF0 | 0.448276545 | 0.708833834 |
| EF8 | 0.450555098 | 0.706858907 |
| EL0 | 0.448898611 | 0.707373251 |
| EL8 | **0.452483161** | **0.705591157** |

EL8 won four of five folds against R0. Its pooled gain was **+0.005385453**, with stored 10,000-draw painting-paired 95% interval **[+0.001897973, +0.008796065]**. This cleared the frozen CV gate, including gain over EF0 and the Log Loss guardrail. The then-correct decision was `CV_GATE_QUALIFIED_NO_DEVELOPMENT_AUTHORIZED`; it did not itself authorize production promotion.

| Scientific comparison | Pooled Contrast change | Stored paired 95% interval |
| --- | ---: | ---: |
| Backbone: L0−F0 | +0.003320759 | [−0.001636755, +0.008300804] |
| Backbone: L8−F8 | +0.003024392 | [−0.001714298, +0.007844872] |
| Auxiliary: F8−F0 | +0.006213320 | [+0.002525743, +0.009889013] |
| Auxiliary: L8−L0 | +0.005916952 | [+0.002436367, +0.009309299] |
| Factorial interaction | −0.000296367 | [−0.004044982, +0.003380295] |
| Ensemble backbone: EL8−EF8 | +0.001928062 | [−0.000467540, +0.004320536] |

The auxiliary effect is more consistent in these stored conditional intervals than the incremental LoRA effect. The interaction interval spans zero. Standalone L8 still trails R0. Do not attribute the whole ensemble gain uniquely to backbone adaptation. The intervals condition on repeatedly examined CV paintings and fixed predictions; they do not correct for the research program's adaptive reuse or guarantee independent generalization.

## Final refit and deployment

TASK-023 explicitly specified the previously absent full-population policy: use the same 1,335 permitted paintings, fold=0's hashed 10% inner stopping partition (seed17029), unchanged BCE stopping/max100/patience10, then reset and refit each seed on all 1,335 paintings for its selected epoch count. No CV checkpoint was substituted for a final model. The original released E50+L component was retained. The original E50+L LR heads were trained on 1,571 paintings; this population difference is a documented limitation, not an established explanation of the outcome.

| Final L8 seed | Selected refit epochs | Stopping-run epochs | Fit seconds | Reload/smoke |
| --- | ---: | ---: | ---: | --- |
| 2027 | 10 | 20 | 545.071 | PASS, zero error |
| 2028 | 11 | 21 | 588.238 | PASS, zero error |
| 2029 | 8 | 18 | 472.459 | PASS, zero error |

The actual Colab run used NVIDIA L4, Python3.12.10, Torch2.7.1+cu126 and Transformers4.49.0. Final checkpoint replay passes for all three seeds on 148 smoke pairs each. No inference auxiliary head, text or labels are required. The three checkpoint SHAs are in the aggregate evidence file.

| Deployment observation | Verified value |
| --- | --- |
| Final status | `SUCCESS_SUBMISSION_READY` |
| Official input coverage | 849/849 pairs; exact order and valid ZIP |
| Total observed wall/GPU allocation time | 2,072.218 s = 34.54 min ≈0.5756 h |
| Training-stage occupancy | 1,610.304 s = 26.84 min ≈0.4473 h; not CUDA kernel-active time |
| Largest recorded final-fit allocated/reserved VRAM | 549,011,456 / 606,076,928 bytes |
| Submission ZIP SHA-256 | `e64fc89307d214ee356cbe72a038d310c236ff6a8dbe573fa2a78629988eec77` |
| Prediction CSV SHA-256 | `89e7fb28c386ed6c9672d0fa426597686c81bb2306d059c8239531e0c4b8f751` |
| Runtime release evidence | `API_RETURNED`; physical deallocation not independently confirmed |

The one-run engineering sequence succeeded: validate → replay → benchmark → missing final fits → verify → real inference → ZIP checks → persistent export → release request. The final artifacts supersede earlier implementation-only reports saying no final bundle existed. Both those reports and the original failures remain intact.

## Owner-reported Codabench comparison and decision

| System | Hosted Development submission | Contrast ↑ | Difference vs production |
| --- | ---: | ---: | ---: |
| Production E50+L | 1046158 | **0.4643** | — |
| EL8 | 1053827 | 0.4537 | **−0.0106** (about −2.28% relative) |

These hosted scores are supplied by the owner in TASK-0024. The submitted ZIP identity is independently verified locally; the private Codabench score records are not independently fetched or reconstructed from Development gold. No EL8 hosted Development Log Loss or paired label-level uncertainty was supplied. No statistical significance is claimed for this difference. Development is separate from fitting/CV, but the project has examined Development outcomes across several systems; it is not a permanently untouched global test set.

**Final decision: NO-GO for EL8 production promotion.** E50+L stays the default and production system; E50/CLIP stay explicit fallbacks. EL8 remains an archived negative candidate, not an open deployment or retraining task. A successful research experiment, executable model and valid submission do not establish a competition improvement.

## Interpretation, limitations and paper findings

Observed: controlled CV gains, exact same-runtime replay, valid final refits and a negative hosted Development comparison. Not established: whether distribution shift, model/epoch selection, adaptive CV overfitting, training-population differences or another factor caused the reversal. It is not evidence that LoRA is universally ineffective. More EL8 tuning to reverse this specific result is not justified or authorized.

Paper-ready findings include (1) the factorial frozen/LoRA × BCE/CE8 comparison, (2) the distinction between auxiliary and backbone effects, (3) the positive conditional CV interval alongside negative independent Development direction, and (4) verified deployment cost/reproducibility. EXP-013 had an earlier positive-CV/negative-Development pattern on a different protocol; it motivates caution, not pooling incompatible scores. Keep all failed configurations in the ablation account and mark the released E50+L system explicitly.

Lessons: retain a genuinely unexamined confirmation stage; distinguish fit populations and final recipes; require matched controls and material effects; preserve fixed probability rounding and per-runtime identities; and make operational success separate from scientific promotion. The next proposed experiment addresses a new target-supervision hypothesis with frozen encoders, rather than reopening EL8. See [proposal](next_exp_proposal.md), [research index](../docs/research_index.md) and [future workflow standard](../docs/experiment_workflow_standard.md).

## Git closure delivery

PR #22 was closed unmerged first (2026-11-15 16:28:07 UTC), then PR #21 (16:28:30 UTC), each with an explicit final NO-GO comment. Both original head SHAs remain unchanged. Before either action the archive report was read back from GitHub at documentation commit `1ba7f3e6e2db05b4b2225604efb1ffa6b104b863`; the training, EXP-022 head, deployment and final-delivery commits were checked as its ancestors. No unrelated downstream PR existed; upstream PRs #16–20 were left untouched. No branch was deleted, no merge performed and no history rewritten.

Documentation-only [PR #23](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/pull/23), branch `codex/024-exp022-closure`, targets production main directly. Its diff contains only README, AGENTS, documentation and reports, so it does not introduce any archived model implementation into production. It remains unmerged for review. The permanent archive also carries these closure documents while preserving the complete original research tree; the original EXP-001–015 archive remains at `cd321f142142120c17c495458390deb6574ad856`. The [machine-readable audit](../docs/task024_git_audit.json) records dependency snapshots, closure readbacks and verification boundaries.

## Reproduction and archival boundary

The permanent archive retains every original source/config/report and commit. Private input ZIPs, all 60 CV models and all three final models remain outside Git. Full local inventory/receipts are under `runs/task024_closure/`; historical inputs/weights stay in their original directories. The downloaded output includes `run/bundle`, `run/final`, `run/submission`, provenance, curves, diagnostics and source bundle; model/ZIP checks can be repeated read-only on CPU. Do not regenerate these artifacts to document closure.

Historical commands and notebooks at the immutable deployment revision reproduce the engineering workflow, but running them now would require a new explicit owner task and budget. They are not pending instructions. Current authorized verification consists of SHA checks, stored-manifest/source inspection, tensor-schema validation and submission coverage validation only. No new training, inference, Development scoring, locked-label inspection or leaderboard submission occurred during this closure.

For read-only archive identification on the prepared Windows machine:

```powershell
Get-FileHash runs/task024_closure/colab_20261115_a4dc9cf4/outputs.zip -Algorithm SHA256
Get-FileHash runs/exp010_clip_large/dev/predictions/E50+L/predictions.csv -Algorithm SHA256
git show c3337fa8ae8010d76fa598536f5ff3f506ec6215:reports/task023_final_one_run.md
```

TASK-024 ran the unchanged production branch's lightweight suite with CUDA hidden: **104 passed, 1 skipped** (Windows symlink fixture), 4.73 seconds. JSON/UTF-8 and new-document relative links passed. The production CSV matches the frozen SHA above; the original production ZIP matches `38c2eca8ee4c0557bf198bf5df5dbf90571a2851404ded6302f40e42dc138e0d` and validates all 849 input pairs without model execution or gold. The pre-closure inventory's **498 protected files** match exactly, including both owner-modified notebooks, all historical CV checkpoints/seals and production bundles. These local checks do not constitute new cloud execution or a new model replay. [Git dependency/closure audit](../docs/task024_git_audit.json).
