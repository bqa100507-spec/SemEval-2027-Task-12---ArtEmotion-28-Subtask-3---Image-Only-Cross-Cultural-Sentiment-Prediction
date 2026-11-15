# ST3 research index and production decision

**Current decision: production E50+L; EXP-022 CLOSED; TASK-023 CLOSED; EL8 NO_GO_FOR_PRODUCTION.** TASK-025 implements EXP-023 for CPU verification only; scientific GPU execution is blocked by unresolved eligibility and advance authorization. No EL8 refit, retuning, upload or promotion is pending. Historical runnable instructions/READY states are archival records, not current authorization. [Final closure](../reports/exp022_final_closure.md) · [aggregate evidence](exp022_closure_evidence.json) · [frozen EXP-023 protocol](../reports/next_exp_proposal.md) · [implementation](../reports/exp023_implementation.md).

Production remains the released default `predict --mode e50l`, with E50/CLIP explicit fallbacks. E50+L's frozen 849-row reference SHA is `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`; owner-reported hosted Contrast is 0.4643 (#1046158). EL8's valid archived submission #1053827 scored 0.4537 according to the owner, so engineering success did not justify replacement.

## Protocols must not be mixed

- **P1:** EXP-002 original folds, 1,571 paintings / 16,019 pairs; used by most EXP-003–015 comparisons.
- **P2:** EXP-017's new grouped protocol, 1,335 paintings / 13,600 pairs plus 236 sealed training-confirmation paintings; used through EXP-022. M0/R0 equals 0.447097707 Contrast. Closure checked exact P2 fold/key/probability identity between EXP-017 and EXP-022.
- **Development:** separate official 849 input pairs / 85 paintings, distinct from P1/P2 OOF populations. Hosted scores below are owner-reported where marked. This closure did not read Development gold or recompute those scores.

Numerical gains on different populations are not directly interchangeable. Stored paired intervals condition on the observed paintings and adaptive reuse; they are not independent confirmations.

## Complete study/task ledger

Reports for EXP-001–015 are retained at [archive `cd321f1`](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/tree/cd321f142142120c17c495458390deb6574ad856/reports). EXP-017–022 and TASK-023 source/config/reports are retained at [delivery `c3337fa`](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/tree/c3337fa8ae8010d76fa598536f5ff3f506ec6215/reports) and its permanent archive descendant. Preparation reports are preserved even when later local manifests established completed NO-GO results.

| ID | Question/change | Relevant evidence | Outcome / exact report basename |
| --- | --- | --- | --- |
| EXP-001 | Official CLIP-B + full interaction LR reproduction | Development Contrast 0.446564; hosted 0.4466 owner-reported | Reproducible baseline; `baseline_dev.md` |
| EXP-002 | Shared vs language-specific visual weights | P1 fold-mean Contrast 0.337542 vs 0.435705; interactions win5/5 | Retain interactions; `exp002_ablation.md` |
| EXP-003 | Low-rank interaction factors / SVD control | Direct best rank4≈0.406309; loss gate fails; SVD16 compression near baseline | NO-GO for direct factor model; `exp003_low_rank.md` |
| EXP-004 | Pairwise Huber / Pearson losses | P1 best mean0.437539; +0.001834 below frozen gate | NO-GO; `exp004_contrast_aware.md` |
| EXP-005 | Frozen DINOv2 vs CLIP | DINO768 mean0.418597, worse loss | NO-GO; `exp005_visual_representation.md` |
| EXP-006 | SigLIP2-Base alone | Mean0.439767; +0.004062 misses +0.005 gate | Single-backbone NO-GO; useful fusion component; `exp006_siglip2.md` |
| EXP-007 | Fixed CLIP-B/SigLIP probability fusion | P1 E50 pooled0.442055; Development0.460026; hosted0.4600 owner-reported | E50 advanced; `exp007_fusion.md` |
| EXP-008 | Full-artwork padding / dual-view CLIP | P1 pooled0.437825 /0.441386 below E50 | NO-GO; `exp008_full_image.md` |
| TASK-009 | E50 production inference | Exact reference parity, resumable encodings | Infrastructure, not a new EXP; `e50_inference_verification.md` |
| EXP-010 | Frozen CLIP-L addition | P1 E50+L pooled0.445643; hosted0.4643 owner-reported | Best released system; `exp010_clip_large.md` |
| TASK-011 | E50+L production integration | All849 original predictions exact | Production default; `e50l_inference_verification.md` |
| EXP-012 | Global/language residual calibration | P1 best correction0.445424 <R0 0.445643 | NO-GO; `exp012_language_residual.md` |
| EXP-013 | Small nonlinear head / fixed blend | P1 blend0.447723; Development0.457525 vs0.464317 | CV qualification did not generalize; retain E50+L; `exp013_nonlinear_head.md` |
| EXP-014 | Small CLIP-L tail LoRA | P1 ensemble gain+0.000891 below+0.003 gate | NO-GO; `exp014_clip_large_lora.md` |
| EXP-015 | Larger frozen SigLIP2 | P1 ensemble gains+0.000565/+0.000473 | NO-GO; `exp015_siglip2_large.md` |
| TASK-016 | Production release / history preservation | PR1–11 integrated; archive1–15 retained | Infrastructure; `docs/competition_release.md` |
| EXP-017 | CLIP/E5 caption supervision with frozen visual features | P2 B2 0.449753673; B2−B1 +0.001167926; misses competition gates | NO-GO; `exp017_text_supervision.md` |
| EXP-018 | Pooled/polarity caption cosine supervision | Actual P2 polarity ensemble0.449566497, +0.002468790 vsR0 | NO-GO; later parent manifests supersede preparation status; `exp018_polarity_captions.md` |
| EXP-019 | Raise text-loss coefficient to1 | Actual P2 ensemble0.446633867; student regresses to0.417569215 | NO-GO; `exp019_gradient_scaling.md` |
| EXP-020 | Emotion-specific vs pooled-polarity targets | Actual P2 EE8 0.449189301 vs EA8 0.449334031 | NO-GO; `exp020_emotion_specific_captions.md` |
| EXP-021 | Empirical vote distribution / reliability audit | Actual P2 ED8 0.449920217; misses competition gate | NO-GO; `exp021_vote_distribution.md` |
| **EXP-022** | Matched frozen/LoRA × BCE/CE8 | P2 EL8 0.452483161, positive conditional CV interval; hosted Development0.4537 | **CLOSED / NO-GO for production**; [final closure](../reports/exp022_final_closure.md) |
| **TASK-023** | Artifact recovery, exact L4 replay and final inference | Real Colab SUCCESS_SUBMISSION_READY; 3 final seeds verified, valid849-row ZIP | **Engineering complete / CLOSED**; no promotion; [closure](../reports/exp022_final_closure.md) |
| TASK-024 | Research closure, archives and next hypothesis | Read-only evidence verification; documentation only | No new model execution |
| **EXP-023 / TASK-025** | Painting-centered caption targets with matched frozen heads | Implemented under the unchanged preregistration; indirect EXP-017 motivation, no new score | **IMPLEMENTED / GPU BLOCKED_ELIGIBILITY**; [implementation](../reports/exp023_implementation.md) |

Never promote a candidate merely because it has a valid ZIP or clears a reused-CV threshold. The future workflow requires frozen controls, verification and conditional independent confirmation before any manual hosted comparison. Its [one-click contract](experiment_workflow_standard.md) includes verified Drive export, optional browser-download attempts and runtime release; it is not retrofitted into the closed EXP-022 notebooks.
