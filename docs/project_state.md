# SemEval-2027 ST3 project state

Updated 26 November 2026. **The active incumbent is the EXP-036 ensemble, 75% Corrected Ridge and 25% epoch-5 CS-CVA.** Corrected Ridge is the primary fallback. E50+L remains the historical released fallback, with E50 and CLIP available explicitly. [Experiment registry](experiment_registry.md) · [Evaluation inference](exp036_evaluation.md) · [E50+L release](competition_release.md)

## Hosted Development

| System | Hosted Contrast | Submission | Repository status |
| --- | ---: | ---: | --- |
| **EXP-036 ensemble** | **0.4703** | **#1065924** | Active incumbent, local Development replay 0.470339 / 0.729709 |
| Corrected Ridge | 0.4655 | #1063064 | Primary fallback, local Development replay 0.465462 / 0.740305 |
| E50+L | 0.4643 | #1046158 | Released production fallback |
| EL8 | 0.4537 | #1053827 | Archived, production NO-GO |

The hosted values and submission IDs come from owner reports. The EXP-036 closure independently rescored the archived 849 Development predictions against the official Development counts. Development selected both the CS-CVA checkpoint and the ensemble coefficient. Earlier Fixed Ridge and Corrected Ridge archive validation also verified their 849 keys and byte-identical saved replay.

| Local Development archive | SHA-256 |
| --- | --- |
| `runs/exp036/closure/exp036_ensemble_submission.zip` | `bd6752c9a8a60f7c3754053afe11eec0187172838e9c67d43880b9c41fb5b274` |
| `runs/exp036/closure/exp036_standalone_submission.zip` | `d4feaa9dc09422eb78bf530b3cf9d36a6c76bcc2fc8e5cc50f57be5acdb25df1` |
| `runs/codabench_dev_release/corrected_ridge/submission.zip` | `39db05b31bb37a82c327a55c0d981af73db957e7133377308200218dfc67f958` |
| `runs/codabench_dev_release/fixed_ridge/submission.zip` | `d77f1bf6881afef898a73ea2a6d41c218dfccbd735295d8017509e3e09069b24` |

Fixed Ridge's full-training heads are in `runs/competition_track1/january_bundle/`. Corrected Ridge uses those heads and the cross-fitted residual bank in `runs/codabench_dev_release/corrected_bundle/`. The [registry](experiment_registry.md#competition-candidates-and-controls) records source and result fingerprints. The official January evaluation inputs will require a fresh prediction run and pair-coverage validation. The published ST3 submission window is January 10–16, 2027 UTC. [Organizer schedule](https://github.com/artemotion-28/artemotion-28/blob/516d7641c3c1d099e63aa8d066971c6b07c89384/README.md#important-dates)

## Exploratory P2 reference scores

| Model | Full Contrast / Macro Log Loss | Test-like Contrast / Macro Log Loss |
| --- | ---: | ---: |
| E50+L | 0.447098 / 0.710554 | 0.455486 / 0.719860 |
| Fixed Ridge | 0.451702 / 0.709247 | 0.460043 / 0.718425 |
| Corrected Ridge | **0.454134 / 0.708500** | **0.463108 / 0.717479** |

P2 uses five grouped folds over 1,335 official training paintings and 13,600 pairs. Test-like P2 excludes English, Arabic and Chinese, then recomputes eligibility on 11,643 pairs. Both populations reuse research paintings, and their results are exploratory. The correction gained **+0.003065** test-like Contrast over Fixed Ridge, with a painting-paired 95% interval of **[+0.000530, +0.005567]** and four fold wins. The gain shrank to **+0.001799** after trimming the largest 5% of absolute painting differences. Log Loss is in bits.

## Evidence guiding the next comparison

- **Cultural interaction is useful.** Refitting matched heads with image–language interactions instead of additive effects gained **+0.109123 full / +0.116100 test-like** P2 Contrast. The earlier **+0.138631** full-P2 term-deletion result measured reliance of an already-fitted model. Sparse votes matter: 90.3% of P2 pairs have five votes, the estimated sampling share of observed cross-language variance is **58.2%** full / **56.7%** test-like under the stated binomial assumptions, and conditional split-panel Pearson is **0.232** full / **0.242** test-like. These are reliability diagnostics. [Diagnostics and validation audits](experiment_registry.md#diagnostics-and-strategy)
- **Qwen reached the adapters, with limited score gains.** EXP-027 verified a nonzero gradient from stabilized within-painting Pearson into Qwen2.5-VL LoRA. Both treatments trailed BCE. The later Qwen3-VL semantic run cached all 1,571 training images and gained **+0.002687** test-like Contrast in joint fusion, below its **+0.005** gate. Direct Qwen3-VL QLoRA improved on frozen Qwen, with **0.344625** versus Corrected Ridge's **0.441599** on test-like fold 3. [Qwen experiments and objective audit](experiment_registry.md#post-release-research)
- **Additional representations have yet to displace the incumbent.** DINOv2-L global features and language-conditioned patch attention failed their comparisons. EmotionCLIP's fixed blend gained **+0.000637** test-like, with an interval spanning zero and a negative result after removing the top painting wins. Cultural concepts improved their own matched-language control while their blend regressed by **0.006065**. EXP-033 spatial adaptation remains closed. [Competition comparisons](experiment_registry.md#competition-candidates-and-controls)

## Completed experiments

- **EXP-037 — complete and artifact-verified, experimental.** Original C6 blend: **0.472818 / 0.728088** full and **0.454041 / 0.750275** Evaluation-like. Separately selected post-hoc 65/15/20: **0.473143 / 0.729246** full and **0.453667 / 0.751549** Evaluation-like. Both are local Development results with unresolved Qwen eligibility. The incumbent stays EXP-036. [Closure](../reports/exp037_final.md) · [Deployment](exp037_evaluation.md)

- **EXP-036 — closed as the selected competition candidate.** Kaggle completed six epochs. Epoch-5 standalone scored **0.437632 / 0.742911**. The frozen 75/25 blend gained **+0.004876 Contrast** and reduced Macro Log Loss by **0.010597 bits** versus Corrected Ridge on Development. The deployment bundle is in `runs/exp036/deployment/`. Fresh CS-CVA GPU inference passed on two new synthetic artworks. Hidden Evaluation remains unmeasured. [Closure report](../reports/exp036_final.md) · [Smoke test](exp037_evaluation.md#closure-replay)

- **EXP-035 — Kaggle fold-3 pilot complete, scientific NO-GO for five-fold expansion.** Both four-epoch arms selected epoch 1. Test-like Contrast / Macro Log Loss was **0.441386 / 0.721157** for frozen Qwen residuals and **0.441396 / 0.721156** for merger LoRA, versus **0.441599 / 0.721110** for Corrected Ridge. Both corrections were small, and the paired Contrast intervals span zero. [Verified final report](../reports/exp035_cultural_residual.md#final-kaggle-outcome)

## Competition-first evaluation strategy

Default to a single resumable Kaggle run on the full authorized official training split. Use official Development for checkpoint selection and bounded model or blend selection. Prioritize Contrast, track Macro Log Loss, and preserve the risks of Development reuse in the record. Development gold remains scorer-only and hidden Evaluation annotations remain out of scope. Every completed fit produces its validated standalone ZIP and applicable ensemble ZIPs. Five-fold and nested CV are opt-in. The owner retains upload control. [Workflow and upload-safe packaging standard](experiment_workflow_standard.md)

## Active, blocked and proposed work

- **EXP-038 — design recommendation pending owner review.** A VLM-free spatial branch with a shared cultural interaction basis is proposed. The evidence points to errors in image-specific ordering and highly correlated centered residuals. Implementation and training remain outside the closure. [Cultural diagnostic](../reports/exp037_cultural_diagnostic.md) · [Design recommendation](../reports/exp038_design_recommendation.md)

- **DINOv3 — access pending.** The official ViT-B/16 gate returned authenticated HTTP 403 for the pinned weights and config. Hugging Face approval is pending in the project record. A valid checkpoint preflight, feature extraction and outer evaluation are all unmeasured. Preserve the existing runner and its frozen scope. [Track 2 access receipt and runner](experiment_registry.md#competition-candidates-and-controls)
- **EXP-034 — paused, proposal only.** No implementation or run is recorded. The latest possible CPU-only direction is a whole-image color, tone and coarse-composition branch. It is an untested idea. More Qwen Pearson fine-tuning was declined in the latest audit. [Diagnostics and proposals](experiment_registry.md#diagnostics-and-strategy)

## Maintenance and history

Every completed competition experiment updates the [registry](experiment_registry.md) in its results commit or PR. Update this page in the same commit or PR whenever the best model, active work, blockers or strategic decision changes. Record the run population, outcome, source revision, artifact and evidence status. Distinguish verified local files, owner-reported hosted results and diagnostic scores. Keep evaluation annotations sealed and retain previous states in Git.

The [pre-sync local snapshot](project_state_history.md) preserves the earlier text verbatim, including superseded readiness instructions. The [reconciliation notes](experiment_registry.md#reconciliation-and-evidence-gaps) identify newer snapshots on other branches and the source records that remain incomplete.
