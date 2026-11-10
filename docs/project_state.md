# Project state

| Component | Status |
| --- | --- |
| Public dataset | Pinned ST3 checkout and local integrity audit complete |
| Baseline | Full CUDA development run complete. Cached rerun byte-identical. CPU embedding smoke passed |
| Local evaluation | 0.446564 Contrast and 0.751995 macro Log Loss for CLIP + LR |
| Submission | `runs/baseline_dev/submission.zip` validated against all 849 public development keys |
| Tests | 26 lightweight tests passed locally. GitHub Actions workflow included |

Run `py scripts/setup.py` on Windows, then use the virtual environment for `scripts/run_st3.py baseline`, `scripts/run_st3.py submission`, `scripts/run_st3.py validate runs/baseline_dev/submission.zip`, and `python -m pytest -q`. See the [README](../README.md) for executable paths and standalone audit and evaluation commands.

The experiment uses the pinned public development split and a sparse representation of the official features. See the [data audit](../reports/data_audit.md) for coverage and [baseline report](../reports/baseline_dev.md) for measured scores and reproducibility checks.

The project owner reported 0.4466 Contrast for Codabench development submission `#1044742`, matching the local Task 001 score at four decimal places.

EXP-002 completed the shared-weights ablation, 20 inference-time image shuffles, a 10,000-draw paired painting bootstrap, and five train-only grouped folds. A and C reproduced Task 001. Shared weights scored 0.370386 Contrast and 0.766309 Macro Log Loss. Interactions gained +0.076178 Contrast with a 95% paired percentile interval of [+0.033904, +0.118978]. The study ran from cached embeddings on CPU. [EXP-002 report and commands](../reports/exp002_ablation.md)

EXP-003 completed ranks 2, 4, 8 and 16 under two seeds on the exact five EXP-002 folds, plus matched-rank SVD compression. Every directly trained rank exceeded the 0.748056 Macro Log Loss guardrail, so C remains the baseline and its development results are unchanged. The full interaction effective rank was 20.6235. SVD-16 came closest to C in grouped CV. [EXP-003 study and artifacts](../reports/exp003_low_rank.md)

EXP-004 tested painting-balanced pairwise Huber and smoothed Pearson losses on top of full Model C using the exact EXP-002 folds. R-small led at 0.437539 Contrast and 0.737824 Macro Log Loss. Its +0.001834 mean Contrast gain fell below the frozen +0.005 threshold. The study retained C and its established development scores. [EXP-004 study](../reports/exp004_contrast_aware.md)

EXP-005 compared CLIP with frozen DINOv2-base CLS features and fold-local PCA to 512 dimensions. CLIP reproduced EXP-002 and beat both DINO conditions on Contrast and Macro Log Loss in every fold. DINO768 scored 0.418597 Contrast and 0.781704 Log Loss, versus CLIP's 0.435705 and 0.738056. The frozen decision retained C-CLIP. [EXP-005 rules, provenance and results](../reports/exp005_visual_representation.md)

EXP-006 completed the final predefined single-backbone comparison. Frozen SigLIP 2 scored 0.439767 Contrast and 0.735555 Macro Log Loss, winning four of five folds. Its +0.004062 mean Contrast gain fell short of the frozen +0.005 gate. The development fallback retained C-CLIP. [EXP-006 report](../reports/exp006_siglip2.md)

The project now prioritizes reliable competition Contrast, with concise paper records. [EXP-001–006 synthesis](../reports/research_synthesis_001_006.md)

EXP-007 selected the fixed 50/50 CLIP and SigLIP 2 blend E50. It scored 0.442055 pooled CV Contrast and 0.460026 development Contrast, with development Macro Log Loss 0.744214. Its separate development ZIP is validated at `runs/exp007_fusion/dev/submission.zip` for manual upload. C-CLIP remains available as the lower-cost fallback. [EXP-007 comparison](../reports/exp007_fusion.md)

EXP-008 audited crop coverage without labels and compared two frozen CLIP views while reusing the original E50 SigLIP 2 probabilities. Full-image and Dual pooled CV Contrast were 0.437825 and 0.441386, respectively. Neither cleared the promotion gate, so E50 remains the competition system. The project owner reported hosted Development Contrast 0.4600 for submission `#1045921`. [EXP-008 audit and comparison](../reports/exp008_full_image.md)
