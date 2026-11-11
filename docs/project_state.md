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

The proposed Task 005 is a controlled frozen-representation comparison using Model C's original objective and the same painting-grouped validation protocol.
