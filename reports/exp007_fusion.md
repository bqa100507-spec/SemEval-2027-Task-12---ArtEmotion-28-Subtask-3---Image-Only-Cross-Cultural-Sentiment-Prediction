# EXP-007 CLIP and SigLIP 2 fusion

Reuse the original CLIP and SigLIP 2 caches, the exact EXP-002 painting folds, and the full interaction LR at `C=0.3`. The competition permits pretrained image models and requires citations. Both checkpoints are already recorded, and all inference uses artwork pixels and the target language. [ST3 Data page](https://www.codabench.org/competitions/18468/) · [CLIP baseline](baseline_dev.md) · [SigLIP 2 provenance](exp006_siglip2.md)

## Fixed comparison and selection

Refit the two reference heads and verify the saved out-of-fold predictions. E25, E50 and E75 use 25%, 50% and 75% CLIP probability, respectively, with SigLIP 2 supplying the remainder. Blend the six-decimal constituent predictions from the same fold and round the result to six decimals. Early fusion fits one original LR head on the concatenation of the two unit vectors divided by `sqrt(2)`. The image cache is built solely from the two frozen features.

Rank alternatives that beat C-CLIP by pooled out-of-fold Contrast. Require all folds and fits to be valid and the mean fold Macro Log Loss to be at most C-CLIP +0.01 bits. Among positive-gain candidates within strictly 0.001 Contrast and at most 0.001 bits of the leader, prefer one backbone, then one head over two. For equal complexity, take higher pooled Contrast. C-CLIP is the fallback when no alternative passes. The EXP-006 decision remains historical, and standalone SigLIP 2 is a single-backbone comparator in this new selection. The decision is saved before any new development scoring.
