# EXP-007 CLIP and SigLIP 2 fusion

Reuse the original CLIP and SigLIP 2 caches, the exact EXP-002 painting folds, and the full interaction LR at `C=0.3`. The competition permits pretrained image models and requires citations. Both checkpoints are already recorded, and all inference uses artwork pixels and the target language. [ST3 Data page](https://www.codabench.org/competitions/18468/) · [CLIP baseline](baseline_dev.md) · [SigLIP 2 provenance](exp006_siglip2.md)

## Fixed comparison and selection

Refit the two reference heads and verify the saved out-of-fold predictions. E25, E50 and E75 use 25%, 50% and 75% CLIP probability, respectively, with SigLIP 2 supplying the remainder. Blend the six-decimal constituent predictions from the same fold and round the result to six decimals. Early fusion fits one original LR head on the concatenation of the two unit vectors divided by `sqrt(2)`. The image cache is built solely from the two frozen features.

Rank alternatives that beat C-CLIP by pooled out-of-fold Contrast. Require all folds and fits to be valid and the mean fold Macro Log Loss to be at most C-CLIP +0.01 bits. Among positive-gain candidates within strictly 0.001 Contrast and at most 0.001 bits of the leader, prefer one backbone, then one head over two. For equal complexity, take higher pooled Contrast. C-CLIP is the fallback when no alternative passes. The EXP-006 decision remains historical, and standalone SigLIP 2 is a single-backbone comparator in this new selection. The decision is saved before any new development scoring.

## Train-only comparison

Each metric cell is Contrast ↑ / Macro Log Loss ↓. Pooled scores apply the scorer once to all held-out predictions. The last two columns give summed CPU LR fit time across five folds and the requirements for inference.

| Model | Fold mean | Pooled OOF | Δ pooled Contrast | Fold wins | Painting wins / regressions | LR fit | Inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C-CLIP | 0.435705 / 0.738056 | 0.435854 / 0.737288 | — | — | — | 20.62 s | CLIP, one head |
| C-SigLIP2 | 0.439767 / 0.735555 | 0.439910 / 0.734760 | +0.004056 | 4/5 | 779 / 717 | 29.35 s | SigLIP 2, one head |
| E25 | 0.441890 / 0.728459 | 0.442036 / 0.727704 | +0.006182 | 5/5 | 809 / 687 | 0 s extra | Both encoders, two heads |
| **E50** | **0.441908 / 0.726666** | **0.442055 / 0.725927** | **+0.006201** | **5/5** | **838 / 658** | 0 s extra | Both encoders, two heads |
| E75 | 0.439875 / 0.729771 | 0.440024 / 0.729028 | +0.004170 | 5/5 | 865 / 631 | 0 s extra | Both encoders, two heads |
| Early fusion | 0.441811 / **0.724955** | 0.441966 / **0.724211** | +0.006112 | 5/5 | 857 / 639 | 48.72 s | Both encoders, one head |

**E50 won the frozen selection** at the 0.748056 mean loss ceiling. Early fusion was only 0.000089 behind in pooled Contrast but lowered mean loss by 0.001710 more than E50, outside the predefined negligible-loss band. All four fusion variants beat CLIP in every fold and reduced mean loss in all 21 macro-eligible CV languages. E50 also improved pooled Contrast by 0.002145 over SigLIP 2 alone. Its fold Contrast SD was 0.022620 versus CLIP's 0.024291.

The 10,000-draw painting-paired percentile intervals for pooled Contrast change versus C-CLIP were **[−0.000963, +0.009266]** for SigLIP 2, **[+0.002372, +0.010093]** for E25, **[+0.003605, +0.008853]** for E50, **[+0.002835, +0.005515]** for E75, and **[+0.003293, +0.008934]** for early fusion. E50 retained a +0.005015 pooled gain after excluding its ten largest gains. Its largest mean language loss reductions were Burmese (−0.030919 bits) and Malay (−0.017061). These intervals are conditional on the saved predictions on the reused CV paintings.

## Frozen development result

Both component heads were refit on the full original training split. The resulting 50/50 blend scored **0.460026 Contrast** and **0.744214 Macro Log Loss**, versus **0.446564** and **0.751995** for the reproduced full-train CLIP reference. Contrast rose **+0.013462**, with paired 95% percentile interval **[−0.002101, +0.029496]**. Macro Log Loss fell **0.007782 bits**. The development interval includes zero.

E50 improved 49 of 82 scored paintings and regressed on 33. The largest gain was +0.280556 on Peter Paul Rubens's *Hagar in the Desert*, and the largest drop was −0.192700 on Georges Seurat's *Farm Women at Work*. Macro-eligible language loss improved in 12 of 17 languages. Tamil (−0.036737 bits) and Korean (−0.030628) improved most, while Kyrgyz (+0.011401) regressed most.

The separate `runs/exp007_fusion/dev/submission.zip` validates against all 849 development pairs and has SHA-256 `c937fb7fe24a9aae3186ea5073a5b14d5e1da5371c10c6f035523b882df16d76`. The baseline archive remains available.

## Cost and verification

CV ran in **129.2 seconds** on CPU. Building the 1,656 × 1,280 unit-normalized fused feature file from the two caches took 0.063 seconds. The development command took **16.8 seconds**, including full-training fits of 4.18 seconds for CLIP and 6.49 seconds for SigLIP 2. For the 849 rows, cached-feature head prediction took 0.041 and 0.058 seconds, respectively, plus 0.011 seconds for the blend. E50 requires two encoders and 33,332 total head parameters versus 13,338 for CLIP. Early fusion uses one 33,306-parameter head and the same two encoders.

On an RTX 4060 Laptop GPU, a synthetic batch of 16 RGB images took a median **0.092 s** through CLIP and **0.198 s** through SigLIP 2. Adding the separate medians gives about **0.290 s per batch** for sequential two-encoder inference, or **3.16×** the CLIP encoder time. This measures preprocessing, forward pass and normalization after two warmups over five timed batches. Disk decode and model loading are outside the timing. The local harness and timings are in `runs/exp007_checks/bench.py` and `benchmark_*.json`. The measured gains justify a manual development trial if the extra encoder cost is acceptable. The development interval and resource trade-off warrant retaining CLIP as fallback.

All 15 trained CV heads and both full-training heads converged. The combined CLIP and SigLIP 2 CV CSV reproduced the EXP-006 file byte for byte, and the CLIP values matched EXP-002 exactly. All three blends replayed exactly from the verified reference probabilities. The audit validated 96,114 CV rows, confirmed the ZIP matches the E50 CSV, and preserved the hashes of 63 earlier run files and five earlier feature files. The full-train CLIP CSV matched Task 001 byte for byte. All 60 tests passed.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py fusion-cv
.venv/Scripts/python.exe scripts/run_st3.py fusion-dev
.venv/Scripts/python.exe scripts/run_st3.py validate runs/exp007_fusion/dev/submission.zip
```

Use E50 for the next manual development submission and keep CLIP available as the lower-cost fallback. The next practical improvement is a batched two-backbone inference command with recoverable feature caching and coverage checks for newly released paintings. Keep the blend fixed while checking the hosted development result.
