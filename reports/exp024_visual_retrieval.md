# EXP-024: culture-conditioned visual retrieval

Standalone CPU method; E50+L is a fixed comparison only. Reuse original P2 1,335 paintings/13,600 training pairs, five grouped folds, and existing pinned CLIP-L/SigLIP2-Base caches. No captions, confirmation annotations, Development labels, new encoder or production changes. Existing pretrained-resource eligibility caveats remain unchanged.

**Frozen comparison before outer scoring.** Concatenate separately unit-normalized CLIP-L and SigLIP2 vectors, divide by sqrt(2), and retrieve by cosine. Prior-only predicts a vote-weighted language prior shrunk toward the global training prior with20 pseudo-votes. Uncentered kNN pools neighbor positive/negative votes for the requested language. Centered retrieval separately estimates image-level sentiment and retrieves each neighbor's language deviation from its vote-weighted painting mean. Neighbor weights are exp((cosine-1)/temperature); language residuals shrink toward language-prior minus global-prior, and image mean toward global-prior. Missing languages fall back to priors; unknown-language centered prediction uses the visual image mean. Probabilities clip to[0,1] and round6 before scoring. IDs/paths only join vectors/check isolation; no text/metadata features.

For each outer fold, use the original deterministic10% inner painting split (seed17029+fold). Independently choose each retrieval control's parameters from k={16,64}, temperature={0.05,0.1}, shrinkage pseudo-votes={5,20}: eight combinations. Rank inner Contrast, then Macro Log Loss, then grid order. Rebuild the bank on the complete outer training split and score the held-out fold once. No outer-score selection, weight search, E50 blend or promotion.

Use the existing published-specification ST3 scorer (painting Pearson with its eligibility/constant rules; base2 clipped Macro Log Loss). Its exact hosted implementation remains unverified, as documented in docs/competition.md. E50+L uses the exact same P2 keys/folds and frozen R0 probabilities. Local CV uses repeatedly inspected training paintings and is exploratory, not independent confirmation.

Run: `.venv/Scripts/python.exe scripts/run_exp024.py --inputs runs/exp023_budget_delivery/final/restore_check --run-dir runs/exp024_visual_retrieval`. New runs refuse to overwrite existing artifacts. OOF predictions, all inner candidates, selected parameters, partitions, per-fold/per-language/per-painting scores and source/input hashes stay under the ignored run directory.

## Measured P2 CV results

| Method | Pooled Contrast | Pooled Macro LL (bits) | Fold-mean Contrast | Fold-mean Macro LL |
| --- | ---: | ---: | ---: | ---: |
| Global + language prior | 0.336155 | 0.884944 | 0.336028 | 0.885261 |
| Uncentered visual kNN | 0.423876 | 0.799796 | 0.423780 | 0.799864 |
| Centered cultural retrieval | 0.426703 | 0.778935 | 0.426627 | 0.779170 |
| E50+L reference | 0.447098 | 0.710554 | 0.446971 | 0.711229 |

Per-fold cells are **Contrast / Macro Log Loss in bits**.

| Fold | Prior | Uncentered kNN | Centered retrieval | E50+L |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.317167 / 0.850268 | 0.406723 / 0.768408 | 0.409501 / 0.744965 | 0.433429 / 0.666222 |
| 2 | 0.351012 / 0.886367 | 0.426624 / 0.812802 | 0.422762 / 0.800709 | 0.459194 / 0.724814 |
| 3 | 0.332873 / 0.885344 | 0.403124 / 0.799051 | 0.416218 / 0.776062 | 0.433536 / 0.713192 |
| 4 | 0.345175 / 0.907165 | 0.445467 / 0.811468 | 0.446955 / 0.787511 | 0.461302 / 0.739182 |
| 5 | 0.333914 / 0.897161 | 0.436960 / 0.807592 | 0.437697 / 0.786600 | 0.447396 / 0.712734 |

All five inner searches independently selected k64 /temperature0.1 /shrinkage5 for both retrieval variants. The grid was not enlarged after scoring. Centering beats uncentered kNN on4/5 folds and improves pooled Contrast by0.002826 and loss by0.020861 bits. Nevertheless, it loses to E50+L on **all5 folds**, with pooled Contrast lower by0.020395 and loss higher by0.068381 bits. Across1269 scored paintings it beats E50+L on526 and regresses on743.

**Recommendation: discard this configuration as a competition candidate; retain the cheap implementation as a control.** Retrieval clearly adds visual signal beyond language priors (+0.090548 Contrast), and centering modestly helps transfer cultural differences. It still does not recover the cross-cultural pattern or probability quality of the learned reference. A plausible explanation is noisy/oversmoothed local cultural response averages under general-purpose visual similarity; this is interpretation, not a demonstrated causal mechanism. No additional grid expansion, blend, Development run or promotion was attempted. Reused P2 results remain exploratory.

CPU execution completed in **12.94s**, four numerical-library threads, no GPU. Implementation source commit: `c83607290f89507b8fb015ca8020b5b2eacf8b73`; actual checkout source/input hashes and Python/package versions are in `runs/exp024_visual_retrieval/run.json`. Predictions, selections, all80 inner evaluations and split identities are preserved there. Fresh-bank replay exactly reproduced all15 retrieval-fold prediction sets; all4 systems were independently rescored. The P2 E50+L reference exactly matches0.44709770744257127 Contrast.

Validation: **247 CPU tests passed, 0 failed, 1 Windows symlink fixture skipped**; includes six essential retrieval tests. All499 protected historical/owner files and E50+L production CSV remain unchanged (SHA256 `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`). EXP-023 was not rerun/repaired/recovered. No captions, sealed confirmation annotations or Development labels were read by EXP-024.
