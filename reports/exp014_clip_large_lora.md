# EXP-014 Parameter-efficient CLIP-L adaptation

## Frozen protocol

The pinned EXP-010 CLIP ViT-L/14 vision checkpoint, official processor and 768-dimensional normalized projection are reused. LoRA adds rank-4 updates with scale 8/4 and no dropout only to query and value projections in vision blocks 22 and 23, using zero-based indices. Its A matrices use seeded Kaiming initialization and its B matrices start at zero. All original parameters and the fold-specific logistic regression head are frozen during adapter optimization. No text tower is used.

The first 22 frozen blocks produce FP32 patch activations once per image, under the original processor and checkpoint. These are reused across epochs and folds. The cache depends on image bytes, encoder pins and runtime. The two final blocks still encode each distinct image once per microbatch, with all eligible painting-language pairs sharing the resulting vector. BF16 autocast is used for CUDA training in the tail. Normalization, the LR readout, weighted BCE, and the accumulator use FP32. Validation and prediction use FP32. No augmentation or activation checkpointing was needed.

Each microbatch has up to four distinct paintings. Four microbatches make an optimizer step, with each microbatch's vote-summed loss divided by the total eligible votes in that step. The last step uses its actual vote count. AdamW uses learning rate 0.0001, weight decay 0.01, seed 2027 and gradient clipping at 1.0.

Epoch selection uses a deterministic grouped 10% stopping subset inside each of the exact five EXP-002 outer training folds. The temporary LR head fits only the remaining inner training paintings, as does the adapter. The lowest stopping BCE selects the epoch, up to six epochs with patience two. LoRA then resets and refits for that epoch count on the full outer training set against its independently fitted, frozen original CLIP-L LR head.

M0 is the exact original E50+L. M1 is adapted CLIP-L, compared with original standalone CLIP-L as a diagnostic. M2 replaces the CLIP-L component and uses the original 50/50 blend with the rounded E50, with final six-decimal rounding. Only M2 can promote. The frozen rule requires five successful folds, at least +0.003 pooled Contrast, at least four fold wins, and mean Macro Log Loss within M0 +0.005 bits. The 10,000-draw paired bootstrap is conditional on the repeatedly examined folds. [Frozen config](../configs/exp014.json)

```powershell
.venv/Scripts/python.exe scripts/run_st3.py lora-pilot
.venv/Scripts/python.exe scripts/run_st3.py lora-cv
.venv/Scripts/python.exe scripts/run_st3.py lora-dev
```

## Resource and correctness pilot

**FEASIBLE** on the RTX 4060 Laptop GPU with the frozen configuration. The first outer fold was run without Contrast scoring. The zero adapter matched the original cached CLIP-L embeddings on 64 paintings with maximum absolute error 1.79×10⁻⁷, below the 5×10⁻⁶ tolerance. The FP32 PyTorch and sklearn readouts agreed within 5.79×10⁻⁷ in logits on the pilot, below 10⁻⁵.

There are exactly **32,768 trainable parameters**. All four B matrices had gradients on the first step. All eight A and B matrices had finite nonzero gradients on the second. Repeating those seeded steps reproduced losses, gradients and parameters exactly. Frozen tail weight hashes matched, and adapter and LR checkpoint reload reproduced every pilot prediction.

The frozen prefix build took 58.95 s for 1,571 images. The pilot took 52.57 s wall-clock using that cache, including 36.34 s for head fits, gradient checks and adapter fitting. It selected epoch 2 after four stopping epochs. Peak CUDA allocation was about 1.87 GiB, with 1.91 GiB reserved, including the activation bank. Individual pilot epochs took roughly 4–4.4 s. The full CV reused this verified pilot fold.

## Train-only results

Each score cell shows Cultural Contrast ↑ / Macro Log Loss ↓.

| System | Fold mean | Pooled out-of-fold | Contrast fold wins vs M0 | Painting wins / regressions | Pooled ordering agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 · E50+L | 0.445498117 / 0.715718675 | 0.445643116 / 0.714879226 | — | — | 0.700935236 |
| M1 · Adapted CLIP-L | 0.441624541 / 0.721247069 | 0.441753420 / 0.720224865 | 1/5 | 714 / 782 | 0.699123805 |
| M2 · E50 + adapted CLIP-L | **0.446389460 / 0.714205741** | **0.446534185 / 0.713354343** | **4/5** | 812 / 684 | **0.701248898** |
| Frozen standalone CLIP-L · diagnostic | 0.440619874 / 0.721874184 | 0.440750719 / 0.720892449 | — | — | — |

| Fold | M0 | M1 | M2 | Δ M2 Contrast |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.462693 / 0.708627 | 0.460055 / 0.705573 | 0.464028 / 0.706190 | +0.001334 |
| 2 | 0.439750 / 0.716017 | 0.427723 / 0.724546 | 0.439245 / 0.714130 | −0.000505 |
| 3 | 0.418114 / 0.713914 | 0.413564 / 0.721063 | 0.421308 / 0.711270 | +0.003193 |
| 4 | 0.477616 / 0.718075 | 0.475157 / 0.726491 | 0.477893 / 0.717726 | +0.000277 |
| 5 | 0.429317 / 0.721959 | 0.431624 / 0.728562 | 0.429474 / 0.721713 | +0.000157 |

M2 gained **+0.000891** pooled Contrast over M0, with a paired 95% interval of **[+0.000262, +0.001515]**. The M1 change versus M0 was −0.003890 [−0.006828, −0.000929]. Adapted standalone CLIP-L gained +0.001003 Contrast and reduced loss by 0.000668 bits versus frozen standalone CLIP-L.

M2 reduced pooled loss in 20 of 21 eligible languages. Indonesian improved most (−0.004102 bits) and English was the one regression (+0.000321 bits). Its largest painting gain was Shevchenko's *Portrait of Y. F. Rudzinsky* (+0.075179), and its largest drop was Berthe Morisot's *Young Girl Resting* (−0.079015). Pair ordering gives half credit for a predicted tie and excludes tied gold pairs.

**NO-GO.** M2 passed the four-fold consistency and loss checks, but fell short of the required +0.003 pooled Contrast gain. The development command exited before scoring or packaging a candidate.

## Runtime and verification

The selected epoch counts were 2, 3, 3, 1, and 1. Stopping BCE increased from its minimum by 0.00158–0.00286 in every fold, while training BCE continued to fall. The refits used only the selected epochs. The original LR head coefficients stayed fixed in both stages.

After the pilot, the CV command took 167.61 s. Across all five folds the measured fit time was 178.83 s, including the two LR fits per fold and the pilot gradient checks, and adapted prediction took 4.80 s. The 30 adapter epochs used 127.72 s in total. Refit microbatches averaged 13.52 ms, including optimization overhead. Peak CV allocation was 1,915 MiB, with 1,968 MiB reserved.

All 48,057 final out-of-fold predictions passed coverage and probability validation. The 16,019 E50+L rows reproduced EXP-010 byte for byte, all five frozen standalone CLIP-L predictions matched EXP-010 exactly, and each adapted checkpoint replayed its predictions. The independent audit rebuilt the grouped stopping splits, verified the selected epoch against each stopping curve, checked saved adapter and LR head checksums, and replayed M2 from rounded E50 and M1. Full artifacts are under `runs/exp014_clip_large_lora/`.

The 849-row E50+L production regression reproduced its CSV exactly. Its original ZIP still validates, and all 721 historical file hashes matched, including 33 cache and checkpoint files. The Windows suite passed 129 tests and skipped one symlink fixture.

Keep the existing E50+L system for competition. A further adaptation study should demonstrate a material Contrast gain on independent paintings before changing that pipeline.
