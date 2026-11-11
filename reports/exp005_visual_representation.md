# EXP-005 frozen visual representations for ST3

## External resources and checkpoint provenance

Before acquiring weights, we verified the official ST3 Codabench Data page through its public competition metadata on 2026-11-10. The page explicitly permits pretrained image and language models, requires citation of every external resource, and bars outside annotations for evaluation paintings, including those in the original ArtELingo release. The public rule covers the pretrained image-model category. It provides no checkpoint-specific clearance or accounting of artwork overlap in pretraining. This study uses only the pinned organizer train/validation release and its public development image list. [Codabench ST3 Data page](https://www.codabench.org/competitions/18468/) · [Public competition metadata](https://www.codabench.org/api/competitions/18468/?format=json) · [Organizer rules](https://github.com/artemotion-28/artemotion-28#competition-rules-and-terms)

The alternative is Meta's `facebook/dinov2-base` at Hugging Face revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, using the published safetensors and the image processor from the same revision. The checkpoint is labeled Apache 2.0, and the model card describes self-supervised visual pretraining. The authors report a curated 142-million-image corpus and distillation into smaller ViTs. The comparison uses normalized 768-dimensional final CLS features. The pinned slow BitImageProcessor uses RGB input, bicubic shortest-edge resize to 256, a 224 × 224 center crop, 1/255 rescaling and the checkpoint's documented RGB mean and standard deviation. [Pinned model card and files](https://huggingface.co/facebook/dinov2-base/tree/f9e44c814b77203eaa57a6bdbbd535f21ede1415) · [Meta license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) · [DINOv2 paper](https://arxiv.org/html/2304.07193#S3)

## Frozen protocol

Use the exact five EXP-002 folds and the original full interaction LR with `C=0.3`, annotation-count weights, original solver settings and unchanged metrics. Compare the official CLIP-512 head, DINOv2-768, and DINOv2 with full-solver, non-whitened PCA to 512 dimensions. Fit PCA only on unique training paintings for each fold, then apply it to held-out images and L2-normalize the projections. This diagnostic controls head width while also changing the feature geometry.

Require mean Contrast gain of at least +0.005, positive gains in at least four folds, Macro Log Loss within +0.01 bits and convergence on all five folds. Among eligible candidates, choose the highest mean Contrast and prefer the 512-dimensional condition within 0.002 of the best. These are engineering thresholds. Save the decision before any development model comparison.

C-CLIP reuses the official `openai/clip-vit-base-patch32` cache at revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Each fold fits the same annotator-weighted classifier on its original eligible pairs. Fold membership and all 16,019 held-out C-CLIP probabilities match the EXP-002 artifact exactly after the original six-decimal rounding. [EXP-002 baseline and fold protocol](exp002_ablation.md)

## Grouped CV results

Means and sample SDs across the five folds are shown below. All 15 heads converged without a scikit-learn convergence warning. All models share the same 1,496 Contrast-eligible held-out paintings.

| Condition | Head parameters | Cultural Contrast ↑ | Macro Log Loss ↓ | Contrast wins vs CLIP |
| --- | ---: | ---: | ---: | ---: |
| **C-CLIP** | 13,338 | **0.435705 ± 0.024291** | **0.738056 ± 0.005381** | — |
| C-DINO768 | 19,994 | 0.418597 ± 0.023775 | 0.781704 ± 0.010512 | 0/5 |
| C-DINO512 | 13,338 | 0.413333 ± 0.022804 | 0.785623 ± 0.009958 | 0/5 |

Each fold entry gives Cultural Contrast / Macro Log Loss.

| Fold | C-CLIP | C-DINO768 | C-DINO512 |
| ---: | --- | --- | --- |
| 1 | 0.450409 / 0.739601 | 0.427019 / 0.789330 | 0.423778 / 0.796683 |
| 2 | 0.434233 / 0.731610 | 0.418913 / 0.774894 | 0.410602 / 0.777207 |
| 3 | 0.411972 / 0.742971 | 0.397066 / 0.791121 | 0.394380 / 0.790521 |
| 4 | 0.468554 / 0.733164 | 0.453669 / 0.766678 | 0.446711 / 0.773161 |
| 5 | 0.413355 / 0.742934 | 0.396316 / 0.786498 | 0.391194 / 0.790541 |

**NO-GO.** DINO768 changed mean Contrast by **−0.017108** and Macro Log Loss by **+0.043648 bits** versus CLIP. DINO512 changed them by **−0.022372** and **+0.047567 bits**. Both exceeded the **0.748056** absolute loss guardrail and trailed CLIP on both metrics in every fold. The frozen decision retained C-CLIP. The development stage wrote a fallback record without generating new predictions, bootstrap results or a ZIP.

The PCA fits used 1,256 unique training images in fold 1 and 1,257 in folds 2–5. They retained **98.14%** of training variance on average. The dimension control matched CLIP's head parameter count and widened the mean performance gap. Under this protocol, the gap persisted at equal head width. PCA truncation and re-normalization also change geometry, and explained feature variance need not preserve sentiment information.

## Paired reliability

- **Languages.** Both DINO conditions had higher mean Log Loss than CLIP in all **21 macro-eligible CV languages**. DINO768's largest increases were Turkish (+0.066508 bits), Tagalog (+0.065947) and Vietnamese (+0.064815). Yoruba had the smallest increase (+0.019522). The deterioration was distributed across languages.
- **Paintings.** DINO768 improved Contrast on 657 of 1,496 paintings and regressed on 839, with a median paired change of −0.014808. DINO512 improved 665 and regressed on 831, with median −0.018577. Both DINO conditions improved the same 624 paintings. The largest DINO768 gain was +0.536890 on Edward Hopper's *Second Story Sunlight*, and the largest drop was −0.917824 on Claude Monet's *The Promenade Near Limetz*. Removing DINO768's ten largest gains moved its painting-pooled mean change from −0.017135 to −0.020266. These are descriptive painting-level differences.

The painting-level wins identify examples where DINO may add complementary information. The fold and language results provide no aggregate evidence of a benefit from combining representations. A fusion claim would require its own held-out comparison. CLIP and DINOv2 differ in encoder architecture, preprocessing and pretraining data as well as supervision. This comparison supports the tested CLIP checkpoint for this head and protocol. It cannot isolate the effect of image-text supervision.

## Compute and reproducibility

DINOv2 ran in inference mode with all encoder parameters frozen. Each of the **1,656** required images went through the encoder once on an RTX 4060 Laptop GPU at batch size 16. Image loading, preprocessing and inference took **46.4 seconds**. The full cache-build function took **61.4 seconds** from locally cached weights. PyTorch peak GPU memory was **565.4 MiB allocated** and **642 MiB reserved**. No batch reduction or CPU fallback occurred in the full extraction. Both DINO heads used the same cached features. CLIP used the original cache, whose historical extraction timing and GPU peak were unrecorded.

| Condition | Mean CPU LR fit per fold | Mean PCA per fold | LR iterations |
| --- | ---: | ---: | ---: |
| C-CLIP | 8.46 s | — | 66–79 |
| C-DINO768 | 11.80 s | — | 61–71 |
| C-DINO512 | 6.97 s | 1.16 s | 55–62 |

The cached-feature CV run took **182.3 seconds** on CPU at project revision `744af5b8f14b325d5a7c4e312a03b11443ea7113`. DINO extraction used PyTorch 2.7.1+cu126, Transformers 4.49.0, Pillow 11.2.1 and the pinned slow processor. The safetensors SHA-256 is `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`. The processor JSON SHA-256 is `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828`. The cache key binds image paths and bytes to the extraction identity. A manifest verifies the resulting feature file checksum and records all checkpoint asset hashes, processor settings and runtime details.

All **52 tests** passed. An offline DINO512 replay reproduced all **3,212** probabilities in fold 1 exactly. A separate offline CPU smoke produced a normalized 768-vector from a synthetic image with the pinned checkpoint. The artifact checks validated all 48,057 CV predictions and verified unchanged hashes for 45 earlier run files and both CLIP caches.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py visual-embed
.venv/Scripts/python.exe scripts/run_st3.py visual-cv
.venv/Scripts/python.exe scripts/run_st3.py visual-dev
.venv/Scripts/python.exe -m pytest -q
```

Artifacts are in `runs/exp005_visual_representation/`. `cv/` includes fold metrics, paired language and painting deltas, predictions, provenance and the frozen selection. `dev/run.json` records the fallback. The embeddings and weights live under `cache/`.

## Next question

Retain full C with the current CLIP representation. Test whether one predeclared, stronger frozen image-text checkpoint improves Contrast under the same head, folds and loss guardrail. That would probe the more promising representation direction without treating DINO's isolated painting-level wins as evidence for fusion.
