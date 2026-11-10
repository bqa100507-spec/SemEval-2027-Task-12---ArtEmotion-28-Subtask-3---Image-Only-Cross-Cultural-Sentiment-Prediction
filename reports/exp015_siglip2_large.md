# EXP-015 Frozen SigLIP 2 Large

## Frozen protocol

The sole new checkpoint is Google's Apache-2.0 `google/siglip2-large-patch16-256` at revision `787800c8990e6f058423089178e718139608408c`. The model config, processor config and full checkpoint SHA-256 digests are pinned. Eligibility follows the organizer's published competition rules and the saved official Codabench Data tab, which permits pretrained image and language models and requires attribution. The live Codabench page did not return the ST3 content during this run. Extraction uses the organizer's artwork pixels and the LR head uses the target language. [Checkpoint card](https://huggingface.co/google/siglip2-large-patch16-256/tree/787800c8990e6f058423089178e718139608408c) · [Organizer rules](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#competition-rules-and-terms) · [Pinned config](../configs/exp015.json)

The slow official image processor takes RGB artwork, resizes directly to 256 × 256 with bilinear interpolation, rescales by 1/255 and normalizes with per-channel mean and standard deviation 0.5. The vision-only model loads 315,956,224 FP32 parameters, including all weights of the trained attention pooling head. It returns 1,024-dimensional L2-normalized FP32 vectors using eager attention and deterministic PyTorch operations. Cache identity includes the image bytes, checkpoint assets, extraction settings and runtime.

Each of the five original EXP-002 folds fits the original annotator-weighted full image–language interaction Logistic Regression with C = 0.3, seed 2027 and the original vocabulary derived from that fold's training rows. There were 25 training languages in each fit. Each painting belongs to one held-out fold. M0 replays E50+L. M1 is the new standalone head, M2 replaces SigLIP 2 Base inside E50+L, and M3 averages CLIP-L and SigLIP 2 Large. Every blend is 50/50 and rounds to six decimals after each stage, including the inner blend of M2.

The gate was committed in `0e63b12` before extraction and scoring. M2 requires a +0.005 pooled Contrast gain and M3 requires +0.002. Both require at least three fold wins, all five fits to converge and fold-mean Macro Log Loss within M0 +0.005 bits. If both qualify, the higher pooled Contrast wins. M1 is diagnostic. [Run commands](../README.md#exp-015-frozen-siglip-2-large)

## Train-only results

Each score cell reports Cultural Contrast ↑ / Macro Log Loss ↓.

| System | Fold mean | Pooled out of fold | Contrast fold wins vs M0 | Painting wins / regressions |
| --- | ---: | ---: | ---: | ---: |
| M0 · Original E50+L | 0.445498117 / 0.715718675 | 0.445643116 / 0.714879226 | — | — |
| M1 · SigLIP 2 Large | 0.442172456 / 0.716609974 | 0.442287069 / 0.715844612 | 1/5 | 721 / 775 |
| M2 · E50+L with SigLIP 2 Large | 0.446070416 / 0.712054066 | **0.446208508** / 0.711211436 | 3/5 | 769 / 727 |
| M3 · CLIP-L + SigLIP 2 Large | 0.445987347 / **0.710637063** | 0.446115879 / **0.709785115** | 2/5 | 762 / 734 |

Each delta cell gives the change in Contrast / Macro Log Loss relative to M0 in that fold.

| Fold | M0 scores | M1 Δ | M2 Δ | M3 Δ |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.462693 / 0.708627 | −0.008531 / +0.009995 | −0.000260 / −0.001636 | −0.001005 / −0.002495 |
| 2 | 0.439750 / 0.716017 | −0.000386 / −0.002252 | +0.000430 / −0.007032 | −0.000263 / −0.007753 |
| 3 | 0.418114 / 0.713914 | +0.002200 / +0.002514 | −0.000066 / −0.000559 | +0.000676 / −0.002813 |
| 4 | 0.477616 / 0.718075 | −0.008115 / +0.004611 | +0.000568 / −0.002939 | −0.001087 / −0.002232 |
| 5 | 0.429317 / 0.721959 | −0.001796 / −0.010411 | +0.002190 / −0.006157 | +0.004125 / −0.010115 |

| System | Pooled Contrast Δ | Paired 95% interval |
| --- | ---: | ---: |
| M1 | −0.003356047 | [−0.007547370, +0.000974747] |
| M2 | +0.000565392 | [−0.000664347, +0.001776748] |
| M3 | +0.000472763 | [−0.001834254, +0.002800081] |

The 10,000 painting-paired draws use the 1,496 Contrast-scored paintings. M2's largest gain was Hassam's `italian-day-may` (+0.153179) and its largest regression was El Greco's `not-identified` (−0.172708). For M1 and M3, Gris's `figs` gained +0.599450 and +0.246116, respectively, and Ghirlandaio's `giovanna-tornabuoni-and-her-accompaniment` fell −0.499008 and −0.297526. These intervals describe the repeatedly examined CV paintings. An independent set would be needed to confirm a gain.

**NO-GO.** All fits converged and both ensembles passed the loss guardrail. M2 fell short of the required Contrast gain despite winning three folds. M3 fell short of the Contrast and fold-win requirements. The development command retained M0 before candidate scoring or packaging.

### Pooled per-language Log Loss change vs M0

The table shows all 21 languages included in the pooled macro average. Negative differences indicate lower loss. The run retains all language and painting records.

| Language | Pairs | M1 Δ bits | M2 Δ bits | M3 Δ bits |
| --- | ---: | ---: | ---: | ---: |
| Arabic | 535 | −0.007134 | −0.004748 | −0.006487 |
| Burmese | 198 | −0.010419 | −0.003497 | −0.011600 |
| Chinese | 1,113 | −0.001262 | −0.003364 | −0.003998 |
| English | 648 | +0.007153 | −0.001288 | +0.000121 |
| Hausa | 930 | +0.002712 | −0.002557 | −0.002161 |
| Hindi | 908 | +0.006864 | −0.001130 | −0.000090 |
| Igbo | 1,269 | −0.000826 | −0.004173 | −0.004932 |
| Indonesian | 296 | +0.000728 | −0.005952 | −0.007217 |
| IsiZulu | 756 | +0.013899 | −0.000010 | +0.001575 |
| Korean | 1,053 | +0.007330 | −0.004372 | −0.003114 |
| Kyrgyz | 642 | +0.008050 | −0.003665 | −0.004863 |
| Malay | 720 | +0.000252 | −0.003620 | −0.007993 |
| Setswana | 333 | +0.003738 | −0.001803 | +0.000429 |
| Swahili | 993 | −0.011486 | −0.005137 | −0.010102 |
| Tagalog | 658 | +0.004162 | −0.002986 | −0.003948 |
| Tamil | 662 | −0.003362 | −0.005924 | −0.009337 |
| Thai | 240 | +0.000880 | −0.008290 | −0.014859 |
| Turkish | 572 | −0.001955 | −0.002259 | −0.002956 |
| Uzbek | 1,044 | +0.002612 | −0.002971 | −0.003551 |
| Vietnamese | 1,228 | −0.006020 | −0.007764 | −0.009989 |
| Yoruba | 1,172 | +0.004358 | −0.001514 | −0.001904 |

## Resource feasibility and inference cost

**FEASIBLE** on the RTX 4060 Laptop GPU. Before the full checkpoint download, disk had 302 GiB free, the host had 15.71 GiB physical RAM with 3.88 GiB free, and CUDA had 5,652 of 8,188 MiB free. The verified full checkpoint was 3,526,204,360 bytes, of which 1,263,824,896 bytes were FP32 vision tensors. The loader instantiated only the vision tower in the Python 3.12.10, PyTorch 2.7.1+cu126 and Transformers 4.51.3 extraction environment.

The four-image probe produced bit-identical repeat outputs with shape 4 × 1,024. It took 17.13 s end to end after download and peaked at 1,298 MiB allocated and 1,304 MiB reserved by PyTorch. Training extraction took 86.65 s including model setup, with 76.69 s in the extraction loop for 1,571 images at batch 16. It peaked at 1,474 MiB allocated and 1,508 MiB reserved. No batch fallback was needed. The training NPZ is 7,283,668 bytes and its manifest is 2,050 bytes.

CPU CV took 58.77 s, including 38.51 s for LR fits and 1.81 s for the new head's held-out predictions. The five fits took 80, 62, 64, 66 and 77 iterations with no convergence warnings.

The inference benchmark used six timed FP32 passes after three warmups on the same seeded synthetic batch of 16 images. Timings include preprocessing and feature output transfer. Ensemble times are sums of component medians, and ensemble peak memory assumes sequential encoder loading. Cold starts and LR scoring are outside this timing.

| Encoder | Seconds per 16 images | Peak allocated / reserved MiB |
| --- | ---: | ---: |
| CLIP-B | 0.09893 | 656 / 710 |
| SigLIP 2 Base | 0.19994 | 507 / 668 |
| CLIP-L | 0.69650 | 1,458 / 1,562 |
| SigLIP 2 Large | 0.70958 | 1,474 / 1,508 |

Estimated sequential encoder time per 16 images was 0.995 s for M0, 0.710 s for M1, 1.505 s for M2 and 1.406 s for M3. The two-encoder M3 is about 7% faster than M2 and 41% slower than M0 in this benchmark. The measured per-encoder peaks imply about 1,562 MiB of peak PyTorch reserved memory for sequential M0, M2 or M3.

## Verification

The independent replay checked all 64,076 OOF rows for coverage, finite probabilities and painting-group integrity and reconstructed both candidate blends with their prescribed rounding. M0's 16,019 reference rows matched EXP-010 byte for byte. The production inference regression used the 849 label-free development inputs and reproduced the original E50+L predictions byte for byte. Its original ZIP still validates. All 773 historical files matched their pre-run SHA-256 digests, including 33 cache and checkpoint files. The Windows suite passed 135 tests and skipped the existing symlink fixture. Both Python environments passed their dependency checks.

Artifacts are under `runs/exp015_siglip2_large/` with the preflight, extraction manifest, CV results and verification logs. Keep the hosted-verified E50+L for competition. This comparison gives no reason to extend the backbone search on the same five folds. Revisit encoder changes if independent paintings or new evidence support a material Contrast gain at an acceptable inference cost.
