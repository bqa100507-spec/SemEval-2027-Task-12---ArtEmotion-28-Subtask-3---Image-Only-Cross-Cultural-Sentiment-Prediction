# EXP-010 Frozen CLIP-L and competition ensembles

## Eligibility and protocol

The organizers' public Codabench Data page allows pretrained image and language models, prohibits outside annotations for evaluation paintings, and requires attribution of external resources. EXP-010 uses only artwork pixels and target language for inference and only original training votes for fitting. The live Codabench endpoint returned unrelated content during this run, so the rule was checked against the previously captured official Data page and the pinned organizer rules. [Codabench ST3](https://www.codabench.org/competitions/18468/) · [Organizer rules](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#competition-rules-and-terms)

OpenAI publishes ViT-L/14 through its MIT-licensed CLIP project. The pinned Hugging Face conversion has no separate license field, and the upstream model card describes research as an intended use. This study uses `openai/clip-vit-large-patch14` at revision `32bd64288804d66eefd0ccbe215aa642df71cc41`. The three artifact SHA-256 pins are in the experiment config. The official image processor resizes the short edge and center crops to 224 pixels. The frozen vision tower and trained projection produce 768-dimensional unit vectors in FP32. [Experiment config](../configs/exp010.json) · [OpenAI license](https://github.com/openai/CLIP/blob/main/LICENSE) · [OpenAI checkpoint registry](https://github.com/openai/CLIP/blob/main/clip/clip.py) · [OpenAI model card](https://github.com/openai/CLIP/blob/main/model-card.md) · [Pinned Hugging Face snapshot](https://huggingface.co/openai/clip-vit-large-patch14/tree/32bd64288804d66eefd0ccbe215aa642df71cc41)

Each CLIP-L head uses the original annotator-weighted full-interaction Logistic Regression with C=0.3, seed 2027 and the exact EXP-002 painting-grouped folds. The existing CLIP-B and SigLIP 2 fold probabilities come from their already-fitted heads on the same folds. They are cross-checked against EXP-002 and EXP-006 and replayed against the EXP-007 E50 probabilities. Blend six-decimal component predictions at 50/50 and round to six decimals. E50+L uses the rounded original E50 as its component.

Commit `99be543` froze selection before extraction and scoring. Require five valid, converged folds and fold-mean Macro Log Loss within +0.01 bits of E50. Rank by pooled Contrast and retain E50 when the leader is within 0.001 Contrast, inclusive. Paired intervals use 10,000 painting draws with seed 102027. They are conditional on the fixed predictions from repeatedly examined folds.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py clip-large-cv
.venv/Scripts/python.exe scripts/run_st3.py clip-large-dev
```

## Train-only results

Each score cell shows Cultural Contrast ↑ / Macro Log Loss ↓. Pooled scores use all 16,019 out-of-fold pairs.

| Fold | E50 | CLIP-L | L+SigLIP | E50+L |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.457082 / 0.725124 | 0.458390 / 0.708405 | 0.463222 / 0.708455 | 0.462693 / 0.708627 |
| 2 | 0.441425 / 0.727116 | 0.429487 / 0.723061 | 0.439233 / 0.722668 | 0.439750 / 0.716017 |
| 3 | 0.419136 / 0.722669 | 0.409095 / 0.722622 | 0.418994 / 0.712020 | 0.418114 / 0.713914 |
| 4 | 0.471097 / 0.726467 | 0.474811 / 0.726684 | 0.475690 / 0.721093 | 0.477616 / 0.718075 |
| 5 | 0.420799 / 0.731953 | 0.431317 / 0.728599 | 0.429622 / 0.723590 | 0.429317 / 0.721959 |
| Fold mean | 0.441908 / 0.726666 | 0.440620 / 0.721874 | 0.445352 / 0.717565 | **0.445498 / 0.715719** |
| Pooled OOF | 0.442055 / 0.725927 | 0.440751 / 0.720892 | 0.445497 / 0.716710 | **0.445643 / 0.714879** |

| Candidate | Δ pooled Contrast | Paired 95% interval | Fold wins | Painting wins / regressions |
| --- | ---: | ---: | ---: | ---: |
| CLIP-L | −0.001304 | [−0.006116, +0.003464] | 3/5 | 740 / 756 |
| L+SigLIP | +0.003442 | [+0.000745, +0.006101] | 3/5 | 794 / 702 |
| E50+L | +0.003588 | [+0.001223, +0.005938] | 3/5 | 796 / 700 |

All five CLIP-L fits converged. E50 replayed exactly and all systems passed the loss guardrail of 0.736666. **E50+L won the frozen rule.** It was only 0.000146 Contrast ahead of the two-encoder L+SigLIP and 0.003588 ahead of E50. E50+L lowered pooled loss by 0.011048 bits versus E50.

Pooled loss improved in 14 of 21 eligible languages for CLIP-L and all 21 for both ensembles. E50+L's largest reductions were Thai (−0.030116 bits) and Kyrgyz (−0.018780 bits). Its largest painting gain was on Van Gogh's *A Sower and a Man with a Spade* (+0.267264), and its largest drop was on Fred Sandback's *Untitled* (−0.291409). Fold and painting records, per-language loss changes, pooled metrics, and bootstrap samples are in `runs/exp010_clip_large/cv/`.

## Compute

CLIP-L extracted 1,571 training embeddings on an RTX 4060 Laptop GPU at batch size 16. Extraction took 99.39 s, and the complete cache build took 106.88 s. Peak CUDA allocation was 1,458 MiB with 1,562 MiB reserved. The real run needed no fallback. An injected CUDA out-of-memory test retried the same image offset at half the batch size, stayed on CUDA, and cached all four fixture images. The CV command took 151.02 s overall. The five CPU CLIP-L fits took 24.92 s, shared by all three alternatives.

A warmed synthetic benchmark used six timed 16-image batches after three warmups per encoder. It measures preprocessing, transfer, forward pass and normalized output readback. Disk decode and model loading are outside the timed section.

| Encoder | Median per 16 images | Peak CUDA allocated | Peak CUDA reserved |
| --- | ---: | ---: | ---: |
| CLIP-B | 0.108 s | 656 MiB | 710 MiB |
| SigLIP 2 | 0.267 s | 507 MiB | 668 MiB |
| CLIP-L | 0.936 s | 1,458 MiB | 1,562 MiB |

Summing separate medians gives about 0.376 s per batch for E50, 0.936 s for CLIP-L, 1.204 s for L+SigLIP, and 1.312 s for E50+L. Sequential E50+L needs three encoder passes and three heads and is about 3.5× E50's measured encoder time. Sequential processing can unload each encoder before loading the next.

The first single-stream weight download stalled. Ranged requests retrieved the pinned 1.71 GB artifact, which passed a full SHA-256 check before model loading.

## Frozen development result

The selected E50+L blend refit all three heads on the original training split, and the two original components replayed the saved E50 predictions exactly. All three fits converged. The separately cached 85 development CLIP-L images took 5.84 s to extract and 12.07 s to cache including startup. The CLIP-L, CLIP-B and SigLIP 2 LR fits took 5.37, 4.18 and 6.22 s respectively on CPU. The complete development command took 36.28 s.

| System | Cultural Contrast ↑ | Macro Log Loss ↓ |
| --- | ---: | ---: |
| E50 | 0.4600261535860795 | 0.7442135227739901 |
| E50+L | **0.46431737014951985** | **0.7409970449028339** |

The Contrast change was +0.004291 with a 10,000-draw painting-paired 95% interval of [−0.009183, +0.018040]. Macro Log Loss fell by 0.003216 bits. E50+L improved 41 of 82 scored paintings and regressed on 41. The largest gain was Rossetti's *The Sermon on the Mount* (+0.181675), and the largest drop was Kirchner's *Girls with Good Luck Charms* (−0.142930). Loss decreased in 10 of 17 eligible languages. English (−0.020019 bits) and Kyrgyz (−0.017944) improved most, while Malay (+0.013368) regressed most.

The development archive `runs/exp010_clip_large/dev/submission.zip` contains the exact candidate CSV and validates against all 849 input pairs. Its SHA-256 is `38c2eca8ee4c0557bf198bf5df5dbf90571a2851404ded6302f40e42dc138e0d`. No archive was uploaded.

## Verification and next step

Verified all 64,076 OOF probabilities, both blends, training feature checksums, exact E50 parity, and the original and new development ZIPs. The Task 009 prediction command reproduced the original 849-row CSV and its 14-row synthetic CSV byte for byte. Both encoders reused their existing per-image entries for the synthetic rerun. All 381 historical artifact hashes, including the frozen E50 head bundle and 14 feature-cache files, matched their pre-run values. The Windows test suite passed 101 tests and skipped its symlink fixture.

Manually score the fixed E50+L archive on hosted development before investing in a production CLIP-L encoder path. Keep the hosted-verified E50 pipeline available for evaluation. The development interval crosses zero and the third encoder raises the cost of processing new artwork.
