# EXP-006 frozen SigLIP 2 representation

## Rules and checkpoint

Before acquiring weights on 2026-11-10, we verified the public Codabench ST3 Data page through its competition metadata. It permits pretrained image models, requires citing external resources in the system paper, and forbids outside annotations for evaluation paintings. This establishes category-level permission. The published rule gives no checkpoint-specific artwork overlap audit. The study uses only organizer images and the permitted original training labels. [Codabench ST3 rules](https://www.codabench.org/competitions/18468/) · [Public competition metadata](https://www.codabench.org/api/competitions/18468/?format=json)

The single candidate is Apache 2.0 `google/siglip2-base-patch16-224` at revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. The Hub reports SHA-256 `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b` for the 1,500,800,904-byte safetensors. The card identifies WebLI as the pretraining dataset. The paper describes multilingual image-text pretraining with captioning and self-supervised losses. [Pinned model card and assets](https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2) · [SigLIP 2 paper](https://arxiv.org/abs/2502.14786) · [Google Big Vision license](https://github.com/google-research/big_vision/blob/main/LICENSE)

This fixed-resolution checkpoint declares `model_type=siglip` and `SiglipImageProcessor`. The pinned slow processor resizes RGB images to 224 × 224 using bilinear interpolation, rescales by 1/255 and normalizes with RGB means and SDs of 0.5. The model-card API is `get_image_features`. In the pinned `SiglipModel` implementation this returns the vision transformer's learned attention-pooling output directly. The fixed-resolution vision path applies the learned pooling head and has no separate projection afterward. The study takes that 768-dimensional image embedding and L2-normalizes it. [Pinned Transformers SigLIP implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/siglip/modeling_siglip.py)

## Frozen protocol

Extract on a separate Python 3.12 `.venv-siglip2` with Transformers 4.51.3. The overlay inherits the baseline's pinned dependencies, including its CUDA-capable PyTorch, while loading Transformers from the overlay. CV and evaluation run with the existing `.venv` and Transformers 4.49.0.

Compare C-CLIP and C-SigLIP2 using the original full interaction LR, `C=0.3`, exact EXP-002 folds, annotator-count weighting, and unchanged scorer. Require +0.005 mean Contrast gain, positive gain in at least four folds, mean Macro Log Loss at most 0.748056 bits, and convergence in every fold. Freeze the decision before development scoring. If SigLIP 2 qualifies, fit once on all original training pairs, use 10,000 painting-paired bootstrap draws against C, and validate a separate ZIP locally.

## Five-fold results and decision

All 16,019 held-out CLIP probabilities reproduced EXP-002 exactly after the original six-decimal rounding. Each fold entry is Cultural Contrast / Macro Log Loss.

| Fold | C-CLIP | C-SigLIP2 | Contrast change | Loss change (bits) |
| ---: | --- | --- | ---: | ---: |
| 1 | 0.450409 / 0.739601 | 0.455601 / 0.730092 | +0.005192 | −0.009509 |
| 2 | 0.434233 / 0.731610 | 0.438698 / 0.745614 | +0.004465 | +0.014005 |
| 3 | 0.411972 / 0.742971 | 0.418258 / 0.724299 | +0.006285 | −0.018671 |
| 4 | 0.468554 / 0.733164 | 0.465921 / 0.738197 | −0.002633 | +0.005033 |
| 5 | 0.413355 / 0.742934 | 0.420355 / 0.739570 | +0.007000 | −0.003364 |
| **Mean** | **0.435705 / 0.738056** | **0.439767 / 0.735555** | **+0.004062** | **−0.002501** |
| Pooled out-of-fold | 0.435854 / 0.737288 | 0.439910 / 0.734760 | +0.004056 | −0.002527 |

Across-fold sample SDs for Contrast / Macro Log Loss were 0.024291 / 0.005381 for CLIP and 0.021068 / 0.008379 for SigLIP 2. Pooled scores apply the same scorer once to the combined predictions, giving Contrast to the 1,496 eligible paintings and Macro Log Loss to 21 eligible languages. The fold means determine selection.

**NO-GO under the frozen rule.** SigLIP 2 met the four-fold reliability, mean Log Loss and convergence conditions. Its mean Contrast gain fell **0.000938** short of the +0.005 condition. The development stage recorded the fallback. The retained CLIP's established development scores are 0.446564 Contrast and 0.751995 Macro Log Loss. A new development comparison, bootstrap and ZIP are inapplicable. [Task 001 development baseline](baseline_dev.md)

## Paired reliability

SigLIP 2 improved Contrast on **779 of 1,496** scored paintings and regressed on **717**. The median paired change was +0.003293. Removing its ten largest gains left a pooled mean change of +0.001785. The largest gain was +0.415019 on Luca Signorelli's *The Holy Family*. The largest drop was −0.599946 on William Baziotes's *Moon Animal*. Mean Log Loss improved for **11 of 21** macro-eligible CV languages and regressed for ten. Burmese (−0.035775 bits) and Malay (−0.013751) improved most, and Vietnamese (+0.010288) regressed most.

The sign and size of the changes vary by painting and language. The fixed C head has **13,338** parameters for CLIP-512 and **19,994** for SigLIP 2-768. The checkpoint, preprocessing, supervision and head width differ jointly. These results measure the tested substitutions, with limited evidence for a generally better multilingual visual representation. [Research synthesis and reuse limitations](research_synthesis_001_006.md)

## Compute, provenance and verification

The resolved vision configuration has 12 layers and 12 attention heads, hidden width 768, MLP width 3,072 and 16-pixel patches at 224 × 224 resolution. The extraction loader verified all vision weights and the learned pooling head were present. The offline CPU smoke used a synthetic RGB image. The vision-only tower and the documented full-model image API returned exactly the same 768 values, with maximum absolute difference **0**. The vision-only CPU forward took about **0.41 seconds**.

The full extraction used the RTX 4060 Laptop GPU at batch size 16. It produced all **1,656** features in **32.3 seconds** of image loading, preprocessing and inference, and **39.8 seconds** for the full cache build from local weights. PyTorch peak GPU allocation was **506.9 MiB** and peak reserve was **668 MiB**. The full extraction required zero batch reductions. The feature cache SHA-256 is `2d781fce2b5fa57bb53a9be77b8298c9dfe316d25296561a2a77c43f276347eb`. Processor JSON SHA-256 is `9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc`. The manifest stores checkpoint asset hashes, image-content identity, feature coverage, processor settings, runtime and memory.

Cached-feature CV took **63.8 seconds** on CPU. Mean LR fit time per fold was **4.02 seconds** for C-CLIP and **5.84 seconds** for C-SigLIP2. All ten heads converged. C used 66–79 iterations, and SigLIP 2 used 66–80. The extraction code revision was `0bc5192`. CV ran at `77948976d81425a68ac9ca4153b76262b1ecd370`.

All **55 tests** passed. A cached fold 1 replay reproduced all **3,212** SigLIP 2 probabilities exactly. The artifact audit validated **32,038** CV probabilities, preserved the hashes of 53 earlier run files and all four earlier feature files, and verified that the baseline environment's package freeze was byte-identical before and after overlay setup. The refactored extractor also reopened the existing 1,656-image DINO cache offline without re-encoding images.

```powershell
.venv/Scripts/python.exe scripts/setup_siglip2.py
.venv-siglip2/Scripts/python.exe scripts/run_st3.py siglip-embed
.venv/Scripts/python.exe scripts/run_st3.py siglip-cv
.venv/Scripts/python.exe scripts/run_st3.py siglip-dev
.venv/Scripts/python.exe -m pytest -q
```

Local artifacts are in `runs/exp006_siglip2/`. `extraction/` contains the feature manifest, `cv/` contains fold and pooled metrics, predictions, language and painting deltas, provenance and the frozen decision, and `dev/run.json` records the fallback.
