# EXP-006 frozen SigLIP 2 representation

## Rules and checkpoint

Before acquiring weights on 2026-11-10, we verified the public Codabench ST3 Data page through its competition metadata. It permits pretrained image models, requires citing external resources in the system paper, and forbids outside annotations for evaluation paintings. This establishes category-level permission. The published rule gives no checkpoint-specific artwork overlap audit. The study uses only organizer images and the permitted original training labels. [Codabench ST3 rules](https://www.codabench.org/competitions/18468/) · [Public competition metadata](https://www.codabench.org/api/competitions/18468/?format=json)

The single candidate is Apache 2.0 `google/siglip2-base-patch16-224` at revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. The Hub reports SHA-256 `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b` for the 1,500,800,904-byte safetensors. The card identifies WebLI as the pretraining dataset. The paper describes multilingual image-text pretraining with captioning and self-supervised losses. [Pinned model card and assets](https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2) · [SigLIP 2 paper](https://arxiv.org/abs/2502.14786) · [Google Big Vision license](https://github.com/google-research/big_vision/blob/main/LICENSE)

This fixed-resolution checkpoint declares `model_type=siglip` and `SiglipImageProcessor`. The pinned slow processor resizes RGB images to 224 × 224 using bilinear interpolation, rescales by 1/255 and normalizes with RGB means and SDs of 0.5. The model-card API is `get_image_features`. In the pinned `SiglipModel` implementation this returns the vision transformer's learned attention-pooling output directly. The fixed-resolution vision path applies the learned pooling head and has no separate projection afterward. The study takes that 768-dimensional image embedding and L2-normalizes it. [Pinned Transformers SigLIP implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/siglip/modeling_siglip.py)

## Frozen protocol

Extract on a separate Python 3.12 `.venv-siglip2` with Transformers 4.51.3. The overlay inherits the baseline's pinned dependencies, including its CUDA-capable PyTorch, while loading Transformers from the overlay. CV and evaluation run with the existing `.venv` and Transformers 4.49.0.

Compare C-CLIP and C-SigLIP2 using the original full interaction LR, `C=0.3`, exact EXP-002 folds, annotator-count weighting, and unchanged scorer. Require +0.005 mean Contrast gain, positive gain in at least four folds, mean Macro Log Loss at most 0.748056 bits, and convergence in every fold. Freeze the decision before development scoring. If SigLIP 2 qualifies, fit once on all original training pairs, use 10,000 painting-paired bootstrap draws against C, and validate a separate ZIP locally.
