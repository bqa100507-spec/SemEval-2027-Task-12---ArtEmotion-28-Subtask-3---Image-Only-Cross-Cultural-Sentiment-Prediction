# EXP-031 · Hierarchical cultural ranking Qwen 7B (research only)

## Why this model

EXP-030 Qwen epoch 3 achieved fold-3 Cultural Contrast **0.419801**, Macro Log Loss **0.733054 bits**, versus matched E50+L **0.433536 / 0.713192**. The held-out painting bootstrap 95% CI for Contrast difference was [−0.033822, +0.007307], so this reused fold does not establish a stable ordering. The prior-only reference was 0.332541. Post-prior image–language residuals had more variation for Qwen but lower alignment with gold. Late-epoch ranking and calibration worsened.

This experiment tests **one** fully specified architecture/recipe rather than many small pilots. It does not assume the ranking objective will outperform EXP-030 or E50+L.

## Frozen architecture

- Pinned native **Qwen2.5-VL-7B-Instruct** revision, NF4 decoder, frozen BF16 native visual encoder, rank-16 alpha-32 decoder Q/V LoRA, official multimodal position handling, FP32 trainable heads. No E50+L features or logits enter the network.
- Image-level neutral Qwen state from an `unspecified` language instruction; computed **once per painting**, independent of which languages are evaluated. A shared linear scalar learns the painting's common sentiment effect.
- Per-language binary logit:
  
  `logit_positive(image, lang) = fitted_language_logit_prior(lang) + shared_image_logit(image) + centered_low_rank_image_x_language(image, lang)`
  
  Language priors use **only fitting-split** vote totals with Jeffreys-style pseudocount 0.5. The interaction embedding initializes at zero and is centered across training languages to separate it from the shared image effect.
- The original language-conditioned Qwen hidden state goes through an eight-output conditional emotion head. Two four-way softmax distributions represent within-positive and within-negative emotion categories, multiplied by the corresponding binary sentiment mass. The resulting eight probabilities always sum to one and the positive four always sum to the binary sigmoid.
- Train using exactly 1,068 fitting paintings (10,879 painting-language pairs), held-out 267 paintings (2,721 pairs). Keep all rows of a painting together for its ranking loss. Each optimizer accumulation is approximately 16 pairs but never divides a painting group, so some updates may be larger.
- Fixed objective: **count-weighted 8-way CE + 0.5 count-weighted binary BCE + 0.15 confidence-weighted within-painting ranking loss + 0.01 centered interaction L2**. The vote term is normalized by the votes in each optimizer update. Ranking is an average over complete paintings; weights depend only on vote proportions and Jeffreys posterior uncertainty, suppressing close/noisy comparisons. No held-out hyperparameter search.
- Fixed 6 epochs, head learning rate 3e-4, adapter learning rate 3e-5, AdamW weight decay 0.05, clipping at norm 1, seed 2031, deterministic math SDPA and strict runtime/dependency pins. Best checkpoint selected by held-out Contrast for exploratory comparison, not for a final unbiased claim.

## GPU / Colab

- [EXP031_Colab.ipynb](../notebooks/EXP031_Colab.ipynb) is **default-off**. Mount the original pinned `exp023_handoff.zip` in Drive, select A100 or L4, explicitly set `AUTHORIZE_EXPERIMENT = True`, and choose **Run all** once. A large full-painting max-image-grid backprop CUDA feasibility test runs as part of the same notebook before paid training. CPU tests execute before pretrained model load.
- 8–12 hours may be configured; **12 hours is recommended** for a six-epoch completion attempt because EXP-030 took roughly 1.6 hours/epoch before the additional neutral forward. Neither EXP-031 epoch time nor full-painting VRAM is measured yet. The first full-painting feasibility pass may fail for L4; do not claim T4 support until separately implemented and measured.
- Outputs: `MyDrive/exp031/attempts/<attempt>/final_report.json` and `outputs.zip`. `MyDrive/exp031/checkpoints/epoch-XX` holds the immutable epoch seal, trainable weights, optimizer state, RNG, metrics and 2,721 predictions. Resume requires identical pinned source, GPU name, software versions and config. Confirm the previous runtime has exited before removing a leftover `.active` directory.
- Keep `AUTO_RELEASE_COLAB = True` to release the GPU upon a normal exit. If using VS Code, a Jupyter disconnect does not alone prove a runtime died; check Drive checkpoints and Colab Manage sessions.

## Interpretation and safeguards

Report Contrast, Macro Log Loss, ranking loss, eight-way CE, binary BCE, confidence/VRAM/throughput and per-epoch learning curve. The measured fold has already influenced architectural choices; improvements here would remain exploratory. No sealed confirmation, Development gold, or production promotion is authorized. Competition model eligibility, paper timelines and eventual T4 inference remain separate engineering checks.

**Before any GPU run:** Have the reviewer verify code syntax/CPU tests and confirm the pinned notebook source and maximal painting-group probe. There is no pretrained GPU result yet.
