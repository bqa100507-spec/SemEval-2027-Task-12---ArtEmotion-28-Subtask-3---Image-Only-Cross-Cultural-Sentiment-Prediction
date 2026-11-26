# EXP-038 · Cultural-deviation ensemble (frozen encoders + interaction heads)

**Status: approved 2026-11-26 · implemented · tested CPU · not run on GPU.** No score in this file is an EXP-038 result. [Evaluation inference](exp038_evaluation.md) · [Superseded shared-basis recommendation](../reports/exp038_design_recommendation.md) · [Config](../configs/exp038.json)

## Question

Contrast depends only on the image-specific language deviation. Earlier gains came from adding partly independent estimators of that deviation (Track 1 three-encoder ridge, the neighborhood residual, the CS-CVA and C6 blends), while new losses, heads, low rank and learned stacking did not help. Centered residual correlations between existing components are 0.96–0.975. **Hypothesis:** frozen encoders with materially different pretraining data and objectives, passed through the unchanged Track 1 full-interaction ridge and the unchanged neighborhood correction, lower that correlation and raise test-like Contrast. Secondary: does a target-language instruction add image-specific signal over the same VLM with a neutral instruction (E4 vs E3)? No new loss, head architecture, low rank, learned stacking, constant retuning or Development-based coefficient search.

## Arms and candidates

| Arm | Features (frozen, L2-normalized) | Pin | License |
| --- | --- | --- | --- |
| R0 | Fixed Ridge (CLIP-B, SigLIP2-Base, CLIP-L at 0.25/0.25/0.50) and Corrected Ridge, exact replay | Track 1 / neighborhood artifacts | as before |
| E1 | SigLIP2 So400m patch14-384 pooled, EXP-036 processor, width 1152 | `google/siglip2-so400m-patch14-384@e8e487298228002f3d8a82e0cd5c8ea9c567f57f` | Apache-2.0 |
| E2 | MetaCLIP 2 worldwide ViT-H/14 quickgelu, `get_image_features`, width 1024 | `facebook/metaclip-2-worldwide-huge-quickgelu@c139061af7b10fdb2e754b60d2b1182a3d5526c2` | **CC-BY-NC-4.0 (non-commercial)** |
| E3 | Qwen3-VL-Embedding-2B, neutral instruction, full width 2048 | `Qwen/Qwen3-VL-Embedding-2B@9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | Apache-2.0 |
| E4 | E3 plus Δe(i,l) = e(i,l) − e_neutral(i) with the target-language instruction | same as E3 | Apache-2.0 |

Weight SHA-256 values and metadata Git blob hashes for every file are in `configs/exp038.json`; Validate hashes every downloaded file before any GPU work.

- Every head is `competition_ridge.fit_head` with the Track 1 rule (`reference_c` 0.3, deviation ratio 0.5, same solver tolerances, unseen-language fallback). E4 adds one language-independent linear term on Δe with the shared-slope ridge strength; E3 and E4 differ only by that term. Δe is computed only for the 16,019 eligible training pairs (which contain all P2 pairs) and the 849 Development pairs.
- Qwen formatting follows the pinned `scripts/qwen3_vl_embedding.py`: the instruction is the system message (a final period is added when missing), the user message holds one image with `min_pixels` 4,096 and `max_pixels` 262,144 (pinned here; up to 256 visual tokens), `apply_chat_template(add_generation_prompt=True)`, `qwen_vl_utils.process_vision_info(image_patch_size=16)`, the processor with `do_resize=False` and right padding, last non-padding token of `last_hidden_state`. FP16 weights, SDPA attention, eval mode, deterministic CUDA flags; L2 normalization is done in FP32 (the script normalizes in the model dtype). Neutral: `Represent the emotion a viewer would feel looking at this artwork.` Language: `Represent the emotion a {language}-speaking viewer would feel looking at this artwork.` with the official language name.
- **C1 (VLM-free)** = round6(0.5 Fixed Ridge + 0.5 mean(E1, E2)); **C2** adds E3; **C3** uses E4 instead of E3. Each then gets the frozen neighborhood correction (k 64, τ 0.10, 20 pseudo-votes, vote weights, painting-centered residuals from three inner painting-grouped fits inside every outer fit, clip and round6). Only the base predictor (the candidate mix) and the similarity space change. **Declared interpretation:** the similarity space concatenates every image encoder feeding the candidate (CLIP-B, SigLIP2-Base, CLIP-L, then E1/E2/E3; E4 contributes E3's image features), each block L2-normalized, divided by √(blocks). R0 keeps its original SigLIP2-Base + CLIP-L space.

## Gate (P2, opt-in, frozen)

P2 is the pinned five-fold painting assignment over 1,335 official training paintings and 13,600 pairs (`protocol.csv` `869b2fe…`, `pairs.csv` `50798dd…`). Each held-out painting is absent from its outer fit, its inner fits and its neighbor bank. Before scoring anything new the worker replays R0: all Fixed Ridge OOF rows exactly, the Corrected Ridge OOF CSV bytes (`fc8273e3…`), and pooled metrics to 1e-12; failure stops the run.

Pass vs Corrected Ridge requires all of: filtered (no English, Arabic, Chinese; eligibility recomputed) Contrast ≥ +0.004; painting-paired bootstrap 95% lower bound > 0 (20,000 draws, seed 203804); ≥ 4/5 filtered fold wins; positive mean after trimming the 5% largest absolute painting deltas; no full-P2 Contrast regression; Macro Log Loss regression ≤ +0.005 bits in both scopes. Among passing candidates the highest filtered Contrast wins; within 0.001 the simpler one (C1 < C2 < C3). None passing is a scientific **NO_GO**. Diagnostics (not gating): standalone arm metrics, centered residual correlation vs Fixed Ridge (interest threshold < 0.95), pair-order agreement, centered RMSE, deviation slope, E4 − E3 paired interval, per-language and per-painting deltas, leave-one-fold-out gains.

## Full refit, Development and blends

All arms and candidates are refitted on the 1,571 official training paintings; each deployment bank is cross-fitted exactly like Corrected Ridge's (inner seed 203201). The refitted R0 must reproduce the Corrected Ridge Development ZIP member bytes (`39db05b3…`), and the R0 compositions must reproduce the EXP-036 ensemble and the EXP-037 original blend byte for byte. Development (849 pairs, full and Evaluation-like) is reporting only.

CS-CVA: two new seeds **203836** and **203936** of `configs/exp036.json` (only the seed changes; six-epoch schedule; training stops after epoch 5; no Development epoch selection). CS-CVA-3 = round6(mean of the original epoch-5 predictions and the two new seeds). B1(Cx) = round6(0.75 Cx + 0.25 CS-CVA), with CS-CVA-3 and with the original; B2(Cx) = round6(0.80 B1 + 0.20 C6) with the saved EXP-037 C6 predictions. Every arm, candidate, B1 and B2 gets a validated ZIP. `exp038_best_submission.zip` is B2(selected) with CS-CVA-3 on GO (original CS-CVA if a seed failed, recorded), or the byte-identical EXP-037 original blend on NO_GO. Labels: C1, B1(C1) and the R0 blends are VLM-free; E3, E4, C2, C3 and every B2 carry `UNCONFIRMED_QWEN_ST3`.

## Run once on Kaggle

1. Attach the existing private dataset built from `runs/exp037/handoff/exp037_inputs.zip` (SHA-256 `eaf16b9301cb131fbdf61a1b86fb191d3617b2141e39398f0c3b12bcf57ffce2`) and a second private dataset from `runs/exp038/handoff/exp038_inputs.zip` (pins below). Import [EXP038_Kaggle.ipynb](../notebooks/EXP038_Kaggle.ipynb). Kaggle may mount the ZIPs intact or extracted; both work. Kaggle also expands archives nested inside an upload, so the overlay (patch 2) stores every archive-like member (the five reference ZIPs and three `.npz` files) as base64 `.b64` text with both the stored and the decoded SHA-256 in its manifest; the launcher decodes them into `/kaggle/temp/exp038/overlay` and re-verifies them, leaving the mounted input read-only. An overlay built before patch 2 fails with an error that names the patch-2 packaging.
2. Accelerator **GPU** (T4 ×2 is fine; only the first GPU is used) and **Internet on** (pinned Python 3.12.10 environment and model downloads).
3. In the Configuration cell set `AUTHORIZE_RUN = True`; leave `RESUME_INPUT` empty for the first session. Click **Run All**.
4. When it ends, save a version or download from **Output**: `exp038_results.zip`, `exp038_best_submission.zip` and their `.sha256` (after a failure: `exp038_diagnostic.zip`; to continue: `exp038_progress.zip`). Do not upload anything to Codabench from the notebook.

Stage order: **Setup → Validate → Benchmark → Feature extraction → CS-CVA seeds → P2 evaluation → Full refit and Development scoring → Submission validation → Export → Release.** Validate verifies both datasets, every image and every model file hash, and runs the R0 CPU replays; any failure stops before GPU work. The launcher sets `OPENBLAS_CORETYPE=Haswell` and the ridge fits use two threads, matching the machine that produced the reference artifacts (local replay recorded OpenBLAS 0.3.29 Haswell kernels).

**Budget.** Session allowance 10.5 h including setup, 20-minute persistence reserve, hard 3 h cap on E4 extraction. Benchmark forecast method: each encoder runs two passes over 32 seeded real artworks (or pairs for E4) on the first GPU; the warm second pass gives items/s, and a stage forecast is items / rate plus a 240 s model-load allowance. CS-CVA uses EXP-036's own two-batch real-image preflight rate for 5 × 1,571 paintings plus Development inference. CPU stages scale a timed real ridge fit by encoder width and fit count, plus twice the measured R0 replay time. The sum is multiplied by 1.25. If it exceeds the remaining time, whole optional pieces are skipped in order (CS-CVA seed 3, seed 2, then E4 with C3 `BUDGET_SKIPPED`) and the reason is recorded; an E4 forecast above its cap skips E4 directly. R0, E1, E2 and C1 are mandatory. E3 and E4 (Qwen3-VL-Embedding) are optional: if E3 fails or is skipped, E4 is skipped as a dependency, C2/C3 are listed as unavailable, and the run still completes with the VLM-free C1 and its ZIPs. Reaching a cap is a recorded `BUDGET_STOP`, not a crash. Two-pass determinism must agree within 1e-3 (bitwise agreement is recorded).

**Resume.** Features (sealed 256-item chunks, then one sealed cache), CS-CVA seeds, Validate, Benchmark, P2 and Development outputs are sealed with source, runtime, GPU and input identity. On resume (attach the progress or results ZIP and set `RESUME_INPUT`, same inputs and GPU type) they are reused; corrupt, stale or incomplete caches stop the run instead of being recomputed. An interrupted CS-CVA seed restarts. Progress shows stage, item, elapsed, remaining and the stop reason; `exp038_progress.zip` is refreshed at stage boundaries and every 15 minutes during extraction.

## Pins

| Pin | Value |
| --- | --- |
| Worker source (embedded in the overlay) | `85122b6b770bf31404c53602ea514094d2516fb9` |
| Overlay `runs/exp038/handoff/exp038_inputs.zip` SHA-256 | `b9aeb02578b8299bf885139cac7dd9871b6df250c3947b381a0f1f575a448702` (24,541,593 bytes, 204 files) |
| Overlay manifest SHA-256 | `36222457bac13e7b3abe8157261f400d060d68a283b0b7a3f47e70ecd35e794e` |
| Base ZIP / manifest SHA-256 | `eaf16b9301cb131fbdf61a1b86fb191d3617b2141e39398f0c3b12bcf57ffce2` / `026720f406c02236aeaf5bfdc806853816323b66267f81f0e3e2dbbfe1ce7e0e` |

## Local verification before handoff

`scripts/run_exp038.py cpu-validate`, run from the packaged overlay's own `runtime/` on the real base and overlay, reproduced on Windows CPU: all 13,600 Fixed Ridge OOF rows and every fold's inner component hash; the Corrected Ridge OOF CSV (`fc8273e3…`); pooled full / filtered Contrast and Log Loss to 1e-12 (Fixed 0.451702 / 0.709247 and 0.460043 / 0.718425; Corrected 0.454134 / 0.708500 and 0.463108 / 0.717479); the full-training Fixed Ridge heads and Corrected Ridge bank bit for bit; the Fixed and Corrected Development ZIP members; and both published compositions (EXP-036 ensemble, EXP-037 original blend) byte for byte. Kaggle Linux is a different platform, so Validate repeats these checks there before any GPU work. Wide-head convergence was checked on a real P2 fold with structured synthetic features: widths 1,152 and 2,048 and the E4 pair head converged in 207–292 iterations with gradient ≤ 2.4e-7 (limit 5e-7). `python -m pytest -q`: 401 passed, 5 skipped.

## Licenses and eligibility

SigLIP2 and Qwen3-VL-Embedding-2B are Apache-2.0. **MetaCLIP 2 worldwide is CC-BY-NC-4.0: attribution is required and commercial use is not permitted.** This competition entry is non-commercial research; the owner should confirm that this is acceptable and that organizer rules allow this pretrained model before any hosted upload. `qwen-vl-utils` 0.0.14 is Apache-2.0. Organizer permission for VLMs in ST3 is still unresolved (public issue #1); every Qwen-dependent ZIP stays `UNCONFIRMED_QWEN_ST3`. The inputs contain official Train annotations and public Development counts (scorer only) and no Evaluation data.
