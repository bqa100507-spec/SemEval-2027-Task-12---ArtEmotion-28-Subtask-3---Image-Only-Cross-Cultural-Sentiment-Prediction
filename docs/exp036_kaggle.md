# EXP-036 · Caption-supervised cultural visual adaptation

**This training run is completed and closed.** This page preserves the original handoff and recipe. [Verified results](../reports/exp036_final.md) · [Active Evaluation inference guide](exp036_evaluation.md)

The private handoff contains 1,571 official training paintings, 16,019 eligible training pairs, the 85 public Development artworks covering 849 pairs, and the sealed Corrected Ridge predictions and bundle. The local Development comparator scored **0.465462 Contrast / 0.740305 Macro Log Loss** and matched the saved replay on every pair. The package puts training captions under `training/`, input-only Development rows under `phase_inputs/dev/`, and Development vote counts under `scoring/`. Training and Development are disjoint by painting, image path and image bytes. [Organizer ST3 specification](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#st3-image-only-cross-cultural-sentiment-prediction)

## Run once on Kaggle

1. Upload the ignored `runs/exp036/handoff/exp036_inputs.zip` as a **private Kaggle dataset**. Attach it to a private notebook and import `notebooks/EXP036_Kaggle.ipynb`. The notebook accepts Kaggle's extracted directory or the intact ZIP.
2. Enable Internet and a CUDA GPU. The worker uses one GPU, including when Kaggle supplies T4 ×2. It installs Python 3.12.10 and the pinned dependencies, then downloads the SigLIP2 weights and verifies their hashes. If you already attached the exact pinned model, set `MODEL_ROOT` to that directory to avoid the download.
3. Set `AUTHORIZE_RUN = True` and click **Run All**. The preflight checks zero-LoRA feature parity, real-caption gradients and microbatch memory. It reports measured forward/backward throughput and an approximate fit duration. The runtime estimate does not gate training. The default session allowance is 10.5 hours, with two minutes reserved for exit and export.
4. Keep the final files private. Download the results and checksums from **Output** or save a private version with those output files. Progress files under `/kaggle/working/` require a saved version or download to survive VM loss.

| Pin | Value |
| --- | --- |
| Packaged worker | `abc23cf9b9a3a138c51e85185b8e2003058436f4` |
| Input ZIP SHA-256 | `3e59a91ea6e4c9cdda0ecc3164a2a557d6425cb5ddc4e0e868bac1d6128ecddc` |
| Input manifest SHA-256 | `ff34edd58a442082b6848f77dabcc4e2804582d15350d73e033de2caf7c2b4d2` |
| SigLIP2 So400m patch14 384 | `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` |

The [SigLIP2 model card](https://huggingface.co/google/siglip2-so400m-patch14-384/tree/e8e487298228002f3d8a82e0cd5c8ea9c567f57f) lists Apache-2.0. The project previously recorded that pretrained models and official training captions are allowed. The current public organizer README still defers external-resource guidance, and the live Codabench rules could not be independently reread during this work. Check the latest organizer terms before a hosted upload. [Project rules audit](competition.md) · [Organizer FAQ](https://github.com/artemotion-28/artemotion-28#faqs)

## Recipe

- Run the official 384-pixel processor and native SigLIP pooling. Freeze the pretrained vision weights and pooling head. Apply rank-32 LoRA to Q, V, and both MLP projections in blocks 21–26 (zero-based), then fit a shared visual coefficient, centered language-specific coefficient deviations and language intercepts. Encode each artwork once per forward.
- Use the matched frozen SigLIP2 text tower only before visual fitting. Deduplicate encoder calls while retaining all annotation multiplicities in pair means. Use valid captions of binary-mappable training votes. Record raw, language-centered, painting-centered and sequentially centered norm diagnostics. The training target subtracts the caption-count-weighted language mean first, then the caption-count-weighted painting mean. Zero or missing targets carry zero caption loss weight.
- Optimize vote-normalized binary cross-entropy plus **0.10** times vote-weighted caption cosine loss, over six epochs using seed **203036**. FP16 autocast and scaling, activation checkpointing and microbatches of 4, 2 or 1 preserve the effective batch of eight paintings. The final vision pooler is frozen because this checkpoint exposes a pretrained pooling head. The inference bundle contains the learned LoRA and sentiment head weights and pins the unchanged base separately.
- Score each epoch on full official Development. Select by Contrast, then Log Loss, then earlier epoch. Search ensemble weights `0, 0.10, 0.25, 0.50, 0.75, 1` with a +0.01-bit guardrail relative to Ridge. Both resulting submissions use six-decimal probabilities and exact official keys. The endpoint `alpha=0` is explicitly recorded if selected. The standalone always contains CS-CVA scores. Repeated decisions on these 849 pairs can overfit Development. No cross-validation or separate control fit runs in this notebook.

EXP-017 used a caption auxiliary loss on frozen image features. EXP-023 prepared painting-centered targets and its complete persistent scientific evidence was unavailable. Track 3 stopped at the teacher screen and never evaluated its image-only student. EXP-022 compared vote supervision and CLIP-B LoRA, and EXP-033 used NaFlex spatial readouts and selective adapters. EXP-036 fits the caption objective through a SigLIP2 vision adapter and prioritizes a usable Development candidate. [Experiment registry](experiment_registry.md)

## Outputs and recovery

On completion, download `exp036_results.zip` and `exp036_results.zip.sha256` from the Output panel. The verified archive contains the selected inference bundle, complete checkpoint and resumption metadata, epoch predictions, scores, and the full weight grid. The standalone and ensemble archives are also available at:

- `/kaggle/working/exp036_run/exp036_standalone_submission.zip`
- `/kaggle/working/exp036_run/exp036_ensemble_submission.zip`

The worker seals optimizer boundaries every 64 steps and publishes a progress ZIP after each scored epoch. A handled session stop attempts to export the last sealed state with the optimizer, scheduler, scaler, RNG, completed steps, and completed caption cache blocks. For a later session, attach the saved result or progress archive privately and set `RESUME` to its mounted ZIP or extracted directory containing `export_manifest.json`. Use the same data, runtime and accelerator model. Abrupt losses can discard work after the last saved export. The diagnostic ZIP or status JSON identifies bootstrap or CUDA failures.

The bundle runner `scripts/run_exp036.py infer --bundle ... --pairs ... --images ... --destination ...` accepts the strict `painting,language,image` schema and infers using the image pixels and target language. `--images` points to a root containing `images/`. The caption teacher, caption targets and gold are absent from this inference path. To use ensemble inference on new inputs, combine those standalone predictions with the existing Corrected Ridge deployment pipeline on the same keys using the recorded alpha.

## Validation

The packaging command `scripts/package_exp036.py --source <worker-sha> --upstream <organizer-root> --ridge <dev-release> --fixed <fixed-ridge-bundle> --output <private-folder> --notebook <ipynb>` regenerated and read back the private handoff. Focused CPU tests exercised the actual tiny SigLIP module layout, LoRA gradients, centered target aggregation, accumulation parity, exact optimizer resume, strict ZIP coverage and mocked notebook launch. The real Corrected Ridge result above was the only Development score measured at handoff. Subsequent verified Kaggle artifacts are recorded in the [closure report](../reports/exp036_final.md).
