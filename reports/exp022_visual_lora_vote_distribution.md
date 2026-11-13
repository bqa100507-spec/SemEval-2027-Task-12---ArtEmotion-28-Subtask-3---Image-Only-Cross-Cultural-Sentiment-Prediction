# EXP-022 — Matched visual LoRA and empirical emotion votes

## Frozen design and execution boundary

Base EXP-021: `9c339a614eca9dfdb173c707ca1f17cdcdc0547f`, PR #20. Before implementation its actual local source/configuration, CV, latest successful replay and thirty checkpoint manifests were verified. Decision: **COMPETITION_NO_GO**. Pooled historical Contrast: R0 0.447097707443; D0 0.437792268384; D8 0.441591226055; ED0 0.448585746756; ED8 0.449920216770. These are exploratory results on repeatedly examined paintings, not independent test results. The older EXP-021 report remains a preparation record; no historical report or artifact is rewritten.

Only the original 1,335 development-training paintings / 13,600 eligible pairs / 69,719 binary-mappable votes are used. The eight counts and original fold identities are copied byte-for-byte from the verified EXP-021 audit. Ambiguous emotions remain excluded; at least five votes per pair. The 236 confirmation paintings' emotion fields, public Development gold, captions and art metadata are excluded. Historical artifact bytes are checksummed opaquely, never parsed as new targets. No Confirm/Dev/submission entry point exists. E50+L remains production.

F0/F8 use the original three frozen feature banks; L0/L8 compute CLIP-B live and keep the other two banks detached. Every condition allocates exactly the same original width-64 sentiment model and compact 128-to-8 auxiliary classifier. F0/L0 train BCE only; F8/L8 train `BCE + 0.1 CE8 / ln(8)`, with vote-count weighting exactly as EXP-021. CE uses empirical counts without smoothing or reweighting. The auxiliary classifier is omitted from normal prediction forwards.

CLIP-B is `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Only the visual tower is instantiated. Rank 8 / alpha 16 / zero dropout adapters wrap `vision_model.encoder.layers.{0..11}.self_attn.{q_proj,v_proj}`: 24 modules, 48 matrices, **294,912 parameters**. Base weights, biases, projection and normalization layers are frozen. The original 224-pixel resize/center-crop/normalization processor and 512-dimensional normalized projection are preserved. No EXP-014 adapted checkpoint is read. Its archived engineering lessons were reviewed at `cd321f142142120c17c495458390deb6574ad856`: zero-B initialization, multiple-step gradient checks, exact feature parity, frozen-weight hashes and accumulation denominators.

Fusion is `concat(CLIP_B_512, SigLIP2_768, CLIP_L_768) / sqrt(3)`. Unit-normalized blocks are not normalized again. Group sampling shuffles distinct paintings with seed+1009*epoch and retains sorted language rows. Each full epoch visits every eligible pair exactly once. Each optimizer step contains four paintings; microbatches follow 4, 2, 1 only for hardware feasibility. Microbatch likelihood numerators are divided by the total votes of the entire effective step. This ratio normalization differs from EXP-021's shuffled 256-pair minibatches; it is identical across all four new conditions. It does not assign equal weights to individual painting-language pairs or discard paintings with many languages.

Frozen before held-out scores: head width 64, dropout .2, head LR .001, LoRA LR .0001, AdamW decay .01, global norm clip 1, no augmentation, maximum 100 epochs, patience 10, original nested inner partition and binary-BCE stopping, then fresh refit for the selected epoch count. Adapter initialization has a separate generator; the task head and dropout RNG are reseeded after adapter allocation. F0/F8/L0/L8 have identical head initialization and group order for a matched seed. The learning rates are engineering starting choices, not validated optima.

Vision tensors and adapters use FP32. Historical cuDNN TF32 convolution is enabled; CUDA matmul TF32 is disabled. Only task-head training uses FP16 autocast and GradScaler; objectives are FP32 and prediction/replay is FP32. Initial scale 1024, at most eight backoffs. Overflow retries the same effective batch with restored dropout RNG and never silently skips an optimizer update. These precision choices were settled during label-free/short-gradient preflight, before screening. The first engineering attempts exposed a backend mismatch and overly high default AMP scale; both failures are retained locally. No tolerance was relaxed.

## Predeclared gates

MiniScreen is fold 1 / seed 2027 / all four conditions, with nested stopping/refit. It is a catastrophic-regression check, not hyperparameter selection. Screen adds fold 3 and seed 2028, reusing completed MiniScreen models: sixteen total fits. Each aggregate rounds the mean raw seed probabilities to six decimals, then applies the existing six-decimal 50/50 blend with R0. Historical D0/D8 comparisons reconstruct the **same folds and same seeds**, rather than using their five-fold/three-seed summaries.

`LORA_PROMISING`: L0−F0 or L8−F8 pooled Contrast >= .001, at least one of two fold improvements, pooled Macro Log Loss <= matched frozen control + .005 bits. `AUXILIARY_PROMISING`: L8−L0 meets the same thresholds. Both require complete replay, finite gradients, nonzero adapter updates, isolation and absence of major prediction collapse. Both flags can be true. Both LoRA variants losing >= .01 Contrast and worsening loss by > .01 bits is `SCREEN_UNPROMISING`; otherwise `SCREEN_INCONCLUSIVE`. The fixed .01 material-loss threshold makes the published qualitative rule executable. MiniScreen stops for review if both lose >= .05 Contrast and worsen loss by > .02 bits. These are resource-allocation heuristics, not statistical significance claims.

Full CV requires a currently verified promising Screen **and** explicit owner `-AllowFullCV`; it is never run automatically. All sixty fold/seed/condition checkpoints are required, reusing the sixteen screen fits. Competition gates retain ensemble gain over R0 >= .003, gain over matched EF0 >= .002, >=3/5 fold wins and fold-mean Macro Log Loss <= R0 + .005. Even a qualifying gate does not authorize confirmation/Development scoring or upload in EXP-022.

Any failed/interrupted verification revokes the current successful decision. Checkpoints are published only after a complete selection/refit/prediction; interrupted current fits restart, completed models are reused. Source/configuration/runtime/input identities and checksums must match. Different hardware gets its own run directory and fresh Preflight/Pilot; completed checkpoints are not silently mixed across runtimes.

## Execution record

Actual final preflight, full-epoch pilot, runtime budget, tests and handoff details are recorded below after execution. No held-out EXP-022 performance has yet been measured.

## Commands and artifacts

From the project root, using the existing pinned `.venv`:

```powershell
.\scripts\run_exp022.ps1 -Stage Preflight -Device cuda -MaxSeconds 600
.\scripts\run_exp022.ps1 -Stage Pilot -Device cuda -MaxSeconds 900
# Read pilot/runtime_projection.json BEFORE allowing long GPU work.
.\scripts\run_exp022.ps1 -Stage MiniScreen -Device cuda -MaxSeconds 14400
.\scripts\run_exp022.ps1 -Stage MiniScreenVerify -Device cuda -MaxSeconds 900
.\scripts\run_exp022.ps1 -Stage Screen -Device cuda -MaxSeconds 14400
.\scripts\run_exp022.ps1 -Stage ScreenVerify -Device cuda -MaxSeconds 1800
# Only after verified screening success and explicit owner allocation:
.\scripts\run_exp022.ps1 -Stage CV -Device cuda -MaxSeconds 86400 -AllowFullCV
.\scripts\run_exp022.ps1 -Stage Verify -Device cuda -MaxSeconds 3600
```

These are staged commands, not a single script that launches all stages. Default `runs/exp022_visual_lora/` is ignored by Git. `-RunDir`, `-DataDir`, `-CacheDir`, `-Exp017Dir` through `-Exp021Dir`, `-Device`, and `-MaxSeconds` are explicit overrides. The Python runner also works on Linux. Exit codes: 0 complete, 1 invalid evidence, 2 invalid arguments/environment, 3 stage/resource blocked, 4 CUDA OOM, 75 resumable budget stop. PowerShell writes unique stage logs and RAM telemetry.

Inputs contain counts, frozen features and OOF references only. `preflight/` seals processor/feature comparisons, exact module names, memory and separate gradients. `pilot/` records all four full-epoch timings and a conservative projection. Completed models live in `work/fold_N/CONDITION_SEED/`; `mini/`, `screen/`, `cv/` contain pooled/fold, language/painting, factorial, bootstrap and prediction-variance tables. Verification stages add replayed seed metrics, gradients and training curves. Each independent comparison includes 10,000 paired painting bootstrap replicates when a study is completed; these are exploratory intervals on reused folds.

`python scripts/package_exp022.py` creates and validates the ignored `exp022_colab_inputs.zip`: only 1,335 allowed images, exact count/features/reference inputs and the three already-pinned CLIP-B assets. It contains no confirmation images, captions, new checkpoint or SigLIP/CLIP-L weights. Use [the Colab handoff](../docs/exp022_colab.md) if local runtime is excessive. No data/weights/checkpoints are pushed.

Resource eligibility remains as previously qualified in [competition constraints](../docs/competition.md). Reusing an existing pretrained resource here does not resolve organizer permissions for the final competition system. No new resource or text supervision is introduced.
