# EXP-022 — Matched visual LoRA and empirical emotion votes

## Frozen design and execution boundary

Base EXP-021: `9c339a614eca9dfdb173c707ca1f17cdcdc0547f`, PR #20. Before implementation its actual local source/configuration, CV, latest successful replay and thirty checkpoint manifests were verified. Decision: **COMPETITION_NO_GO**. Pooled historical Contrast: R0 0.447097707443; D0 0.437792268384; D8 0.441591226055; ED0 0.448585746756; ED8 0.449920216770. These are exploratory results on repeatedly examined paintings, not independent test results. The older EXP-021 report remains a preparation record; no historical report or artifact is rewritten.

Only the original 1,335 development-training paintings / 13,600 eligible pairs / 69,719 binary-mappable votes are used. The eight counts and original fold identities are copied byte-for-byte from the verified EXP-021 audit. Ambiguous emotions remain excluded; at least five votes per pair. The 236 confirmation paintings' emotion fields, public Development gold, captions and art metadata are excluded. Historical artifact bytes are checksummed opaquely, never parsed as new targets. No Confirm/Dev/submission entry point exists. E50+L remains production.

F0/F8 use the original three frozen feature banks; L0/L8 compute CLIP-B live and keep the other two banks detached. Every condition allocates exactly the same original width-64 sentiment model and compact 128-to-8 auxiliary classifier. F0/L0 train BCE only; F8/L8 train `BCE + 0.1 CE8 / ln(8)`, with vote-count weighting exactly as EXP-021. CE uses empirical counts without smoothing or reweighting. The auxiliary classifier is omitted from normal prediction forwards.

CLIP-B is `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Only the visual tower is instantiated. Rank 8 / alpha 16 / zero dropout adapters wrap `vision_model.encoder.layers.{0..11}.self_attn.{q_proj,v_proj}`: 24 modules, 48 matrices, **294,912 parameters**. Base weights, biases, projection and normalization layers are frozen. The original 224-pixel resize/center-crop/normalization processor and 512-dimensional normalized projection are preserved. No EXP-014 adapted checkpoint is loaded or used to initialize training. Its archived engineering lessons were reviewed at `cd321f142142120c17c495458390deb6574ad856`: zero-B initialization, multiple-step gradient checks, exact feature parity, frozen-weight hashes and accumulation denominators.

Fusion is `concat(CLIP_B_512, SigLIP2_768, CLIP_L_768) / sqrt(3)`. Unit-normalized blocks are not normalized again. Group sampling shuffles distinct paintings with seed+1009*epoch and retains sorted language rows. Each full epoch visits every eligible pair exactly once. Each optimizer step contains four paintings; microbatches follow 4, 2, 1 only for hardware feasibility. Microbatch likelihood numerators are divided by the total votes of the entire effective step. This ratio normalization differs from EXP-021's shuffled 256-pair minibatches; it is identical across all four new conditions. It does not assign equal weights to individual painting-language pairs or discard paintings with many languages.

Frozen before held-out scores: head width 64, dropout .2, head LR .001, LoRA LR .0001, AdamW decay .01, global norm clip 1, no augmentation, maximum 100 epochs, patience 10, original nested inner partition and binary-BCE stopping, then fresh refit for the selected epoch count. Adapter initialization has a separate generator; the task head and dropout RNG are reseeded after adapter allocation. F0/F8/L0/L8 have identical head initialization and group order for a matched seed. The learning rates are engineering starting choices, not validated optima.

Vision tensors and adapters use FP32. Historical cuDNN TF32 convolution is enabled; CUDA matmul TF32 is disabled. Only task-head training uses FP16 autocast and GradScaler; objectives are FP32 and prediction/replay is FP32. Initial scale 1024, at most eight backoffs. Overflow retries the same effective batch with restored dropout RNG and never silently skips an optimizer update. These precision choices were settled during label-free/short-gradient preflight, before screening. The first engineering attempts exposed a backend mismatch and overly high default AMP scale; both failures are retained locally. No tolerance was relaxed.

## Predeclared gates

MiniScreen is fold 1 / seed 2027 / all four conditions, with nested stopping/refit. It is a catastrophic-regression check, not hyperparameter selection. Screen adds fold 3 and seed 2028, reusing completed MiniScreen models: sixteen total fits. Each aggregate rounds the mean raw seed probabilities to six decimals, then applies the existing six-decimal 50/50 blend with R0. Historical D0/D8 comparisons reconstruct the **same folds and same seeds**, rather than using their five-fold/three-seed summaries.

`LORA_PROMISING`: L0−F0 or L8−F8 pooled Contrast >= .001, at least one of two fold improvements, pooled Macro Log Loss <= matched frozen control + .005 bits. `AUXILIARY_PROMISING`: L8−L0 meets the same thresholds. Both require complete replay, finite gradients, nonzero adapter updates, isolation and absence of major prediction collapse. Both flags can be true. Both LoRA variants losing >= .01 Contrast and worsening loss by > .01 bits is `SCREEN_UNPROMISING`; otherwise `SCREEN_INCONCLUSIVE`. The fixed .01 material-loss threshold makes the published qualitative rule executable. MiniScreen stops for review if both lose >= .05 Contrast and worsen loss by > .02 bits. These are resource-allocation heuristics, not statistical significance claims.

Full CV requires a currently verified promising Screen **and** explicit owner `-AllowFullCV`; it is never run automatically. All sixty fold/seed/condition checkpoints are required, reusing the sixteen screen fits. Competition gates retain ensemble gain over R0 >= .003, gain over matched EF0 >= .002, >=3/5 fold wins and fold-mean Macro Log Loss <= R0 + .005. Even a qualifying gate does not authorize confirmation/Development scoring or upload in EXP-022.

Any failed/interrupted verification revokes the current successful decision. Checkpoints are published only after a complete selection/refit/prediction; interrupted current fits restart, completed models are reused. Source/configuration/runtime/input identities and checksums must match. Different hardware gets its own run directory and fresh Preflight/Pilot; completed checkpoints are not silently mixed across runtimes.

## Execution record

**PREFLIGHT_PASS / PILOT_PASS / RESOURCE_HANDOFF. No real MiniScreen, Screen or CV checkpoint was trained. No screening decision or held-out EXP-022 score exists.** E50+L stays production; backbone and auxiliary effects on Contrast remain unanswered. Finite gradients or a completed pilot do not establish improvement.

Executed source: `f0d2ac65d875a0875e8b280bc6065545611198e3`. [Unmerged stacked PR #21](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/pull/21), base `exp/021-vote-distribution`. Raw sealed outputs remain in `runs/exp022_visual_lora/`; the committed [aggregate feasibility record](exp022_feasibility.json) contains exact versions, metrics and manifest hashes. Later report-only commits do not change the executed source fingerprint.

### Numerical and gradient verification

Sixteen deterministic, label-independent sampled inner-training artworks passed live/cache parity: maximum absolute error **2.384185791e-7**, minimum cosine **0.9999999404**, maximum unit-norm error **1.192092896e-7**. Zero adapters reproduce the original live model **exactly**, with maximum absolute error zero. The earlier cuDNN-disabled diagnostic failed at 8.7478e-4; restoring the historical convolution setting fixed it without changing preprocessing, weights or thresholds. `engineering/preflight_v1/` and the failed/backend diagnostic JSONs preserve the engineering history.

All four initial head hashes equal `030a8bf9dea54431eb747c4d30a504a6338a364f450ae62512b50d86efd9bb1e`. The frozen visual tower has 87,849,216 parameters / 351,396,864 bytes. L0/L8 train 294,912 adapters plus 133,986 head parameters; F0/F8 train the head only. All 24 target names are sealed in `preflight/target_modules.json`. Pretrained-base digest before/after diagnostic and pilot updates: `bcc1248fea4f234458dd994556d8970f12ce2814185c9db0357b6198aa011f67`.

After three real-image updates, L0/L8 mean embedding L2 shifts from the original cached projection were 0.032738 / 0.032711; frozen shifts remained 8.24e-7 numerical error. Initially A-matrix gradients were zero as expected, while B gradients were nonzero; subsequent A gradients became nonzero. BCE and normalized CE independently reach adapters, including BCE-only conditions where CE is diagnostic and not optimized. Frozen SigLIP2/CLIP-L feature blocks remain detached; their encoders are never instantiated for training. Every pinned original SigLIP2/CLIP-L config/processor/weight asset was independently rehashed afterward and matched its historical pin.

### One full inner-training epoch per condition

Fold 1, seed 2027: 961 inner paintings / 9,759 pairs, plus 107 stopping paintings. Each condition used the same 241 four-painting optimizer steps, with a final partial effective step. One complete epoch per condition is within the requested three-epoch pilot cap. Values below are **optimization diagnostics in nats**, not outer-fold Contrast or Log Loss results.

| Condition | Epoch seconds | Paintings/s | Pairs/s | Train BCE | Inner stopping BCE | Adapter update L2 | Clipped steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F0 | 2.9735 | 323.19 | 3282.02 | 0.568661 | 0.532111 | 0 | 17.01% |
| F8 | 2.5208 | 381.22 | 3871.32 | 0.568416 | 0.530159 | 0 | 19.50% |
| L0 | 20.9428 | 45.89 | 465.98 | 0.568475 | 0.530361 | 1.307534 | 17.43% |
| L8 | 20.4801 | 46.92 | 476.51 | 0.568223 | 0.528345 | 1.313659 | 20.75% |

F8/L8 mean training CE8 was 1.783580 / 1.783263 nats. All losses/gradients were finite, all adapter updates were nonzero where enabled, frozen adapters remained unchanged, and **zero pilot AMP retries** occurred. There is only one epoch of pilot behavior; no convergence or multi-epoch quality conclusion is drawn. Both live models produced varying image-dependent predictions. Peak memory was nearly identical across the two LoRA passes; no runaway allocation was observed during these epochs. Longer-run leak behavior remains unmeasured until screening.

On the fixed post-pilot training diagnostic batch:

| Condition | Head BCE gradient norm | LoRA BCE norm | LoRA CE8/ln(8) norm | Scaled auxiliary/BCE ratio | LoRA gradient cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| F0 | 1.081811 | N/A | N/A | N/A | N/A |
| F8 | 1.096147 | N/A | N/A | N/A | N/A |
| L0 | 1.001293 | 0.289682 | 0.048746 | 0 (disabled) | 0.595889 |
| L8 | 1.005366 | 0.326324 | 0.192108 | 0.058870 | 0.965534 |

These gradients describe one training batch after one epoch, not seed stability or generalization. Full per-adapter initial/three-step gradients are in preflight; screening fits/replay collect final per-seed diagnostics and curves when actually run.

### Actual resource measurements and scheduling decision

GPU: RTX 4060 Laptop, **8,585,216,000 bytes** total VRAM; **7,451,181,056 bytes** free before model load. Four paintings passed the one-GiB safety margin immediately, so sizes two/one were not retried. No checkpointing or rank/resolution change was needed. Maximum pilot allocated/reserved memory: **546,871,808 / 616,562,688 bytes** (0.509 / 0.574 GiB). Peak process RSS: **1,960,288,256 bytes**. L8 AdamW state: **3,431,416 bytes**; F0 state: 1,063,664 bytes. These are measured process/allocator values, not estimates from parameter size.

L0/L8 full-epoch wall-time components (seconds): image decode/processor/copy enqueue 8.006/7.893; encoder forward 4.618/4.378; head forward 0.596/0.721; head backward 0.822/0.812; encoder backward 5.034/4.848. Remaining time is optimizer, synchronization and loop overhead. Backward boundaries are measured with a graph hook and CUDA synchronization without detaching features. Cached frozen controls perform no encoder inference or image decoding. Unseen-image inference would require the original three encoders plus the adapter/head; this experiment adds no new production inference deployment.

The measured full epoch—not startup microbatches—drives the forecast. Historical EXP-021 selected-epoch p90 is 33; planning uses 43 stopping epochs, 33 fresh refit epochs, the measured stopping-pass cost, full-fit image count 1,068, diagnostics/checkpoint allowance and a 25% timing margin. Per-model forecast: F0 5.18 min, F8 4.38 min, L0 36.19 min, L8 35.31 min. **MiniScreen: 4,863.89 s / 81.06 min. Screen: 19,455.55 s / 5.4043 GPU-hours total**, including MiniScreen; incremental after it 4.0532 hours. A 100-selection-plus-100-refit-epoch ceiling is 213.05 min / 14.2035 hours respectively. Grouped optimization may select very different epochs; neither estimate is a guarantee. Independent full screening replay has not been timed; its separate runner budget is explicit.

The 81-minute MiniScreen exceeds the 15-minute automatic limit, and the projected Screen exceeds the four-GPU-hour local limit. **Handed off, not launched.** Calling MiniScreen without an explicit long budget returned stage exit 3 before any fit; zero real screening checkpoints exist. No hyperparameters were changed based on a screening score. The best next step is the same frozen MiniScreen/Screen on a faster CUDA runtime with a fresh measured pilot, rather than a local long run or a learning-rate search.

### Validation and artifact preservation

Successful final commands: PowerShell Preflight (36.06 s process / 29.48 s stage), Pilot (79.93 s process / 73.36 s stage), `python -m pytest -q` (**197 passed / 1 skipped**, 62.21 s), and `python scripts/package_exp022.py` (full ZIP CRC, entry coverage and per-entry SHA checks). The new nineteen tests use a tiny random local twelve-block HF vision model, synthetic counts and an actual nested four-condition fit/replay; no weights/dataset download or expensive training job is part of the suite. Revision, zero initialization, frozen blocks/base, normalization, grouped reuse, accumulation weighting, positive/negative auxiliary paths, RNG, AMP overflow retry, memory failure, unknown languages, identity invariance, resume, corruption and gates are exercised. Real-image preflight supplies the integration check absent from dataset-free CI.

[GitHub source PR CI](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/actions/runs/43436145097): **196 passed / 2 skipped**, success; the local production artifact and CUDA-only fixture account for environment differences. No new production inference code, formula, original training configuration or prediction file changed. All **2,483** snapshotted historical artifact/cache files match before/after; the three frozen checkpoint asset sets match their historical pins. Exact E50+L CSV SHA-256 remains `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3` (849 rows).

Sealed input / preflight / pilot manifest SHAs respectively:

- `fa2f7c4528b49925c400216037a64381bc6a0b35a1b3b11458a6bcbfda924337`
- `fe6daa6893a31307b88bcd6fac863ee30c8a896bec2e8c2185648c23eaa8d760`
- `72cfbb8908271bdfbfda28a26f23d88b1f20ee92b19c89365b95a39ad0ab3a64`

Prepared handoff: `runs/exp022_visual_lora/exp022_colab_inputs.zip`, **705,080,786 bytes**, SHA-256 `bbedbe2ad7e426040aebb63366ea5978e03d6483503b8ac9c88af14238a3d156`. This is an execution-input bundle, not a Codabench submission. No submission ZIP was created.

After privately transferring/extracting that validated bundle and preparing the pinned environment as in the Colab runbook, the **exact next experiment command** is:

```bash
.venv/bin/python scripts/run_exp022.py Preflight --portable-dir /content/exp022-inputs --input-manifest-sha256 fa2f7c4528b49925c400216037a64381bc6a0b35a1b3b11458a6bcbfda924337 --run-dir /content/drive/MyDrive/exp022_colab --device cuda --max-seconds 600
```

Then run the matching Pilot and inspect its forecast before authorizing MiniScreen. Do not infer a LoRA benefit, auxiliary benefit, interaction effect or competition promotion from this feasibility record.

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
