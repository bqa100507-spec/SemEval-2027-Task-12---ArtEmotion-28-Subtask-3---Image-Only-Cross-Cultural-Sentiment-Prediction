# EXP-040 Kaggle handoff — E3-LoRA

Status: **prepared, not run.** Owner approval 2026-11-28 covers implementation, tests and packaging only. The owner runs the Kaggle smoke first, then the full Run All. The analysis side runs the Linux preflight on the exact upload ZIP before any upload.

**Patch 1 (2026-11-28, owner decision):** two-GPU fold parallelism and a vision-tower cache. Both are efficiency changes only. The model, data, seeds, schedule, selection rule and gate are unchanged. The patch-1 overlay replaces the first one (`c386ad41…`, superseded and never uploaded).

## What runs

E3 (Qwen/Qwen3-VL-Embedding-2B @ `9f2f7e7…`) with its exact instruction, pixels and last-token pooling. LoRA (rank 16, alpha 32, dropout 0.05) sits on q/k/v/o of all 28 decoder layers, 6,422,528 parameters; everything else is frozen.

The head is the Track 1 E3 component written as one linear layer (51,225 parameters). It starts at the frozen E3 ridge fitted on the same training paintings and is anchored to it with λ = 1e-3. The rationale for λ is recorded in `configs/exp040.json` (`train.lambda_rationale`).

Training settings:
- Loss: binomial NLL on counts.
- Optimizer: AdamW, LoRA lr 5e-5 (weight decay 0.01), head lr 5e-4 (no decay).
- Schedule: the EXP-036 schedule (5% warmup, then cosine to a 0.1 floor), 4 epochs.
- Batch: 16 paintings per step, with one forward per painting scoring all its eligible languages.
- Numerics: FP16 autocast with GradScaler, gradient checkpointing, deterministic kernels. Each model sits whole on one GPU (`device_map {'': 0}` in a child that sees only that GPU); it is never split.

**Epoch selection.** Inside each P2 outer training fold, a painting-grouped 10% inner split (seed 204002 + fold) picks the epoch in {0,…,4} with the best inner test-like Contrast; ties go to the smaller epoch. Epoch 0 means no adaptation, so E3L becomes the frozen E3.

**Deployment refit.** The full-train refit uses the median of the five selected epochs and the same 4-epoch schedule, stopped after that epoch.

**Gate against the C2 recomputed in the run** (P2 test-like). Candidates:
- F1 = 0.5 Fixed + 0.5 mean(E1, E2, E3L)
- F2 = 0.5 Fixed + 0.5 mean(E1, E2, E3, E3L)

Both use the unchanged neighborhood correction. E3L's adapted embedding enters the similarity space, and inside the bank cross-fit E3L is a Track 1 ridge on the adapted embeddings.

Gate conditions:
- Δ ≥ +0.003.
- Bootstrap lower bound > 0 (20,000 resamples, seed 204003).
- At least 4 of 5 fold wins.
- Positive after the 5% trim.
- No full-P2 Contrast regression.
- Log Loss regression ≤ +0.005 bits in both scopes.

Selection takes the highest passing candidate; within 0.001, F1 wins. Otherwise the result is NO_GO, and `exp040_best_submission.zip` is the byte-identical incumbent `62093a31…`.

If E3L fails technically, or the forecast exceeds the session, the run still exports a NO_GO report with the reason. It never shrinks the protocol.

## Two-GPU fold parallelism (patch 1)

The five P2 outer folds are independent. Each fold runs as its own child process, pinned to one T4 with `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The child asserts that it sees exactly one GPU and loads the whole model on it.

- **Queues.** One worker thread per GPU. Each takes the next unsealed fold, in fold order, when its GPU is free. With equal fold times that gives GPU 0 folds 1, 3 and 5, and GPU 1 folds 2 and 4.
- **Sealing and resume.** A fold's result is sealed in its own folder, with an identity that includes the fold and the cache setting but not the GPU. A fold sealed on either GPU is reused on resume. An in-progress fold resumes from its `state.pt` on whichever GPU takes it.
- **Failure.** A technical failure in one child stops new work and kills the other child; that child's latest checkpoint stays resumable. E3L then becomes a reported `TECHNICAL_FAILURE` NO_GO, as before.
- **Timeout.** A session timeout stops new work. The worker waits for every child to stop at the shared deadline, then stops resumably.
- **Progress exports.** While children run, periodic progress snapshots leave out fold folders that are not sealed yet, so that a snapshot never pairs a newer `state.pt` with older epoch files. The final export after every child has stopped includes everything.
- **Full refit.** It needs the median of all five selected epochs, so it runs after the folds, on GPU 0 alone.
- **Logs.** Each child's lines are relayed once, prefixed `[gpu0]` or `[gpu1]`.
- **One GPU.** If only one GPU is visible, the same code runs one queue, and the forecast says whether that fits.

**Forecast.** For each fold, the per-fold seconds are computed from the slowest GPU as measured in the concurrent benchmark. The folds are then laid out in the execution order above, and the fold phase is the longest queue. The total is that fold phase, plus the full refit (worst case 4 epochs, on one GPU), plus CPU and export. The total is multiplied by 1.15. If the result exceeds the remaining session, the run stops before training exactly as before.

**Fallback, documented only and not implemented.** If two GPUs still do not fit one session, the folds could be spread over two sessions: they are already sealed and resumable per fold. This would need a forecast rule that plans across sessions, and it needs a new owner decision.

## Vision cache (patch 1)

The vision tower and its mergers are frozen and stay in eval mode; LoRA sits only on the language model. So the vision outputs for an image are the same in every epoch. With the cache on, the first forward of an image computes them once, under the same FP16 autocast:
- the merged image embeddings;
- the three deepstack features;
- the processor's text inputs.

Later forwards inject these through `get_image_features` and skip image decoding and the vision tower. The cache is kept in CPU memory, about 3.8 MB per painting.

**Gate.** Validate runs the step-0 held-out pass on fold 1 three times: uncached, cached on first visit, and cached on revisit. The cache is enabled for the run only if both cached passes match the uncached probabilities within 1e-5, and the projected cache fits the configured limits:
- at most 12 GB per child;
- at peak (two fold children at once, or the full refit alone), at most 60% of available memory.

Otherwise the run continues uncached and records why.

Evidence:
- **Unit tests.** One test uses a fake backbone with the `Qwen3VLModel` contract. Another uses a random tiny real `Qwen3VLForConditionalGeneration` (transformers 4.57.6). The cached output is bitwise equal to the uncached output in both.
- **RTX 4060.** The real pinned 2B model was run on the 267 held-out paintings (receipt below).

## Inputs

| Dataset | File | SHA-256 |
| --- | --- | --- |
| Existing base (exempt) | `exp037_inputs.zip` | `eaf16b9301cb131fbdf61a1b86fb191d3617b2141e39398f0c3b12bcf57ffce2` (manifest `026720f4…`) |
| New overlay (patch 1) | `runs/exp040/handoff/exp040_inputs.zip`, 53,578,707 bytes, 243 files | `4f316e9da2bd5bec35c7f63f5c8446bda78456584b91d5a1110715f01ec3a497` |
| Overlay manifest | `exp040_manifest.json` | `c5d2834019fae58444d9c67d9457249589737dc3ad52126ab513ee3ae84a934f` |

The worker source is `53d2c52`, and the notebook is `notebooks/EXP040_Kaggle.ipynb`.

Two earlier overlays are superseded and were never uploaded:
- the first overlay (`c386ad41…`, manifest `e811330d…`, source `e7b9d43`), kept in `runs/exp040/handoff_c386ad4_superseded/`;
- an intermediate patch-1 build (`9e29c366…`, source `753771d`), kept in `runs/exp040/handoff_9e29c36_superseded/`. It was replaced before validation because its cache memory check multiplied the full-refit cache by the GPU count. The carried input members are identical; only `runtime/` changed.

The overlay carries every input member of the exact EXP-039 overlay (`4e918a8c…`):
- references and P2 folds;
- the sealed EXP-038 run-4 E1/E2/E3 caches, P2 OOF and Development predictions;
- the CS-CVA-3 and incumbent ZIPs;
- the C6 Development predictions;
- the E3 key-audit files.

The 8B audit files and the old runtime are dropped. Archive-like members are stored as `.b64`. Pins are generated by `scripts/make_exp040_config.py` from the committed EXP-039 config and the measured EXP-039 overlay hashes.

## Owner steps

1. **Upload.** Upload `exp040_inputs.zip` as a new private dataset and attach it together with the existing EXP-037 input. Select **GPU T4 ×2** (only GPU 0 is used) and **Internet on**.
2. **Smoke.** Keep `SMOKE_ONLY = True`, set `AUTHORIZE_RUN = True`, and click **Run All**. This takes about 30 minutes. Validate runs on GPU 0, then fold 1 trains on GPU 0 and fold 2 on GPU 1 at the same time, 30 optimizer steps each. Download `exp040_smoke.json`; it contains:
   - Validate statuses, the step-0 receipt and the vision-cache decision;
   - per GPU: throughput (first visit and revisit), peak VRAM, placement and the device identity;
   - finite losses and the two-pass determinism result on each GPU;
   - the parallel full-run forecast, with the two queues and the serial total for comparison.

   No submission is produced.
3. **Full run.** Do this only if the smoke passes and `forecast.fits_session` is true. Set `SMOKE_ONLY = False` and click **Run All** once. Save `exp040_results.zip`, `exp040_best_submission.zip` and their `.sha256` files, or `exp040_diagnostic.zip`; use `exp040_progress.zip` to resume.

## Pre-upload checklist evidence

| Item | Evidence |
| --- | --- |
| 1 No nested archives | `encode_members`/`refuse_archives` in the packager; every written member rechecked; `test_packager_refuses_archives_and_survives_recursive_expansion` |
| 2 Generated pins | `make_exp040_config.py`; `check_config_pins` before packaging and in Validate |
| 3 Measured tolerances | the EXP-039 `replay_tolerance` (probability 2e-5, pooled 1e-6, heads/bank 1e-4); receipts carry `exact`/`max_abs_difference`/platform |
| 4 Preflight from the exact ZIP | Windows `cpu-validate` in both layouts (below); **Linux run pending on the analysis side** |
| 5 Collect-all Validate | pins, inputs, snapshot, sealed caches, key audit, CPU head check, GPU step-0, R0 and C2 replays |
| 6 Optional never kills the core | E3L failure → reported NO_GO with C2/R0 Development, ZIPs and the incumbent as best |
| 7 Single relay | EXP-039 launcher and `stream` |
| 8 CPU key audit | E3 checkpoint names vs `Qwen3VLForConditionalGeneration` on meta: PASS (receipt below) |
| 9 Real-hardware smoke + `SMOKE_ONLY` | RTX 4060 receipts below (first smoke and patch 1, one GPU); the two-GPU path runs first in the Kaggle `SMOKE_ONLY` |
| 10 Rerun only with value | stated in the report template; no automatic rerun |

## RTX 4060 smoke (checklist item 9, 2026-11-28)

### Patch 1: vision cache and the concurrent-smoke path (one GPU)

The owner's laptop has one GPU, so two-GPU concurrency itself is covered by tests only, not by real hardware:
- the job scheduler and GPU pinning;
- cancel and timeout handling;
- per-line relay;
- progress snapshots.

On the 4060, the driver `runs/exp040/local_smoke_v2.py` ran the worker's own `step0()`, `vision_cache_decision()` and `concurrent_smoke()` on GPU 0, with the same runtime as below. Receipts:
- `runs/exp040/local_smoke_v2/receipt.json` (cache on);
- `runs/exp040/local_smoke_v2_nocache/receipt.json`, an uncached control of the same 30 steps run straight after with nothing else on the machine.

**Vision-cache check on step 0** (fold 1, 267 held-out paintings): the cached passes (first visit, then revisit) give **max |Δp| = 0 and bitwise-equal embeddings** against the uncached pass. The cache holds 3.85 MB per painting. The cache is kept.

Training, 30 optimizer steps × 16 paintings on fold 1:

| | Uncached control | Cached |
| --- | --- | --- |
| First-visit step | 21.2 s | 21.8 s |
| Revisit step (cache hits) | 21.4 s | 18.6 s |
| Inner forward, first visit | 2.48 paintings/s | 2.52 paintings/s |
| Inner forward, revisit | 2.58 paintings/s | 4.02 paintings/s |
| Peak VRAM allocated | 4.52 GB | 4.52 GB |

Both runs:
- Losses are finite, the eval forward is bitwise repeatable, and the two-pass training check is bitwise identical.
- All 30 losses are **bitwise identical between the cached and uncached runs**. The first 20 also equal the original smoke below.

So the cache changes nothing numerically. Its gain is modest: revisit training steps are 13% faster and revisit forwards 56% faster, while first-visit steps are 3% slower. The LLM forward and backward with gradient checkpointing dominate the step. Today's uncached step (21.2 s) is slower than the 19.2 s of the first smoke; the laptop's speed varies from run to run by about 10%.

The cache for fold 1's first 589 paintings took 2.26 GB of CPU memory, with a largest entry of 4.20 MB. Validate projects the cache from the largest entry:
- each fold child holds up to 1,335 paintings, about 5.6 GB;
- the full refit holds 1,656 paintings, about 7.0 GB.

The peak is the larger of two fold children at once (about 11.2 GB) and the full refit alone (about 7.0 GB). Validate checks that peak against 60% of `MemAvailable` on Kaggle.

### First smoke (2026-11-28, before patch 1)

The exact GPU child (`run_exp040.py gpu-e3l`) ran on the owner's NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) with the pinned snapshot and real training images. Runtime: transformers 4.57.6, qwen-vl-utils 0.0.14, torchvision 0.22.1+cu126 and torch 2.7.1+cu126, all from the Kaggle lock, installed as target directories under `runs/exp040/env`. Receipt: `runs/exp040/local_smoke/receipt.json`.

Step-0 check on P2 fold 1 (267 held-out paintings, 2,764 pairs):
- Max |Δp| is 4.5e-8 against the 1e-5 tolerance, with the LoRA right factor exactly zero.
- Placement is cuda:0 only, with no missing or ignored checkpoint keys. Load takes 26 s.
- Against the sealed E3 (a different forward: batch 8 without autocast, against batch 1 under FP16 autocast), the minimum cosine is 0.99977, the max embedding difference is 1.9e-3 and the max |Δp| is 6.9e-3. This is reported, not gated.

Training, 20 optimizer steps × 16 paintings, batch 1 with accumulation:
- Warm step time is 19.2 s, or 0.82 training paintings/s.
- Peak VRAM is 4.52 GB allocated and 4.57 GB reserved of 8.59 GB.
- Inputs, hidden states, outputs and parameters are all on cuda:0.
- All losses are finite (0.35–0.57 nats per vote). Gradient norms are 0.24–1.03, under clip 1.0 except one step.
- The eval forward is bitwise repeatable, and two identical 2-step training runs gave identical losses and weights (max difference 0).

Inner pass: 107 paintings at 2.4 paintings/s.

### Forecast method and what it says

The forecast for each piece is:
- **Each fold:**
  - load time;
  - ⌈fit/16⌉ × (one first-visit epoch + three revisit epochs) of training steps;
  - the inner passes (epoch 0 at first-visit speed, epochs 1–4 at revisit speed);
  - the embedding pass over all fold paintings (held-out paintings at first-visit speed);
  - the held-out probability pass.

  Revisit rates equal first-visit rates when the cache is off.
- **Fold phase:** the folds are laid out in execution order over the GPU queues, and the phase lasts as long as the longest queue. Rates come from the slowest GPU measured while both run at once.
- **Full refit:** runs after the folds on GPU 0, with the worst case of 4 epochs over the 1,571 training paintings, plus the train and Development embeddings.
- **CPU:** 2 × the measured on-platform C2 replay, plus 600 s.
- **Export:** 600 s.

The run multiplies the total by 1.15 and stops before training if the result exceeds the remaining session (10.5 h minus a 1,200 s reserve, so about 10.17 h usable).

**At today's RTX 4060 rates**, applied to two GPUs of that speed, with CPU at the 900 s default:

| | One GPU | Two GPUs |
| --- | --- | --- |
| Cache on | 12.17 h | **8.73 h** |
| Cache off | 13.59 h | 9.71 h |

All figures include the ×1.15 margin. With the cache on and two GPUs, the queues are:
- GPU 0: folds 1, 3 and 5, 16,148 s;
- GPU 1: folds 2 and 4, 10,766 s.

The full refit adds 8,193 s, then CPU and export.

So at 4060 speed the parallel plan fits one session with about 1.4 h to spare. This is still a projection, not a T4 measurement: a T4 may be slower than the laptop 4060, and two concurrent children share Kaggle's CPUs. The Kaggle `SMOKE_ONLY` run measures both T4s at once and reports `forecast.fits_session`. If it is false, the full run stops at the benchmark with `FORECAST_EXCEEDS_ALLOWANCE` and exports a NO_GO report without training. The protocol is never shrunk.

The two-session resume stays a documented fallback; it is not implemented (see the parallelism section).

## Windows cpu-validate (both layouts)

`scripts/linux_preflight.py --experiment exp040 --allow-non-linux` ran `cpu-validate` on Windows 11 from the patch-1 overlay's own runtime (`53d2c52`), using the notebook's `locate` code and transformers 4.57.6 for the meta-device key audit. It ran in two layouts: the intact ZIP and a simulated recursive Kaggle expansion.

**Both layouts PASS** with identical results:
- **Replay:** all 46 R0/C2 tolerance checks pass. The worst cross-platform difference is 0.40 of its threshold; the checks are not bit-exact, as expected between Windows and the Linux-built references.
- **Key audit (E3):** PASS. The checkpoint has 625 names and the class has 626; the only difference is the tied `lm_head.weight`.
- **CPU head check:** the ridge-initialized linear head reproduces the frozen ridge on fold 1's 2,764 held-out pairs within 1.06e-7.

Receipt: `runs/exp040/handoff/windows_cpu_validate.json`. The Linux preflight (checklist item 4) remains for the analysis side.

## Learning-curve diagnostic (CPU, report only)

The frozen C2 recipe was refitted on 25/50/75/100% of each P2 outer training fold. Each fraction below 100% used three seeded subsamples; 100% is a single run. The diagnostic ran locally on Windows from the packaged overlay with `run_exp040.py learning-curve`, in 2,981 s. It is not repeated on Kaggle: the inputs are the identical sealed features, and no GPU is involved.

| Fraction | Mean training paintings per fold | Filtered P2 Contrast (mean ± sd over seeds) |
| --- | --- | --- |
| 25% | 267 | 0.42350 ± 0.00527 |
| 50% | 534 | 0.44886 ± 0.00285 |
| 75% | 801 | 0.46060 ± 0.00175 |
| 100% | 1,068 | 0.46865 (equals the C2 P2 reference 0.468648) |

The slope of filtered Contrast against log N is **+0.0333 [+0.0283, +0.0378]** (bootstrap over the 10 runs). That is about +0.023 per doubling, with diminishing absolute gains (+0.025, +0.012, +0.008 per step).

The slope is clearly positive, so the frozen C2 system is still **variance-limited**: more effective labels would pay off, and it is not purely representation-limited. This is a descriptive diagnostic and gates nothing.

Receipt: `runs/exp040/learning_curve/learning_curve.json` (`095566fe…`), with rows in `learning_curve.csv`.

