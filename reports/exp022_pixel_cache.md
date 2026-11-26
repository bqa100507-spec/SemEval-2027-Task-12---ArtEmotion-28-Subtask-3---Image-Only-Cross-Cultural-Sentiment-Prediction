# EXP-022 — Exact deterministic CPU pixel cache

Execution source: `eb4a7fac4ebe79971305b0b3de2c0e5102ea5f91`, preserving original execution `41ca0d4` and all earlier commits. Scientific config, CLIP checkpoint/processor, architecture, loss, learning rates, LoRA, folds/seeds, epoch/step grouping, rounding and gates are unchanged. Source changes are CPU pixel reading/caching, execution identity, profiling and a diagnostic stage. Cache mode/cap/reserve enter the run identity. Existing source/configuration seals are not bypassed; old models cannot be silently reused under the new identity.

## Actual measured result — native RTX 4060, not Colab L4

The owner's L4 baseline (about 44 paintings/s, 5.45-hour Screen forecast) is user-reported context. No optimized L4 run was executed from this environment. The local benchmark below establishes exactness and a measured optimization on the available GPU; **the L4 four-hour decision remains pending its own benchmark**.

Two full epochs per reader/condition, each reset from the identical seed-2027 initialization, were counterbalanced original→cached then cached→original. Every pass uses fold-1 inner training: 961 paintings, 9,759 language rows, 241 original effective-four-painting optimizer steps. This is throughput diagnostics, not held-out model fitting or a new scientific comparison. No MiniScreen, Screen, full CV, confirmation or Development scoring ran. Profiling instruments both paths identically with transfer synchronization; timings include this instrumentation.

| Condition | Original epochs (s) | Cached epochs (s) | Median original/cached (s) | Speedup | Median cached paintings/s |
| --- | --- | --- | --- | ---: | ---: |
| L0 | 16.721, 15.849 | 9.608, 10.039 | 16.285 / 9.823 | 1.6578× | 97.83 |
| L8 | 16.039, 16.200 | 10.123, 10.070 | 16.119 / 10.096 | 1.5966× | 95.18 |

F0/F8 were also timed with identical resulting states; they never read pixels and show ordinary timing variation rather than an image-cache benefit. Cache creation took **9.913 seconds**, separately recorded, not included in epoch speedups.

| Median L0 full-epoch component | Original seconds | Cached seconds |
| --- | ---: | ---: |
| JPEG decode / RGB conversion | 1.885 | 0 |
| Official CLIP preprocessing | 4.503 | 0 |
| CPU tensor lookup/copy | 0 | 0.116 |
| Synchronized CPU→GPU transfer | 0.137 | 0.087 |
| Encoder forward (including frozen block retrieval) | 3.613 | 3.423 |

Encoder computation, head/backward and optimizer algorithms are unchanged; small encoder-time differences reflect runtime variation. The main removed work is repeated decoding/preprocessing. Full encoder/head forward/backward timings, per-pass process/GPU memory, losses and state hashes are in [the aggregate record](exp022_pixel_cache.json).

Using the unchanged p90 historical stopping/refit scheduling method and 25% timing margin:

| Forecast on this RTX 4060 | Original | Cached |
| --- | ---: | ---: |
| MiniScreen | 61.526 min | 39.669 min |
| Limited Screen total | 4.1017 GPU-hours | 2.6446 GPU-hours |

These are forecasts derived from measured complete epochs, not completed screening times. The one-time cache cost is recorded separately; original scheduling allowances/margin remain unchanged. Both MiniScreen estimates exceed the automatic 15-minute policy, so neither was launched. The four-hour gate was never relaxed. Do not multiply the user-reported L4 baseline by these ratios and call it a measured L4 result. The updated cloud notebook repeats all measurements on L4.

## Exactness and memory

All **1,335** permitted original-training artwork pixel tensors were compared with the uncached official processor path, plus singleton/batch-boundary checks. Equality is bitwise: FP32, 3×224×224, identical resize/center crop/rescale/normalization. No augmentation, half-precision cache, GPU activation cache or feature normalization change is used. CPU cache lookup returns a copy, so callers cannot mutate the bank. Images/targets remain governed by original input/image checksum checks.

Before timing, separate BCE and normalized CE8 gradients matched bitwise, as did parameters, clipped gradients, AdamW state, GradScaler state and CPU/CUDA RNG after three real LoRA updates. After every complete timed epoch, all task-head/adapter parameters, final gradients, optimizer/scaler and RNG states matched across cached/uncached and repeats for **all four conditions**. This checks dropout draw order, gathering, votes, step counts and real gradient flow rather than merely comparing losses. Frozen visual/projection SHA remained `bcc1248fea4f234458dd994556d8970f12ce2814185c9db0357b6198aa011f67`. SigLIP2/CLIP-L banks remain detached and unmodified.

The RAM cache uses **803,819,520 bytes (0.749 GiB)**. Default safeguards require a 2 GiB cache cap and 1 GiB available reserve plus 64 MiB scratch, checked before allocation and during filling. Defaults initially refused the memory-constrained 16 GiB native machine. An explicit **256 MiB CPU reserve** was then recorded for the local measurement; a concurrent test process caused another clean guard stop during construction. After tests exited, the isolated benchmark completed with that reserve. No OOM was forced, no cache was partially published, and no silent fallback occurred. Peak measured process RSS was **2,737,438,720 bytes**. Colab/Kaggle retain the safer 1 GiB default reserve. This CPU reserve is distinct from the original GPU safety margin and has no effect on scientific computation.

## Reproducible execution

Run from the new source with a **new** run directory and the original data bundle:

```powershell
.venv/Scripts/python.exe scripts/run_exp022.py PixelBenchmark --portable-dir runs/exp022_visual_lora/portable_bundle --input-manifest-sha256 fa2f7c4528b49925c400216037a64381bc6a0b35a1b3b11458a6bcbfda924337 --run-dir runs/exp022_pixel_cache_256m --device cuda --pixel-cache-margin-gib 0.25 --max-seconds 900
```

That exact successful command used uncached context setup and internally compared both readers. `runs/exp022_pixel_cache_256m/pixel_benchmark/` contains sealed results; its source fingerprints were checked against the committed working tree after execution. Its recorded Git parent is `56c1524` because the benchmark ran before committing the execution changes; the captured file hashes, verified afterward, match `eb4a7fa`. Failed capacity checks created no success manifest.

For actual cached execution, use `--pixel-cache ram` consistently in Preflight, Pilot and subsequent owner-invoked stages, with a separate new run directory. Cache-off and cache-on identities deliberately differ. Default cloud workflows perform Preflight first, strict paired PixelBenchmark second and cached Pilot third; no long stage is selected automatically. Advanced original PowerShell stages remain unchanged; new cache flags are available through the cross-platform Python runner.

## Notebooks, tests and preservation

Both updated notebooks pin execution `eb4a7fa`, helper `c3322759c1a216c69079cbdad1999f4bdaeff7ca` / SHA `c93e4df9997a68700773bca3dca4799e2f48c60e3e86e78180f6ad90c67ceb1e`, snapshot `0b4f0b89bc8835a11239664537a761c001238094`. The original input ZIP SHA is unchanged. Existing source/checkpoint identities and offline runtime kits remain archived; offline kits must be rebuilt to include the new source/helper. Old checkpoint restores are rejected. Scientific model/configuration pins are not replaced.

The user's locally edited Colab notebook contained execution outputs/metadata but no changed cell source. Its exact bytes were backed up before regeneration to `runs/exp022_notebook_validation/user_colab_before_pixel_cache.ipynb`, SHA `42d10d2f62cb08a7371310ca621b71360a473cfe350d289768e2ff10c58fbb8d`. The new committed notebooks have clean outputs; the user's record remains local and intact.

Validation so far: full native suite 226 passed / 1 skipped before the final workflow-order regression; final targeted pixel-cache/notebook suite **30 passed**, including memory rejection, immutable lookups, RGB/processor bitwise equality, gradient/update/dropout RNG equality and blocking Pilot after a failed pixel audit. Both notebook schemas and syntax passed. Final native suite: **227 passed / 1 skipped**, 141.44 seconds. Notebook code [CI](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/actions/runs/43575147092) passed schemas and the full CPU suite; final delivery checks are on PR #21. No local test is described as a Colab/Kaggle GPU run. E50+L remains production.
