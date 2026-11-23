# EXP-035 · Qwen3-VL cultural residual

## Frozen pilot

A is the sealed Corrected Ridge fold-3 reference. B adds a residual from one language-neutral Qwen3-VL-4B native decoder hidden state. C adds rank-8 LoRA on `visual.merger.linear_fc2`, the final image merger's output projection. The main vision blocks, three DeepStack mergers and decoder weights are frozen. The same 32-wide head emits a bounded log-odds correction for all fitting languages, with exact zero-output initialization and zero correction for unseen languages. No text is generated. The model uses the earlier verified NF4/FP16 checkpoint, official processor and a fixed neutral prompt. This compares merger adaptation against its own frozen residual control, unlike the earlier decoder-adapter and standalone VLM runs.

Fit on a deterministic four-fifths of the original P2 fold-3 fitting paintings. Use the other fifth for checkpoint selection. The 267 outer paintings enter once for the chosen checkpoint of each arm. Each fitting painting's Corrected Ridge offset comes from a three-way painting cross-fit whose inner nuisance fits and residual bank also exclude that painting. Inner stopping offsets use only the B/C fitting scope, and the sealed outer incumbent was fitted on the full original outer fitting partition. B and C fit on less data than the outer incumbent. Neither the final full-training Ridge bundle nor any production predictions are overwritten.

One painting per update, all observed languages in that update. Binomial BCE uses a fixed mean fitting vote mass per painting, preserving relative vote weights across updates. The centered probability error has unit weight and the log-odds residual mean-square penalty has weight 0.1. The correction is capped smoothly at ±0.5 log-odds. Exact 0 and 1 probabilities remain endpoints at inference. Only the BCE offset logit is clamped for finiteness. AdamW uses head/adapter rates 0.001/0.0001, 0.01 decay and norm cap 1. Four epochs, one seed. Choose highest test-like inner Cultural Contrast among epochs within +0.005 bits of inner incumbent Log Loss in both scopes. If none qualifies, use the minimum test-like inner Log Loss checkpoint for a diagnostic comparison and record guardrail failure. Ties prefer earlier epochs.

The local runner checks 24 fitting artworks with real votes, discards diagnostic updates and proceeds only if memory and the observed forecast plus 35% reserve fit the 55-minute cumulative GPU budget. Full CV requires separate authorization. Report paired painting uncertainty in both scopes and the advantage of C over B and A. A small, consistent advantage can justify further evaluation without requiring +0.005 Contrast.

```powershell
$env:PYTHONPATH = 'runs/competition_semantic_vlm/runtime;runs/competition_track2/transformers_overlay'
.\.venv\Scripts\python.exe scripts/run_exp035.py
.\.venv\Scripts\python.exe scripts/run_exp035.py --run-bounded-gpu
```

Without the GPU flag the runner only prepares and verifies the isolated CPU offsets. With it, the runner uses a time-limited worker, seals completed epochs and evaluates outer predictions once after inner selection. Reuse the same run directory to resume at the latest complete epoch with remaining budget. Artifacts are local under `runs/exp035_cultural_residual/`.

## Measured outcome

**Technical PASS / scientific pilot blocked by its budget.** Executed source `ee1837a1c2b63c13dcdaeb468d0ab1f3b9747d04`. CPU preparation sealed **854 fitting paintings / 8,673 pairs**, **214 inner-stopping paintings / 2,206 pairs**, and **267 outer paintings / 2,721 pairs**. All outer offsets matched the sealed incumbent row for row. The matched incumbent reference remains **0.437806 / 0.710339** on full fold 3 and **0.441599 / 0.721110** on test-like fold 3 (Contrast / Macro Log Loss). These are existing OOF measurements. B versus A, C versus A and C versus B remain unmeasured.

| Real-image RTX 4060 Laptop preflight | Measured |
| --- | ---: |
| Fitting artworks, one neutral forward per artwork | 24 |
| Peak CUDA allocated / reserved | 4.49 / 4.63 GiB |
| Frozen forward, including image I/O and preprocessing | 2.54 artworks/s |
| Live merger forward, backward and update | 1.22 artworks/s |
| Head / merger adapter parameters | 87,897 / 53,248 |
| First / second / maximum merger gradient norm | 0 / 0.0000168 / 0.0001018 |
| LoRA left / right matrix update L2 norms | 0.105436 / 0.085946 |
| Total charged GPU worker time | 65.47 s |

The exact zero-head probabilities, initial hidden-state parity, finite objectives and frozen parameter isolation passed. The first update trained the zero head before gradient reached the merger on the second step. Both LoRA matrices moved after the short diagnostic. No diagnostic checkpoint or loss curve is a held-out model result.

The four-epoch two-arm fold projects to **62.7 minutes** before overhead, **84.7 minutes** with the preregistered 35% reserve. The runner stopped before B or C training because that exceeded its **55-minute** cumulative cap. A separately authorized approximately **90-minute single-fold pilot** with this frozen scientific recipe would answer the matched B/C question. Five-fold expansion is not recommended on technical evidence alone. Require a positive held-out C−B and C−A test-like effect, reasonable paired uncertainty and full-scope calibration before investing in all folds.

The preflight receipt is `runs/exp035_cultural_residual/preflight_1795352743924805600/result.json` (SHA-256 `09b96d52508b0b145aff18a973f2abf24454749dca6b07ad40fc04a0195fea0f`). The prepared offset manifest, GPU time ledger and exact run log are in the parent directory. E50+L, Fixed Ridge heads and the Corrected Ridge production ZIP and bank matched their existing SHA-256 digests.

## Authorized 90-minute continuation

**Technical PASS / PILOT_EXCEEDS_MEASURED_BUDGET.** The original runner already cached B's hidden state once per artwork in CPU RAM and reused it across epochs. The continuation kept the same loss, optimizer, four epochs and fold protocol. On all 24 fixed fitting artworks, independently recomputed frozen features matched the cached tensors bit for bit at the original dtype after a CPU round trip. C used a live Qwen forward. Its 53,248 trainable adapter parameters were isolated to the intended merger projection. The zero-correction check passed, the maximum merger gradient was 0.0001018, both adapter matrix norms changed (0.105436 and 0.085946), and the diagnostic updates did not clip. These temporary probe updates are separate from the scientific pilot.

| RTX 4060 Laptop measurement | Continuation |
| --- | ---: |
| B cold cache fill including image I/O and preprocessing | 21.202 s / 24 artworks (1.13 artworks/s) |
| B warm cached state transfer and head update | 1.342 s / 24 artworks (17.88 artworks/s) |
| C live forward, backward and update | 30.631 s / 24 artworks (0.78 artworks/s) |
| Peak GPU allocated / reserved | 4.48 / 4.60 GiB |
| B / C estimated times before reserve | 23.9 / 89.2 min |
| Combined forecast before / after 35% reserve | 113.1 / 152.7 min |
| Remaining at decision / required including shutdown margin | 86.6 / 153.2 min |
| Time charged to continuation / cumulative with initial preflight | 146.22 / 211.69 s |

The continuation's live paths were slower than in the original preflight on the same sample. The original frozen forward took 9.464 s and the live adapter update took 19.631 s for the 24 artworks. Both model load and live training steps slowed in the continuation. The available telemetry does not isolate the cause. The forecast uses this run's measured rates. C accounts for about 79% of the unreserved forecast, so removing all remaining B work would still leave C above the remaining budget at these rates.

The CPU replay regenerated byte-identical fit, stopping and outer CSVs. It reconstructed the three fitting offset-group receipts and the stopping receipt from the pinned P2 handoff. Each fitting query's painting is outside its Ridge head fitting set and all nested residual-bank fits. The inner-stopping offsets only use the 854 fitting paintings. Outer A matched the sealed fold-3 OOF predictions on all 2,721 pairs. The protocol selects checkpoints on the 214 stopping paintings and reserves the 267 outer paintings for the selected-checkpoint comparison.

| Arm | Full fold-3 Contrast / Macro Log Loss | Test-like fold-3 Contrast / Macro Log Loss |
| --- | ---: | ---: |
| A · Corrected Ridge | 0.437806 / 0.710339 | 0.441599 / 0.721110 |
| B · Frozen Qwen residual | Unmeasured | Unmeasured |
| C · Qwen merger LoRA residual | Unmeasured | Unmeasured |

The worker exited at the revised budget gate before scientific epoch 1. There are no scientific loss curves, selected B/C checkpoints, paired B−A/C−A/C−B deltas or held-out B/C intervals. This is a runtime result. The predictive question remains open. All 35 focused CPU tests passed, covering EXP-035, neighborhood residuals, the shared pilot helpers, scoring and submission validation.

Executed source `bffef12895c8f7abdc5c343d7979f371a8f8e287`. The continuation's prepared manifest, log, budget ledger and status are under `runs/exp035_fold3_pilot/`. Its preflight receipt is `preflight_1795353621822401000/result.json` in that directory, SHA-256 `8e984884df92f80f35996544e3f38bdb6dbcdab608ad2d717b97160bd0c616a9`. The E50+L Development CSV and both Ridge Development ZIPs still match the recorded hashes.

## Final Kaggle outcome

**COMPLETED · SCIENTIFIC NO-GO for five-fold expansion of this configuration.** The 23 November 2026 Kaggle artifact reports `FOLD_PILOT_COMPLETE`. Both B (frozen Qwen residual) and C (merger LoRA with the matched head) completed four epochs. Kaggle resolved the earlier local runtime blocker. A is the sealed Corrected Ridge comparator.

### Run identity and verification

The worker ran at `dc5aea2cff92835b59cae746075b6a11c9abf846` with Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Kaggle exposed two Tesla T4s (driver 580.178.04, 15,360 MiB each). All model tensors were placed on one T4 (compute capability 7.5), using NF4 double quantization and FP16 compute. The sealed runtime lists Linux x86-64 (glibc 2.39), Python 3.12.10, CUDA 12.6, Torch 2.7.1+cu126, Transformers 4.57.6, bitsandbytes 0.48.2 and Accelerate 1.6.0.

| Artifact | SHA-256 |
| --- | --- |
| Kaggle `exp035_results.zip` | `cf5bc8a0365c6cc5f135a6867984ae0e05236b0502549540c05c15b3cb038800` |
| Input handoff manifest | `599e255faa46565cd8225f05c9a81b1c1ba631bf4aba44bc99b89221e121e32a` |
| Prepared split manifest | `baa3c118856a4867af80f734a093024baccb8ff110913e5b2e6c02e9e0753e07` |
| Output `result_manifest.json` | `4e1194649bca2e5632b73aa810ca385a257bf4fd7cd88c86a709c43a0bdf1403` |
| Runtime dependency lock | `b562cb36ac14b55c422b43d63d26248885e40a25ce8ce7f1be33cbdbbd0d1812` |
| `results/predictions.csv` | `293865eeaae15d26b918218442c4fe7732fd5fd4c9cb65895cf5830b702caf4d` |
| `results/metrics.json` | `d1fa602ed8ea3975da970fd23a28320512fd1a750b744f893ff1fcc14f20855f` |
| B selected epoch-1 checkpoint | `648509acfcb90bdfcc661c2fb5aff42e818037e8d3558f8b76e3da00e93fe429` |
| C selected epoch-1 checkpoint | `8042f76faff49b23e1774664527475e601f050c2c2805a428a73ca942a5f6be0` |

The CPU audit verified ZIP integrity and all 34 manifest entries, matched the config and runtime lock to the pinned handoff, checked model pins in the saved receipts and verified all eight checkpoint hashes and their optimizer, scaler and RNG state. The optimizer had 854 steps per completed epoch. The audit reconstructed the 854 fitting, 214 inner-stopping and 267 outer paintings and verified disjointness, vote counts and the fitting and stopping offsets' group receipts. Outer A matched the sealed Corrected Ridge OOF file exactly on all 2,721 keys. B and C's prediction files match the outer seals that point to the selected checkpoints, with complete and unique keys and finite probabilities in [0, 1]. The ZIP and CPU audit receipt are retained outside Git in `runs/exp035_kaggle/closure/`.

Preflight verified exact zero correction, exact frozen-cache parity on its three parity samples, gradient flow to the merger, and frozen-weight isolation in the worker. On the 24-artwork timing sample, cold frozen throughput was 2.65 artworks/s, cached head-update throughput was 76.76 artworks/s, and live merger update throughput was 1.76 artworks/s. The forecast with reserve was 65.95 minutes. The worker ledger charged 3,156.85 seconds (52.61 minutes, including setup and the export allowance) against the five-hour ceiling. Summed epoch times were 6.26 minutes for B and 37.73 for C. Peak allocated / reserved VRAM during the epochs was 2.85 / 4.63 GiB for B and 4.52 / 4.72 GiB for C.

### Recomputed held-out fold-3 comparison

The CPU replay and a separate calculation from the saved probabilities reproduce the archived scorer outputs. Full fold 3 contains 2,721 pairs over 267 paintings, of which 253 score for Contrast (eight have too few languages and six have constant gold). The test-like filter removes English, Arabic and Chinese and leaves 2,330 pairs over 258 paintings. Of those, 252 score for Contrast and six have constant gold. Contrast is the mean within-painting Pearson correlation. Macro Log Loss uses base-2 binary loss with probabilities clipped to [0.01, 0.99], averaged first over pairs in each language and then over the 21 full / 18 test-like languages with at least 20 pairs.

| Arm | Full Contrast | Full Log Loss (bits) | Test-like Contrast | Test-like Log Loss (bits) |
| --- | ---: | ---: | ---: | ---: |
| A · Corrected Ridge | **0.437806** | **0.710339** | **0.441599** | **0.721110** |
| B · Frozen Qwen residual | 0.437655 | 0.710385 | 0.441386 | 0.721157 |
| C · Merger-LoRA residual | 0.437664 | 0.710385 | 0.441396 | 0.721156 |

The paired bootstrap resampled 253 full or 252 test-like Contrast-eligible paintings with replacement within its scope. Each of 10,000 replicates used the mean of paired per-painting correlation differences, with the 2.5th and 97.5th percentiles as the interval and RNG seed 203204. Positive Log Loss changes indicate regression. There were zero tied painting deltas. The intervals reflect variation across paintings in this fold with the saved selected predictions fixed.

| Scope | Comparison | Δ Contrast | 95% paired interval | Painting wins / losses | Δ Log Loss (bits) |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | B−A | −0.00015072 | [−0.00048403, +0.00018169] | 117 / 136 | +0.00004620 |
| Full | C−A | −0.00014202 | [−0.00047569, +0.00019342] | 119 / 134 | +0.00004568 |
| Full | C−B | +0.00000871 | [−0.00000700, +0.00002454] | 133 / 120 | −0.00000053 |
| Test-like | B−A | −0.00021270 | [−0.00056284, +0.00014301] | 117 / 135 | +0.00004720 |
| Test-like | C−A | −0.00020271 | [−0.00055394, +0.00015406] | 116 / 136 | +0.00004645 |
| Test-like | C−B | +0.00001000 | [−0.00000772, +0.00002821] | 131 / 121 | −0.00000075 |

Mean absolute full-scope probability shifts were 0.000618 for B−A, 0.000627 for C−A and 0.00002595 for C−B. Their largest shifts were 0.003227, 0.003204 and 0.00008854, respectively. Test-like mean absolute shifts were 0.000627, 0.000636 and 0.00002874. The corrections changed all saved rows at the recorded precision, with very small magnitudes. Both residual arms slightly worsened clipped Macro Log Loss on the outer fold despite slightly improving it on the inner set. These are probabilistic loss comparisons, with no separate calibration fit.

### Inner selection and optimization

All four epochs passed the frozen +0.005-bit inner Log Loss guardrail in both scopes. Both arms selected **epoch 1** by peak test-like inner Contrast. For context, inner A was 0.460603 / 0.736102 full and 0.477052 / 0.740743 test-like. The saved curves show small drops in training loss and inner Log Loss while inner Contrast fell. The mean absolute inner probability correction grew from 0.000629 to 0.001178 for B and from 0.000639 to 0.001178 for C.

| Arm | Epoch | Training objective | Inner objective | Full inner Contrast / Log Loss | Test-like inner Contrast / Log Loss | Epoch seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B | **1** | 0.5312035 | 0.5540213 | 0.460262 / 0.736092 | 0.476778 / 0.740719 | 356.87 |
| B | 2 | 0.5311690 | 0.5540229 | 0.460188 / 0.736083 | 0.476741 / 0.740707 | 6.23 |
| B | 3 | 0.5311663 | 0.5540192 | 0.460066 / 0.736069 | 0.476620 / 0.740689 | 6.15 |
| B | 4 | 0.5311607 | 0.5540209 | 0.460029 / 0.736054 | 0.476636 / 0.740672 | 6.25 |
| C | **1** | 0.5312168 | 0.5540228 | 0.460258 / 0.736093 | 0.476775 / 0.740721 | 567.99 |
| C | 2 | 0.5311691 | 0.5540238 | 0.460187 / 0.736084 | 0.476739 / 0.740708 | 567.33 |
| C | 3 | 0.5311663 | 0.5540197 | 0.460066 / 0.736070 | 0.476620 / 0.740690 | 565.16 |
| C | 4 | 0.5311607 | 0.5540212 | 0.460029 / 0.736055 | 0.476636 / 0.740672 | 563.17 |

The epoch-1 mean merger gradient norm was 4.35×10⁻⁵, then 4.21×10⁻¹⁰, 1.11×10⁻⁹ and 4.82×10⁻⁹ in epochs 2–4. The checkpoint tensors independently verify that both matrices moved in epoch 1 (L2 changes of 0.229618 and 0.181859). The selected checkpoint's effective LoRA projection delta has Frobenius norm 0.220682. Later matrix movement was much smaller. Logged clipping frequency was zero in both arms at every epoch. The adapter updated with detectable early gradients, while the held-out difference from B remained negligible. The saved evidence cannot isolate the cause of the later decline in gradient magnitude.

### Disposition and evidence limits

Both residual arms trailed A on outer Contrast and Macro Log Loss, every paired interval spans zero, and C's advantage over B is too small to justify repeating the unchanged recipe across five folds. This is a result for one exploratory P2 fold and one seed. The Qwen residuals used only the 854 fitting paintings, while outer A used the full outer-training scope. The archive carries sealed inner curves and outer predictions. The audit rescored those outer predictions and re-applied checkpoint selection to the saved inner curves. Independent rescoring of B and C's inner curves would require inner prediction files, which the archive omits. The audit used existing predictions and CPU checkpoint inspection. A fresh GPU inference replay was outside the closure scope. This result leaves the broader feasibility of trainable VLMs and the next research direction open.

Corrected Ridge remains the best-known competition candidate and E50+L the production fallback. All 35 focused CPU tests passed during closure. EXP-035 is scientifically closed for this configuration.
