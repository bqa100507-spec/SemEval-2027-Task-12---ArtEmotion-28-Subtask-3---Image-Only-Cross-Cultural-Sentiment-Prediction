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
