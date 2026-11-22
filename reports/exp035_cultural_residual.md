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
