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
