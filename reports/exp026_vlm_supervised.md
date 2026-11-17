# EXP-026 — direct VLM sentiment, frozen versus decoder QLoRA

**TECHNICAL_PASS; pilot and scientific fold NOT RUN. Production remains E50+L.** EXP-025 is closed for caption fusion; its final owner-reported scores are in [the short closure](exp025_vlm_clip_probe.md). No EXP-025 extraction or evaluation was rerun.

**Continuation:** the owner subsequently completed a Colab L4 pilot (PILOT_PASS); its saved notebook logs report L0 validation BCE0.644959→0.581847 and about6.74 pairs/s. This supersedes the initial pilot-not-run status above, which records TASK-028's local execution boundary. The13.4-hour scientific fold remains unrun. A separate, unexecuted two-epoch/no-refit exploratory stage is prepared in [the short continuation report](exp026_exploratory.md); original scientific settings below are unchanged.

## Frozen design

`(pixels, target language) → Qwen multimodal decoder state → supervised sigmoid head`. No generated text, caption targets, CLIP fusion or auxiliary loss. Both primary conditions use the identical NF4 Qwen2.5-VL-3B revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, official processor, 128–256 image tokens and final assistant-prefix decoder state (2048 dimensions). Checkpoint/processor hashes and dependencies are pinned in `configs/exp026.json` and `requirements-exp026.lock.txt`.

| Condition | Decoder | Input and prediction |
| --- | --- | --- |
| F0 | Frozen | Real target-language prompt; trained width64 language-conditioned head |
| L0 | NF4 base frozen; decoder Q/V LoRA | Same prompt/head; joint head and adapter optimization |
| F0_image (diagnostic) | Frozen | Neutral-prompt cached state; same language-conditioned head |
| E50+L | External fixed reference | Existing predictions on identical held-out keys only |

The frozen vision tower runs once per image; its post-merger tokens are cached exactly and reused. F0 caches both neutral and language-conditioned frozen decoder states. L0 runs the decoder live on those vision tokens with gradient flow. Language conditioning requires a decoder forward per language, but never another vision forward. Cached features are raw decoder states (no CLIP normalization). Image identifiers locate/checksum files; only pixels and target language enter the model. Unseen languages use the shared visual head.

LoRA: rank8, alpha16, dropout0; exact targets `model.layers.{0..35}.self_attn.{q_proj,v_proj}` (72 modules), no bias/vision/base training. Head: LayerNorm2048 → Linear64 → GELU with a shared logit and language bias/interaction. Head LR0.001, adapter LR0.0001, AdamW decay0.01, clip1, FP16 autocast/scaler, nonreentrant gradient checkpointing. No hyperparameter search.

Count-aware BCE uses `sum(n_i * BCE(logit_i,k_i/n_i))/sum(n_i)` within each effective four-painting window. Eight-pair microbatches accumulate against that same window denominator before one optimizer step. All language rows occur once per epoch, in identical painting/language order across conditions. Stopping uses global vote-weighted inner-validation BCE in natural logs; reporting uses the existing scorer's Contrast and Macro Log Loss in bits. Original P2:1335 training paintings/13600 pairs, original five-fold mapping; fold1 only is implemented for this stage. The236 sealed confirmation paintings' identifiers are checked for disjointness; their annotations are never loaded.

Fixed pilot:192 inner-training/48 inner-stopping paintings, seed2027, two epochs. Scientific fold: original inner split, max12 selection epochs, patience3, then a fresh head/zero-B adapter reset and refit for selected epochs. Pilot/technical weights are never scientific warm starts. Each condition has matched initialization and order. Published checkpoints carry file checksums and exact source/config/runtime/input/partition identities; complete epochs can resume. Final predictions must replay from the saved state to six decimals before scoring. A separate32-painting image shuffle measures image sensitivity; it never replaces ordinary predictions.

## Actual authorized local check — 2026-11-17

The owner authorized one RTX4060 check, one inner-training artwork, at most300 seconds. It completed in **43.829 seconds**, with no OOM or retry. No further GPU work was executed.

| Measurement | Actual result |
| --- | ---: |
| Artwork / eligible language rows used |1 /1 (Arabic) |
| Vision tokens / decoder width |238 /2048 |
| Official assembly vs cached injection max absolute error |0.0 |
| Initial zero-LoRA vs frozen state max absolute error |0.0 |
| Persisted hidden-state cache |Exact parity |
| Trainable adapter / head parameters |1,843,200 /136,922 |
| Three BCE values (nats) |0.702194,0.508813,0.517477 |
| Three head gradient norms |4.508494,0.932235,1.546284 |
| Three adapter gradient norms |0.155196,0.028168,0.027390 |
| Adapter / head update norm |0.174525 /0.616919 |
| Adapted decoder-state shift norm |12.198716 |
| Peak allocated / reserved after model loading |3.39 /3.42 GiB |
| Frozen pretrained and vision weights |Exact digest parity; no vision gradients |

The allocator was capped at half of8GiB; these peaks cover the post-load forward/backward check, **not model-loading peak**. The future runner records loading peaks separately. Frozen parameter digest before/after: `3a4479031ee7af88d91a7d3e632118f6eabe9ccc40994fd03fb455e5eab1420d`.

Local evidence: `runs/exp026_technical/technical_report.json`, its unique process log and technical-only checkpoint. The exact executed source/config snapshot is preserved in `runs/task028/technical_check_source/source_sha256.json`; later pilot/one-fold code was completed with CPU tests and has not been GPU-executed. The technical checkpoint is never eligible for scientific reuse. Three one-image steps prove a connected finite gradient/update path, not convergence, generalization, L4 feasibility or training throughput. No reliable pilot/fold/GPU-hour estimate can be derived from this check.

## Next execution and validation boundary

Use [EXP026_Colab.ipynb](../notebooks/EXP026_Colab.ipynb), the unchanged TASK-025C input ZIP (`2db4a53679940c7eb63c7c8272b91be8ec0618bb6c04a69ebaa15909e20f64e0`), a Colab L4 and **fresh explicit owner authorization**. Default `STAGE="pilot"`, `AUTHORIZE_EXP026=False`, maximum2 allocation-hours including setup. Run all after authorization. Images/weights stay local; compatible caches and completed epochs/models persist under a new `MyDrive/exp026` namespace, separately from EXP-025. Interrupted current epochs may be lost. Export/download attempts precede runtime release; browser arrival/deallocation are not independently confirmed.

The pilot must measure complete-epoch throughput, losses, gradients/updates and peak VRAM. It estimates a conservative max-selection/refit fold cost including frozen controls, validation, cache and overhead. After reviewing that evidence, the owner may separately authorize `STAGE="one-fold"`; the same compatible pilot resumes, and the fold runs only if its measured estimate fits the remaining allocation. This is a resource check, not score-based tuning. Full five-fold QLoRA is neither offered nor recommended before a measured one-fold result. No held-out EXP-026 score is available yet.

CPU tests cover tiny-model zero-LoRA parity/gradient updates/frozen weights, target matching/RNG, vote-weighted accumulation, unknown-language fallback, image cache reuse/checksums, grouping, completed-model resume and hard technical timeout/authorization. Notebook tests use mocks; cloud CUDA and full-size pilot/fold behavior remain unverified. Preservation checks compare existing local cache/artifact hashes and frozen E50+L CSV. The owner's1335-image FP16 Drive cache was not retrieved or modified; its contents are not independently re-audited here.

Local validation: `python -m pytest -q` → **283 passed,1 Windows symlink skip**; notebook nbformat schema and Python AST PASS. All499 protected files and70 local EXP-025 cache files retain their previous hashes. E50+L CSV remains `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`. Existing owner notebook edits were preserved in their original workspaces.

Research only. The pinned checkpoint's [model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3) and license were already retained with asset hashes. Non-commercial Qwen Research License/competition compatibility remains unresolved. No Development gold, confirmation annotations, production changes, automatic promotion or Codabench upload.
