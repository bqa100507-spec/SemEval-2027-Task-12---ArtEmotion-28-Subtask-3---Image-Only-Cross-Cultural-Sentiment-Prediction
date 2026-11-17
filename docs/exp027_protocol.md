# EXP-027 frozen scientific protocol — TASK-029

Implementation/CPU validation authorized; **no GPU run, outer evaluation, full CV, Development scoring, confirmation access or submission authorized**. EXP-026 remains EXPLORATORY_COMPLETED / NOT_PROMOTED; E50+L remains production. This protocol supersedes the [proposal-only preparation](../reports/exp027_proposal.md) for implementation, not GPU permission.

## Question, model and controls

Does a within-painting Pearson term improve language-relative sentiment learning over exactly matched BCE-only decoder QLoRA? Reuse EXP-026's pinned Qwen/Qwen2.5-VL-3B-Instruct revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, official128–256-token processor, NF4 base, frozen vision tower, original width64 sentiment head and decoder-only Q/V LoRA (rank8, alpha16, dropout0,72 targets). No architecture change, generated text, caption supervision, ranking, distillation or CLIP fusion. No previous trained adapter can initialize this experiment.

Common settings: seed2027, head LR0.001, LoRA LR0.0001, AdamW decay0.01, clip1, FP16 autocast/GradScaler, nonreentrant activation checkpointing; four unique paintings per optimizer window, eight language rows per microbatch. Original deterministic grouping/order and every eligible pair once per epoch. Checkpoint selection for **every variant**: earliest minimum vote-weighted inner-validation binary BCE, as computed by EXP-026 validation, across a maximum2 epochs. No fresh refit in this pilot.

| Name | Mode | Lambda |
| --- | --- | ---: |
| BCE | bce_only |0|
| P_low | bce_pearson |0.000682|
| P_high | bce_pearson |0.00196|

The BCE control calls EXP-026's native epoch method unchanged, with bitwise CPU parameter/optimizer regression checks. Treatments have the same initialization, data, batches, denominator, optimizer steps and selection rule. Additional forwards supply complete-group gradients, not additional optimizer steps or training examples.

## Loss and complete-group gradient

Within one effective painting window, `BCE = sum_i n_i * BCEWithLogits(z_i,k_i/n_i)/sum_i n_i`, matching EXP-026 exactly. Pearson uses `q=sigmoid(z)` and `y=k/n`. For each complete painting group, include pairs with n≥5, require at least3 eligible languages and nonconstant target proportions, center q/y, then `r_epsilon = dot(q_centered, y_centered/||y_centered||) / sqrt(sum(q_centered²)+0.01²)`. Average `1-r_epsilon` equally over eligible paintings in that window; zero connected auxiliary loss if none qualify. No annotator weighting inside Pearson. Total is BCE + lambda*Pearson.

The published/locally reconstructed scorer uses unsmoothed Pearson, excludes too-small/constant-gold groups and assigns0 correlation to constant predictions. Training retains those target-eligibility rules but smooths only the predicted norm, as in EXP-004, to obtain a finite nonzero gradient at constant predictions. This is a **training surrogate**, not a replacement scorer. Tiny group aggregation is float64; BCE retains EXP-026's float32 arithmetic and normalization. Invalid/nonfinite counts/logits fail rather than silently changing supervision.

Treatments use two grad-enabled forwards per microbatch, one backward. Pass1 immediately discards each model graph and retains only detached logits plus CPU/CUDA/Python/NumPy RNG snapshots. Full-window loss yields dL/dz across all painting-language rows, including groups spanning microbatches. Pass2 restores each draw/state and backpropagates each slice of that full-window logit gradient; GradScaler is applied once to the VJP. Clip/step once after the entire window. End RNG equals a single pass's consumption. Registered buffer mutation, parameter-version mutation, unequal replay logits or nonfinite/disconnected gradients block training. No graph for more than one microbatch is retained.

The chain rule is mathematically equivalent **conditional on identical replay of the same microbatched forward function**. CPU deterministic models also match unsplit full-batch gradients within documented floating-point error; stochastic tests compare the same partition/draws, not arbitrary dropout masks under different batch partitions. Batch-dependent/state-mutating layers are unsupported and fail. CPU tests do not prove quantized L4 kernel equality. Future L4 preflight compares one complete3–8-language painting's unsplit/two-pass AMP gradients (atol5e-4, rtol2e-3), requires exact same-chunk replay logits, and performs no optimizer updates. Numerical tolerances are not bitwise GPU-equivalence claims.

## Lambda calibration — training only, before GPU outcomes

[Diagnostics](exp027_scale_diagnostics.json) use64 predetermined inner-training paintings/609 pairs, seed2711, four-painting windows and label-independent synthetic logits with SD0.001/0.05/0.2. Outer fold2, examined fold1 and all inner-validation targets are excluded from calibration. No EXP-026 predictions, held-out outcomes or Qwen fit selects the grid.

Unscaled logit-gradient norm ratio (Pearson/BCE) has median73.3418 and90th percentile254.8801. Final engineering targets are5%/15% median auxiliary gradient, capped at50% at the90th percentile, giving lambdas0.000682/0.00196 after three-significant-digit rounding. These correspond to median ratios≈0.050/0.144. EXP-004's0.01/0.05 would imply≈0.733/3.667 median and2.549/12.744 at the90th percentile on these diagnostics, so they are not inherited. Proposed0.1/0.3/1.0 are not tested. This is a scale rationale, **not evidence of optimal lambda or actual Qwen parameter-gradient ratios**; synthetic CPU head/LoRA tests and future GPU component-gradient logging cover the latter pathways without adapting the frozen grid.

## Population and authorized next scope

Use original P2 metadata, nominal outer fold2 excluded before target indexing. Original painting-disjoint inner split (seed17029+2,10%) on the remaining original training paintings. Future pilot selects192 inner-training and48 inner-validation paintings deterministically with EXP-026's ranking rule; actual1931/504 pairs, respectively. All variants use identical populations. The streaming label filter checks painting membership before reading/converting k/n; outer/locked targets are never passed to training/calibration/selection. Input hashes are unchanged. Only image pixels and target language enter inference; identifiers locate/checksum data.

Fold1 has been inspected and cannot tune loss, lambda or stopping. Other P2 folds were historically reused and are not untouched independent confirmation. Any subsequently authorized held-out run needs a fresh task/protocol, fresh pretrained/zero-output LoRA initialization and matched controls. This implementation exposes only a pilot endpoint, not outer scoring or full CV.

## Persistence and resources

New EXP-027 namespace/source identity; compatible frozen **vision** cache reuse only. Completed epochs persist current/best head+adapter state, optimizer/scaler, RNG and counters with checksums; selected models are independently loaded and their inner-validation predictions replayed to six decimals before publication and on resume. Input/model/processor/source/runtime/config/partition/variant changes fail. Owner time limits are operational and may change only within the fixed2-hour cap; no scientific settings change. A stale pilot-success marker is removed before new verification.

Future Colab workflow: one default-OFF authorization/configuration, immutable source clone, ZIP+input-manifest hashes, isolated Python3.12.10/exact EXP-026 dependency lock, permitted pilot image extraction to `/content`, CPU/workflow checks, parity/gradient preflight, three inner-only fits/replays, persistent read-back verified export, optional browser download attempt, then outer-finally runtime release. No mid-run approvals; abrupt VM loss can lose the unfinished epoch. Runtime/download API returns never prove physical deallocation/local arrival. OOM stops once, no automatic precision/rank/batch changes.

EXP-026's L4 pilot observed≈6.74 training pairs/s and≈20.38 forward-only pairs/s. For1931 training pairs×2 epochs, native BCE training projects≈573s; each treatment adds≈190s forward work (≈1.33x, not measured). Three fits plus repeated validation/checkpoint replay, state checks, cache/model setup and export project roughly50–80 minutes. Budget remains≤2 hours including setup; a pessimistic2x treatment-training bound is retained. Expected peak activation memory resembles one EXP-026 microbatch (historical≈4GiB allocated), plus small logit/RNG/buffer copies; use≈4–6GiB only as an engineering planning range, **not a measurement**. Real L4 peaks/throughput must be recorded and can block execution. Resource guards never auto-expand authorization.
