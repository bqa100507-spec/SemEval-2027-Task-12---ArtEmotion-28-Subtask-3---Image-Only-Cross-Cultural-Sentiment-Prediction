# TASK-031 — implementation and CPU readiness

**CPU_VALIDATED / GPU_INFERENCE_READY**, conditional on the documented future
owner authorization. **No GPU inference or training was executed. No H1/H2/H3
finding is available.** EXP-027 stays NO_GO_FOR_EXPANSION; E50+L stays production.

The saved model has two language pathways: target-language Qwen prompt and
explicit head bias/visual interaction. The shared head receives prompt-dependent
decoder states, so it cannot be assumed language-independent. The new probe
measures raw-logit mixed contrasts, with the matching, neutral-prompt, fixed-head
and crossed-head interventions. Disabling each model's adapters with its same
saved head is an additional **OOD matched-head toggle**, not an independent
frozen model or unique causal decomposition. [Frozen protocol](../docs/task031_protocol.md).

## Verified preparation

Original source/config identity, all three checkpoint SHA receipts and original
minimum-inner-BCE epoch1 selection pass on CPU. The same saved vocabularies and
fitting IDs are used. Deterministic sampling produces32 paintings /16 disjoint
painting pairs /48 fully observed rectangles /134 distinct cells. Sampling uses
identifiers/language support only; selected vote counts are read afterward.
Fold1 and outer fold2 are excluded before target indexing. No inner-validation,
Development or sealed confirmation targets are used in the probe.

Actual command, successful on the existing local artifacts:

```powershell
.venv/Scripts/python.exe runs/task031/review/scripts/run_task031.py --stage prepare --inputs runs/exp023_budget_delivery/final/restore_check --checkpoints runs/task030/evidence --run-dir runs/task031/cpu_prepare
```

The preparation output independently matches the committed manifest. It contains
selected training targets only in ignored local storage. No images, model states
or vote counts are committed. Frozen source revision and per-file hashes:
[`task031_source.json`](../docs/task031_source.json).

## Execution cost and future run

| Work | Cold run count |
| --- | ---: |
| Saved models × adapter states × conditioning conditions |3 ×2 ×4|
| Raw keyed logits |3,216|
| Main decoder image batches / sequences |192 /996|
| Exact-determinism repeat batches / sequences |6 /36|
| Official/cached assembly parity decoder calls |2|
| Maximum vision forwards, including fresh and official parity |34|

**Estimated, not measured:** at1–4 decoder sequences/s, decoder work is roughly
4–17 minutes, plus cold dependencies/model loading, pixel/vision checks and export.
Use a maximum30-minute total allocation/wall limit including180s export reserve.
Cold setup can exhaust it; completed image batches persist, and an incomplete run
cannot produce a verified mechanism conclusion. No earlier5–10-minute estimate is
claimed as a measured end-to-end runtime. Actual L4 latency/VRAM remain untested.

Open [`TASK031_Colab.ipynb`](../notebooks/TASK031_Colab.ipynb) after separate owner
authorization, select L4, set `AUTHORIZE_EXPERIMENT=True`, then Run all. Keep
`STAGE="probe"`, both budgets0.5h, confirmation/submission disabled. Default paths:

- Input: `MyDrive/exp023_handoff.zip`, unchanged SHA/manifest pins.
- Checkpoints: `MyDrive/exp027/pilot_run_4bc703af144e/`; only identity and three
  model/seal pairs are copied to local `/content` and checked against closure hashes.
- Frozen vision cache: `MyDrive/exp026/features/`; validated compatible entries only.
- New outputs: `MyDrive/task031/probe_run_<pinned-source-prefix>/` and unique attempts.

No intermediate approval or training stage exists. Complete keyed logit batches
are sealed persistently, source/runtime/GPU changes fail resume, the first batch
per state is exactly replayed, and incomplete/corrupt publications fail closed.
Logs, diagnostics, hashes, raw logits and descriptive analyses export with persistent
read-back, optional browser download and release on success/failure. Download
arrival and physical deallocation remain unverified. The notebook never modifies
the original checkpoint directory or production inference.

Offline verification, using the same pinned source checkout and a downloaded
run directory (no CUDA required):

```text
python scripts/verify_task031.py --run-dir <downloaded-run-directory>
```

This reconstructs cache/source integrity, explicit keys, mixed contrasts and
painting-pair bootstrap analysis; it does not independently replay GPU inference.
A failed verification invalidates previous successful evidence before checking.

## CPU checks and preservation

Synthetic additive/interaction symmetry, exact-key joins, target ties/noise,
paired cluster bootstrap, saved state compatibility, prompt/head pathways,
authentic-image dependence, all-frozen/no-optimizer/no-backward inference, real
tiny-image cache reuse/interruption, cache corruption/source mismatch, pixel
tampering and authorization/deadline handling pass. Notebook syntax/schema,
source pins, workflow contract and mocked failure/export/download/release tests
pass. CPU mocks do not prove L4 numerical determinism or processor parity; actual
GPU parity must pass before completing the probe.

The complete CPU suite passes **367 tests,1 skipped**. Historical preservation
checks:499 protected source/artifact files and70 EXP-025 cache files match their
prior snapshots. Original E50+L CSV SHA remains
`162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`.
Owner-executed EXP-026/027 notebooks remain untouched in their original worktrees.

After future verified inference, inspect H1 weak/additive behavior, H2 existing but
misaligned interactions, H3 pathway concentration, or inconclusive evidence.
The sample is training-only, small and noisy; OOD sensitivity is not a deployment
score. Do not claim CLIP-guided fusion, a larger model, or a new architecture will
succeed from this probe. No successor implementation, refit or promotion is authorized.
