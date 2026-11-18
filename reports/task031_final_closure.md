# TASK-031 — scientific closure

**TASK031 = CLOSED / MECHANISM_ANALYZED.** Engineering stage completed:
GPU_INFERENCE_COMPLETED / MECHANISM_ANALYZED. E50+L remains production; EXP-027
remains NO_GO_FOR_EXPANSION. TASK-032 performed CPU analysis only, no encoder
rerun, fitting, new checkpoint, outer-fold scoring, Development or confirmation access.

## Question and frozen protocol

Do the saved BCE/P_low/P_high Qwen models behave additively in image and language,
or produce useful painting-specific interactions, and through which conditioning
pathway? [Original preregistration](../docs/task031_protocol.md) and
[manifest](../configs/task031_manifest.json) are preserved byte-for-byte in Git.
32 original inner-training paintings, excluding folds1/2, form16 disjoint painting
pairs and48 fully observed language rectangles /134 unique cells. Use raw
`D=z(I,a)-z(I,b)-z(J,a)+z(J,b)`, not sigmoid probabilities. Compare matching,
neutral-prompt, fixed-Igbo-head and cyclic-crossed-head conditions; repeat with
adapters disabled under the **same saved head**, explicitly OOD.

No outcome-based sampling. Near-zero threshold0.001 logits, target tie1e-12,
10,000 bootstrap resamples of16 disjoint painting-pair clusters, seed31032,
retaining all three correlated rectangles per cluster. Noise resolution uses
the preregistered approximate Jeffreys variance, not a significance test.

## Artifact provenance and independent verification

Owner-run Drive directory: `MyDrive/task031/probe_run_8d2d24b6b83b/`.
Inference source `8d2d24b6b83b31fdf5164da6717b13c0f94c9bf7`, delivered in PR#29
at `c5e2288c004606f6b83754ff849b2986adcbe56b`. Original selected model states
remain bound to EXP-027 source `4bc703af144e9121e58ed4cb75c94306c318dff1`.
Qwen revision `66285546d2b821cf421d4f5eb2576359d3770cd3` and all original
processor/quantization/runtime settings are unchanged.

TASK-032 downloaded the actual443,991-byte export ZIP and9 standalone run files.
ZIP SHA-256:
`86f06117ff1e9e09615b8488c0298ef6369d94faf76478f01b16bdc0cbf2d3b1`.
All406 exported-file hashes pass; all9 standalone files match archive bytes.
All17 source hashes match immutable Git blobs.192 sealed image batches have
complete keys/receipts and3,216 raw logits. Frozen manifest, original checkpoint
SHA/config/source/earliest-minimum-BCE selection and selected training targets
independently reconstruct from the existing local original inputs/models.

The pinned analysis implementation was loaded directly from its Git blob and
recomputed on CPU. All1,152 diagnostic rows,24 summaries and original paired
bootstrap intervals reproduce within absolute1e-12; a second direct four-key
formula agrees on every mixed contrast. Original verification/output hashes pass.
[Full CPU receipt, original summaries and intervals](../docs/task031_closure_receipt.json).
Raw logits, votes, CSV and logs remain outside Git in `runs/task032/evidence/`;
original Drive/local records were not rewritten. The saved analysis's automatic
`INCONCLUSIVE_REQUIRES_INTERPRETATION` label is preserved; this closure supplies
the human scientific interpretation separately.

Verified **saved engineering measurements**, not new GPU measurements:

| Measurement | Recorded value |
| --- | ---: |
| GPU | NVIDIA L4 |
| Worker elapsed |200.241881491 seconds|
| Peak allocated /reserved |4,066,377,728 /4,150,263,808 bytes|
| Completed new/resumed image batches |192 /0|
| Official/cache state max error /cosine |0.0 /1.0|
| Fresh/cache vision parity |Exact PASS|
| Deterministic first-batch replay |6/6 PASS|
| Training /outer evaluation |Neither performed|

These checks establish artifact integrity and CPU statistical reproduction, not a
fresh independent encoder replay. Elapsed worker time excludes notebook setup,
input preparation and final Drive export; it is not total allocation duration.
Browser arrival and physical GPU deallocation remain unverified.

## Quantitative findings

Trained-adapter mean absolute mixed contrasts (logit units):

| Condition | BCE | P_low | P_high |
| --- | ---: | ---: | ---: |
| Matching |0.06737750|0.06542680|0.07360923|
| Neutral prompt |0.00180877|0.00181735|0.00187062|
| Fixed head |0.06693808|0.06534502|0.07315844|
| Crossed head |0.06593416|0.06398061|0.07192063|

| Matching diagnostic | BCE | P_low | P_high |
| --- | ---: | ---: | ---: |
| Mean abs(D), pair-cluster95% interval |[0.05267445,0.08340040]|[0.05158071,0.08073926]|[0.05471376,0.09420866]|
| D versus target-D correlation |0.19853474|0.24426124|0.27322600|
| Non-tied sign agreement (support) |0.6053 (38)|0.6154 (39)|0.6154 (39)|
| Noise-resolved sign agreement |5/8|6/8|6/8|
| Language-order reversals |1/48|1/48|2/48|
| Near-zero D fraction |1/48|0/48|0/48|

Eight of48 target differences exceed the approximate noise threshold. Neutral
predictions have only three resolved **non-tied prediction/target** comparisons;
do not mistake that condition-specific denominator for a different gold population.

Paired mean-abs(D) changes, retaining the original cluster bootstrap:

| Change | BCE | P_low | P_high |
| --- | --- | --- | --- |
| Neutral minus matching |−0.06556873 [−0.08180597,−0.05068585]|−0.06360946 [−0.07903536,−0.04940167]|−0.07173861 [−0.09248591,−0.05266352]|
| Fixed head minus matching |−0.00043942 [−0.00164118,+0.00058227]|−0.00008179 [−0.00116913,+0.00104743]|−0.00045079 [−0.00172951,+0.00086916]|
| Trained minus adapter-off, same head |+0.01091138 [+0.00403596,+0.01704046]|+0.01039725 [+0.00419118,+0.01605088]|+0.01010785 [+0.00221300,+0.01697857]|

Intervals describe this training sample; they are not independent generalization
confidence statements. Full distributions, other conditions and main-effect
scales are preserved in the receipt, without selecting a checkpoint or treatment.

## H1/H2/H3 interpretation and limits

**H1, strict additive logits: not supported.** Matching contrasts are measurably
nonzero. Additive language effects can still dominate overall profiles: matching
interaction RMS divided by twice the language main-effect RMS is only0.099–0.109,
and language-order reversals remain rare. Nonzero interaction does not imply a
flexible, accurate cultural ordering or contradict the earlier offset diagnostics.

**H2, existing but poorly aligned interactions: supported descriptively.**
Matching target correlations are modestly positive, not zero; only eight
noise-resolved differences and training-only measurement prevent a strong accuracy
or causal claim. P_high's larger correlation here does not reverse its failed
selected-validation result or reopen EXP-027. Strength is not prediction quality.

**H3, unequal pathway contributions: strong descriptive evidence.** Neutralizing
the prompt sharply suppresses measured mixed interactions while a fixed/crossed
head leaves similar magnitudes. Prompt-dependent decoder representations dominate
these interventions. This does **not** mean the learned head is irrelevant: its
shared function still acts on prompt-dependent states, and all interventions are
OOD. The observations are not a unique causal decomposition. Adapter-on changes
strength, but adapter-off with the trained head is not an independently fitted
frozen-model control and establishes no generalization benefit from adaptation.

Remaining limits:16 clusters /training-only labels; approximate independent-vote
noise model; FP16 numerical effects; one fixed reference language; OOD sensitivity;
prior repeated P2 examination; no outer/Development/confirmation evidence here.
H4 (3B capacity insufficient) remains plausible and untested; do not launch a
larger-model search from this result.

## Implication and reproduction

Test **representation complementarity cheaply first**. There is an existing Qwen
interaction pathway to improve, so an absence-of-interaction argument cannot
justify a large decoder redesign. CLIP-guided Qwen is a reasonable hypothesis,
not an established improvement. Require Qwen to add value beyond CLIP/SigLIP and
equal-capacity controls before considering decoder injection.
[EXP-028 design, controls and resource limits](exp028_design_proposal.md).

From the new checkout, with the unchanged export (no GPU):

```text
python scripts/close_task031.py --export-zip <outputs.zip> --output <new-receipt.json>
```

Add `--inputs <original-handoff-directory> --checkpoints <original-exp027-run>`
to additionally recheck selected targets and the three original checkpoints.
The actual local verification used `runs/task032/evidence/outputs.zip`,
`runs/exp023_budget_delivery/final/restore_check`, `runs/task030/evidence` and
new output `runs/task032/cpu_closure.json`. The verifier checks the frozen ZIP
hash, extracts into a temporary directory and never overwrites the saved run.
No further TASK-031 inference or EXP-027 expansion is an open action item.

## TASK-032 validation and preservation

Full Windows Python3.12.10 CPU suite: **376 passed,1 skipped**; no CUDA work.
Nine new lightweight tests cover read-only synthetic statistical replay, preserved
bootstrap intervals, wrong source/artifact/key/target/replay/analysis rejection,
ZIP hash/coverage/path safety, and design dimension/parameter/boundary checks.
All previous workflow tests remain included. CI status is recorded in the PR.

499 protected historical source/artifact files and70 EXP-025 cache files match
their prior snapshots. The E50+L reference CSV SHA remains
`162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`.
The new branch does not change any pre-existing `src/st3/`, config, notebook,
dependency lock, checkpoint or prediction. Closure utilities are new CPU-only
scripts; EXP-028 is a design artifact, not a model implementation. PR#28/#29
remain unmerged. Existing owner-edited notebooks in other worktrees are untouched.
