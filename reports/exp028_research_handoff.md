# EXP-028 research handoff

**EXP028 = CLOSED / STAGE1 NO-GO. E50+L = UNCHANGED. NEXT_DIRECTION = AWAITING_INDEPENDENT_BRAINSTORM.**

| Evidence | What to carry forward |
| --- | --- |
| EXP-026 tested supervised frozen Qwen and decoder QLoRA on an exploratory P2 fold 1. L0 improved on its frozen comparator and scored 0.340691 Contrast versus 0.433429 for the refitted E50+L. | Adaptation changed outcomes within the supervised setup and still left a large performance gap. The exploratory models used less fitting data than the refitted reference. [EXP-026 audit](exp026_exploratory.md) |
| EXP-027 tested a Pearson auxiliary objective alongside BCE. Both variants had lower selected inner-validation Contrast than BCE alone. | These choices supplied no reason for expansion. The result comes from a small inner-validation pilot with BCE-based checkpoint selection and no outer evaluation. [EXP-027 closure](exp027_final_closure.md) |
| TASK-031 found nonzero, prompt-sensitive image–language interaction in saved model logits. | The tested models can produce non-additive outputs. A small training-only mechanism probe with modest target alignment leaves task accuracy and generalization unresolved. [TASK-031 closure](task031_final_closure.md) |
| EXP-028 tested frozen Qwen as a complement rather than requiring it to carry sentiment prediction alone. C512 beat shuffled-Q by +0.006558 Contrast, improved on matched-width A512 by +0.000833, and trailed A_raw by −0.000605. Its E50+L difference was +0.001476 with an interval spanning zero. | Correct artwork association matters in this pipeline on the reused fold. The tested gain is too small for progression and leaves any robust benefit over the strongest visual systems unresolved. The lowest observed Macro Log Loss is a separate result. [Verified Stage 1 closure](exp028_final_closure.md) |

The one-fold result leaves open whether the signal carries to other paintings, which properties of the saved states account for it, and how much the outcome depends on representation, readout or fitting data. Treat the Qwen-only arms, shuffle result, and TASK-031 mechanisms as evidence about specific setups. Comparing the scalar scores across folds and populations would conflate different protocols.

Expensive reruns of the same decoder QLoRA recipes, the gated nonlinear/decoder-injection continuation and expansion of this screen to five folds require a new scientific justification and authorization. The current evidence does not unlock these paths.

Reusable assets include the checksum-pinned released P2 training subset and painting-grouped manifests, the original frozen visual NPZs, the private Qwen state cache with its extraction identity, the matched E50+L OOF reference, the pin-verified CPU workflow receipts, and the official-style scorer and painting-paired audit. Keep P1, P2, hosted Development, and the sealed confirmation population distinct. The production `e50l` default, fallback modes and frozen reference predictions remain protected. The 236 confirmation paintings stay sealed. [Research index](../docs/research_index.md) · [Competition release](../docs/competition_release.md)
