# TASK-031 — frozen interaction mechanism probe

Implementation/CPU validation only. No GPU work is authorized. Base closure:
`ca1198347f834a60f12949115980ea8577298e17`; EXP-027 stays
GPU_PILOT_COMPLETED / NO_GO_FOR_EXPANSION; production stays E50+L.
This is not EXP-028 and cannot train, tune, refit, generate captions, score outer
folds or create competition submissions. Follow the shared workflow standard
for bounded inference, persistence and cleanup, without invoking training stages.

## Existing mechanism and saved states

`vlm_supervised.SentimentHead` maps the 2048-dimensional final Qwen decoder state
through LayerNorm → Linear(64) → GELU. It adds a shared scalar linear output,
learned language bias, and a dot product with the language-specific 64-vector
divided by sqrt(64). Language also enters the user prompt before the Qwen decoder:
the same image can therefore have different decoder states for different prompts.
The shared head component is not necessarily language-independent. Decoder Q/V
QLoRA across 36 blocks can change both prompt and image processing. The visual
tower, pretrained decoder weights and original processor are frozen.

Use exactly BCE, P_low, P_high selected epoch-1 states from
`pilot_run_4bc703af144e`, original source
`4bc703af144e9121e58ed4cb75c94306c318dff1`. The immutable checkpoint, original
run/config and seal hashes are in `exp027_closure_evidence.json`. Validate all
original scientific source blobs, run identity, original minimum-BCE selection,
language vocabulary, adapter names/shapes and strict head state before inference.
No EXP-026 fitted adapter, new head, new checkpoint or model revision is allowed.

Qwen: `Qwen/Qwen2.5-VL-3B-Instruct`, revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`. Reuse the original NF4 base,
processor/vision-token limits, saved rank-8/alpha-16 decoder adapters and original
FP16 inference autocast. Head logits are promoted to float32 before JSON storage;
CPU arithmetic uses float64. Preserve full round-trip logit precision, not rounded
probabilities. All model/head parameters are frozen, eval and inference-mode;
no optimizer/backward/generation exists in the probe path.

## Sampling, frozen before any counterfactual inference

`configs/task031.json` and `configs/task031_manifest.json` are the preregistration.
Require checksum-identical released P2 pairs/protocol. Restrict to the original
192 EXP-027 fitting painting IDs, then exclude previously examined fold1 and
outer fold2 before indexing any votes. Exclude all validation/confirmation IDs.
Only painting IDs, language presence and image mappings determine sampling.
Painting identities are locators, not predictive features.

Sort permitted paintings by SHA-256 fingerprint `[31031, painting]`; greedily
pair each unused painting with the first unused compatible painting in that order,
requiring at least three common eligible languages. Stop at16 disjoint pairs.
For each pair, rank unordered common-language combinations by fingerprint
`[31031,I,J,a,b]` and retain three. No target magnitude/prediction enters selection.
Only after selection, read k/n for the134 selected cells (minimum5 votes).
The committed manifest contains identifiers and target hash, **no vote counts**.

Fixed head reference is the language with greatest painting presence in the
permitted pool (lexicographically ascending tie break): **Igbo**. Crossed heads
use a one-step cycle of the sorted saved head vocabulary, without fixed points.
All three models use exactly the same grid and same actual pixels.

## Conditioning interventions

| Condition | Decoder prompt | Sentiment-head language | Scope |
| --- | --- | --- | --- |
| A_matching | target language | target language | Original path |
| B_neutral_prompt | original neutral prompt | target language | OOD |
| C_fixed_head | target language | fixed Igbo | OOD |
| D_crossed_head | target language | fixed cyclic permutation | OOD |

For each saved model also disable its adapters **with its identical saved head**.
This is an OOD within-model intervention, not an independently trained frozen
control; adapter-off differences cannot identify a clean training benefit or
capacity effect. Do not compare an untrained head against a trained head.

Per painting, decode each unique prompt once in one batch; reuse its hidden state
for the head interventions. No cache of unrelated-image predictions is accepted.
Read only validated frozen vision caches; independently hash artwork pixels and
compare one fresh original-processor vision extraction exactly against cache.
Run the official-vs-injected multimodal assembly parity check. Repeat the first
painting batch exactly for every model/adapter state; discrepancies stop the run.
Batching differs from historical four-row prediction chunks; this probe does not
claim bitwise parity with historical probabilities. Repeated same-batch logits
must be identical; no precision/resolution/batch fallback on OOM.

## Analysis, uncertainty and interpretation

Raw mixed contrast is `D=z(I,a)-z(I,b)-z(J,a)+z(J,b)`. Additive raw-logit models
have D=0; probability-space nonlinearity cannot substitute for this check.
Near-zero means abs(D)≤0.001 logit units, fixed before inference; it is a numerical
descriptor, not a scientific acceptance gate or guarantee against FP16 error.
Report mean/median absolute D, signed quantiles, near-zero fraction, image and
language main-effect RMS, relative interaction RMS, language-order reversals
between paintings and painting-pair detail rows. Preserve explicit prompt/head
keys and component logits. Head components do not uniquely decompose the prompt.

Target contrast uses empirical positive proportions; target units differ from
logits. Report gap/contrast correlation, sign agreement and support counts.
Target ties have abs(D_target)≤1e-12; predicted ties use the0.001 threshold.
Report all non-tied and approximately noise-resolved targets separately. The
latter have abs(D_target)>1.96×sqrt(sum of four Jeffreys Beta posterior variances
from original binary counts). This independent-vote approximation is descriptive,
not a calibrated cross-cultural hypothesis test. Null correlations/support remain
explicit when unavailable; ties are not counted as correct signs.

10,000 percentile bootstrap resamples use seed31032 and disjoint **painting-pair
clusters**, keeping all three language rectangles together. Never resample the
48 correlated rectangles independently. Report paired absolute-interaction changes
between checkpoints, matching/neutral/fixed/crossed paths and adapter-on/off.
These intervals describe training-data mechanism variability, not generalization.
There is no checkpoint/lambda/model selection or official score/promotion here.

Interpret after verified inference, without forcing exclusive or causal claims:

- H1: near-additive behavior / weak mixed interactions supports investigating a
  more image-dependent language pathway, but does not show that CLIP fusion works.
- H2: interactions exist but align poorly with votes; investigate alignment and
  supervision before assuming additional complexity is necessary.
- H3: useful interactions are concentrated in one pathway; preserve it and study
  where complementary CLIP features could enter. OOD sensitivity is not a unique
  causal decomposition or unbiased deployment result.
- Inconclusive: small training population, noisy votes, precision or OOD effects
  prevent a diagnosis. Request only the minimum additional authorized evidence.

## Resource bound and delivery

Cold grid:32 paintings,48 rectangles,134 unique cells;996 decoder sequences in
192 image batches,6 repeat batches,2 official-assembly parity decoder calls,
and at most34 vision forwards including fresh/cache and official parity checks.
The manifest supplies the exact repeat sequence count.3,216 logical raw logits
reuse those hidden states. A resume repeats the first painting per model/state
as well as the parity check; remaining valid image batches are reused exactly.

Planning only: at1–4 decoder sequences/s, about4–17 minutes of decoder work,
plus cold dependency/model loading, vision validation and export. Allow up to
30 allocation/wall minutes since Run all, including180s reserved for persistence.
This is **not measured TASK-031 throughput** and the earlier5–10-minute inference
guess is not an end-to-end result. Budget/OOM stops retain complete image batches
but cannot yield complete analysis. There is no automatic retry or longer run.

The future L4 notebook defaults authorization OFF. It mounts Drive, verifies the
existing input ZIP/manifest and three original checkpoint receipts, installs the
unchanged Python3.12.10/lock, prepares only permitted targets and extracts only
selected pixels locally. Logs/sealed logit batches stay in a new persistent source
namespace. Offline CPU verification rebuilds hashes/keys/analysis, not encoder
inference. Export/read-back precedes optional browser download and outer-finally
runtime release on success/failure. API requests do not prove download arrival or
physical GPU deallocation. Competition eligibility remains unconfirmed; this
research-only diagnostic never opens confirmation or Development labels.
