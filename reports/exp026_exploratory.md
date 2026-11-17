# EXP-026 — two-epoch exploratory fold

**EXPLORATORY_COMPLETED / NOT PROMOTED. E50+L remains production.** The owner completed the L4 exploratory run. On2026-11-18, a read-only CPU audit downloaded the small prediction/log/receipt files and independently recomputed scores using the existing checksum-pinned P2 training votes. No new GPU run, model fit, checkpoint replay, full CV, Development or confirmation annotation access.

The owner-completed L4 pilot passed: L0 validation BCE0.644959→0.581847; F0 epoch2 BCE0.611536. The saved local Colab notebook also contains the pilot's timing logs. These are owner-run observations, not checkpoint replays performed in this patch. The original scientific fold estimate was48,220.88 seconds (13.4h); it is not launched.

## Verified held-out results and final decision

| Model | Fold1 Contrast | Macro Log Loss (bits) | Selected epoch |
| --- | ---: | ---: | ---: |
| F0_image |0.327045|0.781817|2|
| F0 |0.310775|0.808316|2|
| L0 QLoRA |0.340691|0.704087|1|
| E50+L reference |0.433429|0.666222|Refitted reference|

All2764 pairs /267 paintings /25 languages match the original fold1 keys. Predictions/receipts, run/source identity and published metrics checksum pass; all four scores reproduce within1e-12. The local released-training pairs already existed with the required SHA, so the319MB input ZIP did not need to be fetched. Model checkpoints were not downloaded or replayed; this verifies predictions/scoring, not independent encoder inference.

QLoRA improves Contrast by+0.029917 versus matched F0 and+0.013647 versus F0_image, but loses−0.092738 against E50+L; Macro Log Loss is+0.037865 bits worse. These are descriptive single-fold differences without a significance claim. VLMs use less fitting data than the refitted reference, as detailed below. **Keep EXP-026 exploratory completed; do not promote, add epochs or launch five-fold training.**

### Error analysis with actual votes

- Contrast includes251 paintings: L0 beats E50+L on82 and loses on169, with no numerical ties at1e-12 tolerance. Nine constant-gold paintings and seven with fewer than three eligible languages are excluded for every model.
- Macro Log Loss includes21 languages with at least20 pairs. L0 improves only Arabic (−0.008667 bits); it is worse on the other20. Largest included-language regressions: Kyrgyz+0.154876, Burmese+0.083700, Hausa+0.050702, Yoruba+0.049909 and Korean+0.046069 bits. Emakhuwa/Kinyarwanda have2 pairs each; Darija/Urdu have3 each and do not enter Macro Log Loss. Tiny-group changes must not be presented as scored-language evidence.
- Verified training log: L0 training BCE0.595406→0.530031 while inner-validation BCE0.510690→0.541321. Epoch1 selection is correct. This is evidence consistent with overfitting at epoch2, not proof that every later epoch must be worse. F0_image validation0.572994→0.554877; F0 0.592903→0.584779.
- Prediction-only summaries also reproduce: L0 mean0.692927, mean within-painting language SD0.075251, SD of painting means0.171692; E50+L corresponding SDs0.084646 and0.171757 (population SD, ddof0). L0 is not a language-constant predictor. Greater between-painting variance does not itself reduce within-painting Pearson, which is invariant to shared additive shifts and positive scaling within a painting.

The owner's additional label-free half-painting cross-fit analysis reports fixed-language offsets explaining95.1% /88.0% /76.7% /58.2% of centered variation for F0_image/F0/L0/E50+L, with median profile rank correlations0.999 /0.972 /0.944 /0.846. Those profile statistics are retained as **owner-provided diagnostics**, not independently reproduced: the supplied analysis does not fix its split/estimator details. They support a hypothesis of relatively rigid image–language interactions; they do not prove the main causal failure or which offsets are wrong against votes. Language offsets describe this task/model, not inherent community traits.

Source commit: `881393cb560557d7113b3f896715eba27d7025ec`; run identity: `48dd003ae7dc75240ea5fe271cada051ce2a871f348fd3137c9a775fa0e3bc6b`. [Lightweight verification receipt](../docs/exp026_exploratory_evidence.json). Local raw evidence, `audit.py`, complete per-language errors, per-painting Contrast and checksums remain under ignored `runs/task028_closure/`; no raw votes, predictions or checkpoints are added to Git. Worker elapsed4548.25s (75.8min), excluding notebook setup; post-load allocated/reserved peaks4.01/4.18GiB are owner-run log evidence, not a new benchmark in this audit.

[EXP-027 is proposal-only](exp027_proposal.md): a within-painting objective on the same VLM/LoRA, with a matched BCE control. Do not select its loss/lambda from the now-examined fold1, and never warm-start another held-out fold from EXP-026 adapters that trained on it. New implementation or GPU work requires a new explicit owner task. Preserve every existing artifact and production E50+L.

## Separate screening protocol

`scripts/run_exp026.py --stage exploratory-fold` uses the unchanged `configs/exp026.json` plus `configs/exp026_exploratory.json`. Same checkpoint, processor, NF4/LoRA/head, loss, optimizer, learning rates, seed, painting-grouped batches and original P2 fold1/inner partition. Only this separate stage caps training at2 epochs and skips fresh refit. Each condition independently selects its earliest minimum inner-validation BCE checkpoint; the actual selected state, not the final epoch, is saved for inference. No hyperparameter search or repeated pilot.

| Population | Paintings | Eligible pairs | Use |
| --- | ---: | ---: | --- |
| Inner training |961|9759|F0_image/F0/L0 fitting |
| Inner validation |107|1077|Checkpoint selection only |
| Held-out fold1 |267|2764|One prediction pass per selected model |
| Outer training |1068|10836|Existing refitted E50+L reference |

**VLMs use less fitting data than E50+L. This is exploratory screening, not final model selection or promotion.** The four-row output table reports Contrast and Macro Log Loss on identical held-out keys, selected epochs and fitting counts. No Development gold, confirmation annotations or Codabench access. No blend or production changes. The original scientific one-fold procedure/config and original notebook stay unchanged apart from adding a separate CLI dispatch.

Frozen extraction code and its cache identity are byte-equivalent to the original pinned implementation. The new notebook reuses `MyDrive/exp026/features` only when model/runtime/processor/source/image checksums match. Training uses a fresh `MyDrive/exp026/exploratory/exploratory_run_<source>` namespace. Pilot/scientific heads/adapters are never warm starts. Epoch progress stores current parameters, optimizer/scaler, counters and best-so-far parameters; completed selected models and label-free held predictions have checksums/identities and resume without retraining/re-predicting. New source/config/input/runtime identity requires a fresh run. Abrupt loss of a current incomplete epoch may require restarting that epoch.

## Archived pilot-based forecast, not a new benchmark

| Component | Projection |
| --- | ---: |
| L0 two training epochs at6.73738 pairs/s |2897s /48.3min |
| L0 two validation passes |106s /1.8min |
| L0 held prediction pass |136s /2.3min |
| All frozen state extraction, cold (scaled from164s /240 paintings) |912s /15.2min |
| Total, fully compatible cached features |4881s /81.3min |
| Total, cold features |6021s /100.3min |

Validation/inference scale the pilot's493-pair/24.195s validation measurement. Totals include frozen-head work,120s checkpoint allowance,25% compute margin,10min setup and3min export. Partial pilot cache coverage lies between endpoints if scaling holds. Downloads, Drive I/O, image sizes and sequence lengths may change actual cost. The2-hour allocation is a hard maximum including setup; it is not a guarantee. A budget/OOM stop preserves completed artifacts and reports incomplete execution without changing scientific settings.

## Archived execution entry points — not authorization

The completed [EXP026_Exploratory_Colab.ipynb](../notebooks/EXP026_Exploratory_Colab.ipynb) and unchanged TASK-025C input ZIP are retained for provenance. No rerun is currently authorized. The original notebook defaults GPU authorization to false and caps allocation at2 hours; its persistent checkpoint/export behavior remains unchanged.

CLI inside the pinned environment (future authorized execution only):

```bash
python scripts/run_exp026.py --stage exploratory-fold --inputs /content/exp026_exploratory/inputs --cache /content/drive/MyDrive/exp026/features --run-dir /content/drive/MyDrive/exp026/exploratory/exploratory_run_SOURCE --model-cache /content/exp026/model --max-seconds 6600 --authorize-gpu
```

Unit tests use tiny CPU models and mocks: best-state selection when epoch2 worsens, exact completed-epoch resume, no refit, label-free held prediction reuse, full three-condition selection/scoring flow, split integrity, unequal fitting counts, source/cache preservation, hard2-hour timeout and notebook export/release on failures. They do not establish real L4 training, held-out performance or full-fold throughput. The owner-executed pilot notebook and all historical artifacts are preserved.

Local validation: **297 CPU tests passed,1 Windows symlink skip**; notebook nbformat schema/AST PASS. Original technical/pilot/scientific training functions have identical ASTs; original config and both feature extraction/training files retain their Git-content hashes. All499 protected files and70 local semantic-cache files are unchanged. Frozen E50+L CSV SHA-256 remains `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`. No GPU training was executed for this continuation.
