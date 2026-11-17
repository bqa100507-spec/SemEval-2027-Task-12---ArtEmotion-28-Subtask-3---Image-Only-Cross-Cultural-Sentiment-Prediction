# EXP-026 — two-epoch exploratory fold

**Prepared, not GPU-executed. No held-out VLM Contrast result yet. E50+L remains production.**

The owner-completed L4 pilot passed: L0 validation BCE0.644959→0.581847; F0 epoch2 BCE0.611536. The saved local Colab notebook also contains the pilot's timing logs. These are owner-run observations, not checkpoint replays performed in this patch. The original scientific fold estimate was48,220.88 seconds (13.4h); it is not launched.

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

## Pilot-based cost estimate, not a new benchmark

| Component | Projection |
| --- | ---: |
| L0 two training epochs at6.73738 pairs/s |2897s /48.3min |
| L0 two validation passes |106s /1.8min |
| L0 held prediction pass |136s /2.3min |
| All frozen state extraction, cold (scaled from164s /240 paintings) |912s /15.2min |
| Total, fully compatible cached features |4881s /81.3min |
| Total, cold features |6021s /100.3min |

Validation/inference scale the pilot's493-pair/24.195s validation measurement. Totals include frozen-head work,120s checkpoint allowance,25% compute margin,10min setup and3min export. Partial pilot cache coverage lies between endpoints if scaling holds. Downloads, Drive I/O, image sizes and sequence lengths may change actual cost. The2-hour allocation is a hard maximum including setup; it is not a guarantee. A budget/OOM stop preserves completed artifacts and reports incomplete execution without changing scientific settings.

## Prepared execution

Open [EXP026_Exploratory_Colab.ipynb](../notebooks/EXP026_Exploratory_Colab.ipynb), select L4, use the unchanged TASK-025C `exp023_handoff.zip`, and obtain separate owner authorization before setting `AUTHORIZE_EXP026=True` and Run all. Default GPU authorization is false; `MAX_HOURS=2.0` cannot exceed2. The notebook only permits `exploratory-fold`, exports/checks outputs and attempts browser download before runtime release. Local images/weights remain in `/content`; completed artifacts persist on Drive.

CLI inside the pinned environment (future authorized execution only):

```bash
python scripts/run_exp026.py --stage exploratory-fold --inputs /content/exp026_exploratory/inputs --cache /content/drive/MyDrive/exp026/features --run-dir /content/drive/MyDrive/exp026/exploratory/exploratory_run_SOURCE --model-cache /content/exp026/model --max-seconds 6600 --authorize-gpu
```

Unit tests use tiny CPU models and mocks: best-state selection when epoch2 worsens, exact completed-epoch resume, no refit, label-free held prediction reuse, full three-condition selection/scoring flow, split integrity, unequal fitting counts, source/cache preservation, hard2-hour timeout and notebook export/release on failures. They do not establish real L4 training, held-out performance or full-fold throughput. The owner-executed pilot notebook and all historical artifacts are preserved.

Local validation: **297 CPU tests passed,1 Windows symlink skip**; notebook nbformat schema/AST PASS. Original technical/pilot/scientific training functions have identical ASTs; original config and both feature extraction/training files retain their Git-content hashes. All499 protected files and70 local semantic-cache files are unchanged. Frozen E50+L CSV SHA-256 remains `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`. No GPU training was executed for this continuation.
