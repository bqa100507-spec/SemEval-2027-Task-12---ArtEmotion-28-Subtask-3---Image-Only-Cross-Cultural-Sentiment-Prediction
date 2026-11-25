# EXP-037 archived candidate inference

The official run selected A4, B6 and C6. All three inference bundles and their original standalone and ensemble outputs are preserved. The selected **original** C6 blend uses 80% of the already-rounded EXP-036 incumbent plus 20% C6, then rounds to six decimals. The **post-hoc** alternative uses a flat 65% Corrected Ridge, 15% EXP-036 CS-CVA and 20% C6 mix. This is a separate Development-selected candidate, with lower Evaluation-like Contrast and higher Log Loss than the original C6 blend. EXP-036 remains the incumbent. Both EXP-037 candidates have unresolved Qwen ST3 eligibility. [Closure](../reports/exp037_final.md)

## Bundle and provenance

- Original downloads are unchanged in `runs/exp037/closure/`. All 18 sealed training checkpoints remain in `exp037_results.zip`. The copied inference artifacts are in `runs/exp037/deployment/{A4,B6,C6}/`. Each includes the trained fusion, sentiment heads, both adapter groups, language mapping, config, archived predictions, submission ZIPs and original stage seal. An unseen official language uses the shared logit plus the mean fitted language intercept.
- The sealed SigLIP2 So400m patch14-384 revision is `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`. Qwen3-VL-4B-Instruct is `ebb281ec70b05090aa6165b016eac8ec08e71b17` with NF4 double quantization and FP16 compute. The worker checks weights and processor assets. Its fixed prompt is `Inspect this artwork.` Qwen runs once per painting with native processor geometry and uses final contextual image-token states.
- Use Python 3.12.10 with the pinned CUDA 12.6 environment defined by the copied `requirements-exp035-kaggle.lock.txt` and `requirements-exp036-kaggle.txt`. The Kaggle worker used a T4 and Torch 2.7.1+cu126. Use native Linux/Kaggle or matching runtime and driver. Follow the [EXP-036 runbook](exp036_evaluation.md) for its separate pretrained visual and Ridge runtimes. A fresh image run needs both verified base snapshots. Supply `--siglip` and `--qwen` for mounted snapshots, or enable the pinned model downloads.

| Selected weights or receipt | SHA-256 |
| --- | --- |
| A4 `inference.pt` | `122da706822b48487b8430eaa77bc2e7d7370eed59c40e1b41e754cdad955327` |
| B6 `inference.pt` | `dd4ec802a12ef120c5f95808b8760ff8f8026cd124ebba120f94817e1ea6fa03` |
| C6 `inference.pt` | `24e6853eb141a57272aa055ae52a5898f0073c3c0c3a5d7d24e1819877f167b2` |
| `deployment_index.json` | `57a9ddfdac7408ded386004498682270a210947513560c73de63e0da49a3c98b` |

## Real Evaluation input

Use the official input-only `st3_test.csv` and `images/` tree under a new Evaluation root. First generate the matched EXP-036 outputs for those same keys using [its Evaluation commands](exp036_evaluation.md). Supply the resulting ZIPs on the appropriate branch of this runbook. For path prefixes below, `python` is the isolated EXP-037 CUDA interpreter and paths are examples. Execute from the EXP-037 repository.

```sh
python scripts/deploy_exp037.py infer \
  --bundle /path/to/exp037/deployment --candidate C_original_blend \
  --input /path/to/eval/st3_test.csv --data-dir /path/to/eval \
  --component incumbent=/path/to/exp036/exp036_ensemble_submission.zip \
  --output /path/to/new/exp037_original_eval
```

For the post-hoc recipe, use a distinct output path and supply the separate CS-CVA and Corrected Ridge archives. The standard EXP-036 command emits the CS-CVA archive. Generate Corrected Ridge with the runbook's separate `--ridge-only` command and point to that output.

```sh
python scripts/deploy_exp037.py infer \
  --bundle /path/to/exp037/deployment --candidate posthoc_flat_65_15_20 \
  --input /path/to/eval/st3_test.csv --data-dir /path/to/eval \
  --component ridge=/path/to/ridge-output/corrected_ridge_submission.zip \
  --component cscva=/path/to/exp036/exp036_standalone_submission.zip \
  --output /path/to/new/exp037_posthoc_eval
```

For one standalone, specify `A_standalone`, `B_standalone` or `C_standalone` with no external components. The runner validates each component against the exact target input keys, applies the stored recipe, and writes the final ZIP and checksum receipt. The learned inference bundles omit the caption and emotion heads. Input is the target language and image pixels.

## Closure replay

From the EXP-037 checkout, use Python 3.12 with the project dependencies and exact inputs. Use fresh destination paths. These commands use CPU and reproduce the artifact checks, diagnostic tables and bundle preparation.

```sh
python scripts/audit_exp037.py --inputs /path/to/exp037_inputs.zip \
  --results /path/to/exp037_results.zip --best /path/to/exp037_best_submission.zip \
  --upstream /path/to/pinned/artemotion-28 \
  --ridge /path/to/corrected_ridge/submission.zip \
  --exp036-results /path/to/exp036_results.zip --output /path/to/new/closure
python scripts/diagnose_exp037.py --closure /path/to/new/closure --output /path/to/new/diagnostics
python scripts/deploy_exp037.py prepare --results /path/to/exp037_results.zip \
  --diagnostics /path/to/new/diagnostics --output /path/to/new/deployment
```

The `infer --replay` option uses the bundled Development prediction ZIPs and a pinned public Development tree specified with `--data-dir`. Supply the original saved EXP-036 component ZIPs for the recipe. CPU replay reproduced **both full prediction CSV members byte for byte**, including the six-decimal nested blend. The archived Kaggle worker had already replayed selected A/B/C checkpoints.

On 26 November 2026, fresh-pixel inference passed for the exact EXP-036 CS-CVA and EXP-037 C6 weights on an RTX 4060 Laptop GPU. The two new synthetic artworks yielded eight valid pairs per model, with variation across both image and language. End-to-end command times, including checkpoint read and model load, were 33.3 s for CS-CVA and 77.4 s for C6. Fresh Corrected Ridge CPU inference also passed on those artworks, and composing the real outputs produced validated eight-pair ZIPs for both EXP-037 recipes. These are portability smoke tests with no labels or quality score. The official hidden Evaluation inference run remains pending.
