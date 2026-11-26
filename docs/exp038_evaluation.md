# EXP-038 · Evaluation-phase inference runbook

**Prepared and tested on CPU fixtures only.** A real fresh-pixel GPU smoke test is still owner work. [Kaggle runbook](exp038_kaggle.md)

The deployment path uses artwork pixels and the requested target language only. Captions, labels, metadata and other languages' answers never enter it.

## 1. Prepare the bundle (CPU, once)

After the Kaggle run and its separate audit, extract the deployment bundle from the verified results archive:

```powershell
.venv/Scripts/python.exe scripts/deploy_exp038.py prepare --results runs/exp038/closure/exp038_results.zip --output runs/exp038/deployment --identity <overlay-manifest-sha256>
```

The bundle holds the refitted heads (`heads.npz`, including E4's pair term), the cross-fitted neighbor banks for R0 and every available candidate (`banks.npz`), the 1,571 training paintings' similarity features (`train_features.npz`), the two new CS-CVA seed bundles, `decision.json`, `selection.json` and a hash index.

## 2. Components supplied from other pipelines

- **EXP-036 CS-CVA**: run `scripts/run_exp036.py infer` with the original EXP-036 bundle and with each new seed bundle (`cscva/seed_203836/inference`, `cscva/seed_203936/inference`; same `bundle.json` format) on the same official input, then pack each CSV as a ZIP. Pass one ZIP (original) or three (original, seed 203836, seed 203936; CS-CVA-3 is their round6 mean).
- **EXP-037 C6**: the standalone ZIP from [its inference guide](exp037_evaluation.md) for the same keys.

## 3. Infer

On a CUDA machine with the project `.venv` (Track 1 CLIP-B / CLIP-L workers and the `.venv-siglip2` SigLIP2-Base worker, as in Track 1 `predict`) plus Transformers 4.57.6 and `qwen-vl-utils` 0.0.14 for E1–E4:

```powershell
.venv/Scripts/python.exe scripts/deploy_exp038.py infer --bundle runs/exp038/deployment --input data/eval_st3/st3_test.csv --images data/eval_st3 --cscva cscva_original.zip --cscva cscva_seed_203836.zip --cscva cscva_seed_203936.zip --c6 exp037_c6.zip --output runs/exp038/january
```

It extracts the three Fixed Ridge features through the existing Track 1 workers and E1–E3 with the pinned encoders, E4 only for the requested pairs, applies the heads, mixes and banks, composes B1/B2, validates exact keys, order, six-decimal probabilities and ZIP structure, and writes every ZIP, `exp038_best_submission.zip` for the recipe in `selection.json`, and `inference_receipt.json` with input, image, bundle and component hashes. `--features` accepts a precomputed NPZ for CPU replay. Paintings that are also training bank paintings are rejected.

The CPU fixture test (`tests/test_exp038.py::test_deploy_prepare_and_infer_from_precomputed_features`) reproduces the run's Development ZIP members byte for byte from the prepared bundle. Validate the final ZIP against the official input with `scripts/run_st3.py validate-prediction` before any manual upload; the owner uploads.
