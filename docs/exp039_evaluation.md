# EXP-039 · Evaluation-phase inference runbook

**EXP-039 closed NO_GO: the January recipe stays EXP-038's [`B2_C2_cscva3`](exp038_evaluation.md).** This EXP-039 path is kept for reference only and was tested on CPU fixtures only. [Kaggle runbook](exp039_kaggle.md)

The path uses artwork pixels and the requested target language only.

```powershell
.venv/Scripts/python.exe scripts/deploy_exp039.py prepare --results runs/exp039/closure/exp039_results.zip --output runs/exp039/deployment --identity <overlay-manifest-sha256>
.venv/Scripts/python.exe scripts/deploy_exp039.py infer --selected-only --bundle runs/exp039/deployment --input data/eval_st3/st3_test.csv --images data/eval_st3 --cscva cscva_original.zip --cscva cscva_seed_203836.zip --cscva cscva_seed_203936.zip --c6 exp037_c6.zip --output runs/exp039/january
```

- `--selected-only` extracts only the arms of the selected candidate: E1, E2 and E3, plus the selected E5, E6 or E7.
- The three Fixed Ridge base features come from the Track 1 workers, as in EXP-038.
- CS-CVA-3 is the round6 mean of the three CS-CVA ZIPs; a single precomputed CS-CVA-3 ZIP is also accepted.
- The command writes every ZIP, `exp039_best_submission.zip` and `inference_receipt.json`.
- A CPU fixture test reproduces the run's best ZIP member byte for byte from the prepared bundle and precomputed features.

## Time estimate for 342 images (one T4 unless noted)

| Step | Basis | Estimate |
| --- | --- | --- |
| Fixed Ridge base three, E1, E2, E3 | EXP-038 run 4: E1 18.0, E2 38.8, E3 9.0 img/s | ≈ 1.5 min of GPU + loads, base three a few minutes |
| E6 (if selected) | 3 extra 2B instructions per image at E3's rate | ≈ 342 × 3 / 9.0 ≈ 2 min |
| E7 (if selected) | 1024² ≈ 4× E3's visual tokens; planning rate ≈ 2–3 img/s | ≈ 2–3 min |
| E5 (if selected) | 8B FP16 over two T4s; planning rate ≈ 2–4 img/s | ≈ 2–3 min + 16 GB download |
| CS-CVA-3 and C6 | as in [EXP-038](exp038_evaluation.md) | ≈ 5 min |

The E5–E7 rates are planning estimates until the EXP-039 Benchmark stage measures them. Replace this table with the run's `benchmark/benchmark.json` rates at closure. With setup, a full January run fits in well under an hour.
