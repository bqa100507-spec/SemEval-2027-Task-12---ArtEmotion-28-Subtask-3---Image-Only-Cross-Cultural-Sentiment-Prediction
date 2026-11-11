# ST3 development baseline

Measured locally on 2026-11-09 using project commit `5785925719a0f11828bf817ffcb9731377369ffc`, Windows 11, Python 3.12.10, PyTorch 2.7.1+cu126, Transformers 4.49.0, scikit-learn 1.6.1, and an NVIDIA RTX 4060 Laptop GPU. The run downloaded the pinned 605,247,071-byte CLIP checkpoint and embedded the 1,656 paintings used in training or requested in development. The fit used 16,019 eligible training pairs in 25 languages and converged in 69 L-BFGS iterations.

| Run | Cultural Contrast ↑ | Macro Log Loss ↓ | All-pairs Log Loss ↓ |
| --- | ---: | ---: | ---: |
| Local CLIP + language LR | **0.446564** | **0.751995** | 0.763517 |
| Local training-language prior | 0.364707 | 0.867784 | 0.876041 |
| Organizer CLIP + language LR reference | 0.448 | 0.750 | — |
| Organizer language prior reference | 0.364 | 0.868 | — |

Log Loss is in bits per annotator. Contrast scored 82 of 85 development paintings. Two had fewer than three scored languages and one had constant gold. Macro Log Loss included 17 of 22 development languages with at least 20 pairs. The all-pairs mean matches the starter notebook's aggregation and is shown separately. The CLIP run was 0.001436 below the published Contrast and 0.001995 above the published Log Loss. These are local measurements using the reconstructed scorer. [Pinned organizer baseline results](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/baselines/st3/README.md#results-development-data)

## Commands verified

```powershell
py scripts/setup.py
.venv/Scripts/python.exe scripts/run_st3.py audit --official --output runs/data_audit.json
.venv/Scripts/python.exe scripts/run_st3.py baseline
.venv/Scripts/python.exe scripts/run_st3.py evaluate --predictions runs/baseline_dev/predictions.csv --output runs/baseline_dev/independent_metrics.json
.venv/Scripts/python.exe scripts/run_st3.py submission
.venv/Scripts/python.exe scripts/run_st3.py validate runs/baseline_dev/submission.zip
.venv/Scripts/python.exe scripts/run_st3.py baseline --run-dir runs/baseline_dev_repeat
.venv/Scripts/python.exe -m pytest -q
```

- Setup verified the existing default checkout and separately cloned and verified a fresh organizer checkout at a temporary `--data-dir`. In a clean project clone, `py -3.12 scripts/setup.py --torch cpu --data-dir <pinned-checkout>` created a fresh virtual environment, installed PyTorch 2.7.1+cpu, and passed `pip check`, both audits and the 26 tests. The first CUDA wheel install timed out at 2.3 of 2.7 GB on `download-r2.pytorch.org`. The retry succeeded with pip 25.2, a 120-second timeout and resumable downloads, which are now part of setup.
- The standalone evaluator returned the same scores as the baseline command. A separate standard-library calculation from the ZIP and public validation votes agreed on both metrics to `1e-12`. The cached rerun produced byte-identical prediction CSVs and `metrics.json`. A two-image CPU smoke loaded the cached CLIP checkpoint offline and produced finite, unit-normalized 512-dimensional embeddings.
- All 26 lightweight tests passed. The development ZIP passed validation with 849 unique prediction rows and exactly one top-level `predictions.csv`. Local archive `runs/baseline_dev/submission.zip` is ready for manual upload. Its SHA-256 is `17fa5d0b08f6dc9128444f89aad5a8e42092e4d9b864ea39e9a95f1665fcc83d`.
