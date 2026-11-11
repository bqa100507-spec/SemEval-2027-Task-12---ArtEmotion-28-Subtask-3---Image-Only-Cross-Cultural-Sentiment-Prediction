# ArtEmotion-28 Subtask 3

Image-only cross-cultural sentiment prediction for SemEval 2027 Task 12. The development workflow predicts the positive annotator share for each supplied artwork and target language using the organizers' frozen CLIP ViT-B/32 and weighted logistic regression. It also evaluates a training-only language prior.

## 1. Set up the workspace

Run from the repository root with Git and 64-bit Python. On Windows, the modern Python Install Manager can install Python 3.12 from another Python version. On Linux or WSL, make `python3.12` available first. Network access is needed on first setup and first model load.

| Platform | Setup command |
| --- | --- |
| Windows PowerShell | `py scripts/setup.py` |
| Linux / WSL | `python3.12 scripts/setup.py` |

Setup creates `.venv`, installs the pinned Python dependencies, and clones the pinned organizer release to `data/upstream/` with the public ST3 resources in the working tree. It runs both the organizer verifier and a separate ST3 audit, saving `runs/data_audit.json`. Repeat setup to verify the existing environment and data. Pass `--torch cpu` for a smaller PyTorch download or `--torch cu126` to select the CUDA 12.6 build explicitly. The default attempts the CUDA build when it detects an NVIDIA driver and can fall back to the CPU wheel.

## 2. Train and evaluate

```powershell
.venv/Scripts/python.exe scripts/run_st3.py baseline
```

On Linux or WSL, substitute `.venv/bin/python` in the commands. The first run downloads the pinned CLIP weights (~605 MB), embeds the images, fits the classifier, writes both prediction CSVs, and scores them against public validation annotations. A usable CUDA device is selected automatically, with CPU fallback. The local Hugging Face and image-feature caches default to `cache/`.

The default run is `runs/baseline_dev/`, with `predictions.csv`, `language_prior/predictions.csv`, `run.json` and `metrics.json`. Use `baseline --run-dir runs/baseline_dev_repeat` for a repeat run that can reuse cached embeddings. Use `baseline --device cpu` to force CPU execution. With an external checkout, pass `--data-dir PATH` to setup and place it **before** the `run_st3.py` subcommand.

## 3. Package and validate the development submission

```powershell
.venv/Scripts/python.exe scripts/run_st3.py submission
.venv/Scripts/python.exe scripts/run_st3.py validate runs/baseline_dev/submission.zip
```

The archive contains only a top-level `predictions.csv` with `painting,language,p_positive`. Upload `runs/baseline_dev/submission.zip` manually to the development phase. [ST3 on Codabench](https://www.codabench.org/competitions/18468/)

## Independent checks

```powershell
.venv/Scripts/python.exe scripts/run_st3.py audit --official --output runs/data_audit.json
.venv/Scripts/python.exe scripts/run_st3.py evaluate --predictions runs/baseline_dev/predictions.csv
.venv/Scripts/python.exe -m pytest -q
```

The local scorer implements the published specification. It also reports the starter kit's all-pairs Log Loss as a separate number. See [competition and provenance](docs/competition.md), [data audit](reports/data_audit.md), [baseline results](reports/baseline_dev.md), and [project state](docs/project_state.md).
