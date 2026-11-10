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

## EXP-002 baseline ablation

After the baseline run, reuse its cached CLIP features on CPU. The following commands add scientific figures and run the optional train-only grouped cross-validation.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-analysis.txt
.venv/Scripts/python.exe scripts/run_st3.py ablation --cv --plots
```

The core ablation, shuffling diagnostic, and paired bootstrap also run with both flags omitted. If the baseline used custom paths, pass its `run.json` via `--embedding-run` and its cache directory via `--cache-dir`. Outputs go to `runs/exp002_baseline_ablation/`. Choose a new `--run-dir` for repeats. Inference-time shuffled predictions are diagnostic artifacts. [Study protocol and results](reports/exp002_ablation.md)

## EXP-003 low-rank visual interactions

After EXP-002, run train-only selection and then evaluate the frozen candidate. Install `requirements-analysis.txt` for the spectrum plot.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py low-rank-cv
.venv/Scripts/python.exe scripts/run_st3.py low-rank-dev --plots
```

The commands reuse the Task 001 cache and verify the exact EXP-002 fold membership. Use `--exp002-dir`, `--embedding-run` and `--cache-dir` for custom input locations. The first command saves the decision in `runs/exp003_low_rank/cv/selection.json`. The second verifies that decision and writes the full-training Model C spectrum and, when a low-rank candidate qualifies, its development predictions and analysis. Choose a new `--run-dir` for any repeat. [EXP-003 report](reports/exp003_low_rank.md)

## EXP-004 contrast-aware learning

With the same Task 001 cache and EXP-002 fold artifact, run train-only selection before the development stage.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py contrast-cv
.venv/Scripts/python.exe scripts/run_st3.py contrast-dev
```

The first command writes `runs/exp004_contrast_aware/cv/selection.json`. The second verifies that decision and scores a new development candidate only if the train-only criteria are met. Use a new `--run-dir` for repeats. [EXP-004 protocol and results](reports/exp004_contrast_aware.md)

## EXP-005 frozen visual representations

Reuse the Task 001 CLIP cache and EXP-002 fold artifact. The first command can build the pinned DINOv2 cache separately. The second also builds it if needed and writes the frozen CV decision.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py visual-embed
.venv/Scripts/python.exe scripts/run_st3.py visual-cv
.venv/Scripts/python.exe scripts/run_st3.py visual-dev
```

The default device selection uses CUDA when available. Pass `--device cpu` to run extraction on CPU and use the same device option for both stages. Results are under `runs/exp005_visual_representation/`. A qualifying candidate produces a separate validated ZIP for manual use. Use a new `--run-dir` for a repeat. [Rules, provenance and results](reports/exp005_visual_representation.md)

## EXP-006 frozen SigLIP 2 representation

After EXP-002, use the dedicated extraction environment, then use the unchanged baseline environment for selection and the frozen development stage.

```powershell
.venv/Scripts/python.exe scripts/setup_siglip2.py
.venv-siglip2/Scripts/python.exe scripts/run_st3.py siglip-embed
.venv/Scripts/python.exe scripts/run_st3.py siglip-cv
.venv/Scripts/python.exe scripts/run_st3.py siglip-dev
```

All three run commands share `runs/exp006_siglip2/` and its extraction manifest. Choose the same new `--run-dir` across stages for repeats. Pass `--device cpu` to the extraction command to select CPU. [EXP-006 report](reports/exp006_siglip2.md) · [Research synthesis](reports/research_synthesis_001_006.md)

## EXP-007 CLIP and SigLIP 2 fusion

Reuse the existing CLIP and SigLIP 2 feature caches and the EXP-002 and EXP-006 artifacts.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py fusion-cv
.venv/Scripts/python.exe scripts/run_st3.py fusion-dev
```

CV evaluates the three fixed probability blends and one normalized concatenation, then freezes the selection. The development command fits the chosen head or heads on all original training pairs and validates its separate ZIP. Both stages write to `runs/exp007_fusion/`. Use the same new `--run-dir` for a repeat. [Protocol and results](reports/exp007_fusion.md)

## Independent checks

```powershell
.venv/Scripts/python.exe scripts/run_st3.py audit --official --output runs/data_audit.json
.venv/Scripts/python.exe scripts/run_st3.py evaluate --predictions runs/baseline_dev/predictions.csv
.venv/Scripts/python.exe -m pytest -q
```

The local scorer implements the published specification. It also reports the starter kit's all-pairs Log Loss as a separate number. See [competition and provenance](docs/competition.md), [data audit](reports/data_audit.md), [baseline results](reports/baseline_dev.md), and [project state](docs/project_state.md).
