# ArtEmotion-28 Subtask 3

Image-only cross-cultural sentiment prediction for SemEval 2027 Task 12. The competition release uses **E50+L**, the hosted-verified CLIP-B, SigLIP 2 Base and CLIP-L ensemble. The project owner reported Development Contrast **0.4643** for submission **#1046158**. Use the [competition release manifest and evaluation-day commands](docs/competition_release.md) for frozen inference.

The sections below retain the original baseline setup and EXP-001–011 reproduction commands. The complete EXP-001–015 history is preserved on [archive/exp001-015-verified](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/tree/archive/exp001-015-verified). [PR integration audit](docs/release016_pr_audit.json)

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

## EXP-008 full-image CLIP coverage

Reuse the original CLIP cache and the EXP-002, EXP-006 and EXP-007 artifacts. Run the label-free crop audit, then compare the original, full-image and dual-view E50 variants.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py coverage-audit
.venv/Scripts/python.exe scripts/run_st3.py coverage-cv
.venv/Scripts/python.exe scripts/run_st3.py coverage-dev
```

CV uses CUDA for the new CLIP cache and CPU for the Logistic Regression fits. The development command verifies the frozen CV decision and scores and packages a candidate only if it qualifies. This run retained E50. Results are in `runs/exp008_full_image/`. Use the same new `--run-dir` across stages for repeats. [Protocol and results](reports/exp008_full_image.md)

## Frozen ST3 inference

Use `prepare-e50` and `prepare-e50l` once to persist the verified heads. The independent `predict` command takes an explicit ST3 CSV and artwork directory, writes a validated archive, and resumes new-image encoding from its per-image cache. E50+L is the default. Select `--mode e50` or `--mode clip` for a fallback. [Windows commands and evaluation setup](docs/e50l_inference.md)

## EXP-010 frozen CLIP-L ensembles

Reuse the Task 001, EXP-002, EXP-006 and EXP-007 artifacts, then run `clip-large-cv` followed by `clip-large-dev` with `scripts/run_st3.py`. CV builds the pinned CLIP-L cache on CUDA and compares the four fixed systems on CPU. The second command evaluates a new candidate only if it clears the frozen gate. Both commands use `runs/exp010_clip_large/` by default. Use a shared new `--run-dir` for a repeat. [Protocol and results](reports/exp010_clip_large.md)

## EXP-017 human-caption supervision

The caption study compares matched BCE-only, CLIP-text and provisional multilingual-E5 students against refitted E50+L. It locks a new 15% confirmation holdout and uses five new grouped folds on the remaining original training paintings. Run `caption-audit`, the two `caption-extract --teacher clip/e5` commands, `caption-cv` and the guarded `caption-holdout` and `caption-dev` stages with `scripts/run_st3.py`. Use `caption-token-audit --teacher clip/e5` for all-training tokenizer diagnostics. [Frozen protocol and results](reports/exp017_text_supervision.md)

## EXP-018 polarity-aware captions

Use `scripts/run_exp018.ps1` with `-Stage Preflight`, `Audit`, `Pilot`, `CV`, `Verify`, `Confirm` or `Dev`. The owner completed the active run under `runs/exp018_polarity_captions/prepared/`: all 45 checkpoints passed deep replay, and the frozen decision is NO_GO. Its preparation report remains the historical handoff record; [EXP-019](reports/exp019_gradient_scaling.md) records the verified local completion and reused scores.

## EXP-019: text scaling and gradient transfer

EXP-019 completed locally with COMPETITION_NO_GO after all fifteen new checkpoints passed replay. Lambda=1.0 student pooled Contrast was 0.417569215; its ensemble scored 0.446633867. [EXP-020](reports/exp020_emotion_specific_captions.md) records the verified local evidence; the [EXP-019 report](reports/exp019_gradient_scaling.md) remains the original preparation record.

## EXP-020: eight-emotion caption targets

EXP-020 completed locally with COMPETITION_NO_GO after independent thirty-checkpoint/ten-target replay. E8 pooled Contrast was 0.439654679 versus A8 0.440223388; both ensembles fell below promotion. [EXP-021](reports/exp021_vote_distribution.md) records the verified local completion. The [EXP-020 report](reports/exp020_emotion_specific_captions.md) remains the historical preparation record.

## EXP-021: annotation-only vote distribution

Reliability audit and matched D0/D8 correctness pilot passed: **READY_FOR_LOCAL_RUN**. Eight empirical emotion counts reconstruct the binary votes exactly. No caption/text features or new checkpoint is used. Conservative preparation + eight-fit screening/replay estimate: 25.15 minutes, so Screening is an owner command:

```powershell
.\scripts\run_exp021.ps1 -Stage Screen
.\scripts\run_exp021.ps1 -Stage ScreenVerify
```

For a fresh run, use `Preflight`, `Audit`, then `Pilot`. Only verified PILOT_PROMISING permits a separate `-Stage CV`, followed by `-Stage Verify`; the eight screening checkpoints are reused and only 22 missing fits are added. INCONCLUSIVE/UNPROMISING stops full CV. No confirmation, Development or LoRA stage exists. Paths, device and budgets are explicit options. [Audit assumptions, gradient/CE diagnostics, frozen gates and commands](reports/exp021_vote_distribution.md). Production E50+L remains unchanged.

## Independent checks

```powershell
.venv/Scripts/python.exe scripts/run_st3.py audit --official --output runs/data_audit.json
.venv/Scripts/python.exe scripts/run_st3.py evaluate --predictions runs/baseline_dev/predictions.csv
.venv/Scripts/python.exe -m pytest -q
```

The local scorer implements the published specification. It also reports the starter kit's all-pairs Log Loss as a separate number. See [competition and provenance](docs/competition.md), [data audit](reports/data_audit.md), [baseline results](reports/baseline_dev.md), and [project state](docs/project_state.md).


## EXP-022: controlled visual LoRA and vote supervision

EXP-021 completed local five-fold CV and independent thirty-checkpoint replay with COMPETITION_NO_GO; [EXP-022](reports/exp022_visual_lora_vote_distribution.md) records the verified evidence without rewriting historical reports. The new F0/F8/L0/L8 factorial design matches painting-grouped batches, original student, vote objectives and seeds. Only the pinned CLIP-B visual q/v adapters can update; SigLIP2/CLIP-L remain cached and frozen. Production E50+L is unchanged.

```powershell
.\scripts\run_exp022.ps1 -Stage Preflight -Device cuda -MaxSeconds 600
.\scripts\run_exp022.ps1 -Stage Pilot -Device cuda -MaxSeconds 900
```

Read the measured pilot forecast before starting MiniScreen. Training stops at completed-model boundaries; long stages require an explicit time budget. MiniScreenVerify is mandatory before Screen; ScreenVerify is mandatory before any separately owner-invoked CV. No five-fold run, locked scoring, Development scoring or upload is automatic. See the report for all stages, resource gates and [portable Colab commands](docs/exp022_colab.md).
