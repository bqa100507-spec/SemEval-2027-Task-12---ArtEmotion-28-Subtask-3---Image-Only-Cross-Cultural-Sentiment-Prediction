# EXP-022 on Kaggle

[Download EXP022_Kaggle.ipynb](https://raw.githubusercontent.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/6da257fc508a4039f8881b7ddb1e7f5068ecca2b/notebooks/EXP022_Kaggle.ipynb) and import it into a **private** Kaggle Notebook.

1. Create a **private Kaggle Dataset** with `exp022_colab_inputs.zip`.
2. Add that Dataset under the notebook's **Inputs**.
3. Select an available **GPU** and enable **Internet** for online setup.
4. **Run all**, then inspect the final benchmark summary.

No Google Drive, credentials, handwritten commands or model configuration edits are required. The notebook detects exactly one ZIP named `exp022_colab_inputs.zip` under `/kaggle/input/`; missing or duplicate matches stop with instructions. Set the explicit input-path field only to disambiguate attachments.

The same separately pinned helper and immutable `41ca0d4` training checkout used by Colab prepare Python **3.12.10**, Torch **2.7.1+cu126**, Transformers **4.49.0** and the original requirements. Host Python/Torch/Transformers, GPU/free memory and actual isolated dependency versions are recorded. Online helper/Git/pip errors stop with their first actionable error and full logs; Internet access is not assumed to exist. There is no CPU Torch fallback or dependency substitution. `CUDA_VISIBLE_DEVICES=0` restricts training to **one GPU**, even if two T4s are exposed.

Input assets and the environment use ephemeral local `/kaggle/temp/exp022/`, outside saved Notebook outputs. All experiment outputs use `/kaggle/working/exp022_runs/<runtime-identity>/`; setup logs are saved alongside it. The original ZIP, published input manifest, every extracted file, source and dependency checks must pass before Preflight/Pilot. Strict CLIP feature parity and real LoRA gradient/update audits remain mandatory. **No Kaggle GPU compatibility PASS is claimed here.** Available accelerators/drivers may fail the original strict parity or CUDA/version checks; preserve the failure and investigate rather than changing science.

## Optional training

Default Run all runs **Preflight → Pilot → report** only, with `RUN_LONG_TRAINING=False`. The controls offer MiniScreen, MiniScreenVerify, Screen and ScreenVerify, matching Colab. Review the forecast, peak VRAM and explicit budget, check authorization, then press the button. Prerequisites and the four-hour Screen resource gate are enforced; authorization resets. Full five-fold CV and final verification remain separately owner-authorized and are not exposed in notebook controls.

No stage is launched by a persisted widget state or Run all. A second Run all can reuse compatible sealed benchmark results, clearly labeled as saved timings. The original scientific model, data, losses, optimizer, folds, seeds, rounding and promotion gates remain frozen.

## Preserve results before ending a session

The helper exports timestamped **exp022-checkpoints-…zip** files and matching `.sha256` receipts directly under `/kaggle/working/`, including complete checkpoints, sealed stages, logs, dependency lock and failed replay-attempt records. Partial model directories are excluded. Use the latest export shown in the notebook; **download it and its receipt before ending the session**.

Kaggle session files alone do not guarantee persistence. Preserve the export as private versioned Notebook output or upload it to a private Dataset, and verify it is present in the saved output. A **Save & Run All** starts a fresh execution of the safe benchmark and does not replay interactive button clicks. Export existing interactive checkpoints first; do not assume a fresh commit run will save those earlier models. See [Kaggle's output documentation](https://www.kaggle.com/docs/notebooks) for saved outputs and adding prior outputs as inputs.

On a new session, attach the checkpoint ZIP **and its published receipt**, set `RESTORE_CHECKPOINTS=True`, and Run all. A unique attached `exp022-checkpoints-*.zip` is detected, or an explicit path can be supplied. The helper checks archive contents, scientific seals and the complete source/software/GPU/driver/dependency-lock identity **before** copying anything. Different valid artifacts are never overwritten. Failed verification records are retained, preserving gate invalidation. Recovery is at completed-model boundaries only; a partial current model restarts.

Switching platforms with a different GPU/software identity means starting a separate matched run. The experiment is identical, but its checkpoints are not silently interchangeable. Keep the old artifacts archived. A genuinely matching identity can restore; an incompatible one is rejected, without source/version fingerprint bypasses.

## Internet-disabled setup

The 705 MB data ZIP contains data/model assets and historical input proof, **not** Python, source and all wheels. Offline Kaggle needs two private attachments:

- Original `exp022_colab_inputs.zip` (fixed published SHA).
- Separately prepared `exp022_runtime.zip` **plus its producer's `.sha256` receipt**.

Create the runtime kit using **Build offline Kaggle kit** after a successful online Linux Colab benchmark. It contains the pinned helper, complete Git source bundle, managed relocatable Python 3.12.10, exact dependency lock and wheels. The build can download several GB and requires explicit authorization; it runs no model training. This offline packaging path is implemented and synthetically tested; an actual Linux cloud kit has not been built/tested in this Windows execution environment.

Attach the kit/receipt to Kaggle. `NETWORK_MODE=auto` selects it when present, or select `offline`. The notebook first checks the published runtime ZIP SHA and immutable helper SHA, then verifies every extracted asset, clones from the attached Git bundle and installs dependencies using `--no-index --find-links`. No GitHub/pip network request is needed in that path. Missing/incomplete/incompatible artifacts fail safely with instructions. Do not generate a replacement expected checksum from a failing ZIP.

## Pins and diagnostics

The [Colab provenance section](exp022_colab.md#immutable-provenance) records shared source/helper/input hashes. Both notebooks execute the same original `scripts/run_exp022.py` with the same published input SHA, CUDA device mode and stage budgets; paths differ by platform. The helper stays outside the frozen training checkout and does not alter its scientific fingerprints.

The summary includes stage PASS/FAIL, GPU/env compatibility, cache/zero-adapter parity, actual BCE/CE LoRA gradients/updates, allocated/reserved peaks, epoch time and paintings/s, MiniScreen/Screen estimates, the next-stage resource gate and exact output location. Complete diagnostics stay in saved logs. No held-out scientific decision is inferred from a successful benchmark.

**Validation boundary:** structural/compiled notebooks, mocked platform execution, original ZIP checks, failed environments, stage arguments, offline source/wheel orchestration and checkpoint restore/export were tested locally. This is **not** a Kaggle or Colab GPU runtime certification. The owner still needs to attach artifacts, select a GPU and Run all. Internet/account availability, frontend widgets, driver compatibility, strict numerical parity and offline wheel installation require that real runtime check.
