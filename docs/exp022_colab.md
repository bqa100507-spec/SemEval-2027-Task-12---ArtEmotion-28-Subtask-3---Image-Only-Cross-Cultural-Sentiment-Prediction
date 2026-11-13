# EXP-022 bounded Colab handoff

Choose a CUDA runtime with enough budget for the measured pilot forecast. FP32 CLIP-B is intentional for historical cache parity; do not switch encoder precision, rank, learning rate, preprocessing or folds to improve scores. Local Screen blocks when its forecast exceeds four GPU-hours. A new runtime must pass its own parity, memory and full-epoch pilot checks before allocating more training.

On the original Windows machine:

```powershell
.venv\Scripts\python.exe scripts/package_exp022.py
Get-Content runs/exp022_visual_lora/portable_handoff.json
```

The package is approximately the existing CLIP-B checkpoint plus 1,335 training images. It stays local/ignored. Transfer it to your private Colab storage; do not commit it or make it public. Keep the printed ZIP SHA-256 and input-manifest SHA-256 separately. Packaging is idempotent and checksums every file. Only development-training targets are exported, from the already-verified EXP-021 proof.

Use Python 3.12 and the pinned dependencies. If the Colab interpreter differs, create a Python 3.12 environment first; do not silently substitute a new Torch/Transformers version. In a Colab shell after privately attaching the ZIP:

```bash
git clone --branch exp/022-visual-lora-vote-distribution https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction.git st3
cd st3
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
sha256sum /content/exp022_colab_inputs.zip  # Must equal portable_handoff.json
mkdir -p /content/exp022-inputs
.venv/bin/python -m zipfile -e /content/exp022_colab_inputs.zip /content/exp022-inputs
```

Set the **actual published** input SHA, never a replacement derived from untrusted uploaded bytes:

```bash
INPUT_SHA=fa2f7c4528b49925c400216037a64381bc6a0b35a1b3b11458a6bcbfda924337
BASE="--portable-dir /content/exp022-inputs --input-manifest-sha256 $INPUT_SHA --run-dir /content/drive/MyDrive/exp022_colab --device cuda"
.venv/bin/python scripts/run_exp022.py Preflight $BASE --max-seconds 600
.venv/bin/python scripts/run_exp022.py Pilot $BASE --max-seconds 900
cat /content/drive/MyDrive/exp022_colab/pilot/runtime_projection.json
# Only after inspecting the forecast and allocating the time:
.venv/bin/python scripts/run_exp022.py MiniScreen $BASE --max-seconds 14400
.venv/bin/python scripts/run_exp022.py MiniScreenVerify $BASE --max-seconds 900
.venv/bin/python scripts/run_exp022.py Screen $BASE --max-seconds 14400
.venv/bin/python scripts/run_exp022.py ScreenVerify $BASE --max-seconds 1800
```

Mount private Drive first or replace RunDir with other persistent storage. Stage calls are separate and do not automatically proceed. Repeating a completed stage checks seals; repeating interrupted training reuses complete models and restarts only the interrupted model. A failed replay invalidates promotion until a later complete replay passes. Preserve all logs and directories when transferring results back. Run verification on the same hardware/software profile used for training; do not mix local and Colab checkpoints under one identity.

There is no EXP-022 confirmation/Development/upload command. Full CV remains separately gated and requires explicit owner invocation with `--allow-full-cv`; this handoff does not authorize it. Portable mode trusts the locally verified, checksum-pinned parent proof; it does not claim to replay the entire historical project on Colab. E50+L production remains unchanged.
