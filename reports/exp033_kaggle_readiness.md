# EXP-033 · Kaggle handoff and engineering verification

**Prior local pretrained CUDA preflight PASS. Revised one-click Kaggle T4 execution pending.** The technical receipt covers 128 original P2 fitting paintings from outside fold 1. It contains no held-out performance results. [Preregistration](exp033_preregistration.md) · [Research strategy](exp033_representation_learning_strategy.md)

| Pin or artifact | Value |
| --- | --- |
| Worker commit | [`9fdc3eeaf989d394ebb960a441d1d3496ac8f2ae`](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/commit/9fdc3eeaf989d394ebb960a441d1d3496ac8f2ae) |
| Private data `exp033_inputs.zip` | `3bfb284878c886c7313875f909171d3d015ee7860790375a7f873e904b106d84` |
| Input manifest | `21f47c3d3c44530b2ea55d12368b46bf1050d665de607094bce591283baf1a82` |
| Runtime source bundle `exp033_runtime_oneclick.zip` | `055244538f4c1377f6ab9bd3dc0469018016ebf194c77d6894e074d723a11bbb` |
| Local verified preflight export | `d242d21db2ae6b0e5f92aff75e3899ab9dd18cf975bb33ee2eaa3a24cf4f680e` |

The private data bundle contains 1,335 images, 13,600 P2 pairs, and the matched reference and side-head predictions. Both historical predictions match their archived SHA-256. Reconstructed component probabilities recover every archived ridge blend exactly. The revised runtime bundle comes from the pinned worker commit. It installs pinned direct dependencies in Python 3.12.10 via the notebook and records the resolved environment. Runtime/model downloads need Kaggle Internet unless a verified model directory is already attached. The runtime ZIP does not include Python wheels or weights. Keep the input bundle private. [Kaggle CLI and outputs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

## Local engineering check

In the previous worker revision, the real pinned checkpoint and processor files matched their SHA-256. The RTX 4060 Laptop used FP16 autocast with FP32 trainables and microbatch 4. Native and cached outputs matched with max absolute difference **0** for valid tokens and the native global vector in the parity batch. The zero-adapter check also gave **0**. Across 128 paintings, measured suffix training was **57.68 paintings/s**, cached frozen training **234.00 paintings/s**, and decode plus extraction **5.67 s**. Peak CUDA allocated/reserved was **512.31/576.00 MiB** after the CUDA allocator reset. D had **98,304** trainable adapter weights plus **49,408** pooling and **92,184** head parameters for the 24-language diagnostic subset. Mean adapter gradient norm was **0.018268**. Clipping and scaler skips were both 0/8 in the measured suffix epoch. Both matrices got finite gradient and adapters updated.

Exact adapter replay passed after optimizer/scaler/RNG restoration. An independent process then restored and verified the exported canary from the ZIP. The initial Windows attempt exposed PyTorch's rejection of a dot-prefixed temporary filename. The worker now writes through an open binary stream. A CPU regression also exposed optimizer-state tensor aliasing on repeated load, fixed by copying the loaded optimizer state. Historical failure logs are retained separately. Neither fix changes the scientific recipe.

The measured RTX 4060 rates project about **0.48 GPU-hours for suffix work** and **0.91 GPU-hours of measured-phase compute** for the predefined workload. These are short-sample extrapolations. They exclude platform startup, private artifact round-trips and interruptions, and are not measurements for T4. The actual Kaggle preflight emits a separate projection for its own hardware. The strategy's 3–8 GPU-hour allowance remains provisional until that test.

## Owner run

1. Attach the existing private dataset to [EXP033_Kaggle.ipynb](../notebooks/EXP033_Kaggle.ipynb), select T4 and enable Internet. The notebook finds Kaggle's recursively extracted exp033_inputs and exp033_runtime_oneclick directories, or the original ZIPs when present. It verifies the pinned manifest and runtime files. Provide a path override only if needed.
2. Set `AUTHORIZE_RUN = True` and click **Run all** once. The worker checks the pretrained model on the selected T4 and verifies the local sealed optimizer/scaler checkpoint. On PASS it runs all five folds immediately. A failed preflight ends the run and exports diagnostics.
3. Sealed epochs and logs are written to Kaggle Working as the run proceeds. The final verified ZIP and SHA-256 receipt are created automatically on success or handled failure. Save or download the archive before ending the session.

For optional resume, attach a previously exported one-click run archive and enter its path and SHA-256. Run all rechecks hardware, source and input identity, holds the original microbatch fixed, and restores the last valid completed epoch. It reuses completed folds. A hard runtime loss may discard previously unsaved Kaggle Working files. The older multi-stage handoff and its technical receipt are archival.

CPU tests cover masks, gradients, splits, artifact integrity, epoch replay, and the single-run preflight gate. GPU was not invoked for this workflow patch. Production E50+L, EXP-032 artifacts and PR #37 remain preserved.
