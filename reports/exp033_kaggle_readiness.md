# EXP-033 · Kaggle handoff and engineering verification

**Local pretrained CUDA PASS. Kaggle T4 and cross-session publication pending.** The technical receipt covers 128 original P2 fitting paintings from outside fold 1. It contains no held-out performance results. [Preregistration](exp033_preregistration.md) · [Research strategy](exp033_representation_learning_strategy.md)

| Pin or artifact | Value |
| --- | --- |
| Worker commit | [`d54ad4d7b030aaeaf768088eff93292ae7cd4793`](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/commit/d54ad4d7b030aaeaf768088eff93292ae7cd4793) |
| Private data `exp033_inputs.zip` | `3bfb284878c886c7313875f909171d3d015ee7860790375a7f873e904b106d84` |
| Input manifest | `21f47c3d3c44530b2ea55d12368b46bf1050d665de607094bce591283baf1a82` |
| Runtime source bundle `exp033_runtime.zip` | `1ba0f6c3a7d7be4bc49a65f16022e24377ed0047a5b2fad53a4053dbeb9c9793` |
| Local verified preflight export | `d242d21db2ae6b0e5f92aff75e3899ab9dd18cf975bb33ee2eaa3a24cf4f680e` |

The private data bundle contains 1,335 images, 13,600 P2 pairs, and the matched reference and side-head predictions. Both historical predictions match their archived SHA-256. Reconstructed component probabilities recover every archived ridge blend exactly. The runtime bundle is a Git archive of the worker commit. It installs pinned direct dependencies in Python 3.12.10 via the notebook and records the resolved environment. Runtime/model downloads need Kaggle Internet unless a verified model directory is already attached. The runtime ZIP does not include Python wheels or weights. Keep the input bundle private. [Kaggle CLI and outputs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

## Local engineering check

The real pinned checkpoint and processor files matched their SHA-256. The RTX 4060 Laptop used FP16 autocast with FP32 trainables and microbatch 4. Native and cached outputs matched with max absolute difference **0** for valid tokens and the native global vector in the parity batch. The zero-adapter check also gave **0**. Across 128 paintings, measured suffix training was **57.68 paintings/s**, cached frozen training **234.00 paintings/s**, and decode plus extraction **5.67 s**. Peak CUDA allocated/reserved was **512.31/576.00 MiB** after the CUDA allocator reset. D had **98,304** trainable adapter weights plus **49,408** pooling and **92,184** head parameters for the 24-language diagnostic subset. Mean adapter gradient norm was **0.018268**. Clipping and scaler skips were both 0/8 in the measured suffix epoch. Both matrices got finite gradient and adapters updated.

Exact adapter replay passed after optimizer/scaler/RNG restoration. An independent process then restored and verified the exported canary from the ZIP. The initial Windows attempt exposed PyTorch's rejection of a dot-prefixed temporary filename. The worker now writes through an open binary stream. A CPU regression also exposed optimizer-state tensor aliasing on repeated load, fixed by copying the loaded optimizer state. Historical failure logs are retained separately. Neither fix changes the scientific recipe.

The measured RTX 4060 rates project about **0.48 GPU-hours for suffix work** and **0.91 GPU-hours of measured-phase compute** for the predefined workload. These are short-sample extrapolations. They exclude platform startup, private artifact round-trips and interruptions, and are not measurements for T4. The actual Kaggle preflight emits a separate projection for its own hardware. The strategy's 3–8 GPU-hour allowance remains provisional until that test.

## Owner run sequence

1. Upload the runtime and data ZIPs to private Kaggle datasets and attach both to [EXP033_Kaggle.ipynb](../notebooks/EXP033_Kaggle.ipynb). Adjust paths to match the mounted files. Select GPU T4 and enable Internet for setup. Use one T4 with `DEVICE="cuda:0"`. Keep the second GPU unused for the first feasibility check.
2. Set `STAGE="preflight"` and `AUTHORIZE_STAGE=True`, leave full CV authorization false, and Run all. Read peak reserved VRAM, cache parity, throughput, gradients and the measured-workload projection. Download or publish the verified preflight ZIP and retain its SHA. Publish it as a private dataset before the runtime stops.
3. In a fresh Kaggle session attach the published preflight dataset. Set its mounted ZIP path and SHA, set `STAGE="verify-publication"`, and Run all. The worker loads the hash-checked optimizer/scaler checkpoint in the matching pinned environment. This step is the required observed Kaggle durability check. The local Windows receipt cannot authorize a T4 run.
4. After reviewing those results, select `STAGE="cv"` and explicitly set `AUTHORIZE_FULL_CV=True`. Set the worker budget so enough session time remains to export. The defaults leave the run disabled. For multiple sessions, choose fold IDs, publish the resulting CV ZIP, and attach it as `RESUME_CV_ZIP` with its SHA for the next session. Use the same source, inputs and GPU type. Each verified completed fold can be reused. Checkpoints store complete epochs. Loss of a runtime can lose any epochs still only in Kaggle temporary storage. Outer scoring waits for all five fold seals.

Use the [CPU tests](../tests/test_exp033.py) for mask, equivalence, gradients, split, artifact, epoch replay and gate checks. This task has not run real P2 CV, confirmation, Development, or any leaderboard submission. E50+L, EXP-032 artifacts and PR #37 remain preserved.
