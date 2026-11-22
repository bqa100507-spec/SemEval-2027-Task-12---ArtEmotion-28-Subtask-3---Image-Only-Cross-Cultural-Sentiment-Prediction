# EXP-035 manual Kaggle run

Use the generated `notebooks/EXP035_Kaggle.ipynb` and the matching private handoff
in `runs/exp035_kaggle/handoff/`. The notebook runs the existing matched fold-3
comparison: sealed Corrected Ridge, frozen Qwen residuals with a feature cache,
and the same residual head with live merger LoRA. Each trained arm has four
epochs and uses inner validation for checkpoint selection.

## Upload and attach

1. Create a **private Kaggle Dataset** from `exp035_fold3_inputs.zip` (84.3 MB)
   and attach it to your notebook. It includes the verified Ridge references,
   original prepared splits, P2 training inputs, 1,335 artwork files and the
   pinned worker source. Kaggle can extract the ZIP normally. Its expected
   mount is `/kaggle/input/exp035-fold3-inputs/exp035_inputs`. The notebook
   also finds nested mounts through `exp035_handoff.json`.
2. Attach a **separate private Kaggle Dataset or Model** containing the files
   listed in the manifest's `model.files` object. The existing local folder
   `cache/qwen3-vl-4b-ebb281e/` contains these 13 files. Reuse an existing
   Kaggle attachment if it contains the exact files. Otherwise upload those
   files, including the original `README.md`, from the local folder. Keep the
   model card, two shards, index, tokenizer and processor files together.
   The expected mount is `/kaggle/input/exp035-qwen`. The notebook also finds a
   single complete model folder recursively. Set `MODEL_ROOT` if more than
   one complete candidate is attached. The model revision is
   `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
3. Import `EXP035_Kaggle.ipynb` into a **private** notebook. Select **GPU L4**
   if offered, otherwise **GPU T4 ×2**. The worker sees only GPU 0. **Enable
   Internet** for Python 3.12.10 and dependency installation from PyPI and the
   PyTorch CUDA 12.6 index. Model loading uses the mounted files offline.
4. Set `AUTHORIZE_RUN = True` in Settings, then click **Run All**. Leave
   `RESUME_ARCHIVE` empty for a fresh run. The launcher runs preflight and
   continues automatically if the four-epoch forecast fits the remaining
   five-hour budget. Open the log under `/kaggle/working/exp035_fold3/` for
   progress. A budget stop leaves the measured preflight evidence.

Upload the input ZIP, not the standalone checksum or manifest. The manifest
is already in the ZIP. The ZIP SHA-256 is
`7f26ad070797f339c11de1f0cecaf55c0d91f31f6f69c5150852c66e6a276aed`.
The manifest SHA-256 is
`599e255faa46565cd8225f05c9a81b1c1ba631bf4aba44bc99b89221e121e32a`.
The packaged Kaggle worker revision is
`dc5aea2cff92835b59cae746075b6a11c9abf846`.

## Download and resume

Download `exp035_results.zip` and `exp035_results.zip.sha256` from **Output**
after the run. Saving a private notebook version with outputs is another
way to retain them. The ZIP contains the logs, budget record, config,
provenance, sealed epochs and any verified prediction and metric files.
Check `exp035_fold3/status.json` for the outcome. A successful comparison
reports `FOLD_PILOT_COMPLETE`.

For a controlled interruption, attach the exported ZIP privately in a fresh
session. Set `RESUME_ARCHIVE` to the mounted ZIP or the extracted directory
containing `result_manifest.json`. Run the same notebook on the same GPU
model with the same input and model attachments. The launcher verifies the
export, restores the last completed epoch with its optimizer, scaler and RNG,
and carries the elapsed time forward. An interrupted epoch restarts at the
last sealed boundary. Abrupt VM loss can remove unsaved files.

This handoff was checked offline for archive and manifest integrity, source
and model references, notebook structure and syntax. Kaggle CUDA
compatibility, throughput and scientific results are pending the manual run.
