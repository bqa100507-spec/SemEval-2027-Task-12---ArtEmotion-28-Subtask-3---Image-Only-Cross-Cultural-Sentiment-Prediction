# EXP-022 on Google Colab

[Open EXP022_Colab.ipynb in Colab](https://colab.research.google.com/github/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/e78231c109e8c743735b5ba7222221a0bca98c05/notebooks/EXP022_Colab.ipynb)

1. Upload the existing **exp022_colab_inputs.zip** to the root of **MyDrive**.
2. Open the notebook and choose **Runtime → Change runtime type → GPU**.
3. Choose **Runtime → Run all** and authorize the automatic Drive mount.
4. Read the benchmark summary. No MiniScreen, Screen or full CV starts automatically.

No shell commands are required. The default input is `/content/drive/MyDrive/exp022_colab_inputs.zip`; change only `INPUT_ZIP_PATH` in the configuration cell if necessary. First setup downloads Python and pinned packages and may take several minutes/several GB. The host notebook interpreter is kept separate; no kernel restart is required.

The notebook checks the original ZIP and both published bundle/input manifests before extracting data into local `/content/exp022/inputs`. Training images and pretrained assets are read from local storage, not Drive. It installs managed Python **3.12.10**, Torch **2.7.1+cu126**, Transformers **4.49.0** and the original requirements in an isolated environment, checks exact versions/CUDA, then invokes the existing **Preflight → Pilot** runner. Only GPU 0 is visible. There is no alternative checkpoint download, tolerance relaxation or scientific configuration change.

The summary shows GPU and total/free VRAM, environment checks, stage PASS/FAIL, frozen/cache and zero-LoRA parity, BCE/CE adapter gradients and updates, allocated/reserved GPU peaks, measured paintings/s and epoch seconds, MiniScreen/Screen forecasts, the four-hour resource gate and persistent output location. Pilot diagnostics do not establish held-out quality. A failed stage is displayed as FAIL even if an earlier saved stage had passed; full tracebacks stay in logs.

## After reviewing the benchmark

The stage selector offers **Setup / Benchmark**, **MiniScreen**, **MiniScreenVerify**, **Screen**, **ScreenVerify**. Select a stage, review its forecast/budget and measured VRAM, check authorization, then click **Run selected stage**. Authorization resets after each use. Full CV/final verification are deliberately absent from these controls and remain a separately authorized advanced workflow.

`RUN_LONG_TRAINING=False` is the default. Run all registers buttons but never clicks one. Source prerequisites and latest replay status are checked by the original runner. MiniScreen/Screen are blocked when this runtime's full Screen forecast exceeds four GPU-hours. The widgets require a notebook frontend with ipywidgets support; if absent, optional controls fail closed with a clear message, rather than starting a background job.

## Persistence and resume

Artifacts/logs are written directly to `MyDrive/exp022_runs/<runtime-identity>/`; setup logs are under `MyDrive/exp022_runs/setup_logs/`. Complete model directories and stage manifests survive a normal notebook disconnect once Drive has committed those files. Drive/VM failures can still interrupt a write. **Only completed model boundaries resume**; a partial current model restarts.

Repeat Run all to reopen the same compatible directory. Reused benchmark timings are explicitly labeled as saved measurements, not fresh throughput. A different source/configuration, package lock, GPU, driver or runtime identity receives a separate directory. Do not move checkpoints into another identity directory manually.

Use **Export completed artifacts** to create a versioned private checkpoint ZIP plus `.sha256` receipt in MyDrive. To restore an exported run, place its ZIP/receipt in MyDrive, set `RESTORE_CHECKPOINTS=True`, and optionally its explicit path. All file checksums, original seals and runtime identities must match before any files are copied. Conflicting valid files are never overwritten. Failed verification-attempt records travel with the bundle so restore cannot silently reactivate an invalid decision.

## Moving to Kaggle / offline preparation

After a successful online Linux benchmark, the optional **Build offline Kaggle kit** button prepares `MyDrive/exp022_runtime.zip` and its published `.sha256` receipt. Explicit authorization is required: this may download several GB of dependency wheels but launches no ML training. It packages the pinned Git source, relocatable managed Python, exact dependency wheels/lock and pinned orchestration helper. Keep both runtime/data bundles private.

Kaggle with Internet enabled needs only the original data ZIP. Internet-disabled Kaggle additionally needs this complete runtime ZIP and its receipt. **The original 705 MB data ZIP is not a complete offline environment.** See the [Kaggle runbook](exp022_kaggle.md) for attaching data, restoring compatible checkpoints and preserving outputs.

## Immutable provenance

- Training checkout: `41ca0d4061f129c1821009e6d00614dede08708f`; its scientific source is unchanged from executed `f0d2ac65d875a0875e8b280bc6065545611198e3`.
- Notebook helper: `b18991733a6ec60493e35985fcaff945a164d9f4`; SHA-256 `e585beed804a5d9069a596e75968bd7ebabdf73851190ea7285dc6fd82c9e5c7`.
- Notebook snapshot: `e78231c109e8c743735b5ba7222221a0bca98c05`.
- Data ZIP: `bbedbe2ad7e426040aebb63366ea5978e03d6483503b8ac9c88af14238a3d156`.
- Input manifest: `fa2f7c4528b49925c400216037a64381bc6a0b35a1b3b11458a6bcbfda924337`.

The helper is loaded separately outside the frozen training checkout. It is not added to `src/st3` or either original runner; the existing source/configuration fingerprints are not bypassed or rewritten. Its own commit/hash is pinned in both notebooks. The full resolved dependency lock and extra GPU/driver profile add an orchestration compatibility check; they do not relax the original identity. Future helper changes require explicit regeneration of notebook pins with `scripts/build_exp022_notebooks.py`.

No locked confirmation data, public Development gold, new model experiment or leaderboard submission is introduced. E50+L stays production. Resource-eligibility caveats remain in force. See the [original experiment record](../reports/exp022_visual_lora_vote_distribution.md) and [notebook validation record](../reports/exp022_notebooks.md).

**Validation boundary:** notebook schema, code, mocked Run all, paths, offline/restore logic and the actual published ZIP were tested locally. A Colab GPU runtime has **not** been tested by this change. Run all performs that actual compatibility check; strict numerical parity may reject a particular GPU/runtime. Do not substitute versions or relax tolerances to get PASS.

Runtime/storage behavior follows [Colab guidance](https://research.google.com/colaboratory/faq.html); Python installation uses the documented [uv managed-Python mechanism](https://docs.astral.sh/uv/guides/install-python/), with uv 0.9.8 pinned.
