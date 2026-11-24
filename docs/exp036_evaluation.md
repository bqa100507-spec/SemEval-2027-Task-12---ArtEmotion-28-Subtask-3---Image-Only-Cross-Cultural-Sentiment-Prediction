# EXP-036 Evaluation inference

The active incumbent freezes epoch 5 and a **75% Corrected Ridge / 25% CS-CVA** probability blend. It produces both standalone and ensemble archives from the official input-only CSV and artwork files. Corrected Ridge and E50+L remain available as fallbacks. [Verified closure](../reports/exp036_final.md)

## Prepare the pinned environments and bundle

- Use the existing release environments for the Ridge workers, with Python 3.12 and Transformers 4.49.0 for CLIP-B and CLIP-L and the separate Transformers 4.51.3 SigLIP 2 interpreter. [Release setup](competition_release.md#frozen-model-and-data)
- Provide a CUDA Python interpreter for CS-CVA with the recorded Python 3.12.10, PyTorch 2.7.1+cu126, Torchvision 0.22.1+cu126 and Transformers 4.57.6 stack. The saved training bundle includes the full Linux locks. On Linux, use `uv venv --python 3.12.10 .venv-exp036`, then `uv pip sync --python .venv-exp036/bin/python --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match requirements-exp035-kaggle.lock.txt requirements-exp036-kaggle.txt`. On Windows use matching Windows CUDA wheels and match the package versions in `cscva/bundle.json`. This interpreter is separate from the Ridge workers.
- Keep `runs/exp036/deployment/` together. It holds the matched CS-CVA adapter and sentiment weights, official processor, language mapping, locks, Fixed Ridge heads, Corrected Ridge bank and the content manifest. The pinned base checkpoint is fetched from Hugging Face on the first inference if a local snapshot is unavailable. Pass `--model <snapshot-directory>` to use a local exact copy. No text tower weights are instantiated. For a target language outside CS-CVA's fitted mapping but in the official vocabulary, CS-CVA uses the shared image logit plus the mean fitted language intercept. Ridge uses its existing shared fallback.

## Generate Evaluation submissions

Run using the EXP-036 closure source from PR #43. Put the official Evaluation `st3_test.csv` under `data/eval_st3/`, with artwork under `data/eval_st3/images/` and paths matching the CSV. Choose a fresh output directory.

```powershell
$project = 'D:/ML/Semeval2027_Task12_Subtask3+2'
$repo = "$project/runs/exp036/review"
$py = "$project/.venv/Scripts/python.exe"
$siglip = "$project/.venv-siglip2/Scripts/python.exe"
$cscva = "$project/.venv-exp036/Scripts/python.exe"
$eval = "$project/data/eval_st3"
$bundle = "$project/runs/exp036/deployment"
$out = "$project/runs/exp036/january_eval"
$common = @(
  '--input', "$eval/st3_test.csv", '--data-dir', $eval, '--bundle', $bundle,
  '--cache-dir', "$project/cache", '--feature-cache-dir', "$project/runs/exp036/january_features",
  '--siglip2-python', $siglip
)
& $py "$repo/scripts/predict_exp036.py" @common --cscva-python $cscva --output $out
if ($LASTEXITCODE -ne 0) { throw 'EXP-036 inference failed' }
Get-FileHash "$out/exp036_standalone_submission.zip", "$out/exp036_ensemble_submission.zip"
```

Both ZIPs are checked before publication for exact keys, order, schema and finite probabilities. The command records the input image hashes, model and encoder fingerprints, fixed coefficient and output hashes. The only query inputs are image pixels and target language. Training captions, Development gold, retuning and checkpoint selection are absent from this command. Upload the chosen archive manually.

For the Corrected Ridge fallback, reuse the common arguments with a different output folder. This path runs on CPU with the three sealed encoders and does not need the CS-CVA CUDA interpreter.

```powershell
& $py "$repo/scripts/predict_exp036.py" @common --ridge-only --output "$project/runs/exp036/january_ridge"
if ($LASTEXITCODE -ne 0) { throw 'Corrected Ridge inference failed' }
Get-FileHash "$project/runs/exp036/january_ridge/corrected_ridge_submission.zip"
```

With the variables above set, recover the deployment directory if needed from the verified Kaggle archive and original Ridge bundles.

```powershell
& $py "$repo/scripts/package_exp036_deployment.py" --results "$project/runs/exp036/closure/exp036_results.zip" --fixed "$project/runs/competition_track1/january_bundle" --corrected "$project/runs/codabench_dev_release/corrected_bundle" --output $bundle
```

Both new deployment ZIPs reproduced the archived 849 Development probabilities using archived CS-CVA predictions and verified Ridge features. Fresh-image Ridge and CPU-only orchestration fixtures passed. Actual January inputs and a fresh real-weight CS-CVA GPU replay have yet to be validated. The [E50+L release commands](competition_release.md#evaluation-day-runbook) remain available.
