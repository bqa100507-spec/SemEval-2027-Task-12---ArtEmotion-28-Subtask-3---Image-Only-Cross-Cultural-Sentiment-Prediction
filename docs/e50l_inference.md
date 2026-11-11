# Run frozen E50+L inference on Windows

E50+L is the preferred ST3 candidate. The project owner reported hosted Development Contrast 0.4643 for submission `#1046158`, matching EXP-010's local score at four decimals. E50 and CLIP remain available as explicit fallbacks. [EXP-010 results](../reports/exp010_clip_large.md)

## One-time preparation

Use the pinned Python environments and the original EXP-007 and EXP-010 artifacts. The E50 bundle lives in `runs/e50_inference_models/` and the separate CLIP-L bundle lives in `runs/e50l_inference_models/`. Both are prepared on the machine used for Task 011. If either bundle is missing, prepare it from the original artifacts. Keep each bundle's `heads.npz`, `manifest.json` and `legacy-index.json` together.

```powershell
$py = ".venv/Scripts/python.exe"
if (-not (Test-Path ".venv-siglip2/Scripts/python.exe")) {
    & $py scripts/setup_siglip2.py
    if ($LASTEXITCODE -ne 0) { throw "SigLIP 2 environment setup failed" }
}
if (-not (Test-Path "runs/e50_inference_models/manifest.json")) {
    & $py scripts/run_st3.py prepare-e50
    if ($LASTEXITCODE -ne 0) { throw "E50 head preparation failed" }
}
if (-not (Test-Path "runs/e50l_inference_models/manifest.json")) {
    & $py scripts/run_st3.py prepare-e50l
    if ($LASTEXITCODE -ne 0) { throw "CLIP-L head preparation failed" }
}
```

`prepare-e50l` refits only the original CLIP-L LR head from the frozen EXP-010 training embeddings and verifies the reloaded ensemble byte for byte against the saved 849-row EXP-010 CSV. It requires the original feature archives and public development input CSV and uses original training annotations for fitting. The inference commands below read only the released input CSV, artworks, bundles and encoder caches.

## Predict the released evaluation input

Place the official CSV at `data/eval_st3/st3_test.csv`. Place its artworks under `data/eval_st3/images/`, keeping their relative paths from the CSV. Run from the repository root.

```powershell
$py = ".venv/Scripts/python.exe"
$data = "data/eval_st3"
$csv = "$data/st3_test.csv"
$out = "runs/e50l_eval"

& $py scripts/run_st3.py predict --mode e50l --input $csv --data-dir $data --output-dir $out
if ($LASTEXITCODE -ne 0) { throw "E50+L prediction failed" }
& $py scripts/run_st3.py validate-prediction --input $csv "$out/submission.zip"
```

The default mode is `e50l`. The output directory contains the ordered `predictions.csv`, `run.json` and a validated `submission.zip` for manual upload. The ZIP contains a single top-level CSV. Probabilities use EXP-010's intermediate rounding. First average the two six-decimal CLIP-B and SigLIP 2 components and round E50 to six decimals. Then average that E50 value with the six-decimal CLIP-L component and round again.

Run the same command after interruption. The three workers run sequentially. Each commits completed batches to the per-image cache, and a retry starts from the remaining image contents. A failed run leaves the final output directory unpublished. Change `--output-dir` for a fresh output after success. Use `--clip-large-batch-size 8` if the default batch of 16 exceeds available GPU memory. The CLIP-B and SigLIP 2 batch sizes are independent. `--device auto` selects CUDA when available. `--device cpu` runs all requested encoders on CPU and uses separate incremental device profiles. The verified historical archives remain usable on either device. Keep the same `--feature-cache-dir` on retries if you override the default.

To run a fallback, choose it explicitly and use a separate output directory. These modes need only the original E50 bundle.

```powershell
& $py scripts/run_st3.py predict --mode e50 --input $csv --data-dir $data --output-dir runs/e50_eval
& $py scripts/run_st3.py validate-prediction --input $csv runs/e50_eval/submission.zip

& $py scripts/run_st3.py predict --mode clip --input $csv --data-dir $data --output-dir runs/clip_eval
& $py scripts/run_st3.py validate-prediction --input $csv runs/clip_eval/submission.zip
```

Official target languages absent from training use the original shared image terms. Invalid paths, malformed inputs and unrecognized languages stop the run. A CLIP-L failure stops E50+L. [Verification report](../reports/e50l_inference_verification.md)
