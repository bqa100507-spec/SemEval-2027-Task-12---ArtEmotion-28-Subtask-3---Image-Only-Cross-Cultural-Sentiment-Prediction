# Run frozen E50 inference

E50 is the fixed EXP-007 center-crop CLIP ViT-B/32 and SigLIP 2 Base ensemble. Each of the two annotator-weighted Model C heads rounds its probability to six decimals. The pipeline averages the two probabilities at 50/50 and rounds again to six decimals. EXP-008 retained this system. The project owner reported hosted Development Contrast 0.4600 for submission `#1045921`. [EXP-007](../reports/exp007_fusion.md) · [EXP-008](../reports/exp008_full_image.md)

## One-time preparation on Windows

Use the pinned `.venv` and the original public dataset, Task 001 CLIP cache, EXP-006 SigLIP 2 cache and EXP-007 development run. The preparation command verifies these inputs, reconstructs the two heads on the original training split and checks exact saved-and-reloaded parity against the historical development CSVs. It requires the public development input pairs for that check. It does not score or use development labels.

```powershell
$py = ".venv/Scripts/python.exe"
& $py scripts/setup_siglip2.py
& $py scripts/run_st3.py prepare-e50
```

Skip preparation when `runs/e50_inference_models/manifest.json` and its bundle already exist. Keep the entire bundle directory. It contains `heads.npz` (float64 coefficients and intercepts), `manifest.json` (version, vocabulary, fit settings and provenance), and `legacy-index.json` (verified image-content SHA-256 to historical image-name mapping). Use `--model-dir` to select a different bundle location. Keep the original feature archives in `cache/` to preserve exact historical reuse. A missing archive can be rebuilt incrementally for requested images using the same frozen encoder. A present archive with a mismatched checksum stops prediction.

## Predict a released evaluation input

Place the official input CSV at `data/eval_st3/st3_test.csv` and the accompanying artwork under `data/eval_st3/images/`, keeping each image's relative path from the CSV. The directory needs the input CSV and images only. Run from the repository root.

```powershell
$py = ".venv/Scripts/python.exe"
$dataset = "data/eval_st3"
$inputCsv = "$dataset/st3_test.csv"
$output = "runs/e50_eval"

& $py scripts/run_st3.py predict --input $inputCsv --data-dir $dataset --output-dir $output
& $py scripts/run_st3.py validate-prediction --input $inputCsv "$output/submission.zip"
```

The output directory contains `predictions.csv`, `submission.zip` and `run.json`. The ZIP contains the root-level CSV. Inspect the log and validate the archive before any manual upload. Prediction follows CSV row order and validates coverage against the supplied CSV. It accepts the exact organizer columns `painting,language,image` and checks keys, image mapping and path containment. Official languages absent from training (IsiNdebele, IsiXhosa and Kazakh) use the frozen shared image terms with zero language and interaction blocks. Their row counts appear in the log and manifest. Unrecognized language names fail validation. [Organizer ST3 schema and language list](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#st3-image-only-cross-cultural-sentiment-prediction)

Use `--device cuda` to require CUDA or `--device cpu` for CPU. Default `auto` selects CUDA when available. Set `--clip-batch-size` and `--siglip2-batch-size` independently if memory is constrained. New CLIP images run under `.venv`. New SigLIP 2 images run under `.venv-siglip2`. Use `--siglip2-python` for a relocated overlay.

## Resume and use the fallback

After interruption or encoder failure, rerun the same prediction command with the same cache directory and device. You can lower the batch sizes. The cache reuses completed batches and only encodes the remaining or changed image contents. An unsuccessful run leaves the final output directory unpublished. A successful run protects that directory from overwrite. Use a new `--output-dir` for a fresh copy after success.

To choose the CLIP-only fallback explicitly, use a separate output directory.

```powershell
$fallback = "runs/clip_eval"
& $py scripts/run_st3.py predict --mode clip --input $inputCsv --data-dir $dataset --output-dir $fallback
& $py scripts/run_st3.py validate-prediction --input $inputCsv "$fallback/submission.zip"
```

The cache under `cache/e50-incremental-v1/` is separate from prior experiment caches. It keys embeddings by image bytes and encoder profile, including checkpoint, processor, runtime and device. Different paths containing identical bytes share an embedding. Each completed batch is committed atomically to SQLite. Cached vectors are checksummed and validated for shape and unit norm. Device changes use a separate incremental profile. Verified historical features can still be reused on either device.

## Implementation audit

Previously, CLIP and SigLIP 2 keyed caches by an entire fixed image set and saved each archive after extraction finished. EXP-007 held the fitted heads in memory, and the command dispatch read the public development CSV before invoking a run. General inference now loads a verified head bundle and explicit CSV. It reuses the existing feature projections, original pinned processors, sparse design, prediction rounding, probability blend and ZIP validation. The new workers add per-content lookup and batch persistence. The legacy archives are read-only. See the [verification report](../reports/e50_inference_verification.md) for the development parity and real-encoder smoke tests.
