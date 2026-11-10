# Project conventions

- Prioritize reliable ST3 competition Contrast, competitive Macro Log Loss, and reproducible inference. Keep concise records for the system description paper. Treat work on other subtasks, additional data, or research variants as separate tasks.
- Use Python 3.12, the isolated `.venv`, and the pinned organizer and CLIP revisions in `configs/st3.json`. Keep upstream data, artwork, weights, credentials, caches and per-run artifacts outside Git. Commit code, lightweight tests and aggregate findings.
- Use `painting` and `image_name` for identity and group splits. Check disjointness before fitting. Train on `split=train` and retain a minimum of five binary-mappable annotations per painting-language pair. Use validation labels exclusively for evaluation and auditing.
- For ST3 inference, use artwork pixels and target language. Use painting IDs and image paths to locate files and join results. Keep captions, art metadata, labels and other languages' responses out of model inputs. Keep all evaluation-phase resources outside the development workflow.
- Keep run configuration, input hashes, versions, seeds and coverage with results. Distinguish measured development metrics, organizer references and future hypotheses. Cite external resources and check organizer rules before introducing them.
- Use a new run directory for each controlled experiment. Keep shuffled-image diagnostics separate from submission predictions and restrict grouped cross-validation to original training paintings.
- Use `scripts/run_st3.py` for local checks and `python -m pytest -q` from the project root for unit tests. Tests should use small fixtures and run without downloading the dataset or weights. Generate and validate archives locally for manual Codabench upload.
