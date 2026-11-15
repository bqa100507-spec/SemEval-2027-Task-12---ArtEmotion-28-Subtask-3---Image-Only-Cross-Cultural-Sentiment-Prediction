# Project conventions

- Prioritize reliable ST3 competition Contrast, competitive Macro Log Loss, and reproducible inference. Keep concise records for the system description paper. Treat work on other subtasks, additional data, or research variants as separate tasks.
- Use Python 3.12, the isolated `.venv`, and the pinned organizer and CLIP revisions in `configs/st3.json`. Keep upstream data, artwork, weights, credentials, caches and per-run artifacts outside Git. Commit code, lightweight tests and aggregate findings.
- Use `painting` and `image_name` for identity and group splits. Check disjointness before fitting. Train on `split=train` and retain a minimum of five binary-mappable annotations per painting-language pair. Use validation labels exclusively for evaluation and auditing.
- For ST3 inference, use artwork pixels and target language. Use painting IDs and image paths to locate files and join results. Keep captions, art metadata, labels and other languages' responses out of model inputs. Keep all evaluation-phase resources outside the development workflow.
- Keep run configuration, input hashes, versions, seeds and coverage with results. Distinguish measured development metrics, organizer references and future hypotheses. Cite external resources and check organizer rules before introducing them.
- Use a new run directory for each controlled experiment. Keep shuffled-image diagnostics separate from submission predictions and restrict grouped cross-validation to original training paintings.
- Use `scripts/run_st3.py` for local checks and `python -m pytest -q` from the project root for unit tests. Tests should use small fixtures and run without downloading the dataset or weights. Generate and validate archives locally for manual Codabench upload.

## Closed research boundary — TASK-024

- `EXP022 = CLOSED`; `TASK023 = CLOSED`; `EL8 = NO_GO_FOR_PRODUCTION`; `PRODUCTION = E50+L`.
- Preserve the default E50+L inference, its frozen predictions and explicit E50/CLIP fallbacks. Do not make EL8 the default or treat its retraining, retuning, deployment or re-submission as an open action item.
- EXP-022 was a completed controlled experiment, and TASK-023 completed verified engineering. The owner-reported hosted Development comparison was negative for EL8. Do not rewrite historical metrics, notebooks, checkpoints or failure logs; preparation-era READY/next commands are archival, not authorization.
- Read `reports/exp022_final_closure.md` and `docs/research_index.md` before related work. EXP-023 in `reports/next_exp_proposal.md` is proposal-only; implementation, training, confirmation access and hosted submission require a new explicit owner task.
- Future authorized experiments must follow `docs/experiment_workflow_standard.md`: one notebook/authorization/Run all, frozen gates, resumable complete models, persistent success/failure export, optional `AUTO_BROWSER_DOWNLOAD=True` attempts before Colab runtime release. Do not retrofit this standard into the closed notebooks in a closure task.
