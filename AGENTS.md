# Project conventions

- Prioritize reliable ST3 competition Contrast, competitive Macro Log Loss, and reproducible inference. Keep concise records for the system description paper. Treat work on other subtasks, additional data, or research variants as separate tasks.
- Use Python 3.12, the isolated `.venv`, and the pinned organizer and CLIP revisions in `configs/st3.json`. Keep upstream data, artwork, weights, credentials, caches and per-run artifacts outside Git. Commit code, lightweight tests and aggregate findings.
- Use `painting` and `image_name` for identity and group splits. Check disjointness before fitting. Train on `split=train` and retain a minimum of five binary-mappable annotations per painting-language pair. Use validation labels exclusively for evaluation and auditing.
- For ST3 inference, use artwork pixels and target language. Use painting IDs and image paths to locate files and join results. Keep captions, art metadata, labels and other languages' responses out of model inputs. Keep all evaluation-phase resources outside the development workflow.
- Keep run configuration, input hashes, versions, seeds and coverage with results. Distinguish measured development metrics, organizer references and future hypotheses. Cite external resources and check organizer rules before introducing them.
- Use a new run directory for each controlled experiment. Keep shuffled-image diagnostics separate from submission predictions and restrict grouped cross-validation to original training paintings.
- Use `scripts/run_st3.py` for local checks and `python -m pytest -q` from the project root for unit tests. Tests should use small fixtures and run without downloading the dataset or weights. Generate and validate archives locally for manual Codabench upload.


## Closed research boundary â€” TASK-024

- `EXP022 = CLOSED`; `TASK023 = CLOSED`; `EL8 = NO_GO_FOR_PRODUCTION`; `PRODUCTION = E50+L`.
- Preserve the default E50+L inference, its frozen predictions and explicit E50/CLIP fallbacks. Do not make EL8 the default or treat its retraining, retuning, deployment or re-submission as an open action item.
- EXP-022 was a completed controlled experiment, and TASK-023 completed verified engineering. The owner-reported hosted Development comparison was negative for EL8. Do not rewrite historical metrics, notebooks, checkpoints or failure logs; preparation-era READY/next commands are archival, not authorization.
- Read `reports/exp022_final_closure.md` and `docs/research_index.md` before related work. TASK-025 authorizes EXP-023 implementation and CPU/synthetic verification only. Its scientific preregistration is `reports/next_exp_proposal.md`; GPU training, confirmation access and hosted submission remain unauthorized.
- Future authorized experiments must follow `docs/experiment_workflow_standard.md`: one notebook/authorization/Run all, frozen gates, resumable complete models, persistent success/failure export, optional `AUTO_BROWSER_DOWNLOAD=True` attempts before Colab runtime release. Do not retrofit this standard into the closed notebooks in a closure task.

## Mandatory entry point for every new experiment

Before implementation, read **both** `docs/experiment_workflow_standard.md` (the single authoritative execution specification) and the current experiment's frozen scientific protocol. These requirements apply to EXP-024 and every successor without needing another long owner prompt. Scientific choices belong to their own protocol; do not create competing copies of the workflow in prompts/reports.

- One notebook, one advance owner authorization, one Run all. No intermediate manual Verify, Continue, Export, Submission or Cleanup. Automatic transitions must enforce preregistered GO/NO-GO decisions.
- Automatically verify real artifacts, save completed checkpoint boundaries persistently and resume only checksum/identity-compatible completions. Export success/failure diagnostics and verify persistent read-back.
- Enforce GPU/allocation and wall budgets with persistence reserve and parent deadlines. **No GPU execution** before advance owner authorization, scope-appropriate resource provenance and the reusable workflow-compliance tests pass. `RESEARCH_ONLY_CV` may use the preregistered released training data and pinned permitted artifacts without unresolved competition permission blocking research. `COMPETITION_READY` additionally requires source-backed organizer eligibility.
- Colab must offer automatic browser download requests after verified persistent export and before robust outer-finally runtime release. Requests do not prove local arrival or physical deallocation. Kaggle must not invoke Colab APIs.
- Never select from Development gold or automatically upload to Codabench. Never silently change scientific settings, dependencies or gates to force success. Archive and close failed experiments rather than silently retuning them.
- Run `tests/test_experiment_workflow.py` and the notebook contract validator in addition to experiment-specific tests; syntax checks alone cannot establish execution readiness. Keep the closed EXP-022/TASK-023 artifacts and released E50+L behavior unchanged.

## Execution scopes â€” TASK-025B

- Every new experiment must declare `RESEARCH_ONLY_CV` or `COMPETITION_READY` before Run all. Neither scope grants GPU authorization by itself. This coding task authorizes no GPU execution.
- Research-only CV stops after complete verification, metrics and frozen CV gates. Record `CV_GO_PENDING_ELIGIBILITY` or `CV_NO_GO`; never open sealed confirmation, fit final competition models or create a Development submission. Export verified research outputs, request outputs.zip and release Colab automatically.
- Competition scope preserves genuine source-backed eligibility, advance confirmation authorization and all scientific gates. A genuine organizer-evidence document may be supplied separately; do not rewrite the unresolved frozen eligibility config or invent permission. Never promote/upload automatically.
- Scope/permission changes are authorization records, not scientific hyperparameters. Resume complete verified CV models only under exactly identical source revision/files, config, input hashes and runtime. Do not weaken identity checks, relabel historical fits or silently retrain corrupt/incompatible artifacts. Source changes, including this patch versus TASK-025, require a new run identity unless compatibility is exactly identical; no cross-source migration is implemented.

## Operational budgets — TASK-025C

- Owner GPU/wall limits are operational authorization, not scientific hyperparameters. EXP-023 defaults to maximum 4 GPU allocation-hours /5 wall-hours; these are not target durations or universal limits for future experiments. Finish and release when authorized work is complete.
- Keep the whole-job theoretical max-epoch bound diagnostic. Admit only a conservatively affordable complete original fit, initial independent replay, atomic persistent checkpoint and cleanup reserve. Update forecasts from measured selection/refit epochs and complete-fit timings; label small-sample limitations.
- A budget stop with fewer than all preregistered verified fits is operationally incomplete, never scientific NO-GO. Preserve completed models and resume exact source/config/input/runtime identities; changing only owner time limits must not retrain them. New source releases require new run directories; no cross-source migration/bypass.
- EXP-023 operational defaults live in configs/exp023_runtime.json. The legacy budget block in the byte-preserved scientific configs/exp023.json is archival and ignored by the new runtime. No scientific GPU training is authorized by this coding patch.
