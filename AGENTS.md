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

## Execution scopes — TASK-025B

- Every new experiment must declare `RESEARCH_ONLY_CV` or `COMPETITION_READY` before Run all. Neither scope grants GPU authorization by itself. This coding task authorizes no GPU execution.
- Research-only CV stops after complete verification, metrics and frozen CV gates. Record `CV_GO_PENDING_ELIGIBILITY` or `CV_NO_GO`; never open sealed confirmation, fit final competition models or create a Development submission. Export verified research outputs, request outputs.zip and release Colab automatically.
- Competition scope preserves genuine source-backed eligibility, advance confirmation authorization and all scientific gates. A genuine organizer-evidence document may be supplied separately; do not rewrite the unresolved frozen eligibility config or invent permission. Never promote/upload automatically.
- Scope/permission changes are authorization records, not scientific hyperparameters. Resume complete verified CV models only under exactly identical source revision/files, config, input hashes and runtime. Do not weaken identity checks, relabel historical fits or silently retrain corrupt/incompatible artifacts. Source changes, including this patch versus TASK-025, require a new run identity unless compatibility is exactly identical; no cross-source migration is implemented.

## Operational budgets — TASK-025C

- Owner GPU/wall limits are operational authorization, not scientific hyperparameters. EXP-023 defaults to maximum 4 GPU allocation-hours /5 wall-hours; these are not target durations or universal limits for future experiments. Finish and release when authorized work is complete.
- Keep the whole-job theoretical max-epoch bound diagnostic. Admit only a conservatively affordable complete original fit, initial independent replay, atomic persistent checkpoint and cleanup reserve. Update forecasts from measured selection/refit epochs and complete-fit timings; label small-sample limitations.
- A budget stop with fewer than all preregistered verified fits is operationally incomplete, never scientific NO-GO. Preserve completed models and resume exact source/config/input/runtime identities; changing only owner time limits must not retrain them. New source releases require new run directories; no cross-source migration/bypass.
- EXP-023 operational defaults live in configs/exp023_runtime.json. The legacy budget block in the byte-preserved scientific configs/exp023.json is archival and ignored by the new runtime. No scientific GPU training is authorized by this coding patch.

## TASK-026 current direction

- EXP-023 is CLOSED / NO-GO on the owner's final Colab result, with incomplete persistent scientific artifacts. No rerun, export repair or checkpoint recovery. Previous EXP-023 readiness/resume instructions are archival. Preserve EXP-022 closure and E50+L production.
- TASK-026 authorizes independent EXP-024 retrieval on original P2 training votes and existing frozen visual caches, including small nested training-only selection and local CPU CV. Its explicit simple-CLI scope supersedes notebook/compliance/packaging requirements for this experiment. No new cloud framework, Development tuning, E50 blend or automatic promotion.

## TASK-027 / EXP-025

- EXP-022/023/024 stay closed; production remains E50+L. TASK-027 authorizes the image-only Qwen2.5-VL-3B semantic probe, not VLM fine-tuning or competition promotion. No captions/votes/metadata enter generation. No confirmation or Development labels.
- Keep this method-discovery workflow to extraction/cache plus a lightweight matched-head script and a simple Colab notebook. This explicit task scope supersedes earlier elaborate notebook/compliance/packaging requirements. Preserve exact model/processor/decode/text-encoder identities and separate NF4/FP16 caches.
- Hard technical smoke failures block extraction. Semantic inaccuracies, nationality/artist wording and uncertain details are retained warnings, not automatic vetoes. FP16 L4 quality/throughput remain unverified; do not present partial caches or synthetic tests as real P2 results.

## TASK-027B — finish EXP-025

- Keep the existing model, prompt, decoding, head methods and training-only selection. Run all five P2 folds after technically valid, owner-approved extraction; bootstrap statistics are evaluation evidence, never continuation gates.
- The owner explicitly authorizes exactly one visual quality review after the matched30-image NF4/FP16 comparison. Accept broadly useful semantics despite isolated errors. Serious systematic scene hallucination stops and preserves smoke; recommending direct hidden states/a stronger model does not authorize launching them.
- After approval, full1335-image extraction and all five folds run without further owner pauses. Exact-smoke review and compatible completed folds resume; changed/corrupt data never bypass validation. This task-specific review is an explicit exception to the generic no-midrun-interaction wording. Preserve production E50+L and closed EXP-022/023/024.

## TASK-028 / EXP-026

- EXP-025 is CLOSED / NO-GO for caption-based fusion. Preserve its1335-image FP16 cache, reports and artifacts; no rerun. E50+L stays production.
- EXP-026 uses the pinned Qwen multimodal decoder states directly with a supervised sentiment head: frozen NF4 base versus the same base plus decoder Q/V LoRA. No caption generation/supervision, extra loss, CLIP fusion or new checkpoint. Match the target-language prompt in the primary F0/L0 comparison; neutral cached states are a conditioning diagnostic.
- Current owner GPU authorization is strictly one local RTX4060 technical check, maximum5 minutes, one original inner-training artwork and a few gradient steps. Stop on OOM/time limit. No few-hundred-painting pilot, scientific fold, Colab execution or additional GPU experiment is authorized in this coding session. Future notebook GPU stages require their own explicit initial owner authorization.
- Pilot precedes any one-fold fit; full five-fold QLoRA must never run automatically. Keep P2 splits and sealed confirmation annotations untouched. Existing workflow is simplified to minimal scripts/cache/checkpoints and a Colab notebook for this task; no new compliance/packaging framework.

## EXP-026 exploratory continuation

- Owner-reported L4 pilot passed. The expensive original scientific `one-fold` protocol is not authorized by this continuation. Preserve its config, original notebook, historical fits and E50+L production.
- `exploratory-fold` is separate screening: original fold1/inner partition, at most2 epochs for F0_image/F0/L0, best inner-validation state, no fresh refit. Its VLM fitting population is smaller than refitted E50+L; report the unequal-data caveat and do not promote/select a final system from this screen.
- Use a new exploratory source/run namespace and separate notebook. Reuse only exactly compatible frozen feature caches; never warm-start from pilot/scientific models. No GPU training in this coding task. Future Colab L4 execution requires separate authorization and a maximum2-hour allocation; no full CV, Development/confirmation access or submission.

## EXP-026 final exploratory decision — 2026-11-18

- `EXP026 = EXPLORATORY_COMPLETED`; `PRODUCTION = E50+L`. Fold1 predictions/logs were checked and scores recomputed on the existing checksum-pinned released P2 training votes. L0 Contrast0.340691 /Macro Log Loss0.704087 vs same-fold E50+L0.433429 /0.666222; VLM fits961 paintings, refitted reference1068. No promotion or claim about a fully refitted VLM.
- Preserve all original source, configs, notebooks, caches, checkpoints, predictions and logs. No further EXP-026 GPU run, longer-epoch rerun or five-fold training is an open action item. Earlier run commands are archival, not authorization. No Development or sealed confirmation annotations.
- EXP-027 is proposal-only: test a within-painting objective with the existing VLM/LoRA architecture and a matched BCE control. Do not implement/train it without a new explicit task. Fold1 is already examined and cannot select that loss or lambda. Other P2 folds have also been reused historically; do not call them independent confirmation. A new held-out fold requires fresh pretrained/zero-LoRA initialization, never an EXP-026 fitted checkpoint containing its paintings.

## TASK-029 — EXP-027 implementation authorization

- TASK-029 explicitly authorizes EXP-027 implementation and synthetic/training-only CPU diagnostics, superseding the proposal-only implementation restriction above. It authorizes **no GPU training, outer evaluation, full CV, Development or confirmation access**. EXP-026 and production E50+L remain unchanged.
- Follow `docs/exp027_protocol.md` and `configs/exp027.json`: native matched BCE control, complete-painting Pearson surrogate, fixed0/0.000682/0.00196 grid, common inner-BCE checkpoint selection, fresh pretrained/zero-output LoRA. No auxiliary architecture/ranking/distillation or fitted EXP-026 adapter.
- Two-pass replay must preserve same-chunk logits/RNG/state and complete group gradients. Invalid replay, mutable buffers/parameters, zero/nonfinite gradients, bad provenance, OOM or budget stop cannot yield successful verification. Do not relax safeguards or scientific settings to force PASS.
- The prepared Colab entry point is inner-only pilot, default GPU authorization OFF, max2 allocation-hours. `CPU_VALIDATED / GPU_READY` means implementation/checks ready for a separately authorized pilot, not actual CUDA, scientific improvement or competition eligibility.

## TASK-030 — EXP-027 closed for expansion

- `EXP027 = GPU_PILOT_COMPLETED / NO_GO_FOR_EXPANSION`; production stays E50+L. Three saved inner-validation models/metrics/bootstrap/checksum artifacts were independently audited on CPU. Both Pearson treatments worsen Contrast relative to matched BCE. Preserve all EXP-026/027 science, notebooks, caches, predictions, weights and logs.
- No further training, lambda/epoch/rank search, outer evaluation, full CV, Development/confirmation access or automatic GPU execution is an open action item. Earlier readiness/run instructions are archival, not authorization.
- Read `reports/exp027_final_closure.md` before related work. The failure is one selected inner-validation pilot, not proof that all Pearson losses or QLoRA fail. The proposed factorial conditioning probe is proposal-only; do not implement EXP-028 or run inference/training without a new explicit owner task.
