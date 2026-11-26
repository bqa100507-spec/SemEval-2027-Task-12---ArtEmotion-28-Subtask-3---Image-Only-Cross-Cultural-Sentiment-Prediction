# Claude Code notes for SemEval-2027 Task 12 ST3

@AGENTS.md

`AGENTS.md` holds the data, leakage and inference conventions. Its "Closed research boundary — TASK-024" section is historical: it still says `PRODUCTION = E50+L`. The current incumbent is recorded in `docs/project_state.md` on the newest branch, and that file wins when the two disagree.

## Where the latest work is (audited 2026-11-26)

- This root checkout is on `codex/exp035-qwen3-cultural-residual` (2026-11-22) and is **stale**.
- The newest state is the worktree `runs/exp037/review/` on branch `codex/exp037-progressive-joint-adaptation` at `bcc8d846508428ac5d3e2231af49462ada00faa5`, which is open PR #44 (not merged). `main` is older.
- Open Claude Code in `runs/exp037/review/` for EXP-037/038 work, and read `docs/project_state.md`, `docs/experiment_registry.md`, `docs/experiment_workflow_standard.md`, `reports/exp037_final.md` and `reports/exp038_design_recommendation.md` there.

## Current competition position

- Incumbent: EXP-036 ensemble (75% Corrected Ridge + 25% epoch-5 CS-CVA), owner-reported hosted Dev 0.4703. Fallbacks: Corrected Ridge, then E50+L.
- EXP-037: closed, research-only because Qwen eligibility is unresolved (organizer issue #1). Do not promote.
- EXP-038: design recommendation only, pending owner review. Not implemented, not trained.
- Archive SHA-256 values for EXP-036, Corrected Ridge and Fixed Ridge were re-verified on 2026-11-26 and match `docs/project_state.md`.

## Limits for Claude

- Do not implement or train EXP-038 until the owner approves the design. Do not start Kaggle/Colab GPU runs.
- Development gold is scorer-only. Never use hidden Evaluation labels or external annotations of evaluation artworks.
- Export validated ZIPs (top-level `predictions.csv`, exact keys, finite probabilities, manifest + hash) for every completed fit, but never upload to Codabench.
- Do not merge PR #44 or push without an explicit owner request.
