# Claude Code notes for SemEval-2027 Task 12 ST3

@AGENTS.md

`AGENTS.md` holds the data, leakage and inference conventions. Its "Closed research boundary — TASK-024" section is historical: it still says `PRODUCTION = E50+L`. The current incumbent is recorded in `docs/project_state.md` on the newest branch, and that file wins when the two disagree.

## Where the latest work is (audited 2026-11-26)

- This root checkout is on `codex/exp035-qwen3-cultural-residual` (2026-11-22) and is **stale**.
- The newest state is the worktree `runs/exp037/review/` on branch `codex/exp037-progressive-joint-adaptation` at `bcc8d846508428ac5d3e2231af49462ada00faa5`, which is open PR #44 (not merged). `main` is older.
- Open Claude Code in `runs/exp037/review/` for EXP-037/038 work, and read `docs/project_state.md`, `docs/experiment_registry.md`, `docs/experiment_workflow_standard.md`, `reports/exp037_final.md` and `reports/exp038_design_recommendation.md` there.

## Current competition position

- Prior incumbent: EXP-036 ensemble (75% Corrected Ridge + 25% epoch-5 CS-CVA), owner-reported hosted Dev 0.4703. Fallbacks: Corrected Ridge, then E50+L.
- EXP-037: closed. The owner uploaded `exp037_best_submission.zip` (SHA-256 `b50861a1414fd863b4293bb6a8b3dd336bd6f2bc9071f8e20559b2774f4e1f33`, matches `reports/exp037_final.md`) to Codabench Track 3 as submission #1068597 on 2026-11-26 02:16; hosted score owner-reported **0.4728**, matching the local original C6 blend (0.472818) at four decimals. For comparison, #1065924 (EXP-036 ensemble) is 0.4703. Qwen eligibility is still unresolved (organizer issue #1); the owner accepted that risk knowingly. Whether EXP-037 replaces EXP-036 as the incumbent is the owner's decision; do not change it on your own. The post-hoc 65/15/20 blend (local 0.473143) was not the one submitted.
- EXP-038: design recommendation only, pending owner review. Not implemented, not trained.
- Archive SHA-256 values for EXP-036, Corrected Ridge and Fixed Ridge were re-verified on 2026-11-26 and match `docs/project_state.md`.

## Limits for Claude

- Do not implement or train EXP-038 until the owner approves the design. Do not start Kaggle/Colab GPU runs.
- Development gold is scorer-only. Never use hidden Evaluation labels or external annotations of evaluation artworks.
- The owner plans to reuse the Track 3 model plus an adapter for Track 2 later; not started.
- Export validated ZIPs (top-level `predictions.csv`, exact keys, finite probabilities, manifest + hash) for every completed fit, but never upload to Codabench.
- Do not merge PR #44 or push without an explicit owner request.
