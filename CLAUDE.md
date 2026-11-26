# Claude Code notes for SemEval-2027 Task 12 ST3

@AGENTS.md

`AGENTS.md` holds the data, leakage and inference conventions. Its "Closed research boundary — TASK-024" section is historical: it still says `PRODUCTION = E50+L`. The current incumbent is recorded in `docs/project_state.md` on the newest branch, and that file wins when the two disagree.

## Where the latest work is (updated 2026-11-26)

- The root checkout is on `codex/exp035-qwen3-cultural-residual` (2026-11-22) and is **stale**.
- EXP-037 is the worktree `runs/exp037/review/` on `codex/exp037-progressive-joint-adaptation` at `bcc8d846508428ac5d3e2231af49462ada00faa5`, open PR #44 (not merged).
- **EXP-038** is the worktree `runs/exp038/review/` on `codex/exp038-cultural-deviation-ensemble`, branched from that EXP-037 commit (draft PR, not merged). Open Claude Code there for EXP-038 work and read `docs/project_state.md`, `docs/experiment_registry.md`, `docs/experiment_workflow_standard.md`, `docs/exp038_kaggle.md` and `configs/exp038.json`.

## Current competition position

- Prior incumbent: EXP-036 ensemble (75% Corrected Ridge + 25% epoch-5 CS-CVA), owner-reported hosted Dev 0.4703. Fallbacks: Corrected Ridge, then E50+L.
- EXP-037: closed. The owner uploaded `exp037_best_submission.zip` (SHA-256 `b50861a1414fd863b4293bb6a8b3dd336bd6f2bc9071f8e20559b2774f4e1f33`) as Codabench Track 3 submission #1068597 on 2026-11-26; hosted score owner-reported **0.4728**, matching the local original C6 blend (0.472818). Qwen eligibility is unresolved (organizer issue #1). Whether EXP-037 replaces EXP-036 as incumbent is the owner's decision.
- **EXP-038: approved by the owner on 2026-11-26 (cultural-deviation ensemble; the shared-basis recommendation is SUPERSEDED). Implemented and tested on CPU; overlay packaged; no Kaggle/GPU run yet.** The owner attaches both datasets and clicks Run All once. A separate audit verifies the returned outputs; do not start a successor experiment.

## Limits for Claude

- Do not start Kaggle/Colab GPU runs. Development gold is scorer-only. Never use hidden Evaluation labels or external annotations of evaluation artworks.
- Export validated ZIPs for every completed fit, but never upload to Codabench.
- Do not merge PR #44 or the EXP-038 draft PR, and push only on an explicit owner request.
- The owner plans to reuse the Track 3 model plus an adapter for Track 2 later; not started.
