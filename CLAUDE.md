# Claude Code notes for SemEval-2027 Task 12 ST3

@AGENTS.md

`AGENTS.md` holds the data, leakage and inference conventions. Its "Closed research boundary — TASK-024" section is historical: it still says `PRODUCTION = E50+L`. The current incumbent is recorded in `docs/project_state.md` on the newest branch, and that file wins when the two disagree.

## Where the latest work is (updated 2026-11-28)

- The root checkout is on `codex/exp035-qwen3-cultural-residual` (2026-11-22) and is **stale**.
- EXP-037 is the worktree `runs/exp037/review/` on `codex/exp037-progressive-joint-adaptation` at `bcc8d846508428ac5d3e2231af49462ada00faa5`, open PR #44 (not merged).
- **EXP-040** (E3-LoRA, prepared, not run) is the worktree `runs/exp040/review/` on `codex/exp040-e3-lora`, branched from `codex/h1-gate0` at `8961b86`. It is the newest branch; read its `docs/project_state.md` and `docs/exp040_kaggle.md`.
- **H1 Gate 0** (CPU analysis) is the worktree `runs/h1_gate0/review/` on `codex/h1-gate0`, branched from the EXP-039 tip `39adfa8`.
- **EXP-038** is the worktree `runs/exp038/review/` on `codex/exp038-cultural-deviation-ensemble`, branched from that EXP-037 commit (draft PR #47, not merged; EXP-038 closed). Open Claude Code there for EXP-038 work and read `docs/project_state.md`, `docs/experiment_registry.md`, `docs/experiment_workflow_standard.md`, `docs/exp038_kaggle.md` and `configs/exp038.json`.

## Current competition position

- **Active incumbent: EXP-038 `B2_C2_cscva3`** = round6(0.80 × round6(0.75 × C2 + 0.25 × CS-CVA-3) + 0.20 × C6), adopted by the owner on 2026-11-27. Owner-reported hosted Dev **0.4737** (#1071328, best ZIP SHA-256 `62093a31da2cd1d864bad7b7818adf082524c09180e75ca0953d5177250ee4e6`), matching local 0.473743. It depends on Qwen (E3, C6): `UNCONFIRMED_QWEN_ST3`; E2 MetaCLIP 2 is CC-BY-NC-4.0.
- **VLM-free fallback: the EXP-036 ensemble** (75% Corrected Ridge + 25% epoch-5 CS-CVA), hosted 0.4703 (#1065924), then Corrected Ridge, then E50+L.
- EXP-037 original C6 blend: hosted 0.4728 (#1068597), closed.
- **EXP-039 (VLM-embedding scaling): closed NO_GO 2026-11-28; the incumbent is unchanged.** D2/D3/D4 failed against on-platform C2; E5/8B untested (patch 2 not executed by owner decision). Report `reports/exp039_final.md`, audit `runs/exp039/closure/audit.json`. No successor designed.
- **EXP-040 (E3-LoRA): prepared, not run.** Patch 1 overlay `runs/exp040/handoff/exp040_inputs.zip` (`4f316e9d…`, source `53d2c52`): two-GPU fold queues and a gated vision cache, no scientific change. The owner runs `SMOKE_ONLY` (both T4s at once) on Kaggle first. The two-session resume is a documented fallback only; do not change the protocol without an owner decision.
- **H1 Gate 0: STOP_H1 (2026-11-28).** Gold en/ar/zh deviations do not inform the 22 scored languages: oracle over C2 −0.0143 [−0.0211, −0.0076], split-reference r −0.009 [−0.020, +0.004]. Do not build H1 (ArtELingo en/ar/zh deviation pretraining). ArtELingo eligibility is pending organizer issue #2. Report `reports/h1_gate0.md`.
- **EXP-038: closed GO (C2).** Report `reports/exp038_final.md`, audit `runs/exp038/closure/audit.json`. The old shared-basis design is SUPERSEDED. Its successor EXP-039 is closed NO_GO (above).
- Before any new Kaggle handoff, follow the pre-upload checklist in `docs/experiment_workflow_standard.md` (Linux preflight from the exact upload ZIP).

## Limits for Claude

- Do not start Kaggle/Colab GPU runs. Development gold is scorer-only. Never use hidden Evaluation labels or external annotations of evaluation artworks.
- Export validated ZIPs for every completed fit, but never upload to Codabench.
- Do not merge PR #44 or the EXP-038 draft PR, and push only on an explicit owner request.
- The owner plans to reuse the Track 3 model plus an adapter for Track 2 later; not started.
