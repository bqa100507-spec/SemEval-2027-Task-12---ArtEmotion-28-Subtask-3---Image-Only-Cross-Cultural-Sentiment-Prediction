# EXP-019 — Text loss scaling and gradient transfer

## Frozen design and evidence boundary

Base: EXP-018 `7d8e5640b4ad56e1ed4f13499254d7ccb127123c`. Production E50+L remains unchanged. Only C2 changes `text_weight` from 0.1 to 1.0. The complete EXP-018 architecture, initialization sequence, optimizer, shuffling generator, losses, masking, painting-balanced auxiliary weights, stopping BCE, seeds and partition-local targets are reused. C2 starts fresh for each selection and refit; it never resumes from C1 weights. No embeddings, existing models or PCA transforms are regenerated.

C0/C1 reuse M1/M3. E0/E1 reuse B1/B3. M0 is the exact original E50+L OOF baseline. C2 averages three unrounded seed probabilities, rounds to six decimals, then E2 uses the existing 50/50 probability blend and six-decimal rounding. This retains nested M0 rounding. There are fifteen new CV fits, each with selection and full outer-training refit: thirty training stages. A short pilot is separate and never continued into CV.

Before implementation, actual local EXP-018 artifacts were checked: all 45 checkpoints were replay-verified, deep PCA replay passed, all 291 historical hashes matched, and its locked decision was NO_GO. Neither confirmation nor public Development had run. Actual EXP-018 pooled Contrast / Macro Log Loss: M0 0.447097707 / 0.710554182; C0 0.437792268 / 0.716045717; C1 0.440121979 / 0.715230806; E0 0.448585747 / 0.706474066; E1 0.449566497 / 0.705998902. These are reused, repeatedly examined CV results, not independent test results.

`configs/exp019.json` pins the parent configuration and every existing sealed stage, geometry, prediction and seed manifest. Preflight verifies original source keys byte for byte and runtime/device/library identity while permitting additive EXP-019 source files. Original source files and entry points remain unchanged. Checkpoint files are verified against their manifests, seed painting identities against the original protocol, and the latest parent replay must be successful. Preflight snapshots every EXP-018 file; Verify checks its complete preservation. No old preparation-only status is substituted for actual completed results.

## Training-only gradient protocol

Folds 1, 3 and 5; all three seeds; three fixed, nonoverlapping batches where partition size permits. A separate CPU generator seeded `192027 + fold` creates batches without touching training RNG. Diagnostics use evaluation mode, one forward for both losses, and preserve RNG, model mode and parameter gradients. No stopping, outer-validation or locked paintings enter diagnostic batches. All initialization/final/early-refit comparisons use the same full outer-training PCA targets on inner-training pairs. Actual model training uses the original inner PCA during epoch selection and the original full PCA during refit.

Decompose BCE and unscaled cosine gradients on first visual Linear weight/bias, language interaction parameters, and the shared hidden representation. Report BCE norm, text norm, scaled norm/ratio, cosine, combined norm, orthogonal fraction and opposing fraction. Orthogonal fraction is `sqrt(1-cos^2)`; opposing fraction is `max(0,-cos)`, the negative parallel length divided by auxiliary length. Undefined alignment/ratio and unused parameters are explicit. For C0, the auxiliary branch is untrained: its unscaled gradients describe that unused branch, and its actual scaled contribution is zero.

The concatenated normalized input `Z` is a differentiable leaf. Its blocks are CLIP-B 512, SigLIP2-Base 768 and CLIP-L 768. Both L2 norm and per-coordinate RMS are recorded to distinguish block width from gradient strength. These are feature-input gradients, with no graph through any pretrained encoder. Activation/feature gradients are not reported as directly clipped. For parameter gradients, post-clipping components use the single global combined-loss clipping factor, matching training rather than separately clipping each loss.

EXP-018 did not retain early checkpoints. Mark those unavailable; do not retrain C0/C1 to synthesize them. C2 saves a full-refit epoch-5 checkpoint if its selected duration reaches five epochs. Final checkpoint diagnostics never influence stopping or promotion. Telemetry records combined parameter norms before/after clipping, clipping frequency, first-batch update norm, BCE/auxiliary training curves and stopping BCE. An exact optimizer/RNG regression test compares the observational epoch wrapper with the original implementation.

## Frozen decisions

Competition: E2 must gain at least 0.003 pooled Contrast over M0 and 0.002 over E0, win at least three folds over M0, remain within M0 +0.005 bits fold-mean Macro Log Loss, and complete all seeds/isolation/checksum/replay checks. Verify independently replays checkpoints, averaged/rounded predictions, ensembles, pooled/fold-mean metrics and fold wins before classifying. A failed later verification attempt blocks any earlier success. Incomplete execution is READY_FOR_LOCAL_RUN, never COMPETITION_NO_GO.

Research is separate. With longitudinal C2 evidence, RESEARCH_PROMISING requires a higher median scaled gradient ratio at the shared representation, at most 20% of final diagnostics having cosine below -0.5, and C2-C1 pooled Contrast at least +0.002. RESEARCH_UNPROMISING requires no stronger median signal, at least 80% strong conflict, and C2-C1 Contrast at most -0.003. Otherwise report RESEARCH_INCONCLUSIVE. These conservative descriptive criteria were frozen before C2 CV; inspect fold/seed distributions, bootstrap uncertainty and validation curves alongside the classification. None proves successful encoder adaptation or authorizes backbone training.

Local metrics use the existing published-specification reconstruction (`docs/competition.md`). The exact hosted organizer scorer is unavailable in the pinned checkout; this experiment cannot claim hosted-scorer parity. It retains the original Contrast eligibility and base-2 language-macro loss rules. Resource eligibility remains provisional pending organizer clarification.

## Owner workflow and conditional prerequisites

From the project root:

```powershell
.\scripts\run_exp019.ps1 -Stage Preflight
.\scripts\run_exp019.ps1 -Stage GradientPilot
.\scripts\run_exp019.ps1 -Stage CV
.\scripts\run_exp019.ps1 -Stage Verify
.\scripts\run_exp019.ps1 -Stage Confirm
.\scripts\run_exp019.ps1 -Stage Dev
```

Paths are explicit optional runner parameters: `-RunDir`, `-Exp018Dir`, `-Exp017Dir`, `-DataDir`, `-CacheDir`, `-FullTextDir`. The isolated Python environment and original device/runtime must match. Logs include process duration, exit status and available RAM. Separate stages do not automatically chain long work. `-MaxSeconds` overrides the time budget. Exit 75 stops at an epoch boundary and retains successful sealed seeds; an unpublished current seed restarts, never partially resumes optimizer state. Corrupt/partial/incompatible published checkpoints fail validation. CV, Verify and guarded scoring remain separate commands.

Exit 0 means completed stage, 1 validation failure, 2 invocation/environment error, 3 missing prerequisite or blocked promotion gate, 75 resumable budget stop. Confirm and Dev check the latest verified CV decision before opening their labels. Confirmation refits M0, C0 and C2 on the 1,335 development-training paintings, uses the established +0.001 / positive incremental / +0.005 bits confirmation gate, and writes only EXP-019 outputs. A failed confirmation blocks Dev.

Dev requires an already existing, provenance-verified CLIP text cache covering all original training paintings, supplied through `-FullTextDir`. Such a cache is not assumed to exist after EXP-018 NO_GO. EXP-019 never extracts new embeddings to manufacture this prerequisite. If absent, it stops before public-development scoring and reports that precise blocker. Any future cache preparation requires separate scope consistent with organizer rules. No automatic Codabench upload or production modification occurs.

## Execution record

Implementation and lightweight tests are complete. Actual preflight, gradient pilot and measured runtime decision are recorded below after execution. Full C2 CV, competition selection and C2 longitudinal research conclusions remain pending until those stages actually complete.
