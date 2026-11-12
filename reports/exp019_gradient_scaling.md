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

**READY_FOR_LOCAL_RUN.** Preflight and GradientPilot completed with exit 0. Full C2 CV, competition selection, locked confirmation and public Development have not run. There is no candidate archive or EXP-019 competition decision. Research is currently **RESEARCH_INCONCLUSIVE** because C2 longitudinal and held-out evidence is absent. The measured forecast exceeds 15 minutes, so full training is handed off.

### Actual runtime and conservative estimate

| Item | Measured or estimated value |
| --- | ---: |
| Actual Preflight process | 11.91 s |
| Actual GradientPilot process | 19.56 s |
| Python pilot including validation/loading and diagnostics | 13.85 s |
| Training-only diagnostic passes | 135, taking 5.65 s |
| New short C2 pilot | 4 epochs; 0.2642 s mean after warm-up |
| Actual historical C1 selection / refit epochs | 601 / 376 |
| Actual historical C1 cumulative fitting | 703.17 s |
| Last EXP-018 resumed CV invocation, not total fitting | 146.57 s |
| Conservative C2 CV estimate | 1,054.76 s = 17.58 min |
| Diagnostic/scoring/verification allowance | 300 s |
| Estimated total including measured diagnostics | 1,360.41 s = 22.67 min |

The short-epoch forecast alone was 296.82 s. The estimate takes the maximum of that forecast and the historical 703.17 s, applies a 1.5 margin and adds the allowance. This avoids mistaking the faster pilot or a resumed stage's duration for the complete workload. Cache checks and scoring/verification overhead are covered by the allowance. Verify has a separate budget of at least 120 s, increased by 50%. Future stopping at lambda=1 is unknown. Default CV allowance is about 34 minutes; override `-MaxSeconds` if needed. Successful seeds persist after a budget stop.

The four C2 pilot epochs reduced training BCE from 0.561127 to 0.505188, auxiliary cosine loss from 0.936677 to 0.111911, and inner stopping BCE from 0.541489 to 0.518445. These short curves are a correctness/runtime check without a matched short C1 comparison. They do not establish better generalization. No outer CV painting was scored by this pilot.

### Actual gradient observations

Entries are medians of 27 deterministic batches per condition/checkpoint: three folds, three seeds and three batches. Ratios are medians of batch ratios, not ratios of separately aggregated norms. C1/C2 at initialization have identical states and different coefficients.

| Site | C1 initial scaled text/BCE | C2 initial scaled text/BCE | C1 final scaled text/BCE | Initial cosine | C1 final cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| First visual Linear | 0.009378 | 0.093784 | 0.006185 | 0.039508 | 0.060204 |
| Language interaction | 0.043793 | 0.437934 | 0.087986 | 0.000650 | -0.008043 |
| Shared representation | 0.068426 | 0.684258 | 0.005722 | -0.008547 | -0.001740 |
| CLIP-B feature input | 0.066075 | 0.660745 | 0.002654 | -0.002958 | -0.009358 |
| SigLIP2 feature input | 0.068994 | 0.689944 | 0.002667 | -0.002855 | -0.005253 |
| CLIP-L feature input | 0.065194 | 0.651940 | 0.002577 | -0.003967 | -0.008619 |

Initial median first-visual BCE norm was 5.517251 and unscaled text norm 0.532095; scaled text norms are 0.053210 and 0.532095. Shared-representation BCE norm was 0.013220 and unscaled text norm 0.009146. All diagnostics were finite. C0's actual scaled auxiliary contribution is zero; its unused branch still has measurable unscaled gradients.

Directions are predominantly orthogonal, with small signed projections. First-visual cosines span -0.227484 to 0.347919 initially and -0.148282 to 0.211067 in C1 final refits. Final CLIP-L feature cosines span -0.052938 to 0.030389. Shared-representation orthogonal-fraction medians are 0.999963 initially and 0.999990 for final C1. This supports nonredundancy at these interfaces, without proving useful backbone updates.

Initially CLIP-L has the highest median raw auxiliary feature norm, 0.123028, narrowly above SigLIP2 0.122780; CLIP-B is 0.100514. Per-coordinate RMS is essentially equal (about 0.000278). Raw ranking largely reflects dimensionality and does not identify a preferred trainable backbone. Final C1 norms are 0.006287, 0.006167 and 0.004890 respectively. The feature signal shrinks substantially during C1 training, while the language-interaction ratio rises. C2's final/early behavior awaits actual training.

For fold 1 / seed 2027 / batch 1, eval BCE is 0.706186 and auxiliary loss 0.995908. Scaled losses are 0.099591 and 0.995908. Global clipping factors are 0.223412 and 0.221921. Scaling changes the combined clipping factor: tenfold pre-clip auxiliary signal does not imply tenfold updates. Historical clipping/update telemetry is unavailable; C2 records it prospectively. Full decompositions are in `gradient_pilot/gradients.json`; the local derived table is `gradient_pilot_summary.csv`.

### Validation and next execution

Windows tests: **157 passed, 1 existing symlink skip**. Thirteen new tests cover decomposition, missing/zero gradients, exact initialization and optimizer/RNG parity, feature blocks, clipping, deterministic batches, parent prerequisites/source integrity, frozen gates, inference without text, and tiny real Torch CV/replay/resume fixtures. Fixture scores are not actual EXP-019 outcomes; tests download no data or weights.

Preflight preserved **289 EXP-018 files** and checked **291 historical hashes**. Parent geometry, 45 seed checkpoints and prediction manifests match the pinned local result. No existing model, text cache, PCA transform or image feature was regenerated. Python 3.12.10 / Torch 2.7.1+cu126 / RTX 4060 Laptop and package/source signatures are recorded in stage manifests. CLIP-L teacher revision remains `32bd64288804d66eefd0ccbe215aa642df71cc41`; visual checkpoints and production head bundles retain their EXP-018 pins.

Actual production regression re-generated all 849 E50+L, E50 and CLIP probabilities byte for byte using existing cached embeddings and validated all three ZIPs plus the original E50+L archive. No encoder ran. E50+L CSV SHA-256 remains `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`. These are unchanged release regression outputs, not a new EXP-019 candidate or development evaluation.

Next owner command:

```powershell
.\scripts\run_exp019.ps1 -Stage CV
```

Run `-Stage Verify` separately afterward. Only verified PASS permits Confirm; only successful confirmation permits Dev, subject to the documented pre-existing full-text-cache prerequisite. Production E50+L remains the release. A future competition NO-GO will not by itself determine research value.
