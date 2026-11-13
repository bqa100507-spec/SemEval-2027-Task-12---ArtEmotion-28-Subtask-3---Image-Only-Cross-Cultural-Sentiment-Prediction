# Project state

| Component | Status |
| --- | --- |
| Public dataset | Pinned ST3 checkout and local integrity audit complete |
| Baseline | Full CUDA development run complete. Cached rerun byte-identical. CPU embedding smoke passed |
| Local evaluation | 0.446564 Contrast and 0.751995 macro Log Loss for CLIP + LR |
| Submission | `runs/baseline_dev/submission.zip` validated against all 849 public development keys |
| Tests | 26 lightweight tests passed locally. GitHub Actions workflow included |

Run `py scripts/setup.py` on Windows, then use the virtual environment for `scripts/run_st3.py baseline`, `scripts/run_st3.py submission`, `scripts/run_st3.py validate runs/baseline_dev/submission.zip`, and `python -m pytest -q`. See the [README](../README.md) for executable paths and standalone audit and evaluation commands.

The experiment uses the pinned public development split and a sparse representation of the official features. See the [data audit](../reports/data_audit.md) for coverage and [baseline report](../reports/baseline_dev.md) for measured scores and reproducibility checks.

The project owner reported 0.4466 Contrast for Codabench development submission `#1044742`, matching the local Task 001 score at four decimal places.

EXP-002 completed the shared-weights ablation, 20 inference-time image shuffles, a 10,000-draw paired painting bootstrap, and five train-only grouped folds. A and C reproduced Task 001. Shared weights scored 0.370386 Contrast and 0.766309 Macro Log Loss. Interactions gained +0.076178 Contrast with a 95% paired percentile interval of [+0.033904, +0.118978]. The study ran from cached embeddings on CPU. [EXP-002 report and commands](../reports/exp002_ablation.md)

EXP-003 completed ranks 2, 4, 8 and 16 under two seeds on the exact five EXP-002 folds, plus matched-rank SVD compression. Every directly trained rank exceeded the 0.748056 Macro Log Loss guardrail, so C remains the baseline and its development results are unchanged. The full interaction effective rank was 20.6235. SVD-16 came closest to C in grouped CV. [EXP-003 study and artifacts](../reports/exp003_low_rank.md)

EXP-004 tested painting-balanced pairwise Huber and smoothed Pearson losses on top of full Model C using the exact EXP-002 folds. R-small led at 0.437539 Contrast and 0.737824 Macro Log Loss. Its +0.001834 mean Contrast gain fell below the frozen +0.005 threshold. The study retained C and its established development scores. [EXP-004 study](../reports/exp004_contrast_aware.md)

EXP-005 compared CLIP with frozen DINOv2-base CLS features and fold-local PCA to 512 dimensions. CLIP reproduced EXP-002 and beat both DINO conditions on Contrast and Macro Log Loss in every fold. DINO768 scored 0.418597 Contrast and 0.781704 Log Loss, versus CLIP's 0.435705 and 0.738056. The frozen decision retained C-CLIP. [EXP-005 rules, provenance and results](../reports/exp005_visual_representation.md)

EXP-006 completed the final predefined single-backbone comparison. Frozen SigLIP 2 scored 0.439767 Contrast and 0.735555 Macro Log Loss, winning four of five folds. Its +0.004062 mean Contrast gain fell short of the frozen +0.005 gate. The development fallback retained C-CLIP. [EXP-006 report](../reports/exp006_siglip2.md)

The project now prioritizes reliable competition Contrast, with concise paper records. [EXP-001–006 synthesis](../reports/research_synthesis_001_006.md)

EXP-007 selected the fixed 50/50 CLIP and SigLIP 2 blend E50. It scored 0.442055 pooled CV Contrast and 0.460026 development Contrast, with development Macro Log Loss 0.744214. Its separate development ZIP is validated at `runs/exp007_fusion/dev/submission.zip` for manual upload. C-CLIP remains available as the lower-cost fallback. [EXP-007 comparison](../reports/exp007_fusion.md)

EXP-008 audited crop coverage without labels and compared two frozen CLIP views while reusing the original E50 SigLIP 2 probabilities. Full-image and Dual pooled CV Contrast were 0.437825 and 0.441386, respectively. Neither cleared the promotion gate, so E50 remains the competition system. The project owner reported hosted Development Contrast 0.4600 for submission `#1045921`. [EXP-008 audit and comparison](../reports/exp008_full_image.md)

TASK 009 persisted the frozen E50 heads, added label-free prediction and explicit CLIP fallback, and verified exact development CSV parity. A new per-image cache supports batch commits and resumes for both encoders. [Evaluation workflow](e50_inference.md) · [Verification report](../reports/e50_inference_verification.md)

EXP-010 selected the fixed E50+L candidate on the original train-only folds. It scored 0.445643 pooled CV Contrast and 0.464317 development Contrast, with development Macro Log Loss 0.740997. Its separate development ZIP is validated at `runs/exp010_clip_large/dev/submission.zip` for manual upload. E50 remains the hosted-verified production system. [EXP-010 comparison](../reports/exp010_clip_large.md)

TASK 011 made E50+L the default production mode after the project owner reported 0.4643 hosted Contrast on submission `#1046158`. Its separate CLIP-L head bundle and the independent prediction command reproduced all 849 EXP-010 development rows byte for byte. E50 and CLIP remain explicit fallbacks. [Windows inference commands](e50l_inference.md) · [Verification report](../reports/e50l_inference_verification.md)

TASK 016 merged PRs #1–11 into main in dependency order with ordinary merge commits. The production integration tree exactly matched Task 011. The full EXP-001–015 history is remotely preserved at archive head `cd321f142142120c17c495458390deb6574ad856`; PRs #12–15 were closed after verifying remote report access and ancestry. [Release manifest](competition_release.md) · [PR audit](release016_pr_audit.json)

EXP-017 completed the matched BCE-only, CLIP-caption and provisional E5-caption comparison on five new folds over 1,335 original training paintings. The CLIP ensemble gained 0.002656 pooled Contrast over refitted E50+L and 0.001168 over its matched BCE-only blend, below both frozen requirements. E5 also fell short. The decision retained E50+L, with the 236-painting confirmation set and public development unscored. [Caption-supervision results](../reports/exp017_text_supervision.md)

EXP-018 was subsequently completed locally by the owner. Task 019 validated all 45 seed checkpoints and the successful deep replay. B3 pooled Contrast is 0.449566497, a gain of 0.002468790 over M0; the frozen decision is NO_GO. Locked confirmation and public Development remain unscored. The EXP-018 preparation report and all local files are preserved.

EXP-019 was completed locally and independently replayed: fifteen new checkpoints, COMPETITION_NO_GO. Lambda=1 student pooled Contrast was 0.417569215 versus 0.440121979 for lambda=0.1. Its ensemble scored 0.446633867. Task 020 verified the actual local manifests, source/runtime, predictions and decision. Confirmation and public Development remained sealed; the old preparation report and all local files are preserved.

EXP-020 completed locally and was independently replayed: thirty checkpoints and ten target partitions, COMPETITION_NO_GO. E8 pooled Contrast was 0.439654679 versus A8 0.440223388; ensembles were 0.449189301 and 0.449334031. Task 021 verified actual manifests/source/decisions before implementation. Prior preparation reports and all artifacts remain intact; locked/public scoring stayed sealed.

EXP-021 is READY_FOR_LOCAL_RUN. Annotation-only eight-way counts reconstruct all 13,600 eligible training pairs exactly. The reliability audit found 90.29% with n=5, 34.53% tied language comparisons and 51.52% posterior confidence below 0.80 under the stated descriptive model. D0/D8 matched pilot passed; auxiliary CE decreases with finite feature/head gradients, without caption inputs or consistency loss. Conservative eight-fit Screen + preparation/replay estimate: 25.15 minutes. No real Screen/CV decision exists. Next: `.\scripts\run_exp021.ps1 -Stage Screen`, then ScreenVerify. Only verified PROMISING permits manual full CV, reusing eight checkpoints. No Confirm/Dev/LoRA implementation. Windows 178 passed / 1 skip. [Audit, matched objective and execution boundary](../reports/exp021_vote_distribution.md)


EXP-022 verified EXP-021's actual completed CV/Verify before implementation: all thirty checkpoints replayed, COMPETITION_NO_GO, D0 0.437792268 / D8 0.441591226 / ED0 0.448585747 / ED8 0.449920217 pooled Contrast on the same historical five-fold population. The earlier EXP-021 preparation record above is preserved. EXP-022 adds a four-condition matched visual LoRA screen, real-image parity/memory/gradient preflight, bounded pilot, independent replay and a guarded owner-only CV stage. Actual execution and next command are recorded in the [EXP-022 report](../reports/exp022_visual_lora_vote_distribution.md). No production promotion or locked/public scoring is authorized.


EXP-022 execution: RTX 4060 preflight and one complete grouped pilot epoch for each F0/F8/L0/L8 passed. Live/cache error 2.384e-7; zero LoRA exactly matches the original model; real BCE/CE gradients and adapter updates verified. LoRA throughput 45.89/46.92 paintings/s, peak CUDA allocated/reserved 546,871,808/616,562,688 bytes. Forecast MiniScreen 81.06 min / Screen 5.4043 GPU-hours exceeds automatic/local limits: no real screen or CV fit was launched, no held-out effect or screening decision exists. Validated private Colab input ZIP prepared (705,080,786 bytes); next step is the checksum-pinned portable Preflight/Pilot in the report. All 2,483 historical files and exact E50+L prediction SHA match. Windows 197 passed/1 skipped; source PR CI 196 passed/2 skipped. Executed source f0d2ac65d875a0875e8b280bc6065545611198e3; unmerged PR #21 against EXP-021.


EXP-022-N adds thin Colab and Kaggle notebooks over frozen source 41ca0d4, with separately checksum-pinned orchestration, benchmark-only Run all, explicit long-stage controls, strict runtime identity transfer, private checkpoint exports and an optional offline Kaggle kit. No scientific implementation or historical artifact changed and no cloud GPU experiment ran. Native validation: 217 passed / 1 skipped; both notebook schemas and mocked Run all flows passed, and the original data ZIP was independently extracted/verified. Cloud runtime compatibility and real offline installation remain owner checks. See the [workflow validation record](../reports/exp022_notebooks.md), [Colab](exp022_colab.md) and [Kaggle](exp022_kaggle.md) runbooks.


EXP-022-N corrected GitHub CI passed notebook schemas and 216 tests / 2 skips after fetching the pinned helper history; no scientific pin check was weakened. Final read-only checks matched all 2,483 historical artifact/cache hashes and the exact E50+L CSV. No Colab/Kaggle GPU compatibility result or new held-out model score exists.


EXP-022-N final notebook snapshot 6da257fc508a4039f8881b7ddb1e7f5068ecca2b pins helper b18991733a6ec60493e35985fcaff945a164d9f4, including a simulated older-host hashing API regression while retaining training Python 3.12.10. Final validation: 218 passed / 1 skipped locally; notebook schemas and 217 passed / 2 skipped in GitHub CI. Colab launch links and Kaggle downloads point to this immutable notebook snapshot. Cloud GPU compatibility and a real Linux offline-kit installation remain explicitly untested.
