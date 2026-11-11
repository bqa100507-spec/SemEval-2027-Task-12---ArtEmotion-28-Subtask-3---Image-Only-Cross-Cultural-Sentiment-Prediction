# ST3 competition release

Released on 2026-11-11. The selected system is **E50+L** with project-owner-reported hosted Development Contrast **0.4643**, submission **#1046158**. The measured local development Contrast is 0.46431737014951985 and Macro Log Loss is 0.7409970449028339. [Task 011 verification](../reports/e50l_inference_verification.md)

The production integration main commit is `7a02e63f7cdfdf1e89e92929e3297b6a61b9b508`. Its complete Git tree is identical to Task 011 head `64c38204c31ae9915a5241f057cd8dd49ef349e1`. The subsequent release documentation commit changes README and documentation only. Resolve that delivery commit with `git rev-parse HEAD` after checkout. The execution manifest at `runs/release016/release_manifest.json` records the final delivery SHA on the prepared machine. [PR audit](release016_pr_audit.json)

## Frozen model and data

| Encoder | Immutable revision | Features | Interpreter |
| --- | --- | --- | --- |
| `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | 512-wide normalized FP32 image projection | `.venv` |
| `google/siglip2-base-patch16-224` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | 768-wide normalized FP32 trained attention pool | `.venv-siglip2` |
| `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` | 768-wide normalized FP32 image projection | `.venv` |

All three image encoders are frozen and use their pinned official processors. Encoder weights and processor/config hashes are verified by the prediction command against the bundle manifests and [E50 pins](../configs/e50_inference.json) and [CLIP-L pins](../configs/exp010.json).

Each original LR component uses full image–language interactions, C = 0.3, annotator-count weighting and seed 2027. Component probabilities are rounded to six decimals. First average CLIP-B and SigLIP 2 at 50/50 and round E50 to six decimals. Then average rounded E50 and rounded CLIP-L at 50/50 and round the final value to six decimals.

Training used 16,019 eligible pairs from 1,571 original training paintings, with at least five binary-mappable votes per pair. Painting-level train/development isolation is checked. The organizer revision is `cf7cc444cf56617f3b8417497c6930ad172c5c99` and the original annotation CSV SHA-256 is `070194c3f786a8116e1a02bbe30bea8ac1d4f42404b4a56649bc667a489b7787`. [Data provenance](../reports/data_audit.md) · [Pinned organizer configuration](../configs/st3.json)

Use Python 3.12. The prepared machine uses Python 3.12.10, PyTorch 2.7.1+cu126 and Transformers 4.49.0 in `.venv`. SigLIP 2 uses the existing `.venv-siglip2` overlay with Transformers 4.51.3 over the same pinned runtime. Both environments passed `pip check`. For a fresh installation, follow [workspace setup](../README.md#1-set-up-the-workspace) and [bundle preparation](e50l_inference.md#one-time-preparation).

## Prepared head bundle checksums

Keep the three files in each bundle together. These SHA-256 digests identify the prepared release bundles.

| File | SHA-256 |
| --- | --- |
| `runs/e50_inference_models/heads.npz` | `7ce9b709757ae9d44f7f50662436fd7ef887f992fec2af43b8907eb66100d17f` |
| `runs/e50_inference_models/legacy-index.json` | `35759e1aa427c46e42f1899c496e792ea41434174a4830d254c4e2c34df01748` |
| `runs/e50_inference_models/manifest.json` | `743c7b24aa50fb62dc3b0427e143e5c4410ff2bb27294ba21c3f23423aef89bd` |
| `runs/e50l_inference_models/heads.npz` | `4d3a213db3c9ec86e16d56f82566ab5f715fb4905d0e946b8a32920da9a13a42` |
| `runs/e50l_inference_models/legacy-index.json` | `34052e5388e8b660f6f9a4a5efa465e334600284d67abb3c375e8b5af33908ad` |
| `runs/e50l_inference_models/manifest.json` | `76a0d74f41f9b27c74b0a9974575b5ba4b5fa9cca830a4cbe3ad9449924f0330` |

## Evaluation-day runbook

Run from the project root after checking out the release delivery SHA. Put the official evaluation CSV and artwork under the separate `data/eval_st3/` directory, with artwork paths matching the CSV. Use a fresh output directory. The command reads only the input CSV, artwork, prepared bundles and encoder caches.

```powershell
$py = ".venv/Scripts/python.exe"
$data = "data/eval_st3"
$csv = "$data/st3_test.csv"
$out = "runs/e50l_eval_release016"

& $py scripts/run_st3.py predict --input $csv --data-dir $data --output-dir $out
if ($LASTEXITCODE -ne 0) { throw "E50+L inference failed" }
& $py scripts/run_st3.py validate-prediction --input $csv "$out/submission.zip"
if ($LASTEXITCODE -ne 0) { throw "Submission validation failed" }
Get-FileHash "$out/predictions.csv", "$out/submission.zip"
```

The default mode is `e50l`. On interruption, repeat the same prediction command and output path. Completed batches persist by image-content hash, and the final output is published after success. Preserve `run.json` with the CSV and ZIP. CUDA is selected automatically; `--device cpu` selects CPU. `--clip-large-batch-size 8` reduces the CLIP-L batch if needed. Review the input coverage and ZIP validation result, then upload the ZIP manually.

Explicit fallbacks use separate output paths and the original E50 bundle.

```powershell
& $py scripts/run_st3.py predict --mode e50 --input $csv --data-dir $data --output-dir runs/e50_eval_release016
if ($LASTEXITCODE -ne 0) { throw "E50 inference failed" }
& $py scripts/run_st3.py validate-prediction --input $csv runs/e50_eval_release016/submission.zip

& $py scripts/run_st3.py predict --mode clip --input $csv --data-dir $data --output-dir runs/clip_eval_release016
if ($LASTEXITCODE -ne 0) { throw "CLIP inference failed" }
& $py scripts/run_st3.py validate-prediction --input $csv runs/clip_eval_release016/submission.zip
```

The validator checks the single top-level `predictions.csv`, exact column order, complete pair coverage, duplicate keys and finite probabilities in [0, 1]. Bundle and encoder pins are verified during prediction. Use `Get-FileHash` on bundle files to compare with the table above. [Detailed inference and retry instructions](e50l_inference.md)

## Preserved research history

The remote branch `archive/exp001-015-verified` points to `cd321f142142120c17c495458390deb6574ad856`. Every original PR head is reachable from this archive. PRs #1–11 were merged into main with ordinary merge commits in dependency order, with successful head checks and exact original-head tree parity after every merge. PRs #12–15 were closed after verifying their ancestry and fetching the archived reports through GitHub. Each closure has an explanatory discussion comment. [Complete archive](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/tree/cd321f142142120c17c495458390deb6574ad856)

| Archived study | Decision | Exact report |
| --- | --- | --- |
| EXP-012 | Residual corrections failed the frozen CV gate | [Residual calibration](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/cd321f142142120c17c495458390deb6574ad856/reports/exp012_language_residual.md) |
| EXP-013 | CV-selected blend fell to 0.457525 development Contrast versus 0.464317 for E50+L | [Nonlinear head](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/cd321f142142120c17c495458390deb6574ad856/reports/exp013_nonlinear_head.md) |
| EXP-014 | +0.000891 pooled CV Contrast fell below +0.003 | [CLIP-L LoRA](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/cd321f142142120c17c495458390deb6574ad856/reports/exp014_clip_large_lora.md) |
| EXP-015 | M2 +0.000565 and M3 +0.000473 fell below their promotion thresholds | [SigLIP 2 Large](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/cd321f142142120c17c495458390deb6574ad856/reports/exp015_siglip2_large.md) |

The complete branch list is retained. The pre-release inventory covers 1,057 local files under runs and cache, totaling 15,851,297,675 bytes. All 1,057 post-verification hashes matched. Checkpoint weights and experiment artifacts stay in their existing ignored directories. Release verification records are under `runs/release016/`.

## Release regression results

The complete production suite passed 104 tests with one Windows symlink fixture skipped. Both Python environments passed their dependency checks. GitHub CI passed on each of the eleven main merge commits.

Each production mode replayed all 849 development pairs byte for byte and produced a coverage-validated ZIP. The E50 and CLIP regressions used an absent CLIP-L bundle path to exercise their explicit fallback dependencies.

| Mode | Development CSV SHA-256 |
| --- | --- |
| Default E50+L | `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3` |
| Explicit E50 | `89591c9ccefe47b570933ab2f7765e3927882d0912ec3288914b6c692e50b0d1` |
| Explicit CLIP | `e4c0fc06ed640f10b702b4b2a95451b1cad4665c86f2cbf170ef2c7c3cce9e47` |

The original E50+L ZIP still validates and has SHA-256 `38c2eca8ee4c0557bf198bf5df5dbf90571a2851404ded6302f40e42dc138e0d`.

A fresh synthetic CUDA fixture had 20 pairs, 19 paths and 18 distinct image contents, including an alias and an unseen-training language. The release smoke stopped its own process tree after four CLIP-L features committed. The retry reused all 18 CLIP-B and SigLIP 2 features plus those four CLIP-L features, extracted the remaining 14 CLIP-L features and published a validated archive. A warm replay encoded zero images and reproduced its CSV exactly. The smoke took 126.86 s including the interruption, with 35.21 s for the retry and 14.31 s for the warm run. This verifies fresh-image inference and resume behavior on synthetic inputs.
