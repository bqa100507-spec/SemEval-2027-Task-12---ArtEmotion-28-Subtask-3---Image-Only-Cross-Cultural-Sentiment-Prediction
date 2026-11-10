# TASK 009 E50 inference verification

The original EXP-007 system remains the default. Head preparation refit the two existing Model C regressions on 16,019 eligible pairs from 1,571 original training paintings, with seed 2027 and C=0.3. Both heads converged at the original iteration counts (CLIP 69, SigLIP 2 65). Coefficients and intercepts use non-pickle NPZ serialization. Fitted and reloaded heads gave exactly the same component probabilities, and two independent preparations produced the same head file SHA-256 `7ce9b709757ae9d44f7f50662436fd7ef887f992fec2af43b8907eb66100d17f`.

## Development parity

The general `predict` command read an isolated directory with the public development CSV and its 85 artworks. The directory had no annotations. Both modes reproduced all 849 EXP-007 predictions, in input order, byte for byte.

| Mode | Prediction CSV SHA-256 | New encoder passes |
| --- | --- | ---: |
| E50 | `89591c9ccefe47b570933ab2f7765e3927882d0912ec3288914b6c692e50b0d1` | 0 |
| CLIP | `e4c0fc06ed640f10b702b4b2a95451b1cad4665c86f2cbf170ef2c7c3cce9e47` | 0 |

E50 used 85 verified historical features per encoder. Both archives passed exact-coverage and root-level CSV checks. These are prediction parity checks against the frozen EXP-007 run. Its established local metrics remain 0.460026 Contrast and 0.744214 Macro Log Loss.

## New-image smoke

The fixture used seeded synthetic RGB PNGs. It had 14 prediction rows, 13 image paths and 12 distinct image contents. It included repeated language requests for one painting, a byte-identical alias under another path, and one Kazakh row to test the shared image terms for an official language absent from training.

- On an RTX 4060 Laptop GPU, forcibly stopped the process tree during CLIP extraction after seven vectors had committed. The same command encoded the remaining five CLIP images and proceeded to SigLIP 2.
- Forcibly stopped during SigLIP 2 extraction after seven vectors had committed. The next retry reused all 12 CLIP vectors and seven SigLIP 2 vectors, encoded the remaining five SigLIP 2 images, and published a valid 14-row E50 archive. Neither killed attempt published an output directory.
- A full rerun with a different batch size reused all 12 vectors per encoder and reproduced the prediction CSV byte for byte. A reordered three-row subset reused two image contents per encoder. Replacing one image at the same path caused one new extraction per encoder and changed only that image's prediction.
- Missing artwork, path traversal, duplicate keys, inconsistent painting-to-image mapping and an unrecognized language all returned errors before publication. A missing SigLIP 2 interpreter caused an explicit E50 failure. Explicit CLIP mode could use the same input independently.
- A separate CPU run encoded one image through both real encoders and produced a validated two-row ZIP. For two synthetic images, each new CUDA worker's vectors were byte-identical to the original extraction path at the same batch size.

The resumed CUDA run took 31.78 s and the warm-cache run 20.25 s, including fresh worker process startup. The changed-image run took 46.05 s. These are local smoke timings.

## Checks and artifacts

Local prediction, bundle, and synthetic smoke artifacts are under `runs/e50_inference_smoke/`. The prepared default bundle is under `runs/e50_inference_models/`. No evaluation artwork, labels, or hosted submission were used in the smoke tests. SHA-256 checks confirmed that 205 historical files were unchanged, including 10 feature-cache files and the original E50 package. The Windows unit run passed 89 tests and skipped the symlink fixture because the host does not permit creating a test symlink. Both Python environments passed dependency checks.

See the [Windows runbook](../docs/e50_inference.md) for release-time placement, prediction, resume, validation and fallback commands.
